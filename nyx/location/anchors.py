# nyx/location/anchors.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import httpx
from nyx.conversation.snapshot_store import ConversationSnapshotStore

@dataclass
class GeoAnchor:
    lat: Optional[float] = None
    lon: Optional[float] = None
    neighborhood: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    label: Optional[str] = None  # human-friendly

_store = ConversationSnapshotStore()

def _deg_box(lat: float, lon: float, km: float) -> Tuple[float, float, float, float]:
    dlat = km / 111.0
    dlon = km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    return (lon - dlon, lat + dlat, lon + dlon, lat - dlat)

async def _geocode_city_once(city: str, region: Optional[str], country: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    # Resolve an admin area to a point (centroid-like). Cached in snapshot_store.
    q = ", ".join([p for p in [city, region, country] if p])
    if not q.strip():
        return None, None
    headers = {"User-Agent": "nyx/worldsense/1.0"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "jsonv2", "limit": "1", "addressdetails": "1"},
            headers=headers,
        )
        r.raise_for_status()
        data = r.json() or []
        if isinstance(data, list) and data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None

async def derive_geo_anchor(meta: Dict[str, Any], user_id: str = "0", conversation_id: str = "0") -> GeoAnchor:
    """
    Build a robust anchor with lat/lon. Never rely on fictional scene names.
    Priority:
      1) incoming meta.lat/lon
      2) last persisted anchor from snapshot_store
      3) city/region/country (geocode once, cache)
    """
    snap = _store.get(str(user_id), str(conversation_id))
    a = GeoAnchor()

    # 1) explicit lat/lon in meta
    li = meta.get("locationInfo") or {}
    ss = meta.get("scene_scope") or {}
    for scope in (li.get("geo") or {}, ss):
        try:
            if a.lat is None and "lat" in scope and "lon" in scope:
                a.lat = float(scope["lat"]); a.lon = float(scope["lon"])
        except Exception:
            pass

    # 2) try snapshot
    if a.lat is None and isinstance(snap, dict):
        try:
            if "lat" in snap and "lon" in snap:
                a.lat = float(snap["lat"]); a.lon = float(snap["lon"])
        except Exception:
            pass

    # 3) admin names from meta
    for scope in (li, ss, meta.get("world") or {}):
        for src, dst in [
            ("neighborhood","neighborhood"), ("district","neighborhood"),
            ("city","city"), ("region","region"), ("state","region"),
            ("country","country")
        ]:
            v = scope.get(src)
            if isinstance(v, str) and getattr(a, dst) is None:
                setattr(a, dst, v.strip())

    # 4) geocode city if still no lat/lon
    if a.lat is None or a.lon is None:
        city = a.city or (meta.get("world") or {}).get("primary_city")
        if city:
            lat, lon = await _geocode_city_once(city, a.region, a.country)
            a.lat, a.lon = lat, lon
            # persist to snapshot for quick reuse
            if lat is not None and lon is not None:
                snap = dict(snap or {})
                snap["lat"], snap["lon"] = lat, lon
                if a.city: snap["city"] = a.city
                _store.put(str(user_id), str(conversation_id), snap)

    # label for UI
    if a.neighborhood and a.city:
        a.label = f"{a.neighborhood}, {a.city}"
    elif a.city:
        a.label = a.city

    return a

def build_nominatim_params_for_poi(poi: str, anchor: GeoAnchor, radius_km: float = 3.0, limit: int = 5) -> Dict[str, str]:
    """
    Critical behavior:
      - If we have lat/lon: ALWAYS use a bounding box. Do NOT append fictional labels.
      - If we have no lat/lon but have a city: q = "<poi>, <city>".
      - Otherwise: q = "<poi>" (last resort).
    """
    params = {"format": "jsonv2", "limit": str(limit), "addressdetails": "1"}
    brand = (poi or "").replace("’", "'").strip()
    if anchor.lat is not None and anchor.lon is not None:
        w, n, e, s = _deg_box(anchor.lat, anchor.lon, km=radius_km)
        params["q"] = brand
        params["viewbox"] = f"{w:.6f},{n:.6f},{e:.6f},{s:.6f}"
        params["bounded"] = "1"
    elif anchor.city:
        params["q"] = f"{brand}, {anchor.city}"
    else:
        params["q"] = brand
    return params

def nearest_airport_label(anchor: GeoAnchor) -> Tuple[str, float, float]:
    """
    No hardcoded airport tables. Prefer the closest major airport via simple heuristics:
    - If anchor is near SF (rough bbox), choose SFO coords as a known good default.
    - Otherwise return a neutral placeholder near the anchor (so flight plans still work).
    """
    if anchor.lat and anchor.lon:
        # quick SF bbox
        if 37.60 <= anchor.lat <= 37.90 and -122.55 <= anchor.lon <= -122.20:
            return "San Francisco International Airport", 37.6213, -122.3790
        # generic “nearest airport” placeholder ~10km away; world director can refine
        return "Nearest International Airport", anchor.lat + 0.09, anchor.lon + 0.09
    return "Nearest International Airport", 0.0, 0.0
