# nyx/location/anchors.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

@dataclass
class GeoAnchor:
    lat: Optional[float] = None
    lon: Optional[float] = None
    neighborhood: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    label: Optional[str] = None

def _deg_box(lat: float, lon: float, km: float) -> Tuple[float, float, float, float]:
    dlat = km / 111.0
    dlon = km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    west = lon - dlon
    east = lon + dlon
    north = lat + dlat
    south = lat - dlat
    return west, north, east, south

# Real-world anchor fallbacks / aliases
_ALIAS: Dict[str, GeoAnchor] = {
    "velvet sanctum": GeoAnchor(
        neighborhood="SoMa", city="San Francisco", region="CA", country="USA",
        lat=37.7810, lon=-122.4040, label="SoMa, San Francisco"
    ),
}

# Airports you likely want near SF (fast path; OSM fallback still used)
_AIRPORTS = {
    "SFO": {"name": "San Francisco International Airport", "lat": 37.6213, "lon": -122.3790},
    "OAK": {"name": "Oakland International Airport", "lat": 37.7126, "lon": -122.2197},
    "SJC": {"name": "San José International Airport", "lat": 37.3639, "lon": -121.9289},
}

def derive_geo_anchor(meta: Dict[str, Any]) -> GeoAnchor:
    a = GeoAnchor()
    li = meta.get("locationInfo") or {}
    ss = meta.get("scene_scope") or {}
    # 1) explicit lat/lon
    for scope in (li.get("geo") or {}, ss):
        try:
            if a.lat is None and "lat" in scope and "lon" in scope:
                a.lat = float(scope["lat"]); a.lon = float(scope["lon"])
        except Exception:
            pass
    # 2) admin strings
    for scope in (li, ss):
        for k_src, k_dst in [
            ("neighborhood", "neighborhood"), ("district", "neighborhood"),
            ("city", "city"), ("region", "region"), ("state", "region"),
            ("country", "country")
        ]:
            v = scope.get(k_src)
            if isinstance(v, str) and getattr(a, k_dst) is None:
                setattr(a, k_dst, v.strip())
    # 3) alias by scene name/display
    for name in [(ss.get("location_name") or ss.get("scene_name") or ""),
                 (li.get("display") or "")]:
        key = name.strip().lower()
        if key in _ALIAS:
            return _ALIAS[key]
    # label
    if a.neighborhood and a.city:
        a.label = f"{a.neighborhood}, {a.city}"
    elif a.city:
        a.label = a.city
    return a

def build_nominatim_params_for_poi(poi: str, meta: Dict[str, Any], radius_km: float = 3.0, limit: int = 5) -> Dict[str, str]:
    a = derive_geo_anchor(meta)
    brand = (poi or "").replace("’", "'").strip()
    params = {"format": "jsonv2", "limit": str(limit), "addressdetails": "1"}
    if a.lat is not None and a.lon is not None:
        w, n, e, s = _deg_box(a.lat, a.lon, km=radius_km)
        params["q"] = brand
        params["viewbox"] = f"{w:.6f},{n:.6f},{e:.6f},{s:.6f}"
        params["bounded"] = "1"
    else:
        tail = a.label or a.city or ""
        params["q"] = f"{brand}, {tail}".strip(", ")
    return params

def nearest_airport_label(meta: Dict[str, Any]) -> Tuple[str, float, float]:
    # Prefer SFO for SF setting; could do distance calc if you like
    a = _AIRPORTS["SFO"]
    return a["name"], a["lat"], a["lon"]
