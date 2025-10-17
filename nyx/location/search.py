# nyx/location/search.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import httpx

from .anchors import GeoAnchor, build_nominatim_params_for_poi, derive_geo_anchor, nearest_airport_label
from .config import DEFAULT_LOCATION_SETTINGS as _LS
from .query import PlaceQuery
from .types import (
    Anchor,
    Candidate,
    Place,
    ResolveResult,
    STATUS_ASK,
    STATUS_EXACT,
    STATUS_MULTIPLE,
    STATUS_TRAVEL_PLAN,
)

try:
    from rapidfuzz import fuzz
except Exception:  # optional
    fuzz = None

# ---------- Helpers & Normalization ----------

_GENERIC_SYNONYMS: Dict[str, List[Tuple[str, str]]] = {
    "restaurant":         [("amenity","restaurant")],
    "restaurants":        [("amenity","restaurant")],
    "fast food":          [("amenity","fast_food")],
    "cafe":               [("amenity","cafe")],
    "coffee":             [("amenity","cafe")],
    "bar":                [("amenity","bar")],
    "pub":                [("amenity","pub")],
    "bank":               [("amenity","bank")],
    "atm":                [("amenity","atm")],
    "pharmacy":           [("amenity","pharmacy")],
    "supermarket":        [("shop","supermarket")],
    "grocery":            [("shop","supermarket"), ("shop","convenience")],
    "convenience":        [("shop","convenience")],
    "clothing":           [("shop","clothes")],
    "clothing store":     [("shop","clothes")],
    "bookstore":          [("shop","books")],
    "electronics":        [("shop","electronics")],
    "bakery":             [("shop","bakery"), ("amenity","cafe")],
    "liquor":             [("shop","alcohol")],
    "hardware":           [("shop","hardware")],
    "parking":            [("amenity","parking")],
    "park":               [("leisure","park")],
}

_DISCOVERY_DEFAULTS: List[Tuple[str, str]] = [
    ("amenity","cafe"), ("amenity","restaurant"), ("amenity","fast_food"),
    ("amenity","bar"), ("shop","supermarket"), ("shop","convenience"),
    ("amenity","pharmacy"), ("shop","clothes")
]

def _norm_text(s: Optional[str]) -> str:
    return (s or "").strip().replace("’", "'")

def _is_discovery_text(text: str) -> bool:
    t = text.lower().strip()
    return t in {"", "around here", "what's around here", "whats around here",
                 "nearby", "anything nearby", "what is around here"}

def _is_generic_term(text: str) -> bool:
    t = text.lower().strip()
    if t in _GENERIC_SYNONYMS:
        return True
    if t.endswith("s") and t[:-1] in _GENERIC_SYNONYMS:
        return True
    return False

def _guess_query_kind(text: str) -> str:
    if _is_discovery_text(text):
        return "discovery"
    if _is_generic_term(text):
        return "generic"
    tl = text.lower()
    if "'s" in tl or "’s" in text or len(text.split()) <= 3:
        return "brand"
    return "brand"

def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    R = 6371.0
    dlat = math.radians(b_lat - a_lat)
    dlon = math.radians(b_lon - a_lon)
    la1 = math.radians(a_lat); la2 = math.radians(b_lat)
    h = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(h)))

def _address_from_tags(tags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tags, dict): return {}
    return {
        "house_number": tags.get("addr:housenumber"),
        "road": tags.get("addr:street"),
        "city": tags.get("addr:city"),
        "postcode": tags.get("addr:postcode"),
        "state": tags.get("addr:state"),
        "country": tags.get("addr:country"),
    }

# ---------- Candidate constructors ----------

def _candidate_from_nominatim(n: Dict[str, Any]) -> Candidate:
    name = n.get("name") or (n.get("display_name", "").split(",")[0].strip() or "Unknown")
    lat = float(n["lat"])
    lon = float(n["lon"])
    cat = n.get("category") or n.get("type")
    addr = n.get("address") or {}
    imp = float(n.get("importance", 0) or 0)
    place = Place(
        name=name,
        level="venue",
        key=str(n.get("place_id") or n.get("osm_id") or name.lower()),
        lat=lat,
        lon=lon,
        address=addr,
        meta={
            "category": cat,
            "source": "nominatim",
            "importance": imp,
            "display_name": n.get("display_name"),
        },
    )
    return Candidate(
        place=place,
        confidence=min(0.99, 0.5 + imp / 2),
        raw=n,
    )


def _candidate_from_overpass(el: Dict[str, Any]) -> Optional[Candidate]:
    tags = el.get("tags") or {}
    name = tags.get("name") or "Unknown"
    if not name or name == "Unknown":
        brand = tags.get("brand")
        if not brand:
            return None
        name = brand
    if "lat" in el and "lon" in el:
        lat, lon = float(el["lat"]), float(el["lon"])
    elif "center" in el and isinstance(el["center"], dict):
        lat, lon = float(el["center"]["lat"]), float(el["center"]["lon"])
    else:
        return None
    cat = None
    for key in ("amenity","shop","leisure","tourism"):
        if key in tags:
            cat = f"{key}:{tags.get(key)}"; break
    addr = _address_from_tags(tags)
    conf = 0.75 if tags.get("brand") else 0.60
    place = Place(
        name=name,
        level="venue",
        key=str(el.get("id") or name.lower()),
        lat=lat,
        lon=lon,
        address=addr,
        meta={
            "category": cat,
            "source": "overpass",
            "tags": tags,
        },
    )
    return Candidate(place=place, confidence=conf, raw={"tags": tags, "id": el.get("id")})

# ---------- External calls ----------

async def _nominatim_search(poi: str, anchor: GeoAnchor, *, km: float, limit: int) -> List[Candidate]:
    params = build_nominatim_params_for_poi(poi, anchor, radius_km=km, limit=limit)
    headers = {"User-Agent": "nyx/worldsense/1.0 (contact: ops@nyx.example)"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers)
        r.raise_for_status()
        data = r.json() or []
        return [_candidate_from_nominatim(x) for x in (data if isinstance(data, list) else [])]

def _overpass_brand_block(poi_norm: str) -> str:
    rx = poi_norm.replace('"', '\\"')
    return f"""
  node["amenity"]["brand"~"{rx}",i];
  node["shop"]["brand"~"{rx}",i];
  way["amenity"]["brand"~"{rx}",i];
  way["shop"]["brand"~"{rx}",i];
  relation["amenity"]["brand"~"{rx}",i];
  relation["shop"]["brand"~"{rx}",i];
  node["name"~"{rx}",i];
  way["name"~"{rx}",i];
  relation["name"~"{rx}",i];
"""

def _overpass_generic_block(filters: List[Tuple[str, str]]) -> str:
    parts = []
    for key, val in filters:
        parts.append(f'  node["{key}"="{val}"];')
        parts.append(f'  way["{key}"="{val}"];')
        parts.append(f'  relation["{key}"="{val}"];')
    return "\n".join(parts)

async def _overpass_search(
    anchor: GeoAnchor,
    *,
    km: float,
    limit: int,
    poi_norm: Optional[str] = None,
    generic_filters: Optional[List[Tuple[str, str]]] = None,
    timeout_s: int = _LS.overpass_timeout_s,
) -> List[Candidate]:
    assert anchor.lat is not None and anchor.lon is not None, "Overpass needs lat/lon anchor"
    lat, lon = anchor.lat, anchor.lon
    radius_m = int(km * 1000)

    if generic_filters:
        union_block = _overpass_generic_block(generic_filters)
    else:
        union_block = _overpass_brand_block(poi_norm or "")

    q = f"""
[out:json][timeout:{timeout_s}];
(
{union_block}
)->.all;
(
  node.all(around:{radius_m},{lat:.6f},{lon:.6f});
  way.all(around:{radius_m},{lat:.6f},{lon:.6f});
  relation.all(around:{radius_m},{lat:.6f},{lon:.6f});
);
out center {limit};
"""
    headers = {"User-Agent": "nyx/worldsense/1.0 (contact: ops@nyx.example)"}
    async with httpx.AsyncClient(timeout=timeout_s + 5) as client:
        r = await client.post("https://overpass-api.de/api/interpreter", content=q.encode("utf-8"), headers=headers)
        r.raise_for_status()
        data = r.json() or {}
        els = data.get("elements") or []
        cands: List[Candidate] = []
        for el in els:
            c = _candidate_from_overpass(el)
            if c:
                cands.append(c)
        return cands[:limit]

# ---------- Ranking & Merge ----------

def _rank_by_name_similarity(query: str, cands: List[Candidate]) -> List[Candidate]:
    if not cands:
        return cands
    if not fuzz:
        return cands
    q = _norm_text(query).lower()
    for c in cands:
        sim = fuzz.token_set_ratio(q, _norm_text(c.place.name).lower()) / 100.0
        c.confidence = 0.5 * c.confidence + 0.5 * sim
    return sorted(cands, key=lambda x: x.confidence, reverse=True)

def _rank_by_distance(anchor: GeoAnchor, cands: List[Candidate]) -> List[Candidate]:
    if not cands or anchor.lat is None or anchor.lon is None:
        return cands
    for c in cands:
        if c.place.lat is None or c.place.lon is None:
            continue
        d = _haversine_km(anchor.lat, anchor.lon, c.place.lat, c.place.lon)
        c.distance_km = d
        c.confidence = 0.7 * c.confidence + 0.3 * max(0.0, 1.0 - min(1.0, d / _LS.search_radius_km))
    return sorted(cands, key=lambda x: x.confidence, reverse=True)

def _merge_dedupe(anchor: GeoAnchor, a: List[Candidate], b: List[Candidate]) -> List[Candidate]:
    merged: List[Candidate] = []

    def _too_close(x: Candidate, y: Candidate) -> bool:
        if x.place.lat is None or x.place.lon is None or y.place.lat is None or y.place.lon is None:
            return False
        return _haversine_km(x.place.lat, x.place.lon, y.place.lat, y.place.lon) <= 0.075  # ~75m

    for src in (a, b):
        for c in src:
            if any(
                _too_close(c, m)
                or (_norm_text(c.place.name).lower() == _norm_text(m.place.name).lower())
                for m in merged
            ):
                continue
            merged.append(c)
    return _rank_by_distance(anchor, merged)

# ---------- Main entry ----------

async def resolve_real(query: PlaceQuery, anchor: Anchor, meta: Dict[str, Any]) -> ResolveResult:
    """
    Multistep resolution:
      1) Travel plans handled first.
      2) Use geo anchor (lat/lon) without fictional strings.
      3) If discovery: Overpass multiple common categories within radius.
      4) If brand: Overpass brand/name → fallback Nominatim → widen radius.
      5) If generic: Overpass category → fallback Nominatim → widen radius.
      6) Rank, dedupe, return EXACT/MULTIPLE/ASK with canonical ops.
    """

    geo_anchor = anchor.hints.get("geo_anchor") if isinstance(anchor.hints, dict) else None
    if not isinstance(geo_anchor, GeoAnchor):
        try:
            geo_anchor = await derive_geo_anchor(meta)
            anchor.hints = dict(anchor.hints or {})
            anchor.hints["geo_anchor"] = geo_anchor
        except Exception:
            geo_anchor = GeoAnchor()

    # ---- 1) Travel
    if getattr(query, "is_travel", False) and query.target:
        airport_label, alat, alon = nearest_airport_label(geo_anchor)
        origin_label = anchor.label or anchor.primary_city or "Current area"
        legs = [
            {
                "kind": "local",
                "origin_label": origin_label,
                "dest_label": airport_label,
                "dest": (alat, alon),
                "estimate_min": 35,
                "notes": "Local transfer",
            },
            {
                "kind": "flight",
                "origin_label": airport_label,
                "dest_label": f"{query.target} International Airport",
                "estimate_min": 600,
            },
            {
                "kind": "local",
                "origin_label": f"{query.target} International Airport",
                "dest_label": f"{query.target} center",
                "estimate_min": 45,
            },
        ]
        operations = [
            {
                "op": "travel.plan",
                "legs": legs,
                "arrival": {"city": query.target},
            }
        ]
        return ResolveResult(
            status=STATUS_TRAVEL_PLAN,
            message=f"Planning trip to {query.target}.",
            operations=operations,
            anchor=anchor,
            scope=anchor.scope,
        )

    # ---- 2) Anchor sanity
    if geo_anchor.lat is None or geo_anchor.lon is None:
        return ResolveResult(
            status=STATUS_ASK,
            message="I need a real anchor (coords or city) to search nearby. Which city are we in?",
            anchor=anchor,
            scope=anchor.scope,
        )

    radius_km = _LS.search_radius_km
    widen_km = _LS.widen_radius_km
    limit_n = min(_LS.nominatim_limit, 12)
    limit_o = min(_LS.overpass_limit, 24)

    # ---- 3) Discovery vs Brand/Generic
    tgt = _norm_text(getattr(query, "target", None) or "")
    if not tgt or _is_discovery_text(tgt):
        try:
            cands_o = await _overpass_search(geo_anchor, km=radius_km, limit=limit_o, generic_filters=_DISCOVERY_DEFAULTS)
        except Exception:
            cands_o = []
        cands = _rank_by_distance(geo_anchor, cands_o)[:8]
        if not cands:
            return ResolveResult(
                status=STATUS_ASK,
                message="Nothing obvious within ~5 miles. Try a specific place or category?",
                anchor=anchor,
                scope=anchor.scope,
            )
        return ResolveResult(
            status=STATUS_MULTIPLE,
            candidates=cands,
            message="Here’s what’s around within ~5 miles.",
            choices=[c.place.name for c in cands],
            operations=[
                {
                    "op": "poi.suggest",
                    "items": [
                        {
                            "name": c.place.name,
                            "lat": c.place.lat,
                            "lon": c.place.lon,
                            "category": c.place.meta.get("category"),
                        }
                        for c in cands
                    ],
                }
            ],
            anchor=anchor,
            scope=anchor.scope,
        )

    kind = _guess_query_kind(tgt)

    cands_overpass: List[Candidate] = []
    cands_nominatim: List[Candidate] = []

    # Pass 1 (~5 miles)
    try:
        if kind == "brand":
            cands_overpass = await _overpass_search(geo_anchor, km=radius_km, limit=limit_o, poi_norm=tgt)
        else:
            filters = _GENERIC_SYNONYMS.get(tgt.lower(), [])
            cands_overpass = await _overpass_search(geo_anchor, km=radius_km, limit=limit_o, generic_filters=filters or _DISCOVERY_DEFAULTS)
    except Exception:
        cands_overpass = []

    try:
        cands_nominatim = await _nominatim_search(tgt, geo_anchor, km=radius_km, limit=limit_n)
    except Exception:
        cands_nominatim = []

    merged = _merge_dedupe(geo_anchor, cands_overpass, cands_nominatim)
    ranked = _rank_by_name_similarity(tgt, merged) if kind == "brand" else merged

    # Widen if empty (~12 km)
    if not ranked:
        try:
            if kind == "brand":
                wide_o = await _overpass_search(geo_anchor, km=widen_km, limit=limit_o, poi_norm=tgt)
            else:
                filters = _GENERIC_SYNONYMS.get(tgt.lower(), [])
                wide_o = await _overpass_search(geo_anchor, km=widen_km, limit=limit_o, generic_filters=filters or _DISCOVERY_DEFAULTS)
        except Exception:
            wide_o = []
        try:
            wide_n = await _nominatim_search(tgt, geo_anchor, km=widen_km, limit=limit_n)
        except Exception:
            wide_n = []
        merged_wide = _merge_dedupe(geo_anchor, wide_o, wide_n)
        ranked = _rank_by_name_similarity(tgt, merged_wide) if kind == "brand" else merged_wide

    if not ranked:
        city_label = anchor.primary_city or anchor.label or "this area"
        msg = f"Couldn’t find “{tgt}” within ~5–7.5 miles of {city_label}. Try a nearby alternative or broaden the area?"
        return ResolveResult(status=STATUS_ASK, message=msg, anchor=anchor, scope=anchor.scope)

    top = ranked[0]
    if len(ranked) == 1 or (kind == "brand" and top.confidence >= 0.80):
        return ResolveResult(
            status=STATUS_EXACT,
            candidates=[top],
            operations=[{"op":"poi.navigate","label":top.place.name,"lat":top.place.lat,"lon":top.place.lon,
                            "category":top.place.meta.get("category"), "context_hint":{"use_geo_anchor": True}}],
            message=f"Heading to {top.place.name}.",
            anchor=anchor,
            scope=anchor.scope,
        )

    take = ranked[: min(6, len(ranked))]
    return ResolveResult(
        status=STATUS_MULTIPLE,
        candidates=take,
        message=f"I found a few matches for “{tgt}”. Which one?",
        choices=[c.place.name for c in take],
        operations=[{"op":"poi.suggest","items":[{"name":c.place.name,"lat":c.place.lat,"lon":c.place.lon,"category":c.place.meta.get("category")} for c in take]}],
        anchor=anchor,
        scope=anchor.scope,
    )
