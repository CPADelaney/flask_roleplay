# nyx/location/search.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import json
import httpx

from .types import (
    PlaceCandidate, PlaceQuery, SettingProfile, ResolutionResult,
    ResolutionStatus, TravelPlan, TravelLeg
)
from .anchors import GeoAnchor, derive_geo_anchor, build_nominatim_params_for_poi
from .config import DEFAULT_LOCATION_SETTINGS as _LS

try:
    from rapidfuzz import fuzz
except Exception:  # optional
    fuzz = None

# ---------- Helpers & Normalization ----------

_GENERIC_SYNONYMS: Dict[str, List[Tuple[str, str]]] = {
    # maps user generic term -> list of (key, value) Overpass tag filters
    # amenity
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
    # crude heuristics for plurals like "restaurants", "cafes"
    if t.endswith("s") and t[:-1] in _GENERIC_SYNONYMS:
        return True
    return False

def _guess_query_kind(text: str) -> str:
    if _is_discovery_text(text):
        return "discovery"
    if _is_generic_term(text):
        return "generic"
    # Heuristic: brand if it contains "'s" or is 1–3 words not in synonyms
    tl = text.lower()
    if "'s" in tl or "’s" in text or len(text.split()) <= 3:
        return "brand"
    return "brand"  # default to brand if not matched

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

def _candidate_from_nominatim(n: Dict[str, Any]) -> PlaceCandidate:
    name = n.get("name") or (n.get("display_name","").split(",")[0].strip() or "Unknown")
    lat = float(n["lat"]); lon = float(n["lon"])
    cat = n.get("category") or n.get("type")
    addr = n.get("address") or {}
    imp = float(n.get("importance", 0) or 0)
    return PlaceCandidate(
        name=name, lat=lat, lon=lon, address=addr, category=cat,
        confidence=min(0.99, 0.5 + imp/2)
    )

def _candidate_from_overpass(el: Dict[str, Any]) -> Optional[PlaceCandidate]:
    tags = el.get("tags") or {}
    name = tags.get("name") or "Unknown"
    if not name or name == "Unknown":
        # Prefer named POIs for UX; still allow if brand match
        brand = tags.get("brand")
        if not brand:
            return None
        name = brand
    # Coordinates:
    if "lat" in el and "lon" in el:
        lat, lon = float(el["lat"]), float(el["lon"])
    elif "center" in el and isinstance(el["center"], dict):
        lat, lon = float(el["center"]["lat"]), float(el["center"]["lon"])
    else:
        return None
    # Category guess:
    cat = None
    for key in ("amenity","shop","leisure","tourism"):
        if key in tags:
            cat = f"{key}:{tags.get(key)}"; break
    addr = _address_from_tags(tags)
    # Confidence: if brand tag present, start higher
    conf = 0.75 if tags.get("brand") else 0.60
    return PlaceCandidate(name=name, lat=lat, lon=lon, address=addr, category=cat, confidence=conf)

# ---------- External calls ----------

async def _nominatim_search(poi: str, anchor: GeoAnchor, *, km: float, limit: int) -> List[PlaceCandidate]:
    params = build_nominatim_params_for_poi(poi, anchor, radius_km=km, limit=limit)
    headers = {"User-Agent": "nyx/worldsense/1.0 (contact: ops@nyx.example)"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers)
        r.raise_for_status()
        data = r.json() or []
        return [_candidate_from_nominatim(x) for x in (data if isinstance(data, list) else [])]

def _overpass_brand_block(poi_norm: str) -> str:
    # brand or name regex (case-insensitive), escape quotes
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
    # Build a union of the provided tag filters
    parts = []
    for key, val in filters:
        parts.append(f'  node["{key}"="{val}"];')
        parts.append(f'  way["{key}"="{val}"];')
        parts.append(f'  relation["{key}"="{val}"];')
    return "\n".join(parts)

async def _overpass_search(
    anchor: GeoAnchor, *, km: float, limit: int,
    poi_norm: Optional[str] = None,
    generic_filters: Optional[List[Tuple[str, str]]] = None,
    timeout_s: int = _LS.overpass_timeout_s
) -> List[PlaceCandidate]:
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
        cands = []
        for el in els:
            c = _candidate_from_overpass(el)
            if c: cands.append(c)
        return cands[:limit]

# ---------- Ranking & Merge ----------

def _rank_by_name_similarity(query: str, cands: List[PlaceCandidate]) -> List[PlaceCandidate]:
    if not cands: return cands
    if not fuzz:
        return cands
    q = _norm_text(query).lower()
    for c in cands:
        # Blend existing confidence with name similarity
        sim = fuzz.token_set_ratio(q, _norm_text(c.name).lower()) / 100.0
        c.confidence = 0.5 * c.confidence + 0.5 * sim
    return sorted(cands, key=lambda x: x.confidence, reverse=True)

def _rank_by_distance(anchor: GeoAnchor, cands: List[PlaceCandidate]) -> List[PlaceCandidate]:
    if not cands or anchor.lat is None or anchor.lon is None:
        return cands
    for c in cands:
        d = _haversine_km(anchor.lat, anchor.lon, c.lat, c.lon)
        # Softly adjust confidence closer = higher
        c.confidence = 0.7 * c.confidence + 0.3 * max(0.0, 1.0 - min(1.0, d / _LS.search_radius_km))
    return sorted(cands, key=lambda x: x.confidence, reverse=True)

def _merge_dedupe(anchor: GeoAnchor, a: List[PlaceCandidate], b: List[PlaceCandidate]) -> List[PlaceCandidate]:
    merged: List[PlaceCandidate] = []
    def _too_close(x: PlaceCandidate, y: PlaceCandidate) -> bool:
        return _haversine_km(x.lat, x.lon, y.lat, y.lon) <= 0.075  # ~75m
    for src in (a, b):
        for c in src:
            if any(_too_close(c, m) or (_norm_text(c.name).lower() == _norm_text(m.name).lower()) for m in merged):
                continue
            merged.append(c)
    return _rank_by_distance(anchor, merged)

# ---------- Main entry ----------

async def resolve_real(query: PlaceQuery, setting: SettingProfile, meta: Dict[str, Any]) -> ResolutionResult:
    """
    Multistep resolution:
      1) Travel plans handled first (unchanged).
      2) Build anchor (lat/lon) without fictional strings.
      3) If discovery: Overpass multiple common categories within radius.
      4) If brand: Overpass brand/name → fallback Nominatim → widen radius.
      5) If generic: Overpass category → fallback Nominatim → widen radius.
      6) Rank, dedupe, return EXACT/MULTIPLE/ASK with canonical ops.
    """
    # ---- 1) Travel (unchanged) ----
    if getattr(query, "is_travel", False) and query.target:
        anchor = await derive_geo_anchor(meta)
        airport_label, alat, alon = nearest_airport_label(anchor)
        plan = TravelPlan(legs=[
            TravelLeg(kind="local", origin_label=anchor.label or setting.primary_city or "Current area",
                      dest_label=airport_label, dest=(alat, alon), estimate_min=35, notes="Local transfer"),
            TravelLeg(kind="flight", origin_label=airport_label,
                      dest_label=f"{query.target} International Airport", estimate_min=600),
            TravelLeg(kind="local", origin_label=f"{query.target} International Airport",
                      dest_label=f"{query.target} center", estimate_min=45),
        ], arrival_setting=SettingProfile(kind=setting.kind, primary_city=query.target))
        return ResolutionResult(status=ResolutionStatus.TRAVEL_PLAN,
                                message=f"Planning trip to {query.target}.",
                                canonical_ops=[{"op":"travel.plan","plan":plan}])

    # ---- 2) Anchor ----
    anchor = await derive_geo_anchor(meta)
    if anchor.lat is None or anchor.lon is None:
        # We can’t bound a search without coordinates; ask for a city
        return ResolutionResult(status=ResolutionStatus.ASK,
                                message="I need a real anchor (coords or city) to search nearby. Which city are we in?")

    radius_km = _LS.search_radius_km
    widen_km  = _LS.widen_radius_km
    limit_n   = min(_LS.nominatim_limit, 12)
    limit_o   = min(_LS.overpass_limit, 24)

    # ---- 3) Discovery mode ----
    tgt = _norm_text(getattr(query, "target", None) or "")
    if not tgt or _is_discovery_text(tgt):
        try:
            cands_o = await _overpass_search(anchor, km=radius_km, limit=limit_o,
                                             generic_filters=_DISCOVERY_DEFAULTS)
        except Exception:
            cands_o = []
        # No Nominatim in discovery (it’s free-text oriented), we already covered wide common POIs
        cands = _rank_by_distance(anchor, cands_o)[:8]
        if not cands:
            return ResolutionResult(status=ResolutionStatus.ASK,
                                    message="Nothing obvious within ~5 miles. Try a specific place or category?")
        return ResolutionResult(
            status=ResolutionStatus.MULTIPLE,
            candidates=cands,
            message="Here’s what’s around within ~5 miles.",
            choices=[c.name for c in cands],
            canonical_ops=[{"op":"poi.suggest","items":[{"name":c.name,"lat":c.lat,"lon":c.lon,"category":c.category} for c in cands]}]
        )

    # ---- 4/5) Brand vs Generic ----
    kind = _guess_query_kind(tgt)

    cands_overpass: List[PlaceCandidate] = []
    cands_nominatim: List[PlaceCandidate] = []

    # First pass (radius = 5 miles)
    try:
        if kind == "brand":
            cands_overpass = await _overpass_search(anchor, km=radius_km, limit=limit_o, poi_norm=tgt)
        else:  # generic
            filters = _GENERIC_SYNONYMS.get(tgt.lower(), [])
            cands_overpass = await _overpass_search(anchor, km=radius_km, limit=limit_o, generic_filters=filters or _DISCOVERY_DEFAULTS)
    except Exception:
        cands_overpass = []

    try:
        # Nominatim fallback in parallel style flow
        cands_nominatim = await _nominatim_search(tgt, anchor, km=radius_km, limit=limit_n)
    except Exception:
        cands_nominatim = []

    merged = _merge_dedupe(anchor, cands_overpass, cands_nominatim)
    ranked = _rank_by_name_similarity(tgt, merged) if kind == "brand" else merged

    # Widen pass if still empty (radius ~12 km)
    if not ranked:
        try:
            if kind == "brand":
                wide_o = await _overpass_search(anchor, km=widen_km, limit=limit_o, poi_norm=tgt)
            else:
                filters = _GENERIC_SYNONYMS.get(tgt.lower(), [])
                wide_o = await _overpass_search(anchor, km=widen_km, limit=limit_o, generic_filters=filters or _DISCOVERY_DEFAULTS)
        except Exception:
            wide_o = []
        try:
            wide_n = await _nominatim_search(tgt, anchor, km=widen_km, limit=limit_n)
        except Exception:
            wide_n = []
        ranked = _rank_by_name_similarity(tgt, _merge_dedupe(anchor, wide_o, wide_n)) if kind == "brand" else _merge_dedupe(anchor, wide_o, wide_n)

    # Decide result shape
    if not ranked:
        # We explicitly *looked*—be constructive
        city_label = anchor.city or (anchor.label or "this area")
        msg = f"Couldn’t find “{tgt}” within ~5–7.5 miles of {city_label}. Try a nearby alternative or broaden the area?"
        return ResolutionResult(status=ResolutionStatus.ASK, message=msg)

    top = ranked[0]
    if len(ranked) == 1 or (kind == "brand" and top.confidence >= 0.80):
        return ResolutionResult(
            status=ResolutionStatus.EXACT,
            candidates=[top],
            canonical_ops=[{"op":"poi.navigate","label":top.name,"lat":top.lat,"lon":top.lon,
                            "category":top.category, "context_hint":{"use_geo_anchor": True}}],
            message=f"Heading to {top.name}."
        )

    # Multiple good options
    take = ranked[: min(6, len(ranked))]
    return ResolutionResult(
        status=ResolutionStatus.MULTIPLE,
        candidates=take,
        message=f"I found a few matches for “{tgt}”. Which one?",
        choices=[c.name for c in take],
        canonical_ops=[{"op":"poi.suggest","items":[{"name":c.name,"lat":c.lat,"lon":c.lon,"category":c.category} for c in take]}]
    )
