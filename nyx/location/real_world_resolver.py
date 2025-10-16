from __future__ import annotations
from typing import Any, Dict, List
import httpx

from .types import *
from .anchors import build_nominatim_params_for_poi, derive_geo_anchor, nearest_airport_label

try:
    from rapidfuzz import fuzz
except Exception:  # optional
    fuzz = None

def _candidate_from_nominatim(n: Dict[str, Any]) -> PlaceCandidate:
    name = n.get("name") or (n.get("display_name","").split(",")[0].strip() or "Unknown")
    lat = float(n["lat"]); lon = float(n["lon"])
    cat = n.get("category") or n.get("type")
    addr = n.get("address") or {}
    imp = float(n.get("importance", 0) or 0)
    return PlaceCandidate(name=name, lat=lat, lon=lon, address=addr, category=cat, confidence=min(0.99, 0.5 + imp/2))

async def _nominatim_search(poi: str, anchor: GeoAnchor, *, km: float, limit: int) -> List[PlaceCandidate]:
    params = build_nominatim_params_for_poi(poi, anchor, radius_km=km, limit=limit)
    headers = {"User-Agent": "nyx/worldsense/1.0"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers)
        r.raise_for_status()
        data = r.json() or []
        return [_candidate_from_nominatim(x) for x in (data if isinstance(data, list) else [])]

def _rank_by_name_similarity(query: str, cands: List[PlaceCandidate]) -> List[PlaceCandidate]:
    if not cands or not fuzz:
        return cands
    q = query.lower()
    for c in cands:
        c.confidence = 0.5 * c.confidence + 0.5 * (fuzz.token_set_ratio(q, c.name.lower()) / 100.0)
    return sorted(cands, key=lambda x: x.confidence, reverse=True)

async def resolve_real(query: PlaceQuery, setting: SettingProfile, meta: Dict[str, Any]) -> ResolutionResult:
    # Travel plan (e.g., "fly to Tokyo")
    if query.is_travel and query.target:
        anchor = await derive_geo_anchor(meta)
        airport_label, alat, alon = nearest_airport_label(anchor)
        plan = TravelPlan(legs=[
            TravelLeg(kind="local", origin_label=anchor.label or setting.primary_city or "Current area",
                      dest_label=airport_label, dest=(alat, alon), estimate_min=35, notes="Local transfer"),
            TravelLeg(kind="flight", origin_label=airport_label,
                      dest_label=f"{query.target} International Airport", estimate_min=600),
            TravelLeg(kind="local", origin_label=f"{query.target} International Airport",
                      dest_label=f"{query.target} center", estimate_min=45),
        ], arrival_setting=SettingProfile(kind=SettingKind.REAL, primary_city=query.target))
        return ResolutionResult(status=ResolutionStatus.TRAVEL_PLAN, message=f"Planning trip to {query.target}.", canonical_ops=[{"op":"travel.plan","plan":plan}])

    if not query.target:
        return ResolutionResult(status=ResolutionStatus.AMBIGUOUS, message="Where to?")

    anchor = await derive_geo_anchor(meta)
    cands = await _nominatim_search(query.target, anchor, km=3.0, limit=6)
    if not cands and anchor.city:
        # widen to whole city string (still no scene labels)
        cands = await _nominatim_search(f"{query.target}, {anchor.city}", anchor, km=12.0, limit=8)

    if not cands:
        return ResolutionResult(status=ResolutionStatus.ASK, message=f"I can’t place “{query.target}” from here. Do you mean it in {anchor.city or 'this city'}?", choices=[f"{query.target} in {anchor.city or 'this city'}","Somewhere else"])

    ranked = _rank_by_name_similarity(query.target, cands)
    top = ranked[0]
    # exact vs multiple decision
    if len(ranked) == 1 or top.confidence >= 0.80:
        return ResolutionResult(
            status=ResolutionStatus.EXACT,
            candidates=[top],
            canonical_ops=[{"op":"poi.navigate","label":top.name,"lat":top.lat,"lon":top.lon,"category":top.category,"context_hint":{"use_geo_anchor": True}}],
            message=f"Heading to {top.name}.",
        )
    return ResolutionResult(
        status=ResolutionStatus.MULTIPLE,
        candidates=ranked[:4],
        message=f"I found a few matches for “{query.target}”. Which one?",
        choices=[c.name for c in ranked[:4]],
    )
