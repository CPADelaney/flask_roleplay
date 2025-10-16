# nyx/location/real_world_resolver.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple
import httpx

from .types import *
from .anchors import build_nominatim_params_for_poi, derive_geo_anchor, nearest_airport_label

_BRAND_FIXUPS = {
    # common misspellings → canonical
    "ghiradelli": "Ghirardelli",
    "ghiradelli square": "Ghirardelli Square",
    "yank-sing": "Yank Sing",
    "mc donalds": "McDonald's",
    "mcdonalds": "McDonald's",
}

def _normalize_target(t: str) -> str:
    key = (t or "").strip().lower().replace("’", "'")
    return _BRAND_FIXUPS.get(key, t)

def _candidate_from_nominatim(n: Dict[str, Any]) -> PlaceCandidate:
    name = n.get("display_name", "").split(",")[0].strip() or n.get("name") or "Unknown"
    lat = float(n["lat"]); lon = float(n["lon"])
    cat = n.get("category") or n.get("type")
    addr = n.get("address") or {}
    conf = 0.9 if n.get("importance", 0) >= 0.5 else 0.6
    return PlaceCandidate(name=name, lat=lat, lon=lon, address=addr, category=cat, confidence=conf)

async def search_poi(poi: str, meta: Dict[str, Any], *, widen_once: bool = True) -> List[PlaceCandidate]:
    params = build_nominatim_params_for_poi(_normalize_target(poi), meta, radius_km=3.0, limit=5)
    headers = {"User-Agent": "nyx/worldsense/1.0"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers)
        r.raise_for_status()
        data = r.json() or []
        if isinstance(data, list) and data:
            return [_candidate_from_nominatim(x) for x in data]
        # widen search if bounded empty
        if widen_once:
            # remove viewbox and neighborhood tail, keep city
            a = derive_geo_anchor(meta)
            if a.city:
                params2 = {"format": "jsonv2", "limit": "5", "addressdetails": "1", "q": f"{poi}, {a.city}"}
                r2 = await client.get("https://nominatim.openstreetmap.org/search", params=params2, headers=headers)
                r2.raise_for_status()
                data2 = r2.json() or []
                if isinstance(data2, list) and data2:
                    return [_candidate_from_nominatim(x) for x in data2]
    return []

async def resolve_real(query: PlaceQuery, setting: SettingProfile, meta: Dict[str, Any]) -> ResolutionResult:
    # Travel outside the city? (simple “fly to X” detection)
    if query.is_travel and query.target:
        airport_label, alat, alon = nearest_airport_label(meta)
        dest_city = query.target
        plan = TravelPlan(legs=[
            TravelLeg(kind="local", origin_label=setting.label or setting.primary_city or "Current area",
                      dest_label=airport_label, origin=None, dest=(alat, alon), estimate_min=35, notes="Drive/ride-share or BART"),
            TravelLeg(kind="flight", origin_label=airport_label,
                      dest_label=f"{dest_city} International Airport", estimate_min=600, notes="Generate carrier later"),
            TravelLeg(kind="local", origin_label=f"{dest_city} International Airport",
                      dest_label=f"{dest_city} center", estimate_min=45),
        ], arrival_setting=SettingProfile(kind=SettingKind.REAL, primary_city=dest_city, country=None))
        return ResolutionResult(
            status=ResolutionStatus.TRAVEL_PLAN,
            message=f"Planning trip to {dest_city} via {airport_label}.",
            canonical_ops=[{"op": "travel.plan", "plan": plan}],
        )

    # POI/landmark inside the city
    if not query.target:
        return ResolutionResult(status=ResolutionStatus.AMBIGUOUS, message="What place are we heading to?")

    cands = await search_poi(query.target, meta)
    if not cands:
        # Suggest city‑level nudge instead of deny
        a = derive_geo_anchor(meta)
        city = a.city or setting.primary_city or "the current city"
        return ResolutionResult(
            status=ResolutionStatus.ASK,
            message=f'I can’t place “{query.target}” from here. Do you mean it in {city}?',
            choices=[f"{query.target} in {city}", "Somewhere else"],
        )

    # Heuristic: one strong match → EXACT, else MULTIPLE
    if len(cands) == 1 or cands[0].confidence >= 0.85:
        c = cands[0]
        return ResolutionResult(
            status=ResolutionStatus.EXACT,
            candidates=[c],
            canonical_ops=[{
                "op": "poi.navigate",
                "label": c.name,
                "lat": c.lat,
                "lon": c.lon,
                "category": c.category,
                "context_hint": {"use_geo_anchor": True}
            }],
            message=f"Heading to {c.name}.",
        )
    # Multiple plausible
    names = [c.name for c in cands[:4]]
    return ResolutionResult(
        status=ResolutionStatus.MULTIPLE,
        candidates=cands,
        message=f"I found a few matches for “{query.target}”. Which one?",
        choices=names,
    )
