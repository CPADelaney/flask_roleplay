# nyx/location/gemini_maps_adapter.py
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx

from .query import PlaceQuery
from .types import (
    Anchor,
    Candidate,
    Place,
    ResolveResult,
    STATUS_EXACT,
    STATUS_MULTIPLE,
    STATUS_NOT_FOUND,
    STATUS_TRAVEL_PLAN,
)

LOGGER = logging.getLogger(__name__)

# --- Config ------------------------------------------------------------------

# Models that support Maps + Search grounding (Oct 2025):
# 2.5 Pro / 2.5 Flash / 2.5 Flash-Lite / 2.0 Flash
# Default to low-latency; override with GOOGLE_GEMINI_MODEL if needed.
_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash")
_GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent"

# Maps (Places & Routes) endpoints
_PLACES_BASE = "https://places.googleapis.com/v1"
_ROUTES_ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Keys: use a single restricted key permitted for Gemini + Maps if you like
_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")

# Structured output schema (small and robust to avoid 400s).
# See: ai.google.dev/gemini-api/docs/structured-output
_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "places": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "formattedAddress": {"type": "STRING"},
                    "lat": {"type": "NUMBER"},
                    "lon": {"type": "NUMBER"},
                    "placeId": {"type": "STRING"},  # prefer "places/ChIJ..." if available
                    "websiteUri": {"type": "STRING"},
                    "phone": {"type": "STRING"},
                    "primaryType": {"type": "STRING"},
                    "hours": {
                        "type": "OBJECT",
                        "properties": {
                            "weekdayText": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                            },
                            "openNow": {"type": "BOOLEAN"},
                        },
                        "propertyOrdering": ["weekdayText", "openNow"],
                    },
                },
                "propertyOrdering": [
                    "name",
                    "formattedAddress",
                    "lat",
                    "lon",
                    "placeId",
                    "websiteUri",
                    "phone",
                    "primaryType",
                    "hours",
                ],
            },
        },
        # Optional showtimes / one-off events from Search grounding
        "events": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "start": {"type": "STRING"},  # ISO 8601 if available
                    "end": {"type": "STRING"},
                    "venueName": {"type": "STRING"},
                    "address": {"type": "STRING"},
                    "link": {"type": "STRING"},
                    "price": {"type": "STRING"},
                },
                "propertyOrdering": [
                    "title",
                    "start",
                    "end",
                    "venueName",
                    "address",
                    "link",
                    "price",
                ],
            },
        },
    },
    "propertyOrdering": ["places", "events"],
}

# --- Helpers -----------------------------------------------------------------

def _anchor_context(anchor: Anchor) -> str:
    parts: List[str] = []
    if anchor.label:
        parts.append(f"Current location: {anchor.label}")
    if anchor.primary_city:
        parts.append(f"City: {anchor.primary_city}")
    if anchor.region:
        parts.append(f"Region/State: {anchor.region}")
    if anchor.country:
        parts.append(f"Country: {anchor.country}")
    if anchor.lat is not None and anchor.lon is not None:
        parts.append(f"Near coordinates: {anchor.lat:.6f}, {anchor.lon:.6f}")
    return "\n".join(parts)


def _build_prompt(query: PlaceQuery, anchor: Anchor) -> str:
    """
    Guide the model to use BOTH Maps (for places/hours) and Search (for events/showtimes),
    and return structured JSON that matches _RESPONSE_SCHEMA.
    """
    target = query.target or query.raw_text or ""
    ctx = _anchor_context(anchor) or "Unknown location"
    return f"""Task: Resolve real places and any time-sensitive events/showtimes related to:
{target}

Context:
{ctx}

Instructions:
- Use Google Maps grounding to identify exact places and their details.
- Also use Google Search grounding to find showtimes or one-off events relevant to the query and area.
- Output ONLY JSON matching the schema. If a section has no results, return an empty array.
- Prefer ISO 8601 for 'start'/'end' when possible.
- For places include exact name, address, coordinates, and hours if available. If hours aren't known, set hours to null.
"""


def _extract_text_candidates(payload: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for cand in payload.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_grounding_metadata(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for cand in payload.get("candidates") or []:
        gm = cand.get("groundingMetadata")
        if gm:
            return gm
    return None


def _maps_chunks(grounding_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ch in grounding_metadata.get("groundingChunks") or []:
        if isinstance(ch, dict) and "maps" in ch and isinstance(ch["maps"], dict):
            out.append(ch["maps"])
    return out


def _web_chunks(grounding_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ch in grounding_metadata.get("groundingChunks") or []:
        if isinstance(ch, dict) and "web" in ch and isinstance(ch["web"], dict):
            out.append(ch["web"])
    return out


def _extract_coords_from_maps_url(url: str) -> Tuple[Optional[float], Optional[float]]:
    if not url:
        return None, None
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _normalize_place_resource(place_id_or_name: Optional[str]) -> Optional[str]:
    """
    Accept either 'places/ChIJ...' or bare 'ChIJ...'; return 'places/...'
    """
    if not place_id_or_name:
        return None
    return place_id_or_name if place_id_or_name.startswith("places/") else f"places/{place_id_or_name}"


# --- Places + Routes enrichment ----------------------------------------------

async def _places_get_details(
    client: httpx.AsyncClient, place_res_name: str
) -> Optional[Dict[str, Any]]:
    """
    Enrich a place using Places Details (New).
    We request a tight field mask: address, location, type, contact, hours.
    """
    if not _MAPS_KEY:
        return None
    if not place_res_name:
        return None

    url = f"{_PLACES_BASE}/{place_res_name}"
    headers = {
        "X-Goog-Api-Key": _MAPS_KEY,
        "Content-Type": "application/json",
        # Avoid '*' in prod; keep fields lean for latency/cost.
        "X-Goog-FieldMask": ",".join(
            [
                "id",
                "displayName",
                "formattedAddress",
                "location",
                "googleMapsUri",
                "primaryType",
                "websiteUri",
                "nationalPhoneNumber",
                "currentOpeningHours",
                "currentOpeningHours.openNow",
                "currentOpeningHours.weekdayDescriptions",
                "regularOpeningHours.weekdayDescriptions",
            ]
        ),
    }
    try:
        resp = await client.get(url, headers=headers, timeout=8.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        LOGGER.warning("Places details failed for %s: %s", place_res_name, exc)
        return None


async def _compute_eta(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    mode: str = "DRIVE",
    traffic: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Compute ETA/distance using Routes API computeRoutes.
    Returns seconds and meters only.
    """
    if not _MAPS_KEY:
        LOGGER.warning("GOOGLE_MAPS_API_KEY missing for routes")
        return None

    ox, oy = origin
    dx, dy = dest

    body = {
        "origin": {"location": {"latLng": {"latitude": ox, "longitude": oy}}},
        "destination": {"location": {"latLng": {"latitude": dx, "longitude": dy}}},
        "travelMode": mode,  # DRIVE, WALK, BICYCLE, TRANSIT, TWO_WHEELER
    }
    if mode == "DRIVE" and traffic:
        body["routingPreference"] = "TRAFFIC_AWARE_OPTIMAL"
        body["departureTime"] = datetime.now(timezone.utc).isoformat()

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _MAPS_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_ROUTES_ENDPOINT, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        LOGGER.warning("Routes computeRoutes failed: %s", exc)
        return None

    routes = (data or {}).get("routes") or []
    if not routes:
        return None

    r0 = routes[0]
    duration_iso = r0.get("duration")
    meters = r0.get("distanceMeters")
    seconds = None
    if isinstance(duration_iso, str) and duration_iso.endswith("s"):
        try:
            seconds = int(float(duration_iso[:-1]))
        except ValueError:
            seconds = None

    return {"seconds": seconds, "distance_meters": meters, "raw": r0}


# --- Core: Gemini call with Maps + Search + JSON -----------------------------

async def _gemini_structured_places_and_events(
    query: PlaceQuery,
    anchor: Anchor,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Calls Gemini with BOTH googleMaps and google_search tools enabled and requests
    structured JSON output per _RESPONSE_SCHEMA.

    Returns (parsed_json, grounding_metadata_dict or None).
    """
    if not _GEMINI_KEY:
        raise RuntimeError("gemini_api_key_missing")

    prompt = _build_prompt(query, anchor)

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [
            {"googleMaps": {}},       # Maps grounding for place facts
            {"google_search": {}},    # Web Search grounding for events/showtimes
        ],
        "toolConfig": {
            "retrievalConfig": {
                "latLng": (
                    {"latitude": anchor.lat, "longitude": anchor.lon}
                    if anchor.lat is not None and anchor.lon is not None
                    else {}
                )
            }
        },
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": _GEMINI_KEY,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(_GEMINI_ENDPOINT, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    text = _extract_text_candidates(data)
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError:
        parsed = {}

    gm = _extract_grounding_metadata(data)
    return (parsed or {}, gm)


async def _candidates_from_grounding_and_json(
    parsed_json: Dict[str, Any],
    grounding_metadata: Optional[Dict[str, Any]],
    anchor: Anchor,
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    Produce Candidate list from Maps grounding; enrich with Places Details;
    also return a simple list of events extracted from the structured JSON.
    """
    events = parsed_json.get("events") or []
    candidates: List[Candidate] = []

    maps_chunks = _maps_chunks(grounding_metadata or {})
    seen = set()

    async with httpx.AsyncClient() as client:
        # 1) Build candidates from Maps chunks
        for m in maps_chunks:
            title = m.get("title") or "Unknown place"
            uri = m.get("uri") or m.get("googleMapsUri") or ""
            place_res = _normalize_place_resource(m.get("placeId") or m.get("name"))

            key = place_res or uri or title
            if not key or key in seen:
                continue
            seen.add(key)

            lat, lon = _extract_coords_from_maps_url(uri)

            # Minimal Place object first
            place = Place(
                name=title,
                level="venue",
                lat=lat if lat is not None else anchor.lat,
                lon=lon if lon is not None else anchor.lon,
                address={"line1": title},
                meta={
                    "source": "gemini_maps_grounding",
                    "uri": uri,
                    "placeId": place_res,
                    "grounded": True,
                },
            )

            cand = Candidate(
                place=place, confidence=0.85, rationale="google_maps_grounding", raw=m
            )

            # Optional enrichment via Places Details (address/hours/website/phone)
            details = await _places_get_details(client, place_res) if place_res else None
            if details:
                addr = details.get("formattedAddress")
                loc = (details.get("location") or {})
                hours_now = (details.get("currentOpeningHours") or {})
                reg_hours = (details.get("regularOpeningHours") or {})

                place.address = {
                    "line1": addr or title,
                    "_normalized_admin_path": {
                        "city": anchor.primary_city,
                        "region": anchor.region,
                        "country": anchor.country,
                    },
                }
                if loc:
                    place.lat = loc.get("latitude", place.lat)
                    place.lon = loc.get("longitude", place.lon)
                # attach more metadata commonly needed for RP context
                place.meta.update(
                    {
                        "primaryType": details.get("primaryType"),
                        "websiteUri": details.get("websiteUri"),
                        "phone": details.get("nationalPhoneNumber"),
                        "hours": {
                            "weekdayText": (hours_now.get("weekdayDescriptions")
                                            or reg_hours.get("weekdayDescriptions")),
                            "openNow": hours_now.get("openNow"),
                        },
                    }
                )

            candidates.append(cand)

        # 2) If the model returned places JSON but no Maps chunks, fall back to that
        if not candidates:
            for item in (parsed_json.get("places") or []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not name:
                    continue
                place = Place(
                    name=name,
                    level="venue",
                    lat=item.get("lat", anchor.lat),
                    lon=item.get("lon", anchor.lon),
                    address={"line1": item.get("formattedAddress") or name},
                    meta={
                        "source": "gemini_structured_output",
                        "placeId": _normalize_place_resource(item.get("placeId")),
                        "websiteUri": item.get("websiteUri"),
                        "phone": item.get("phone"),
                        "primaryType": item.get("primaryType"),
                        "hours": item.get("hours"),
                    },
                )
                candidates.append(
                    Candidate(
                        place=place, confidence=0.65, rationale="json_structured", raw=item
                    )
                )

    return candidates, events


# --- Public API --------------------------------------------------------------

async def resolve_location_with_gemini(
    query: PlaceQuery,
    anchor: Anchor,
) -> ResolveResult:
    """
    Resolve places with Google Maps grounding AND fetch time-sensitive events via Google Search grounding.
    Returns Candidates (places) and an 'events' operation payload so your RP logic can weave it into narration.
    """
    base = ResolveResult(
        status=STATUS_NOT_FOUND,
        anchor=anchor,
        scope=anchor.scope or "real",
        message="Unable to resolve.",
    )

    try:
        parsed, gm = await _gemini_structured_places_and_events(query, anchor)
    except Exception as exc:
        LOGGER.error("Gemini request failed: %s", exc)
        base.errors.append(f"gemini_request_failed:{exc}")
        base.message = "Gemini lookup failed."
        return base

    candidates, events = await _candidates_from_grounding_and_json(parsed, gm, anchor)

    if not candidates and not events:
        base.errors.append("gemini_no_results")
        base.message = f"No matches for '{query.target or query.raw_text}'."
        return base

    status = STATUS_EXACT if len(candidates) == 1 else (STATUS_MULTIPLE if candidates else STATUS_NOT_FOUND)
    message = (
        f"Found: {candidates[0].place.name}" if status == STATUS_EXACT
        else f"Found {len(candidates)} places"
        if status == STATUS_MULTIPLE
        else f"Found {len(events)} event(s)" if events else "No places found"
    )

    # Pack a simple "events" op for your text engine to consume
    operations: List[Dict[str, Any]] = []
    if events:
        operations.append({"op": "events", "items": events})

    # No widget token, no UI plumbing. You’re welcome.
    return ResolveResult(
        status=status,
        message=message,
        candidates=candidates,
        operations=operations,
        choices=[c.place.name for c in candidates] if status == STATUS_MULTIPLE else [],
        anchor=anchor,
        scope=anchor.scope or "real",
    )


async def eta_between_with_gemini(
    origin_query: PlaceQuery,
    destination_query: PlaceQuery,
    anchor: Anchor,
    mode: str = "DRIVE",
) -> ResolveResult:
    """
    Resolve origin/destination (with Maps grounding) then compute ETA via Routes API.
    Also carries through events attached to either endpoint query if present.
    """
    origin_res = await resolve_location_with_gemini(origin_query, anchor)
    dest_res = await resolve_location_with_gemini(destination_query, anchor)

    if not origin_res.candidates or not dest_res.candidates:
        msg = "Could not resolve both endpoints."
        if not origin_res.candidates:
            msg += f" Missing origin '{origin_query.target or origin_query.raw_text}'."
        if not dest_res.candidates:
            msg += f" Missing destination '{destination_query.target or destination_query.raw_text}'."
        return ResolveResult(
            status=STATUS_NOT_FOUND,
            message=msg,
            candidates=[],
            operations=[],
            anchor=anchor,
            scope=anchor.scope or "real",
            errors=["eta_endpoints_unresolved"],
        )

    o = origin_res.candidates[0].place
    d = dest_res.candidates[0].place
    if o.lat is None or o.lon is None or d.lat is None or d.lon is None:
        return ResolveResult(
            status=STATUS_NOT_FOUND,
            message="Resolved endpoints but coordinates missing for routing.",
            candidates=[origin_res.candidates[0], dest_res.candidates[0]],
            operations=[],
            anchor=anchor,
            scope=anchor.scope or "real",
            errors=["eta_coords_missing"],
        )

    eta = await _compute_eta((o.lat, o.lon), (d.lat, d.lon), mode=mode, traffic=True)

    operations: List[Dict[str, Any]] = [
        {
            "op": "route.eta",
            "mode": mode,
            "from": {"name": o.name, "lat": o.lat, "lon": o.lon},
            "to": {"name": d.name, "lat": d.lat, "lon": d.lon},
            "eta_seconds": eta.get("seconds") if eta else None,
            "distance_meters": eta.get("distance_meters") if eta else None,
        }
    ]

    # Bubble up any event lists discovered on either step
    for res in (origin_res, dest_res):
        for op in res.operations or []:
            if op.get("op") == "events" and op.get("items"):
                operations.append(op)

    msg = (
        f"ETA {eta['seconds']}s, distance {eta['distance_meters']} m"
        if eta
        else "ETA unavailable"
    )

    return ResolveResult(
        status=STATUS_TRAVEL_PLAN,
        message=msg,
        candidates=[origin_res.candidates[0], dest_res.candidates[0]],
        operations=operations,
        anchor=anchor,
        scope=anchor.scope or "real",
    )
