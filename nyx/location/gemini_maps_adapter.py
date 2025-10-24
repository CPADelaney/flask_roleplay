from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
# --- SDK IMPORTS START ---
import google.generativeai as genai
from google.generativeai import types
from google.api_core import exceptions as google_exceptions
# --- SDK IMPORTS END ---

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

# ---------------------------------------------------------------------------
# Config & SDK Initialization
# ---------------------------------------------------------------------------

# API keys: you may use one restricted key enabled for Gemini + Maps or separate keys.
_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")

# --- SDK CONFIGURATION START ---
if _GEMINI_KEY:
    genai.configure(api_key=_GEMINI_KEY)
_GEMINI_MODEL_NAME = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash-latest")
_GEMINI_MODEL = genai.GenerativeModel(_GEMINI_MODEL_NAME) if _GEMINI_KEY else None
# --- SDK CONFIGURATION END ---

# Routes + Places (v1) endpoints (still needed for httpx calls)
_ROUTES_ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"
_PLACES_BASE = "https://places.googleapis.com/v1"

# Events-only schema (unchanged)
_EVENTS_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "events": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "start": {"type": "STRING"},
                    "end": {"type": "STRING"},
                    "venueName": {"type": "STRING"},
                    "address": {"type": "STRING"},
                    "link": {"type": "STRING"},
                    "price": {"type": "STRING"},
                },
            },
        }
    },
}

# ---------------------------------------------------------------------------
# Prompting (Unchanged)
# ---------------------------------------------------------------------------

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

def _build_prompt_for_tools(query: PlaceQuery, anchor: Anchor) -> str:
    target = (query.target or query.raw_text or "").strip()
    ctx = _anchor_context(anchor) or "Unknown location"
    return f"""Task: Resolve real places and relevant events/showtimes for:
{target}

Context:
{ctx}

Use Google Maps grounding to identify exact places. Also use Google Search grounding to find any showtimes or one-off events that match the query and area. Provide concise factual details in your answer text so they can be reformatted later."""

# ---------------------------------------------------------------------------
# HTTP helpers (for Maps Platform APIs, httpx is still used)
# ---------------------------------------------------------------------------

# This is no longer needed for Gemini calls, but kept for context.
# The SDK handles retries and error parsing internally.
def _is_tool_json_conflict(err_text: str) -> bool:
    return "response mime type" in err_text.lower() and "unsupported" in err_text.lower()

# ---------------------------------------------------------------------------
# Grounding -> Candidates (Unchanged, uses httpx for Places API)
# ---------------------------------------------------------------------------

def _maps_chunks(grounding_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ch in grounding_metadata.get("grounding_chunks") or []:
        if isinstance(ch, dict) and "maps" in ch and isinstance(ch["maps"], dict):
            out.append(ch["maps"])
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
    if not place_id_or_name:
        return None
    return place_id_or_name if place_id_or_name.startswith("places/") else f"places/{place_id_or_name}"

async def _places_get_details(client: httpx.AsyncClient, place_res_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not _MAPS_KEY or not place_res_name:
        return None
    url = f"{_PLACES_BASE}/{place_res_name}"
    headers = {
        "X-Goog-Api-Key": _MAPS_KEY,
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "id,displayName,formattedAddress,location,googleMapsUri,primaryType,websiteUri,nationalPhoneNumber,currentOpeningHours,regularOpeningHours",
    }
    try:
        resp = await client.get(url, headers=headers, timeout=8.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        LOGGER.warning("Places details failed for %s: %s", place_res_name, exc)
        return None

async def _candidates_from_maps_grounding(gm: Dict[str, Any], anchor: Anchor) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen = set()
    maps_chunks = _maps_chunks(gm)
    async with httpx.AsyncClient() as client:
        for m in maps_chunks:
            title = m.get("title") or "Unknown place"
            uri = m.get("uri") or m.get("googleMapsUri") or ""
            place_res = _normalize_place_resource(m.get("placeId") or m.get("name"))
            key = place_res or uri or title
            if not key or key in seen:
                continue
            seen.add(key)
            lat, lon = _extract_coords_from_maps_url(uri)
            details = await _places_get_details(client, place_res) if place_res else None
            addr = title
            primary_type, website, phone, hours_obj = None, None, None, None
            if details:
                addr = details.get("formattedAddress") or addr
                loc = details.get("location") or {}
                lat = loc.get("latitude", lat)
                lon = loc.get("longitude", lon)
                primary_type = details.get("primaryType")
                website = details.get("websiteUri")
                phone = details.get("nationalPhoneNumber")
                now_hours = details.get("currentOpeningHours") or {}
                reg_hours = details.get("regularOpeningHours") or {}
                weekday = now_hours.get("weekdayDescriptions") or reg_hours.get("weekdayDescriptions")
                hours_obj = {"weekdayText": weekday, "openNow": now_hours.get("openNow")}
            place = Place(
                name=title,
                level="venue",
                lat=lat if lat is not None else anchor.lat,
                lon=lon if lon is not None else anchor.lon,
                address={"line1": addr, "_normalized_admin_path": {"city": anchor.primary_city, "region": anchor.region, "country": anchor.country}},
                meta={"source": "gemini_maps_grounding", "uri": uri, "placeId": place_res, "grounded": True, "primaryType": primary_type, "websiteUri": website, "phone": phone, "hours": hours_obj},
            )
            candidates.append(Candidate(place=place, confidence=0.85, rationale="google_maps_grounding", raw=m))
    return candidates

# ---------------------------------------------------------------------------
# Gemini calls (REFACTORED with Python SDK)
# ---------------------------------------------------------------------------

async def _gemini_tools_call(query: PlaceQuery, anchor: Anchor) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Step A: Run Gemini with Maps + Search tools ON using the Python SDK.
    Returns (free_text, grounding_metadata).
    """
    if not _GEMINI_MODEL:
        raise RuntimeError("gemini_api_key_missing")

    prompt = _build_prompt_for_tools(query, anchor)

    # --- MODIFICATION START ---
    # The 'google_search_retrieval' tool is the correct one to enable all grounding,
    # including Google Maps when a location context is provided.
    # There is no separate 'GoogleMaps' tool in this SDK.
    tools = [
        types.Tool(google_search_retrieval={})
    ]
    # --- MODIFICATION END ---

    # Define the tool config for providing the anchor location (this part was already correct)
    tool_config = None
    if anchor.lat is not None and anchor.lon is not None:
        tool_config = types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=anchor.lat, longitude=anchor.lon)
            )
        )

    generation_config = types.GenerationConfig(temperature=0.2)

    try:
        # Use asyncio.to_thread to run the synchronous SDK call in an async context
        response = await asyncio.to_thread(
            _GEMINI_MODEL.generate_content,
            contents=[prompt],
            tools=tools,
            tool_config=tool_config,
            generation_config=generation_config
        )
        
        grounding_metadata = response.candidates[0].grounding_metadata if response.candidates else None
        
        # Convert grounding_metadata from protobuf-like object to a plain dict
        gm_dict = types.to_dict(grounding_metadata) if grounding_metadata else None
        
        return response.text, gm_dict
        
    except Exception as e:
        LOGGER.error(f"Gemini SDK call failed: {e}", exc_info=True)
        raise

async def _schema_only_events(text_source: str) -> List[Dict[str, Any]]:
    """
    Step B: Use the SDK to extract structured events/showtimes from text.
    """
    if not _GEMINI_MODEL or not text_source:
        return []

    prompt = (
        "Extract any showtimes or one-off events from the following text. "
        "Return strict JSON matching the provided schema. "
        "If none are found, return an empty 'events' array.\n\n"
        f"{text_source}"
    )

    generation_config = types.GenerationConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=_EVENTS_SCHEMA,
    )

    try:
        response = await asyncio.to_thread(
            _GEMINI_MODEL.generate_content,
            contents=[prompt],
            generation_config=generation_config
        )
        parsed = json.loads(response.text)
        return parsed.get("events") or []
    except (json.JSONDecodeError, google_exceptions.GoogleAPICallError) as e:
        LOGGER.warning(f"Failed to extract structured events: {e}")
        return []

# ---------------------------------------------------------------------------
# Routing / ETA (Unchanged, uses httpx for Routes API)
# ---------------------------------------------------------------------------

async def _compute_eta(origin: Tuple[float, float], dest: Tuple[float, float], mode: str = "DRIVE", traffic: bool = True) -> Optional[Dict[str, Any]]:
    if not _MAPS_KEY:
        LOGGER.warning("GOOGLE_MAPS_API_KEY missing for routes")
        return None
    ox, oy = origin
    dx, dy = dest
    body = {"origin": {"location": {"latLng": {"latitude": ox, "longitude": oy}}}, "destination": {"location": {"latLng": {"latitude": dx, "longitude": dy}}}, "travelMode": mode}
    if mode == "DRIVE" and traffic:
        body["routingPreference"] = "TRAFFIC_AWARE_OPTIMAL"
        body["departureTime"] = datetime.now(timezone.utc).isoformat()
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": _MAPS_KEY, "X-Goog-FieldMask": "routes.duration,routes.distanceMeters"}
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
    seconds = int(float(duration_iso[:-1])) if isinstance(duration_iso, str) and duration_iso.endswith("s") else None
    return {"seconds": seconds, "distance_meters": meters, "raw": r0}

# ---------------------------------------------------------------------------
# Public API (Updated error handling for SDK)
# ---------------------------------------------------------------------------

async def resolve_location_with_gemini(query: PlaceQuery, anchor: Anchor) -> ResolveResult:
    base = ResolveResult(status=STATUS_NOT_FOUND, anchor=anchor, scope=anchor.scope or "real", message="Unable to resolve location.")
    if not _GEMINI_KEY:
        base.errors.append("gemini_api_key_missing")
        LOGGER.error("GOOGLE_GEMINI_API_KEY / GOOGLE_API_KEY missing")
        return base
    try:
        text, gm = await _gemini_tools_call(query, anchor)
    except google_exceptions.GoogleAPICallError as exc:
        base.errors.append(f"gemini_api_error:{exc}")
        base.message = "Gemini lookup failed due to an API error."
        LOGGER.error("Gemini API error: %s", exc)
        return base
    except Exception as exc:
        base.errors.append(f"gemini_request_failed:{exc}")
        base.message = "Gemini lookup failed."
        LOGGER.exception("Gemini request failed", exc_info=exc)
        return base

    candidates: List[Candidate] = []
    if gm:
        LOGGER.info("✓ Google Maps grounding present")
        try:
            candidates = await _candidates_from_maps_grounding(gm, anchor)
        except Exception as exc:
            LOGGER.warning("Grounding->candidates failed: %s", exc)
    else:
        LOGGER.warning("✗ No Google Maps grounding in response")

    events = await _schema_only_events(text) if text else []
    if not candidates and not events:
        base.errors.append("gemini_no_results")
        base.message = f"No matches for '{query.target or query.raw_text}'."
        return base

    status = STATUS_EXACT if len(candidates) == 1 else (STATUS_MULTIPLE if len(candidates) > 1 else STATUS_NOT_FOUND)
    message = (f"Found: {candidates[0].place.name}" if status == STATUS_EXACT else f"Found {len(candidates)} places" if status == STATUS_MULTIPLE else (f"Found {len(events)} event(s)" if events else "No places found"))
    operations: List[Dict[str, Any]] = [{"op": "events", "items": events}] if events else []
    return ResolveResult(status=status, message=message, candidates=candidates, operations=operations, choices=[c.place.name for c in candidates] if status == STATUS_MULTIPLE else [], anchor=anchor, scope=anchor.scope or "real")

async def eta_between_with_gemini(origin_query: PlaceQuery, destination_query: PlaceQuery, anchor: Anchor, mode: str = "DRIVE") -> ResolveResult:
    origin_res = await resolve_location_with_gemini(origin_query, anchor)
    dest_res = await resolve_location_with_gemini(destination_query, anchor)
    if not origin_res.candidates or not dest_res.candidates:
        msg = "Could not resolve both endpoints."
        if not origin_res.candidates: msg += f" Missing origin '{origin_query.target or origin_query.raw_text}'."
        if not dest_res.candidates: msg += f" Missing destination '{destination_query.target or destination_query.raw_text}'."
        return ResolveResult(status=STATUS_NOT_FOUND, message=msg, anchor=anchor, scope=anchor.scope or "real", errors=["eta_endpoints_unresolved"])
    o, d = origin_res.candidates[0].place, dest_res.candidates[0].place
    if o.lat is None or o.lon is None or d.lat is None or d.lon is None:
        return ResolveResult(status=STATUS_NOT_FOUND, message="Resolved endpoints but coordinates missing for routing.", candidates=[o, d], anchor=anchor, scope=anchor.scope or "real", errors=["eta_coords_missing"])
    eta = await _compute_eta((o.lat, o.lon), (d.lat, d.lon), mode=mode, traffic=True)
    operations: List[Dict[str, Any]] = [{"op": "route.eta", "mode": mode, "from": {"name": o.name, "lat": o.lat, "lon": o.lon}, "to": {"name": d.name, "lat": d.lat, "lon": d.lon}, "eta_seconds": eta.get("seconds") if eta else None, "distance_meters": eta.get("distance_meters") if eta else None}]
    for res in (origin_res, dest_res):
        for op in res.operations or []:
            if op.get("op") == "events" and op.get("items"):
                operations.append(op)
    msg = f"ETA {eta['seconds']}s, distance {eta['distance_meters']} m" if eta else "ETA unavailable"
    return ResolveResult(status=STATUS_TRAVEL_PLAN, message=msg, candidates=[o, d], operations=operations, anchor=anchor, scope=anchor.scope or "real")
