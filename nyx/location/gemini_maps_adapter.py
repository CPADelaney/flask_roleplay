# nyx/location/gemini_maps_adapter.py
from __future__ import annotations

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Models that support Maps + Search grounding (Oct 2025): 2.5 Pro / 2.5 Flash / 2.5 Flash-Lite / 2.0 Flash
# Default to low-latency flash; override via env if you need pro.
_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash")
_GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent"

# Routes + Places (v1) endpoints
_ROUTES_ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"
_PLACES_BASE = "https://places.googleapis.com/v1"  # GET /v1/places/{placeId}

# API keys: you may use one restricted key enabled for Gemini + Maps or separate keys.
_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")

# Events-only schema (used in Step B reformat; places we assemble ourselves deterministically)
_EVENTS_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "events": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "start": {"type": "STRING"},  # ISO 8601 if known
                    "end": {"type": "STRING"},
                    "venueName": {"type": "STRING"},
                    "address": {"type": "STRING"},
                    "link": {"type": "STRING"},
                    "price": {"type": "STRING"},
                },
                "propertyOrdering": ["title", "start", "end", "venueName", "address", "link", "price"],
            },
        }
    },
    "propertyOrdering": ["events"],
}

# ---------------------------------------------------------------------------
# Prompting
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
    # Tools-on prompt: ask for places details and any relevant events/showtimes. Free text is fine here.
    return f"""Task: Resolve real places and relevant events/showtimes for:
{target}

Context:
{ctx}

Use Google Maps grounding to identify exact places. Also use Google Search grounding to find any showtimes or one-off events that match the query and area. Provide concise factual details in your answer text so they can be reformatted later."""
# We intentionally do NOT request structured output here to avoid tool+schema 400s.

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _strip_structured_output(payload: dict) -> dict:
    """Remove structured-output options from a request payload."""
    p2 = copy.deepcopy(payload)
    gen = p2.get("generationConfig", {})
    gen.pop("responseMimeType", None)
    gen.pop("responseSchema", None)
    gen.pop("responseJsonSchema", None)
    if not gen:
        p2.pop("generationConfig", None)
    return p2


def _is_tool_json_conflict(err_json: dict) -> bool:
    msg = (err_json or {}).get("error", {}).get("message", "")
    return "response mime type" in msg.lower() and "unsupported" in msg.lower()


async def _post_gemini(payload: dict, api_key: str) -> dict:
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(_GEMINI_ENDPOINT, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()

        # Try to surface detailed error
        try:
            err = resp.json()
        except Exception:
            err = {"error": {"code": resp.status_code, "message": resp.text}}

        # Auto-retry if the tool+schema combo is the culprit
        if resp.status_code == 400 and _is_tool_json_conflict(err):
            LOGGER.info("Gemini: tool+JSON conflict. Retrying without structured output.")
            payload2 = _strip_structured_output(payload)
            resp2 = await client.post(_GEMINI_ENDPOINT, headers=headers, json=payload2)
            resp2.raise_for_status()
            return resp2.json()

        LOGGER.error("Gemini %s error: %s", resp.status_code, json.dumps(err))
        resp.raise_for_status()
        return {}  # never reached


def _extract_text_candidates(payload: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for c in payload.get("candidates") or []:
        content = c.get("content") or {}
        for part in content.get("parts") or []:
            t = part.get("text")
            if isinstance(t, str):
                chunks.append(t)
    return "\n".join(chunks).strip()


def _extract_grounding_metadata(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for c in payload.get("candidates") or []:
        gm = c.get("groundingMetadata")
        if gm:
            return gm
    return None


# ---------------------------------------------------------------------------
# Grounding -> Candidates
# ---------------------------------------------------------------------------

def _maps_chunks(grounding_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ch in grounding_metadata.get("groundingChunks") or []:
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
    """Accept either 'places/ChIJ...' or bare 'ChIJ...'; return 'places/...'; or None."""
    if not place_id_or_name:
        return None
    return place_id_or_name if place_id_or_name.startswith("places/") else f"places/{place_id_or_name}"


async def _places_get_details(client: httpx.AsyncClient, place_res_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Enrich with Places Details v1. Keep the field mask tight for speed and cost.
    """
    if not _MAPS_KEY or not place_res_name:
        return None

    url = f"{_PLACES_BASE}/{place_res_name}"
    headers = {
        "X-Goog-Api-Key": _MAPS_KEY,
        "Content-Type": "application/json",
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


async def _candidates_from_maps_grounding(gm: Dict[str, Any], anchor: Anchor) -> List[Candidate]:
    """
    Convert Maps-grounded chunks to Candidate objects; enrich with Places Details for deterministic data.
    """
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

            # Initial rough coords from URL if present
            lat, lon = _extract_coords_from_maps_url(uri)

            # Enrich via Places Details for address/hours/phone/type/coords
            details = await _places_get_details(client, place_res) if place_res else None

            addr = title
            primary_type = None
            website = None
            phone = None
            hours_obj = None

            if details:
                addr = details.get("formattedAddress") or addr
                loc = (details.get("location") or {})
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
                address={
                    "line1": addr,
                    "_normalized_admin_path": {
                        "city": anchor.primary_city,
                        "region": anchor.region,
                        "country": anchor.country,
                    },
                },
                meta={
                    "source": "gemini_maps_grounding",
                    "uri": uri,
                    "placeId": place_res,
                    "grounded": True,
                    "primaryType": primary_type,
                    "websiteUri": website,
                    "phone": phone,
                    "hours": hours_obj,
                },
            )

            candidates.append(
                Candidate(place=place, confidence=0.85, rationale="google_maps_grounding", raw=m)
            )

    return candidates


# ---------------------------------------------------------------------------
# Gemini calls
# ---------------------------------------------------------------------------

async def _gemini_tools_call(query: PlaceQuery, anchor: Anchor) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Step A: Run Gemini with Maps + Search tools ON and NO structured-output.
    Returns (free_text, grounding_metadata).
    """
    if not _GEMINI_KEY:
        raise RuntimeError("gemini_api_key_missing")

    prompt = _build_prompt_for_tools(query, anchor)

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        # REST casing: camelCase for Maps, snake_case for Search
        "tools": [{"googleMaps": {}}, {"google_search": {}}],
        "toolConfig": {
            "retrievalConfig": {
                # Only include latLng when we actually have it
                "latLng": (
                    {"latitude": anchor.lat, "longitude": anchor.lon}
                    if anchor.lat is not None and anchor.lon is not None
                    else None
                )
            }
        },
        "generationConfig": {
            "temperature": 0.2
        },
    }

    # Clean nulls from payload (latLng None can upset picky parsers)
    if payload.get("toolConfig", {}).get("retrievalConfig", {}).get("latLng") is None:
        try:
            del payload["toolConfig"]["retrievalConfig"]["latLng"]
        except KeyError:
            pass

    data = await _post_gemini(payload, _GEMINI_KEY)
    text = _extract_text_candidates(data)
    gm = _extract_grounding_metadata(data)
    return text, gm


async def _schema_only_events(text_source: str) -> List[Dict[str, Any]]:
    """
    Step B: If you want structured events/showtimes, call Gemini again with NO tools,
    responseMimeType=application/json and a minimal events schema. Feed it the Step A text.
    """
    if not _GEMINI_KEY or not text_source:
        return []

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Extract any showtimes or one-off events from the following text. "
                            "Return strict JSON matching the provided schema. "
                            "If none found, return events: [].\n\n"
                            f"{text_source}"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": _EVENTS_SCHEMA,
        },
    }

    data = await _post_gemini(payload, _GEMINI_KEY)
    # Get the JSON string from the first candidate
    for c in data.get("candidates", []):
        for part in (c.get("content") or {}).get("parts", []):
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                try:
                    parsed = json.loads(t)
                    return parsed.get("events") or []
                except Exception:
                    continue
    return []


# ---------------------------------------------------------------------------
# Routing / ETA
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_location_with_gemini(query: PlaceQuery, anchor: Anchor) -> ResolveResult:
    """
    Resolve places with Google Maps grounding and optionally capture time-sensitive events via Search.
    - Step A: tools ON (Maps + Search), NO structured-output. Parse grounding metadata.
    - Enrich places via Places Details for address/hours/phone/coords.
    - Step B: schema-only events extraction from Step A text (no tools).
    """
    base = ResolveResult(
        status=STATUS_NOT_FOUND,
        anchor=anchor,
        scope=anchor.scope or "real",
        message="Unable to resolve location.",
    )

    if not _GEMINI_KEY:
        base.errors.append("gemini_api_key_missing")
        LOGGER.error("GOOGLE_GEMINI_API_KEY / GOOGLE_API_KEY missing")
        return base

    try:
        text, gm = await _gemini_tools_call(query, anchor)
    except httpx.HTTPStatusError as exc:
        base.errors.append(f"gemini_http_error:{exc.response.status_code}")
        base.message = "Gemini lookup failed."
        LOGGER.error("Gemini HTTP %s: %s", exc.response.status_code, exc.response.text)
        return base
    except Exception as exc:
        base.errors.append(f"gemini_request_failed:{exc}")
        base.message = "Gemini lookup failed."
        LOGGER.exception("Gemini request failed", exc_info=exc)
        return base

    # Candidates from Maps grounding
    candidates: List[Candidate] = []
    if gm:
        LOGGER.info("✓ Google Maps grounding present")
        try:
            candidates = await _candidates_from_maps_grounding(gm, anchor)
        except Exception as exc:
            LOGGER.warning("Grounding->candidates failed: %s", exc)
    else:
        LOGGER.warning("✗ No Google Maps grounding in response")

    # Events via schema-only pass on the text content
    events = await _schema_only_events(text) if text else []

    if not candidates and not events:
        base.errors.append("gemini_no_results")
        base.message = f"No matches for '{query.target or query.raw_text}'."
        return base

    status = (
        STATUS_EXACT if len(candidates) == 1 else
        (STATUS_MULTIPLE if len(candidates) > 1 else STATUS_NOT_FOUND)
    )

    message = (
        f"Found: {candidates[0].place.name}" if status == STATUS_EXACT
        else f"Found {len(candidates)} places" if status == STATUS_MULTIPLE
        else (f"Found {len(events)} event(s)" if events else "No places found")
    )

    operations: List[Dict[str, Any]] = []
    if events:
        operations.append({"op": "events", "items": events})

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
    Resolve origin/destination (Maps grounding), then compute ETA via Routes API.
    Also surfaces any events extracted during each endpoint resolution.
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

    # Bubble up any event lists discovered on either endpoint
    for res in (origin_res, dest_res):
        for op in res.operations or []:
            if op.get("op") == "events" and op.get("items"):
                operations.append(op)

    msg = f"ETA {eta['seconds']}s, distance {eta['distance_meters']} m" if eta else "ETA unavailable"

    return ResolveResult(
        status=STATUS_TRAVEL_PLAN,
        message=msg,
        candidates=[origin_res.candidates[0], dest_res.candidates[0]],
        operations=operations,
        anchor=anchor,
        scope=anchor.scope or "real",
    )
