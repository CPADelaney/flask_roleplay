# nyx/location/gemini_maps_adapter.py
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from .query import PlaceQuery
from .types import (
    Anchor,
    Candidate,
    Place,
    ResolveResult,
    STATUS_ASK,
    STATUS_EXACT,
    STATUS_MULTIPLE,
    STATUS_NOT_FOUND,
    STATUS_TRAVEL_PLAN,
)

LOGGER = logging.getLogger(__name__)

_GEMINI_ENDPOINT = os.getenv(
    "GOOGLE_GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
)


def _anchor_context(anchor: Anchor) -> str:
    parts: List[str] = []
    if anchor.label:
        parts.append(f"Anchor label: {anchor.label}")
    if anchor.primary_city:
        parts.append(f"City: {anchor.primary_city}")
    if anchor.region:
        parts.append(f"Region/State: {anchor.region}")
    if anchor.country:
        parts.append(f"Country: {anchor.country}")
    if anchor.lat is not None and anchor.lon is not None:
        parts.append(f"Coordinates: {anchor.lat:.6f}, {anchor.lon:.6f}")
    focus = anchor.focus
    if focus and focus.meta:
        focus_meta = ", ".join(f"{k}={v}" for k, v in focus.meta.items())
        if focus_meta:
            parts.append(f"Focus meta: {focus_meta}")
    return "\n".join(parts)


def _build_prompt(query: PlaceQuery, anchor: Anchor) -> str:
    context_lines = _anchor_context(anchor)
    target = query.target or query.raw_text
    prompt = f"""
You are an assistant that maps natural language requests to real-world places.
Use the anchor context to reason about which venues or points of interest the user likely means.
Return JSON only. The JSON must follow this schema:
{{
  "status": "exact | multiple | travel_plan | ask | not_found",
  "message": "short status message",
  "candidates": [
    {{
      "name": "place name",
      "level": "venue|city|region|route",
      "latitude": number,
      "longitude": number,
      "confidence": 0.0-1.0,
      "address": {{"line1": str, "city": str, "region": str, "country": str, "postal_code": str}},
      "travel_time_minutes": number | null,
      "category": "category or type",
      "notes": "optional details"
    }}
  ],
  "operations": [
    {{
      "op": "poi.navigate" | "travel.plan" | "travel.estimate",
      "label": "name",
      "lat": number,
      "lon": number,
      "travel_time_min": number | null
    }}
  ]
}}
Anchor context:
{context_lines or "(unknown)"}
User query: {target}
If you are uncertain, return status "ask" with an explanatory message. Ensure numbers are valid floats.
"""
    return prompt.strip()


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    # Remove markdown fences such as ```json ... ```
    fence_match = re.match(r"```(json)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(2).strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    return text


def _extract_text_candidates(payload: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _normalise_status(raw_status: Optional[str], *, fallback: str) -> str:
    if not raw_status:
        return fallback
    s = str(raw_status).strip().lower()
    if s in {STATUS_EXACT, STATUS_MULTIPLE, STATUS_TRAVEL_PLAN, STATUS_ASK, STATUS_NOT_FOUND}:
        return s
    return fallback


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_operations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for op in data.get("operations") or []:
        if isinstance(op, dict):
            ops.append(op)
    return ops


def _build_candidate(item: Dict[str, Any]) -> Optional[Candidate]:
    name = item.get("name") or item.get("label")
    if not isinstance(name, str) or not name.strip():
        return None
    level = item.get("level") or "venue"
    lat = _safe_float(item.get("latitude") or item.get("lat"))
    lon = _safe_float(item.get("longitude") or item.get("lon"))
    address = item.get("address") if isinstance(item.get("address"), dict) else {}
    confidence = _safe_float(item.get("confidence")) or 0.0
    travel_time_min = _safe_float(item.get("travel_time_minutes") or item.get("travel_time_min"))
    meta: Dict[str, Any] = {
        "source": "gemini",
        "scope": "real",
    }
    if "category" in item:
        meta["category"] = item.get("category")
    if item.get("notes"):
        meta["notes"] = item.get("notes")
    if travel_time_min is not None:
        meta["travel_time_minutes"] = travel_time_min
    place = Place(
        name=name.strip(),
        level=level if isinstance(level, str) else "venue",
        key=item.get("id") or item.get("place_id") or name.strip().lower(),
        lat=lat,
        lon=lon,
        address=dict(address),
        meta=meta,
    )
    return Candidate(
        place=place,
        confidence=max(0.0, min(1.0, confidence)),
        raw=dict(item),
    )


async def resolve_location_with_gemini(query: PlaceQuery, anchor: Anchor) -> ResolveResult:
    """Resolve a real-world place using the Gemini API.

    The adapter is best-effort: if Gemini cannot be reached or the response cannot be
    parsed we return a not_found ResolveResult containing the encountered errors so
    callers can fall back to other providers.
    """

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    base_result = ResolveResult(
        status=STATUS_NOT_FOUND,
        anchor=anchor,
        scope=anchor.scope or "real",
        message="Gemini resolution unavailable.",
    )
    if not api_key:
        base_result.errors.append("gemini_api_key_missing")
        return base_result

    prompt = _build_prompt(query, anchor)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(f"{_GEMINI_ENDPOINT}?key={api_key}", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # pragma: no cover - network failure guard
        LOGGER.exception("Gemini request failed", exc_info=exc)
        base_result.errors.append(f"gemini_request_failed:{exc}")
        base_result.message = "Gemini location lookup failed."
        return base_result

    text = _extract_text_candidates(data)
    if not text:
        base_result.errors.append("gemini_empty_response")
        base_result.message = "Gemini returned no content."
        return base_result

    cleaned = _clean_json_text(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Gemini response parse error: %s", cleaned)
        base_result.errors.append(f"gemini_parse_error:{exc}")
        base_result.message = "Gemini response unparseable."
        return base_result

    candidates_raw = parsed.get("candidates") if isinstance(parsed, dict) else None
    candidates: List[Candidate] = []
    for item in candidates_raw or []:
        if isinstance(item, dict):
            cand = _build_candidate(item)
            if cand:
                candidates.append(cand)
    status = _normalise_status(parsed.get("status") if isinstance(parsed, dict) else None, fallback=STATUS_MULTIPLE if len(candidates) > 1 else STATUS_EXACT if candidates else STATUS_NOT_FOUND)

    operations = _collect_operations(parsed if isinstance(parsed, dict) else {})
    if not operations and candidates:
        for cand in candidates:
            travel = cand.raw.get("travel_time_minutes") if isinstance(cand.raw, dict) else None
            operations.append(
                {
                    "op": "poi.navigate",
                    "label": cand.place.name,
                    "lat": cand.place.lat,
                    "lon": cand.place.lon,
                    "travel_time_min": travel,
                }
            )

    message = parsed.get("message") if isinstance(parsed, dict) else None
    if not isinstance(message, str) or not message.strip():
        if status == STATUS_EXACT and candidates:
            message = f"Heading to {candidates[0].place.name}."
        elif status == STATUS_MULTIPLE and candidates:
            message = "Here are some possibilities from Gemini."
        elif status == STATUS_TRAVEL_PLAN:
            message = "Here is a travel plan suggestion."
        elif status == STATUS_ASK:
            message = "Gemini needs more detail about this place."
        else:
            message = "Gemini could not resolve the location."

    choices: List[str] = [c.place.name for c in candidates] if status == STATUS_MULTIPLE else []

    result = ResolveResult(
        status=status,
        message=message,
        candidates=candidates,
        operations=operations,
        choices=choices,
        anchor=anchor,
        scope=anchor.scope or "real",
    )

    if not candidates and status == STATUS_NOT_FOUND:
        result.errors.append("gemini_no_candidates")
    return result
