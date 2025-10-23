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

# Use the correct model that supports grounding
_GEMINI_ENDPOINT = os.getenv(
    "GOOGLE_GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",  # Changed!
)


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
    """Build a prompt that leverages Google Maps grounding."""
    context_lines = _anchor_context(anchor)
    target = query.target or query.raw_text
    
    # Simplified prompt - let Gemini use Maps grounding naturally
    prompt = f"""I'm looking for: {target}

Current context:
{context_lines or "Unknown location"}

Please help me find this place. Provide:
1. The exact name and address
2. Coordinates (latitude, longitude)
3. Distance/travel time from my current location if applicable
4. Any relevant details (business hours, category, etc.)

If you find multiple matches, list the top 3-5 options."""
    
    return prompt.strip()


def _clean_json_text(text: str) -> str:
    """Remove markdown fences and extract JSON."""
    text = text.strip()
    if not text:
        return text
    fence_match = re.match(r"```(json)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(2).strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    return text


def _extract_text_candidates(payload: Dict[str, Any]) -> str:
    """Extract text from Gemini response."""
    chunks: List[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_grounding_metadata(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract Google Maps grounding metadata from response."""
    for candidate in payload.get("candidates") or []:
        grounding_metadata = candidate.get("groundingMetadata")
        if grounding_metadata:
            return grounding_metadata
    return None


def _parse_grounded_response(
    text: str, 
    grounding_metadata: Optional[Dict[str, Any]],
    anchor: Anchor
) -> List[Candidate]:
    """Parse Gemini response with Google Maps grounding data."""
    candidates: List[Candidate] = []
    
    # Extract places from grounding metadata
    if grounding_metadata:
        grounding_supports = grounding_metadata.get("groundingSupports", [])
        
        for support in grounding_supports:
            # Check if this is a Maps grounding support
            grounding_chunk_indices = support.get("groundingChunkIndices", [])
            segment = support.get("segment", {})
            
            # Look for place information in the grounding chunks
            for chunk in grounding_metadata.get("groundingChunks", []):
                # Check if this chunk is referenced by this support
                if chunk.get("web"):  # This is a web/maps result
                    web_info = chunk.get("web", {})
                    
                    # Try to extract structured place data
                    place_name = web_info.get("title", "Unknown Place")
                    uri = web_info.get("uri", "")
                    
                    # Parse coordinates from URI if it's a Google Maps link
                    lat, lon = _extract_coords_from_maps_url(uri)
                    
                    if not lat and anchor.lat:
                        lat = anchor.lat
                    if not lon and anchor.lon:
                        lon = anchor.lon
                    
                    place = Place(
                        name=place_name,
                        level="venue",
                        lat=lat,
                        lon=lon,
                        address={
                            "line1": web_info.get("title"),
                            "_normalized_admin_path": {
                                "city": anchor.primary_city,
                                "region": anchor.region,
                                "country": anchor.country,
                            }
                        },
                        meta={
                            "source": "gemini_maps_grounding",
                            "uri": uri,
                            "grounded": True,
                        },
                    )
                    
                    candidate = Candidate(
                        place=place,
                        confidence=0.85,  # High confidence for grounded results
                        rationale="google_maps_grounding",
                        raw=chunk,
                    )
                    candidates.append(candidate)
    
    # If no grounding metadata, try to parse structured response from text
    if not candidates:
        candidates = _parse_text_fallback(text, anchor)
    
    return candidates


def _extract_coords_from_maps_url(url: str) -> tuple[Optional[float], Optional[float]]:
    """Extract coordinates from a Google Maps URL."""
    if not url:
        return None, None
    
    # Pattern: @lat,lon,zoom
    match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', url)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # Pattern: ?q=lat,lon
    match = re.search(r'[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)', url)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    return None, None


def _parse_text_fallback(text: str, anchor: Anchor) -> List[Candidate]:
    """Fallback parser if grounding metadata is not available."""
    candidates: List[Candidate] = []
    
    # Try to parse as JSON first
    try:
        cleaned = _clean_json_text(text)
        parsed = json.loads(cleaned)
        
        if isinstance(parsed, dict):
            places = parsed.get("places") or parsed.get("results") or [parsed]
            for item in places if isinstance(places, list) else [places]:
                if not isinstance(item, dict):
                    continue
                
                name = item.get("name") or item.get("title")
                if not name:
                    continue
                
                lat = item.get("lat") or item.get("latitude")
                lon = item.get("lon") or item.get("longitude")
                
                place = Place(
                    name=name,
                    level="venue",
                    lat=float(lat) if lat else anchor.lat,
                    lon=float(lon) if lon else anchor.lon,
                    address=item.get("address", {}),
                    meta={
                        "source": "gemini_text_parsing",
                        "category": item.get("category") or item.get("type"),
                    },
                )
                
                candidates.append(Candidate(
                    place=place,
                    confidence=0.6,
                    rationale="text_parsing",
                    raw=item,
                ))
    except json.JSONDecodeError:
        LOGGER.debug("Could not parse response as JSON, using text analysis")
    
    return candidates


async def resolve_location_with_gemini(query: PlaceQuery, anchor: Anchor) -> ResolveResult:
    """
    Resolve a real-world place using Gemini with Google Maps grounding.
    
    This uses Gemini's built-in Google Maps integration to find real places.
    """
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    
    base_result = ResolveResult(
        status=STATUS_NOT_FOUND,
        anchor=anchor,
        scope=anchor.scope or "real",
        message="Gemini resolution unavailable.",
    )
    
    if not api_key:
        base_result.errors.append("gemini_api_key_missing")
        LOGGER.warning("GOOGLE_GEMINI_API_KEY not set")
        return base_result

    prompt = _build_prompt(query, anchor)
    
    # THIS IS THE KEY CHANGE: Enable Google Maps grounding
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "tools": [
            {
                "googleSearchRetrieval": {
                    "dynamicRetrievalConfig": {
                        "mode": "MODE_DYNAMIC",
                        "dynamicThreshold": 0.7
                    }
                }
            }
        ],
        # Optional: Add generation config for better structured output
        "generationConfig": {
            "temperature": 0.2,  # Lower temperature for factual responses
            "topP": 0.8,
            "topK": 40,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(
                f"{_GEMINI_ENDPOINT}?key={api_key}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            LOGGER.debug(f"Gemini response: {json.dumps(data, indent=2)}")
            
    except httpx.HTTPStatusError as exc:
        LOGGER.error(f"Gemini HTTP error: {exc.response.status_code} - {exc.response.text}")
        base_result.errors.append(f"gemini_http_error:{exc.response.status_code}")
        base_result.message = "Gemini location lookup failed."
        return base_result
    except Exception as exc:
        LOGGER.exception("Gemini request failed", exc_info=exc)
        base_result.errors.append(f"gemini_request_failed:{exc}")
        base_result.message = "Gemini location lookup failed."
        return base_result

    text = _extract_text_candidates(data)
    if not text:
        base_result.errors.append("gemini_empty_response")
        base_result.message = "Gemini returned no content."
        return base_result

    # Extract grounding metadata (this is where Google Maps data lives)
    grounding_metadata = _extract_grounding_metadata(data)
    
    if grounding_metadata:
        LOGGER.info("✓ Gemini used Google Maps grounding")
    else:
        LOGGER.warning("✗ No Google Maps grounding in response")

    # Parse candidates from grounded response
    candidates = _parse_grounded_response(text, grounding_metadata, anchor)
    
    if not candidates:
        base_result.errors.append("gemini_no_candidates")
        base_result.message = f"Couldn't find '{query.target}' near {anchor.label or 'your location'}."
        return base_result

    # Determine status
    if len(candidates) == 1:
        status = STATUS_EXACT
        message = f"Found: {candidates[0].place.name}"
    elif len(candidates) > 1:
        status = STATUS_MULTIPLE
        message = f"Found {len(candidates)} possible matches"
    else:
        status = STATUS_NOT_FOUND
        message = "No matches found"

    # Build operations
    operations = []
    for cand in candidates:
        operations.append({
            "op": "poi.navigate",
            "label": cand.place.name,
            "lat": cand.place.lat,
            "lon": cand.place.lon,
            "category": cand.place.meta.get("category"),
            "grounded": cand.place.meta.get("grounded", False),
        })

    return ResolveResult(
        status=status,
        message=message,
        candidates=candidates,
        operations=operations,
        choices=[c.place.name for c in candidates] if status == STATUS_MULTIPLE else [],
        anchor=anchor,
        scope=anchor.scope or "real",
    )
