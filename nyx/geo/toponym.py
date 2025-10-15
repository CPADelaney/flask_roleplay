"""Toponym geocoding helpers with Postgres-backed caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "nominatim"
DEFAULT_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # one week
DEFAULT_HTTP_TIMEOUT = 10.0
USER_AGENT = "nyx-toponym/1.0"


@dataclass
class GeocodeResult:
    """Structured representation of a geocode lookup."""

    query: str
    normalized_query: str
    provider: str
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: float
    payload: Dict[str, Any]


def _normalize(text: str) -> str:
    """Normalize free-form user text for cache lookups."""

    return " ".join((text or "").strip().lower().split())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _confidence_from_payload(payload: Dict[str, Any]) -> float:
    raw_importance = payload.get("importance")
    if raw_importance is None:
        return 0.0
    try:
        confidence = float(raw_importance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


async def _get_cached_entry(
    normalized_query: str,
    provider: str = DEFAULT_PROVIDER,
) -> Optional[Dict[str, Any]]:
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            """
            SELECT response, confidence, expires_at
            FROM geo_cache
            WHERE provider = $1 AND normalized_query = $2
            """,
            provider,
            normalized_query,
        )

    if not row:
        return None

    expires_at = row.get("expires_at")
    if expires_at and expires_at <= _now():
        return None

    payload = row.get("response") or {}
    confidence = row.get("confidence")
    if confidence is None:
        confidence = _confidence_from_payload(payload)

    return {
        "provider": provider,
        "normalized_query": normalized_query,
        "payload": payload,
        "confidence": max(0.0, min(float(confidence), 1.0)),
    }


async def _write_cache_entry(
    query: str,
    normalized_query: str,
    payload: Dict[str, Any],
    confidence: float,
    provider: str = DEFAULT_PROVIDER,
    ttl_seconds: Optional[int] = DEFAULT_CACHE_TTL_SECONDS,
) -> None:
    expires_at = None
    if ttl_seconds:
        expires_at = _now() + timedelta(seconds=ttl_seconds)

    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO geo_cache (
                provider,
                query,
                normalized_query,
                response,
                confidence,
                expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (provider, normalized_query) DO UPDATE SET
                query = EXCLUDED.query,
                response = EXCLUDED.response,
                confidence = EXCLUDED.confidence,
                expires_at = EXCLUDED.expires_at,
                updated_at = NOW()
            """,
            provider,
            query,
            normalized_query,
            payload,
            confidence,
            expires_at,
        )


async def _upsert_world_location(
    normalized_name: str,
    canonical_name: str,
    payload: Dict[str, Any],
    confidence: float,
) -> None:
    address = payload.get("address") or {}

    country_code = address.get("country_code")
    admin1 = address.get("state") or address.get("region")
    admin2 = address.get("county") or address.get("city") or address.get("municipality")

    latitude = payload.get("lat")
    longitude = payload.get("lon")

    try:
        lat_value = float(latitude) if latitude is not None else None
        lon_value = float(longitude) if longitude is not None else None
    except (TypeError, ValueError):
        lat_value = None
        lon_value = None

    feature_class = payload.get("class")
    feature_code = payload.get("type")

    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO world_locations (
                name,
                normalized_name,
                country_code,
                admin1,
                admin2,
                latitude,
                longitude,
                feature_class,
                feature_code,
                data_source,
                confidence
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (normalized_name) DO UPDATE SET
                name = EXCLUDED.name,
                country_code = COALESCE(EXCLUDED.country_code, world_locations.country_code),
                admin1 = COALESCE(EXCLUDED.admin1, world_locations.admin1),
                admin2 = COALESCE(EXCLUDED.admin2, world_locations.admin2),
                latitude = COALESCE(EXCLUDED.latitude, world_locations.latitude),
                longitude = COALESCE(EXCLUDED.longitude, world_locations.longitude),
                feature_class = COALESCE(EXCLUDED.feature_class, world_locations.feature_class),
                feature_code = COALESCE(EXCLUDED.feature_code, world_locations.feature_code),
                data_source = EXCLUDED.data_source,
                confidence = GREATEST(world_locations.confidence, EXCLUDED.confidence),
                updated_at = NOW()
            """,
            canonical_name,
            normalized_name,
            country_code,
            admin1,
            admin2,
            lat_value,
            lon_value,
            feature_class,
            feature_code,
            DEFAULT_PROVIDER,
            confidence,
        )


async def geocode(
    query: str,
    *,
    near: Optional[str] = None,
    provider: str = DEFAULT_PROVIDER,
    http_client: Optional[httpx.AsyncClient] = None,
    ttl_seconds: Optional[int] = DEFAULT_CACHE_TTL_SECONDS,
) -> Optional[GeocodeResult]:
    """Resolve a free-form location string into coordinates with caching."""

    normalized_query = _normalize(query)
    if not normalized_query:
        return None

    cached = await _get_cached_entry(normalized_query, provider=provider)
    if cached:
        payload = cached["payload"]
        confidence = cached["confidence"]
        lat_value = payload.get("lat")
        lon_value = payload.get("lon")
        try:
            latitude = float(lat_value) if lat_value is not None else None
            longitude = float(lon_value) if lon_value is not None else None
        except (TypeError, ValueError):
            latitude = None
            longitude = None

        return GeocodeResult(
            query=query,
            normalized_query=normalized_query,
            provider=provider,
            latitude=latitude,
            longitude=longitude,
            confidence=confidence,
            payload=payload,
        )

    close_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        close_client = True

    near_hint = (str(near).strip() if near is not None else "")
    request_parts = [str(query).strip()]
    if near_hint:
        request_parts.append(near_hint)
    request_query = ", ".join(part for part in request_parts if part)

    try:
        response = await http_client.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": request_query,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
            },
        )
        response.raise_for_status()
        payloads = response.json()
    except Exception:
        logger.exception("Failed to geocode query", extra={"query": request_query})
        payloads = []
    finally:
        if close_client:
            await http_client.aclose()

    if not payloads:
        return None

    payload = payloads[0] if isinstance(payloads, list) else payloads
    confidence = _confidence_from_payload(payload)

    await _write_cache_entry(
        query=request_query,
        normalized_query=normalized_query,
        payload=payload,
        confidence=confidence,
        provider=provider,
        ttl_seconds=ttl_seconds,
    )

    canonical_name = payload.get("name") or (payload.get("display_name") or query).split(",")[0].strip()
    normalized_name = _normalize(canonical_name)

    if confidence >= 0.5 and normalized_name:
        await _upsert_world_location(normalized_name, canonical_name, payload, confidence)

    try:
        latitude = float(payload.get("lat")) if payload.get("lat") is not None else None
        longitude = float(payload.get("lon")) if payload.get("lon") is not None else None
    except (TypeError, ValueError):
        latitude = None
        longitude = None

    return GeocodeResult(
        query=query,
        normalized_query=normalized_query,
        provider=provider,
        latitude=latitude,
        longitude=longitude,
        confidence=confidence,
        payload=payload,
    )


async def plausibility_score(
    toponym: str,
    *,
    near: Optional[str] = None,
    provider: str = DEFAULT_PROVIDER,
    http_client: Optional[httpx.AsyncClient] = None,
) -> float:
    """Return a heuristic plausibility score (0..1) for a location name."""

    normalized = _normalize(toponym)
    if not normalized:
        return 0.0

    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            """
            SELECT confidence
            FROM world_locations
            WHERE normalized_name = $1
            """,
            normalized,
        )

    if row and row.get("confidence") is not None:
        confidence = float(row["confidence"])
        return max(0.0, min(confidence, 1.0))

    cached = await _get_cached_entry(normalized, provider=provider)
    if cached:
        return cached["confidence"]

    result = await geocode(toponym, near=near, provider=provider, http_client=http_client)
    if not result:
        return 0.0

    return max(0.0, min(result.confidence, 1.0))
