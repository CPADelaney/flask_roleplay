"""Async Redis helpers for deep lore bundles."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from lore.cache_version import compute_lore_version_for_location

try:  # pragma: no cover - optional dependency shim for tests
    import redis.asyncio as redis_async
except Exception:  # pragma: no cover - redis is optional in some environments
    redis_async = None  # type: ignore


logger = logging.getLogger(__name__)

_KEY_TEMPLATE = "lore:deep:{user_id}:{conversation_id}:{location_id}"
_TTL_SECONDS = 60 * 60
_CLIENT: Optional["redis_async.Redis"] = None
_CLIENT_LOCK = asyncio.Lock()


async def _get_client() -> Optional["redis_async.Redis"]:
    """Return a shared Redis client for deep lore bundles."""

    if redis_async is None:
        return None

    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    async with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT

        redis_url = (
            os.getenv("NYX_LORE_DEEP_REDIS")
            or os.getenv("NYX_LORE_VERSION_REDIS")
            or os.getenv("NYX_SNAPSHOT_REDIS")
            or os.getenv("REDIS_URL")
        )
        if not redis_url:
            return None

        try:
            client = redis_async.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,
            )
            await client.ping()
        except Exception:  # pragma: no cover - best effort connection initialisation
            logger.debug("Failed to initialise Redis client for deep lore cache", exc_info=True)
            return None

        _CLIENT = client
        return _CLIENT


def _build_key(user_id: Any, conversation_id: Any, location_id: Any) -> Optional[str]:
    if user_id is None or conversation_id is None or location_id is None:
        return None
    return _KEY_TEMPLATE.format(
        user_id=str(user_id),
        conversation_id=str(conversation_id),
        location_id=str(location_id),
    )


async def get_deep_lore_bundle(
    user_id: Any,
    conversation_id: Any,
    location_id: Any,
) -> Optional[Any]:
    """Return cached deep lore data when the version matches the live marker."""

    key = _build_key(user_id, conversation_id, location_id)
    if not key:
        return None

    client = await _get_client()
    if client is None:
        return None

    try:
        raw_value = await client.get(key)
    except Exception:  # pragma: no cover - Redis failures should not bubble
        logger.debug("Failed to read deep lore bundle from Redis", exc_info=True)
        return None

    if not raw_value:
        return None

    if isinstance(raw_value, bytes):
        encoded = raw_value.decode("utf-8", errors="ignore")
    elif isinstance(raw_value, str):
        encoded = raw_value
    else:
        return None

    try:
        payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    cached_version = payload.get("lore_version")
    current_version = compute_lore_version_for_location(str(location_id))
    if not cached_version or cached_version != current_version:
        return None

    return payload.get("data")


async def set_deep_lore_bundle(
    user_id: Any,
    conversation_id: Any,
    location_id: Any,
    data: Any,
) -> bool:
    """Store a deep lore bundle with the current location lore version marker."""

    key = _build_key(user_id, conversation_id, location_id)
    if not key:
        return False

    client = await _get_client()
    if client is None:
        return False

    version = compute_lore_version_for_location(str(location_id))
    payload = {"lore_version": version, "data": data}

    try:
        encoded = json.dumps(payload)
    except (TypeError, ValueError):
        logger.debug("Deep lore payload is not JSON serialisable", exc_info=True)
        return False

    try:
        await client.set(key, encoded, ex=_TTL_SECONDS)
        return True
    except Exception:  # pragma: no cover - Redis failures should not bubble
        logger.debug("Failed to store deep lore bundle in Redis", exc_info=True)
        return False


__all__ = ["get_deep_lore_bundle", "set_deep_lore_bundle"]

