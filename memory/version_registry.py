"""Lightweight registry for tracking per-player memory cache versions."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

try:
    from utils.caching import enhanced_main_cache
except Exception:  # pragma: no cover - defensive import guard
    enhanced_main_cache = None  # type: ignore

logger = logging.getLogger(__name__)

_MEMORY_VERSION_PREFIX = "memory:version:"
_local_versions: Dict[int, int] = {}
_lock = threading.RLock()


def _normalize_user_id(user_id: int) -> Optional[int]:
    """Normalize incoming IDs to ints; return None when invalid."""
    try:
        value = int(user_id)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _get_redis_client():
    """Return the shared Redis client, if available."""
    if enhanced_main_cache is None:
        return None
    return getattr(enhanced_main_cache, "redis_client", None)


def _key(user_id: int) -> str:
    return f"{_MEMORY_VERSION_PREFIX}{user_id}"


def get_memory_version(user_id: int, *, default: int = 0, force_refresh: bool = False) -> int:
    """Fetch the current memory version for a player.

    Falls back to an in-process cache when Redis is unavailable.
    """
    normalized = _normalize_user_id(user_id)
    if normalized is None:
        return default

    if not force_refresh:
        with _lock:
            cached = _local_versions.get(normalized)
            if cached is not None:
                return cached

    version = default
    client = _get_redis_client()
    if client is not None:
        try:
            raw = client.get(_key(normalized))
            if raw is not None:
                version = int(raw)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug("memory.version_registry: redis get failed for %s: %s", normalized, exc)

    with _lock:
        _local_versions[normalized] = version
    return version


def set_memory_version(user_id: int, version: int) -> int:
    """Explicitly set the memory version for a player."""
    normalized = _normalize_user_id(user_id)
    if normalized is None:
        return 0

    value = int(version or 0)
    client = _get_redis_client()
    if client is not None:
        try:
            client.set(_key(normalized), value)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug("memory.version_registry: redis set failed for %s: %s", normalized, exc)

    with _lock:
        _local_versions[normalized] = value
    return value


def bump_memory_version(user_id: int) -> int:
    """Atomically increment and return the player's memory version."""
    normalized = _normalize_user_id(user_id)
    if normalized is None:
        return 0

    client = _get_redis_client()
    new_version: Optional[int] = None
    if client is not None:
        try:
            new_version = int(client.incr(_key(normalized)))
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug("memory.version_registry: redis incr failed for %s: %s", normalized, exc)
            new_version = None

    if new_version is None:
        with _lock:
            new_version = _local_versions.get(normalized, 0) + 1
            _local_versions[normalized] = new_version
    else:
        with _lock:
            _local_versions[normalized] = new_version

    return new_version


def clear_memory_version(user_id: int) -> None:
    """Remove any cached version information for a player."""
    normalized = _normalize_user_id(user_id)
    if normalized is None:
        return

    client = _get_redis_client()
    if client is not None:
        try:
            client.delete(_key(normalized))
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug("memory.version_registry: redis delete failed for %s: %s", normalized, exc)

    with _lock:
        _local_versions.pop(normalized, None)


__all__ = [
    "get_memory_version",
    "set_memory_version",
    "bump_memory_version",
    "clear_memory_version",
]
