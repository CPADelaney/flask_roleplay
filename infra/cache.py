"""Redis cache utilities for hot-path operations.

This module provides fast, cache-first utilities for reading precomputed
results from background workers. All operations are non-blocking and optimized
for the hot path (game loop, event handlers).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)

# Global Redis client singleton
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """Get or create the global Redis client."""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        logger.info(f"Initialized Redis client: {redis_url}")
    return _redis_client


# Convenience alias for direct access
redis_client = get_redis_client()


class VersionedKeyRegistry:
    """Manage per-player version counters for cache keys."""

    def __init__(self, namespace: str = "cache:versions") -> None:
        self._namespace = namespace

    def _normalize_player(self, player_id: Any) -> str:
        if player_id is None:
            raise ValueError("player_id is required for versioned cache keys")
        return str(player_id)

    def _normalize_label(self, label: str) -> str:
        if not label:
            raise ValueError("label is required for versioned cache keys")
        return str(label)

    def _counter_key(self, player_id: Any, label: str) -> str:
        player = self._normalize_player(player_id)
        normalized_label = self._normalize_label(label)
        return f"{self._namespace}:{player}:{normalized_label}"

    def get_version(self, player_id: Any, label: str) -> int:
        """Return the current version counter for the player/label pair."""

        client = get_redis_client()
        raw = client.get(self._counter_key(player_id, label))
        try:
            return int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            return 0

    def bump(self, player_id: Any, label: str) -> int:
        """Atomically increment the version counter for the player/label pair."""

        client = get_redis_client()
        return int(client.incr(self._counter_key(player_id, label)))

    def suffix(self, player_id: Any, label: str) -> str:
        """Return the formatted suffix for the current version."""

        version = self.get_version(player_id, label)
        return f"v{version}"

    def bump_suffix(self, player_id: Any, label: str) -> str:
        """Increment the counter and return the formatted suffix."""

        version = self.bump(player_id, label)
        return f"v{version}"


_VERSIONED_REGISTRY = VersionedKeyRegistry()


def _versioned_cache_key(player_id: Any, label: str, *parts: Any) -> str:
    suffix = _VERSIONED_REGISTRY.suffix(player_id, label)
    return cache_key(label, player_id, *parts, suffix)


def conflict_cache_key_with_version(player_id: Any, *parts: Any) -> str:
    """Build a conflict cache key including the player's version suffix."""

    return _versioned_cache_key(player_id, "conflict", *parts)


def bump_conflict_cache_version(player_id: Any) -> str:
    """Bump the conflict cache counter for this player and return the suffix."""

    return _VERSIONED_REGISTRY.bump_suffix(player_id, "conflict")


def memory_cache_key_with_version(player_id: Any, *parts: Any) -> str:
    """Build a memory cache key including the player's version suffix."""

    return _versioned_cache_key(player_id, "memory", *parts)


def bump_memory_cache_version(player_id: Any) -> str:
    """Bump the memory cache counter for this player and return the suffix."""

    return _VERSIONED_REGISTRY.bump_suffix(player_id, "memory")


def lore_cache_key_with_version(player_id: Any, *parts: Any) -> str:
    """Build a lore cache key including the player's version suffix."""

    return _versioned_cache_key(player_id, "lore", *parts)


def bump_lore_cache_version(player_id: Any) -> str:
    """Bump the lore cache counter for this player and return the suffix."""

    return _VERSIONED_REGISTRY.bump_suffix(player_id, "lore")


def get_json(key: str, default: Any = None) -> Any:
    """Get a JSON value from Redis cache (hot path optimized).

    Args:
        key: Redis key
        default: Default value if key not found or JSON decode fails

    Returns:
        Decoded JSON value or default
    """
    try:
        client = get_redis_client()
        value = client.get(key)
        if value is None:
            return default
        return json.loads(value)
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.warning(f"Cache read failed for key={key}: {e}")
        return default


def set_json(key: str, value: Any, ex: Optional[int] = None) -> bool:
    """Set a JSON value in Redis cache.

    Args:
        key: Redis key
        value: JSON-serializable value
        ex: Expiration time in seconds (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_redis_client()
        serialized = json.dumps(value)
        client.set(key, serialized, ex=ex)
        return True
    except (redis.RedisError, TypeError, ValueError) as e:
        logger.error(f"Cache write failed for key={key}: {e}")
        return False


def delete_keys(pattern: str) -> int:
    """Delete keys matching pattern.

    Args:
        pattern: Redis key pattern (e.g., "social_bundle:*")

    Returns:
        Number of keys deleted
    """
    try:
        client = get_redis_client()
        keys = client.keys(pattern)
        if keys:
            return client.delete(*keys)
        return 0
    except redis.RedisError as e:
        logger.error(f"Cache delete failed for pattern={pattern}: {e}")
        return 0


@contextmanager
def redis_lock(name: str, ttl: int = 15, blocking: bool = False, timeout: int = 1):
    """Context manager for distributed Redis lock.

    Args:
        name: Lock name (key)
        ttl: Lock TTL in seconds
        blocking: If True, wait for lock; if False, raise if unavailable
        timeout: Max seconds to wait if blocking=True

    Yields:
        True if lock acquired

    Example:
        with redis_lock("social_bundle:scene123:lock", ttl=15):
            # Protected critical section
            generate_social_bundle.delay(scene_context)
    """
    client = get_redis_client()
    lock = client.lock(name, timeout=ttl, blocking=blocking, blocking_timeout=timeout)

    acquired = lock.acquire(blocking=blocking, blocking_timeout=timeout if blocking else 0)

    if not acquired:
        raise RuntimeError(f"Failed to acquire lock: {name}")

    try:
        yield True
    finally:
        try:
            lock.release()
        except redis.exceptions.LockError:
            # Lock already expired or released
            pass


def cache_key(*parts: str) -> str:
    """Build a cache key from parts.

    Args:
        *parts: Key components

    Returns:
        Joined cache key

    Example:
        >>> cache_key("conflict", "123", "transition")
        'conflict:123:transition'
    """
    return ":".join(str(p) for p in parts)


def get_or_compute(
    key: str,
    compute_fn: callable,
    ex: Optional[int] = None,
    force_refresh: bool = False,
) -> Any:
    """Get value from cache or compute and cache it.

    Args:
        key: Redis key
        compute_fn: Function to compute value if not cached
        ex: Cache expiration in seconds
        force_refresh: If True, always recompute

    Returns:
        Cached or computed value
    """
    if not force_refresh:
        cached = get_json(key)
        if cached is not None:
            return cached

    value = compute_fn()
    set_json(key, value, ex=ex)
    return value
