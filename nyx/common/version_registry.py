"""Redis-backed version counters for world, memory, and lore state."""

from __future__ import annotations

import logging
from typing import Union

import redis

from infra.cache import get_redis_client

logger = logging.getLogger(__name__)

_DEFAULT_VERSION = 0
_WORLD_VERSION_KEY = "ver:world"
_LORE_VERSION_KEY = "ver:lore"

PlayerIdentifier = Union[str, int]


def _memory_version_key(player_id: PlayerIdentifier) -> str:
    return f"ver:memory:{player_id}"


def _get_counter(key: str) -> int:
    try:
        client = get_redis_client()
        raw_value = client.get(key)
        if raw_value is None:
            return _DEFAULT_VERSION
        return int(raw_value)
    except (ValueError, TypeError):
        logger.warning("Invalid version value for key %s; resetting to default", key)
        return _DEFAULT_VERSION
    except redis.RedisError as exc:
        logger.warning("Failed to read version for key %s: %s", key, exc)
        return _DEFAULT_VERSION


def _bump_counter(key: str) -> int:
    try:
        client = get_redis_client()
        return int(client.incr(key))
    except redis.RedisError as exc:
        logger.error("Failed to bump version for key %s: %s", key, exc)
        return _DEFAULT_VERSION


# World versions

def get_world_version() -> int:
    """Return the current world version counter."""

    return _get_counter(_WORLD_VERSION_KEY)


def bump_world_version() -> int:
    """Atomically increment and return the world version counter."""

    return _bump_counter(_WORLD_VERSION_KEY)


# Memory versions

def get_memory_version(player_id: PlayerIdentifier) -> int:
    """Return the current memory version counter for the given player."""

    return _get_counter(_memory_version_key(player_id))


def bump_memory_version(player_id: PlayerIdentifier) -> int:
    """Atomically increment and return the memory version counter for the given player."""

    return _bump_counter(_memory_version_key(player_id))


# Lore versions

def get_lore_version() -> int:
    """Return the current lore version counter."""

    return _get_counter(_LORE_VERSION_KEY)


def bump_lore_version() -> int:
    """Atomically increment and return the lore version counter."""

    return _bump_counter(_LORE_VERSION_KEY)
