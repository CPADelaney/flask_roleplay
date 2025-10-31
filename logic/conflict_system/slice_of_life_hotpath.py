"""Hot-path helpers for slice-of-life conflict system (no blocking LLM calls).

This module provides fast, cache-first functions for slice-of-life conflicts:
- Retrieve cached tension analysis from Redis
- Queue tension detection to background tasks
- Fast fallback responses when cache misses

The slow LLM calls for pattern analysis have been moved to background tasks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from infra.cache import redis_client, cache_key, get_json, set_json, redis_lock

logger = logging.getLogger(__name__)


def _compute_context_hash(user_id: int, conversation_id: int, lookback_hours: int = 72) -> str:
    """Compute stable hash for tension analysis context."""
    # Include user, conversation, and a time window to allow periodic refresh
    time_bucket = int(datetime.utcnow().timestamp() / (lookback_hours * 3600))
    context = {"user_id": user_id, "conversation_id": conversation_id, "time_bucket": time_bucket}
    serialized = json.dumps(context, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def get_brewing_tensions_cached(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
    """Get cached brewing tensions for a user/conversation (hot path).

    Returns cached tension analysis from recent pattern detection.
    If cache miss, triggers background generation and returns empty list.

    Args:
        user_id: User ID
        conversation_id: Conversation ID

    Returns:
        List of tension dicts with type, intensity, description, evidence
    """
    context_hash = _compute_context_hash(user_id, conversation_id)
    key = cache_key("slice_of_life_tensions", context_hash)
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for slice-of-life tensions: user={user_id} conv={conversation_id}")
        return cached.get("tensions", [])

    # Cache miss - trigger background generation with lock to prevent stampede
    lock_key = cache_key("slice_of_life_tensions", context_hash, "lock")
    try:
        with redis_lock(lock_key, ttl=20, blocking=False):
            # Queue background task to analyze patterns and detect tensions
            from nyx.tasks.background.slice_of_life_tasks import update_slice_tensions_cache

            update_slice_tensions_cache.delay({
                "user_id": user_id,
                "conversation_id": conversation_id,
                "context_hash": context_hash
            })
            logger.debug(f"Queued slice-of-life tension analysis for user={user_id} conv={conversation_id}")
    except RuntimeError:
        # Lock already held, another request is generating
        logger.debug(f"Slice-of-life tension analysis already in progress: user={user_id} conv={conversation_id}")

    # Return empty list while generation is in progress
    # This prevents blocking the hot path while LLM analysis happens in background
    return []


def get_daily_conflicts_cached(
    user_id: int,
    conversation_id: int,
    time_of_day: str
) -> List[Dict[str, Any]]:
    """Get cached appropriate conflicts for time of day (hot path).

    Returns conflicts that are appropriate for the current time of day.
    Uses cached results from background processing.

    Args:
        user_id: User ID
        conversation_id: Conversation ID
        time_of_day: Time period (e.g., "morning", "afternoon", "evening")

    Returns:
        List of conflict dicts appropriate for the time
    """
    key = cache_key("slice_daily_conflicts", user_id, conversation_id, time_of_day)
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for daily conflicts: user={user_id} conv={conversation_id} time={time_of_day}")
        return cached.get("conflicts", [])

    # Cache miss - trigger background update
    lock_key = cache_key("slice_daily_conflicts", user_id, conversation_id, time_of_day, "lock")
    try:
        with redis_lock(lock_key, ttl=15, blocking=False):
            from nyx.tasks.background.slice_of_life_tasks import update_daily_conflicts_cache

            update_daily_conflicts_cache.delay({
                "user_id": user_id,
                "conversation_id": conversation_id,
                "time_of_day": time_of_day
            })
            logger.debug(f"Queued daily conflicts update for user={user_id} time={time_of_day}")
    except RuntimeError:
        logger.debug(f"Daily conflicts update already in progress: user={user_id} time={time_of_day}")

    # Return empty list as fallback
    return []


def cache_tension_analysis(
    user_id: int,
    conversation_id: int,
    tensions: List[Dict[str, Any]],
    ttl: int = 3600
) -> None:
    """Cache tension analysis results (called by background tasks).

    Args:
        user_id: User ID
        conversation_id: Conversation ID
        tensions: List of detected tensions
        ttl: Cache TTL in seconds (default 1 hour)
    """
    context_hash = _compute_context_hash(user_id, conversation_id)
    key = cache_key("slice_of_life_tensions", context_hash)

    data = {
        "tensions": tensions,
        "generated_at": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "conversation_id": conversation_id
    }

    set_json(key, data, ttl=ttl)
    logger.info(f"Cached {len(tensions)} slice-of-life tensions for user={user_id} conv={conversation_id}")


def cache_daily_conflicts(
    user_id: int,
    conversation_id: int,
    time_of_day: str,
    conflicts: List[Dict[str, Any]],
    ttl: int = 1800
) -> None:
    """Cache daily conflict appropriateness results (called by background tasks).

    Args:
        user_id: User ID
        conversation_id: Conversation ID
        time_of_day: Time period
        conflicts: List of appropriate conflicts
        ttl: Cache TTL in seconds (default 30 minutes)
    """
    key = cache_key("slice_daily_conflicts", user_id, conversation_id, time_of_day)

    data = {
        "conflicts": conflicts,
        "time_of_day": time_of_day,
        "generated_at": datetime.utcnow().isoformat()
    }

    set_json(key, data, ttl=ttl)
    logger.info(f"Cached {len(conflicts)} daily conflicts for user={user_id} time={time_of_day}")
