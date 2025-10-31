"""Background tasks for slice-of-life conflict system.

These tasks generate slice-of-life content in the background:
- Pattern analysis for emerging tensions (LLM-based)
- Daily conflict appropriateness checks
- Conflict manifestation generation

The hot path reads from cache; on miss, it dispatches these tasks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from celery import shared_task

from infra.cache import cache_key, set_json
from nyx.tasks.utils import with_retry, run_coro
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _idempotency_key_tensions(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for tension analysis."""
    context_hash = payload.get("context_hash", "")
    return f"slice_tensions:{context_hash}"


def _idempotency_key_daily_conflicts(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for daily conflicts."""
    user_id = payload.get("user_id", 0)
    conversation_id = payload.get("conversation_id", 0)
    time_of_day = payload.get("time_of_day", "")
    return f"slice_daily:{user_id}:{conversation_id}:{time_of_day}"


async def _analyze_tensions_async(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
    """Analyze patterns and detect emerging tensions (slow LLM call).

    This is the blocking LLM operation that was previously in the hot path.
    Now it runs in the background via Celery.
    """
    from logic.conflict_system.slice_of_life_conflicts import EmergentConflictDetector

    detector = EmergentConflictDetector(user_id, conversation_id)

    # This calls the LLM to analyze patterns - slow operation!
    tensions = await detector.detect_brewing_tensions()

    return tensions


async def _check_daily_conflicts_async(
    user_id: int,
    conversation_id: int,
    time_of_day: str
) -> List[Dict[str, Any]]:
    """Check which conflicts are appropriate for the time of day (may involve LLM).

    This operation queries the DB and may use LLM to determine appropriateness.
    """
    from logic.conflict_system.slice_of_life_conflicts import ConflictDailyIntegration

    integration = ConflictDailyIntegration(user_id, conversation_id)

    # This may involve LLM calls to determine appropriateness
    conflicts = await integration.get_conflicts_for_time_of_day(time_of_day)

    return conflicts


@shared_task(
    name="nyx.tasks.background.slice_of_life_tasks.update_slice_tensions_cache",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_tensions)
def update_slice_tensions_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update slice-of-life tensions cache for a user/conversation.

    This task is dispatched when the hot path detects a cache miss for brewing
    tensions. It analyzes memory patterns and relationship dynamics using LLM
    and caches the detected tensions in Redis.

    Args:
        payload: Dict with keys:
            - user_id: int
            - conversation_id: int
            - context_hash: str
            - ttl: int (optional, cache TTL in seconds, default 3600)

    Returns:
        Dict with status and cache key
    """
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    context_hash = payload.get("context_hash", "")
    ttl = payload.get("ttl", 3600)

    if not user_id or not conversation_id:
        raise ValueError("user_id and conversation_id are required")

    logger.info(f"Analyzing slice-of-life tensions for user={user_id} conv={conversation_id}")

    # Analyze patterns (involves LLM - slow!)
    tensions = run_coro(_analyze_tensions_async(user_id, conversation_id))

    # Cache the result using the hotpath helper
    from logic.conflict_system.slice_of_life_hotpath import cache_tension_analysis
    cache_tension_analysis(user_id, conversation_id, tensions, ttl=ttl)

    logger.info(f"Cached {len(tensions)} slice-of-life tensions for user={user_id}")

    return {
        "status": "updated",
        "user_id": user_id,
        "conversation_id": conversation_id,
        "tension_count": len(tensions),
        "context_hash": context_hash
    }


@shared_task(
    name="nyx.tasks.background.slice_of_life_tasks.update_daily_conflicts_cache",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_daily_conflicts)
def update_daily_conflicts_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update daily conflicts cache for time-appropriate conflicts.

    This task checks which active conflicts are appropriate for the given
    time of day and caches the results.

    Args:
        payload: Dict with keys:
            - user_id: int
            - conversation_id: int
            - time_of_day: str
            - ttl: int (optional, cache TTL in seconds, default 1800)

    Returns:
        Dict with status and cache key
    """
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    time_of_day = payload.get("time_of_day", "")
    ttl = payload.get("ttl", 1800)

    if not user_id or not conversation_id:
        raise ValueError("user_id and conversation_id are required")

    logger.info(f"Checking daily conflicts for user={user_id} time={time_of_day}")

    # Check conflicts (may involve LLM calls)
    conflicts = run_coro(_check_daily_conflicts_async(user_id, conversation_id, time_of_day))

    # Cache the result using the hotpath helper
    from logic.conflict_system.slice_of_life_hotpath import cache_daily_conflicts
    cache_daily_conflicts(user_id, conversation_id, time_of_day, conflicts, ttl=ttl)

    logger.info(f"Cached {len(conflicts)} daily conflicts for user={user_id} time={time_of_day}")

    return {
        "status": "updated",
        "user_id": user_id,
        "conversation_id": conversation_id,
        "time_of_day": time_of_day,
        "conflict_count": len(conflicts)
    }
