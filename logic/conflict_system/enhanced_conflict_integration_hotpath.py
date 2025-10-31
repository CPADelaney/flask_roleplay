"""Hot-path helpers for the enhanced conflict integration subsystem."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from infra.cache import cache_key, get_json, set_json

logger = logging.getLogger(__name__)


def _redis_key(prefix: str, user_id: int, conversation_id: int, identifier: str) -> str:
    return cache_key("enhanced_conflict", prefix, user_id, conversation_id, identifier)


def queue_scene_tension_analysis(
    user_id: int,
    conversation_id: int,
    scope_key: str,
    scene_context: Dict[str, Any],
) -> None:
    """Enqueue the slow-path scene tension analysis task."""
    if not scope_key:
        logger.debug("Skipping tension analysis queue due to empty scope key")
        return

    try:
        from nyx.tasks.background.conflict_integration_tasks import (
            analyze_scene_tension,
        )

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "scope_key": scope_key,
            "scene_context": scene_context,
            "queued_at": datetime.utcnow().isoformat(),
        }
        analyze_scene_tension.delay(payload)
        logger.debug(
            "Queued tension analysis for conversation=%s scope=%s",
            conversation_id,
            scope_key,
        )
    except Exception as exc:
        logger.warning(
            "Failed to queue tension analysis for conversation=%s: %s",
            conversation_id,
            exc,
        )


def cache_tension_result(
    user_id: int,
    conversation_id: int,
    scope_key: str,
    summary: Dict[str, Any],
    ttl: int = 3600,
) -> None:
    """Store a tension summary in the cross-process cache."""
    if not scope_key or not summary:
        return
    key = _redis_key("tension", user_id, conversation_id, scope_key)
    try:
        set_json(key, summary, ex=ttl)
    except Exception as exc:  # pragma: no cover - cache is best effort
        logger.debug("Failed to set Redis tension cache for %s: %s", scope_key, exc)


def get_cached_tension_result(
    user_id: int, conversation_id: int, scope_key: str
) -> Optional[Dict[str, Any]]:
    """Fetch a previously cached tension summary from Redis."""
    if not scope_key:
        return None
    key = _redis_key("tension", user_id, conversation_id, scope_key)
    cached = get_json(key)
    if cached:
        logger.debug(
            "Redis tension cache hit for conversation=%s scope=%s",
            conversation_id,
            scope_key,
        )
    return cached


def queue_contextual_conflict_generation(
    user_id: int,
    conversation_id: int,
    context_key: str,
    tension_data: Dict[str, Any],
    npcs: List[int],
) -> None:
    """Enqueue contextual conflict generation."""
    if not context_key:
        logger.debug("Skipping contextual conflict queue due to empty key")
        return

    try:
        from nyx.tasks.background.conflict_integration_tasks import (
            generate_contextual_conflict,
        )

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "context_key": context_key,
            "tension_data": tension_data,
            "npcs": npcs,
            "queued_at": datetime.utcnow().isoformat(),
        }
        generate_contextual_conflict.delay(payload)
        logger.debug(
            "Queued contextual conflict for conversation=%s key=%s",
            conversation_id,
            context_key,
        )
    except Exception as exc:
        logger.warning(
            "Failed to queue contextual conflict for conversation=%s: %s",
            conversation_id,
            exc,
        )


def cache_contextual_conflict(
    user_id: int,
    conversation_id: int,
    context_key: str,
    conflict: Dict[str, Any],
    ttl: int = 3600,
) -> None:
    if not context_key or not conflict:
        return
    key = _redis_key("contextual", user_id, conversation_id, context_key)
    try:
        set_json(key, conflict, ex=ttl)
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to cache contextual conflict %s: %s", context_key, exc)


def get_cached_contextual_conflict(
    user_id: int, conversation_id: int, context_key: str
) -> Optional[Dict[str, Any]]:
    if not context_key:
        return None
    key = _redis_key("contextual", user_id, conversation_id, context_key)
    cached = get_json(key)
    if cached:
        logger.debug(
            "Redis contextual conflict cache hit for conversation=%s key=%s",
            conversation_id,
            context_key,
        )
    return cached


def queue_activity_integration(
    user_id: int,
    conversation_id: int,
    integration_key: str,
    activity: str,
    conflicts: List[Dict[str, Any]],
) -> None:
    """Enqueue activity integration generation."""
    if not integration_key:
        logger.debug("Skipping activity integration queue due to empty key")
        return

    try:
        from nyx.tasks.background.conflict_integration_tasks import integrate_activity

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "integration_key": integration_key,
            "activity": activity,
            "conflicts": conflicts,
            "queued_at": datetime.utcnow().isoformat(),
        }
        integrate_activity.delay(payload)
        logger.debug(
            "Queued activity integration for conversation=%s key=%s",
            conversation_id,
            integration_key,
        )
    except Exception as exc:
        logger.warning(
            "Failed to queue activity integration for conversation=%s: %s",
            conversation_id,
            exc,
        )


def cache_activity_integration(
    user_id: int,
    conversation_id: int,
    integration_key: str,
    integration: Dict[str, Any],
    ttl: int = 3600,
) -> None:
    if not integration_key or not integration:
        return
    key = _redis_key("activity", user_id, conversation_id, integration_key)
    try:
        set_json(key, integration, ex=ttl)
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to cache activity integration %s: %s", integration_key, exc)


def get_cached_activity_integration(
    user_id: int, conversation_id: int, integration_key: str
) -> Optional[Dict[str, Any]]:
    if not integration_key:
        return None
    key = _redis_key("activity", user_id, conversation_id, integration_key)
    cached = get_json(key)
    if cached:
        logger.debug(
            "Redis activity integration cache hit for conversation=%s key=%s",
            conversation_id,
            integration_key,
        )
    return cached

