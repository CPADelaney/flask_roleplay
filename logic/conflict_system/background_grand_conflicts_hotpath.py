"""Hot-path helpers for background grand conflict Celery tasks."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from infra.cache import cache_key, redis_client

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from .background_grand_conflicts import BackgroundConflict


def _serialize_conflict(conflict: "BackgroundConflict") -> Dict[str, Any]:
    return {
        "conflict_id": conflict.conflict_id,
        "conflict_type": conflict.conflict_type.value,
        "name": conflict.name,
        "description": conflict.description,
        "intensity": conflict.intensity.value,
        "progress": conflict.progress,
        "factions": list(conflict.factions or []),
        "current_state": conflict.current_state,
        "recent_developments": list(conflict.recent_developments or []),
        "impact_on_daily_life": list(conflict.impact_on_daily_life or []),
        "player_awareness_level": conflict.player_awareness_level,
        "news_count": conflict.news_count,
        "last_news_generation": conflict.last_news_generation,
    }


def _serialize_conflicts(conflicts: Iterable["BackgroundConflict"]) -> List[Dict[str, Any]]:
    return [_serialize_conflict(conflict) for conflict in conflicts]


def _pop_json(key: str) -> Optional[Any]:
    try:
        raw = redis_client.lpop(key)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to read background conflict cache %s: %s", key, exc)
        return None

    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Discarded malformed background conflict payload for key %s", key)
        return None


def fetch_generated_conflict(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    key = cache_key("background_conflict", user_id, conversation_id, "generated")
    return _pop_json(key)


def queue_conflict_generation(user_id: int, conversation_id: int, *, metadata: Optional[Dict[str, Any]] = None) -> bool:
    try:
        from nyx.tasks.background.conflict_grand_conflicts import generate_grand_conflict

        payload = {"user_id": user_id, "conversation_id": conversation_id}
        if metadata:
            payload.update(metadata)
        generate_grand_conflict.delay(payload)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to queue background conflict generation for user=%s conversation=%s: %s",
            user_id,
            conversation_id,
            exc,
        )
        return False


def fetch_conflict_event(conflict_id: int) -> Optional[Dict[str, Any]]:
    key = cache_key("background_conflict", conflict_id, "events")
    return _pop_json(key)


def queue_conflict_advance(
    user_id: int,
    conversation_id: int,
    conflict: "BackgroundConflict",
) -> bool:
    try:
        from nyx.tasks.background.conflict_grand_conflicts import advance_grand_conflict

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "conflict": _serialize_conflict(conflict),
        }
        advance_grand_conflict.delay(payload)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to queue conflict advance for conflict=%s: %s",
            conflict.conflict_id,
            exc,
        )
        return False


def fetch_conflict_news(conflict_id: int) -> Optional[Dict[str, Any]]:
    key = cache_key("background_conflict", conflict_id, "news")
    return _pop_json(key)


def queue_conflict_news(
    user_id: int,
    conversation_id: int,
    conflict: "BackgroundConflict",
    *,
    news_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        from nyx.tasks.background.conflict_grand_conflicts import generate_conflict_news

        payload: Dict[str, Any] = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "conflict": _serialize_conflict(conflict),
        }
        if news_type:
            payload["news_type"] = news_type
        if metadata:
            payload.update(metadata)
        generate_conflict_news.delay(payload)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to queue conflict news for conflict=%s: %s",
            conflict.conflict_id,
            exc,
        )
        return False


def fetch_conflict_ripples(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    key = cache_key("background_conflict", user_id, conversation_id, "ripples")
    return _pop_json(key)


def queue_conflict_ripples(
    user_id: int,
    conversation_id: int,
    conflict: "BackgroundConflict",
) -> bool:
    try:
        from nyx.tasks.background.conflict_grand_conflicts import generate_conflict_ripples

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "conflict": _serialize_conflict(conflict),
        }
        generate_conflict_ripples.delay(payload)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to queue ripple generation for conflict=%s: %s",
            conflict.conflict_id,
            exc,
        )
        return False


def fetch_conflict_opportunities(user_id: int, conversation_id: int) -> Optional[List[Dict[str, Any]]]:
    key = cache_key("background_conflict", user_id, conversation_id, "opportunities")
    result = _pop_json(key)
    if result is None:
        return None
    if isinstance(result, list):
        return result
    return [result]


def queue_conflict_opportunity_check(
    user_id: int,
    conversation_id: int,
    conflicts: Iterable["BackgroundConflict"],
    *,
    player_skills: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        from nyx.tasks.background.conflict_grand_conflicts import check_conflict_opportunities

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "conflicts": _serialize_conflicts(conflicts),
            "player_skills": player_skills or {},
        }
        check_conflict_opportunities.delay(payload)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Failed to queue opportunity check for conversation=%s: %s",
            conversation_id,
            exc,
        )
        return False


__all__ = [
    "fetch_generated_conflict",
    "queue_conflict_generation",
    "fetch_conflict_event",
    "queue_conflict_advance",
    "fetch_conflict_news",
    "queue_conflict_news",
    "fetch_conflict_ripples",
    "queue_conflict_ripples",
    "fetch_conflict_opportunities",
    "queue_conflict_opportunity_check",
]
