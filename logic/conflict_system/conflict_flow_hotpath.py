"""Hot-path helpers for conflict flow (no blocking LLM calls).

This module provides fast, cache-first functions for conflict flow:
- Numeric conflict state updates (intensity, progress, phase)
- Queue narration generation to background tasks
- Fast retrieval of cached transition narratives

The slow LLM calls have been moved to nyx/tasks/background/flow_tasks.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from infra.cache import redis_client, cache_key, get_json, set_json
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


def _flow_init_cache_key(conflict_id: int) -> str:
    return cache_key("conflict", conflict_id, "flow_init")


def _event_analysis_cache_key(conflict_id: int, event_id: Optional[str]) -> str:
    suffix = event_id or "latest"
    return cache_key("conflict", conflict_id, "event", suffix)


def _transition_cache_key(conflict_id: int) -> str:
    return cache_key("conflict", conflict_id, "transition_narrative")


def _beat_cache_key(conflict_id: int, beat_id: str) -> str:
    return cache_key("conflict", conflict_id, "beat", beat_id)


def apply_event_math(conflict: Any, event: Dict[str, Any]) -> None:
    """Apply numeric updates to conflict state (pure math, no LLM).

    Args:
        conflict: Conflict object with intensity, progress attributes
        event: Event with intensity_delta, progress_delta
    """
    intensity_delta = event.get("intensity_delta", 0.0)
    progress_delta = event.get("progress_delta", 0.0)

    # Clamp values to [0.0, 1.0]
    conflict.intensity = max(0.0, min(1.0, conflict.intensity + intensity_delta))
    conflict.progress = max(0.0, min(1.0, conflict.progress + progress_delta))

    logger.debug(
        f"Applied event math: conflict={conflict.conflict_id}, "
        f"intensity={conflict.intensity:.2f}, progress={conflict.progress:.2f}"
    )


def queue_phase_narration(
    conflict_id: int,
    from_phase: Optional[str],
    to_phase: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None,
    intensity: Optional[float] = None,
    momentum: Optional[float] = None,
) -> None:
    """Queue background task to generate phase transition narration.

    Args:
        conflict_id: Conflict ID
        from_phase: Previous phase
        to_phase: New phase
        context: Additional context for narration
    """
    try:
        from nyx.tasks.background.flow_tasks import narrate_phase_transition

        payload = {
            "conflict_id": conflict_id,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "intensity": intensity,
            "momentum": momentum,
        }

        narrate_phase_transition.delay(payload)
        logger.debug(
            f"Queued phase narration: conflict={conflict_id}, "
            f"{from_phase} -> {to_phase}"
        )
    except Exception as e:
        logger.warning(f"Failed to queue phase narration: {e}")


def get_cached_transition_text(conflict_id: int) -> Optional[str]:
    """Get cached phase transition narrative from Redis.

    Args:
        conflict_id: Conflict ID

    Returns:
        Cached narrative text or None
    """
    key = _transition_cache_key(conflict_id)
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for transition narrative: conflict={conflict_id}")
        return cached.get("text") or cached.get("prose")

    return None


def get_cached_transition_payload(conflict_id: int) -> Optional[Dict[str, Any]]:
    """Return the cached transition payload if present."""

    cached = get_json(_transition_cache_key(conflict_id))
    if cached:
        return cached
    return None


def queue_beat_description(
    conflict_id: int,
    beat_type: str,
    beat_meta: Dict[str, Any]
) -> None:
    """Queue background task to generate dramatic beat description.

    Args:
        conflict_id: Conflict ID
        beat_type: Type of beat (revelation, setback, triumph, etc.)
        beat_meta: Metadata about the beat
    """
    try:
        from nyx.tasks.background.flow_tasks import generate_beat_description

        payload = {
            "conflict_id": conflict_id,
            "beat_type": beat_type,
            "beat_meta": beat_meta,
            "timestamp": datetime.utcnow().isoformat(),
        }

        generate_beat_description.delay(payload)
        logger.debug(
            f"Queued beat description: conflict={conflict_id}, type={beat_type}"
        )
    except Exception as e:
        logger.warning(f"Failed to queue beat description: {e}")


def get_cached_beat_text(conflict_id: int, beat_id: str) -> Optional[str]:
    """Get cached dramatic beat description from Redis.

    Args:
        conflict_id: Conflict ID
        beat_id: Beat identifier

    Returns:
        Cached beat text or None
    """
    key = _beat_cache_key(conflict_id, beat_id)
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for beat: conflict={conflict_id}, beat={beat_id}")
        return cached.get("text") or cached.get("prose")

    return None


def get_cached_beat_payload(conflict_id: int, beat_id: str) -> Optional[Dict[str, Any]]:
    """Get cached beat payload (metadata + prose)."""

    cached = get_json(_beat_cache_key(conflict_id, beat_id))
    if cached:
        return cached
    return None


def queue_flow_initialization(
    conflict_id: int,
    user_id: int,
    conversation_id: int,
    conflict_type: str,
    context: Dict[str, Any],
    *,
    cache_ttl: int = 3600,
) -> None:
    """Queue background initialization for conflict flow."""

    try:
        from nyx.tasks.background.flow_tasks import initialize_conflict_flow  # type: ignore

        payload = {
            "conflict_id": conflict_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "conflict_type": conflict_type,
            "context": context,
            "ttl": cache_ttl,
            "timestamp": datetime.utcnow().isoformat(),
        }
        initialize_conflict_flow.delay(payload)
        logger.debug("Queued conflict flow initialization for %s", conflict_id)
    except Exception as exc:
        logger.warning("Failed to queue flow initialization for %s: %s", conflict_id, exc)


def get_cached_flow_bootstrap(conflict_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve cached flow bootstrap result if present."""

    cached = get_json(_flow_init_cache_key(conflict_id))
    if cached:
        return cached
    return None


def queue_flow_event_analysis(
    conflict_id: int,
    user_id: int,
    conversation_id: int,
    event: Dict[str, Any],
    flow_state: Dict[str, Any],
    *,
    cache_ttl: int = 900,
) -> None:
    """Queue background flow event analysis."""

    try:
        from nyx.tasks.background.flow_tasks import analyze_flow_event  # type: ignore

        payload = {
            "conflict_id": conflict_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "event": event,
            "flow_state": flow_state,
            "ttl": cache_ttl,
            "timestamp": datetime.utcnow().isoformat(),
        }
        analyze_flow_event.delay(payload)
        logger.debug(
            "Queued flow event analysis: conflict=%s event=%s",
            conflict_id,
            event.get("event_id") or event.get("type") or "unknown",
        )
    except Exception as exc:
        logger.warning("Failed to queue flow event analysis for %s: %s", conflict_id, exc)


def get_cached_event_analysis(
    conflict_id: int,
    event_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Retrieve cached flow event analysis."""

    cached = get_json(_event_analysis_cache_key(conflict_id, event_id))
    if cached:
        return cached
    return None


def queue_dramatic_beat_generation(
    conflict_id: int,
    user_id: int,
    conversation_id: int,
    beat_id: str,
    flow_state: Dict[str, Any],
    context: Dict[str, Any],
    *,
    cache_ttl: int = 1800,
) -> None:
    """Queue background beat generation."""

    try:
        from nyx.tasks.background.flow_tasks import generate_dramatic_beat  # type: ignore

        payload = {
            "conflict_id": conflict_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "beat_id": beat_id,
            "flow_state": flow_state,
            "context": context,
            "ttl": cache_ttl,
            "timestamp": datetime.utcnow().isoformat(),
        }
        generate_dramatic_beat.delay(payload)
        logger.debug("Queued dramatic beat generation: conflict=%s beat=%s", conflict_id, beat_id)
    except Exception as exc:
        logger.warning("Failed to queue dramatic beat for %s: %s", conflict_id, exc)


async def get_flow_state(conflict_id: int) -> Dict[str, Any]:
    """Get conflict flow state (hot path - DB only, no LLM).

    Args:
        conflict_id: Conflict ID

    Returns:
        Flow state dict with phase, intensity, progress
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    current_phase,
                    intensity,
                    progress,
                    pacing_score,
                    last_beat_type,
                    last_beat_at
                FROM conflicts
                WHERE conflict_id = $1
                """,
                conflict_id
            )

            if row:
                return {
                    "phase": row["current_phase"],
                    "intensity": float(row["intensity"] or 0.5),
                    "progress": float(row["progress"] or 0.0),
                    "pacing_score": float(row["pacing_score"] or 0.5),
                    "last_beat_type": row["last_beat_type"],
                    "last_beat_at": row["last_beat_at"].isoformat() if row["last_beat_at"] else None,
                }

            return {}
    except Exception as e:
        logger.error(f"Failed to get flow state for conflict {conflict_id}: {e}")
        return {}


def should_trigger_beat(conflict: Any) -> Optional[str]:
    """Fast rule-based check if a dramatic beat should be triggered (no LLM).

    Args:
        conflict: Conflict object

    Returns:
        Beat type to trigger or None
    """
    # Simple heuristics
    intensity = getattr(conflict, "intensity", 0.5)
    progress = getattr(conflict, "progress", 0.0)

    # High intensity + mid progress = climax
    if intensity > 0.8 and 0.4 < progress < 0.7:
        return "climax"

    # High progress = resolution approaching
    if progress > 0.9:
        return "resolution"

    # Low intensity early = setup/exposition
    if intensity < 0.3 and progress < 0.3:
        return "exposition"

    # Mid-range = complications
    if 0.4 < intensity < 0.7 and 0.3 < progress < 0.6:
        return "complication"

    return None
