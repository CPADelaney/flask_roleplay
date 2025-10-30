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
    from_phase: str,
    to_phase: str,
    context: Optional[Dict[str, Any]] = None
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
    key = cache_key("conflict", conflict_id, "transition_narrative")
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for transition narrative: conflict={conflict_id}")
        return cached.get("text")

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
    key = cache_key("conflict", conflict_id, "beat", beat_id)
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for beat: conflict={conflict_id}, beat={beat_id}")
        return cached.get("text")

    return None


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
