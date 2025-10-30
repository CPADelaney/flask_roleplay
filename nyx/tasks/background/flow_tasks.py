"""Background tasks for conflict flow narration and dramatic beats.

These tasks generate prose narration for conflict phase transitions and
dramatic beats. The numeric flow updates happen synchronously in the hot path;
the prose is generated asynchronously and cached in Redis for later retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from infra.cache import cache_key, set_json
from nyx.tasks.utils import with_retry, run_coro
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _idempotency_key_transition(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for phase transition narration."""
    conflict_id = payload.get("conflict_id")
    from_phase = payload.get("from_phase", "")
    to_phase = payload.get("to_phase", "")
    return f"flow_transition:{conflict_id}:{from_phase}:{to_phase}"


def _idempotency_key_beat(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for dramatic beat narration."""
    conflict_id = payload.get("conflict_id")
    beat_type = payload.get("beat_meta", {}).get("type", "")
    beat_id = payload.get("beat_meta", {}).get("id", "")
    return f"flow_beat:{conflict_id}:{beat_type}:{beat_id}"


async def _narrate_transition_async(
    conflict_id: int, from_phase: str, to_phase: str, context: Dict[str, Any]
) -> str:
    """Generate prose for a phase transition (slow LLM call)."""
    # Import dynamically to avoid circular deps
    from logic.conflict_system.conflict_flow import transition_narrator

    # This is the blocking LLM call
    prose = transition_narrator(conflict_id, from_phase, to_phase, context)
    return prose


async def _narrate_beat_async(conflict_id: int, beat_meta: Dict[str, Any]) -> str:
    """Generate prose for a dramatic beat (slow LLM call)."""
    from logic.conflict_system.conflict_flow import beat_narrator

    prose = beat_narrator(conflict_id, beat_meta)
    return prose


@shared_task(
    name="nyx.tasks.background.flow_tasks.narrate_phase_transition",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_transition)
def narrate_phase_transition(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate prose narration for a conflict phase transition (slow LLM call).

    The hot path has already updated the conflict.current_phase field in the DB.
    This task generates rich prose narration and caches it in Redis for the
    next scene sync to retrieve.

    Args:
        payload: Dict with keys:
            - conflict_id: int
            - from_phase: str
            - to_phase: str
            - context: dict (optional, additional scene data)
            - ttl: int (optional, cache TTL in seconds, default 3600)

    Returns:
        Dict with status and cache key
    """
    conflict_id = payload.get("conflict_id")
    from_phase = payload.get("from_phase", "")
    to_phase = payload.get("to_phase", "")
    context = payload.get("context", {})
    ttl = payload.get("ttl", 3600)

    if not conflict_id or not to_phase:
        raise ValueError("conflict_id and to_phase are required")

    logger.info(f"Generating transition narration for conflict {conflict_id}: {from_phase} -> {to_phase}")

    # Run the slow LLM call
    prose = run_coro(_narrate_transition_async(conflict_id, from_phase, to_phase, context))

    # Cache the result in Redis
    key = cache_key("conflict", str(conflict_id), "transition_narrative")
    set_json(key, {"prose": prose, "from_phase": from_phase, "to_phase": to_phase}, ex=ttl)

    logger.info(f"Cached transition narration at {key}")

    return {"status": "narrated", "conflict_id": conflict_id, "cache_key": key, "prose_length": len(prose)}


@shared_task(
    name="nyx.tasks.background.flow_tasks.generate_beat_description",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_beat)
def generate_beat_description(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate prose description for a dramatic beat (slow LLM call).

    The hot path has already detected the beat via rules and recorded the
    numeric metadata. This task generates rich prose and caches it.

    Args:
        payload: Dict with keys:
            - conflict_id: int
            - beat_meta: dict (beat type, intensity, etc.)
            - ttl: int (optional, cache TTL in seconds, default 1800)

    Returns:
        Dict with status and cache key
    """
    conflict_id = payload.get("conflict_id")
    beat_meta = payload.get("beat_meta", {})
    ttl = payload.get("ttl", 1800)

    if not conflict_id or not beat_meta:
        raise ValueError("conflict_id and beat_meta are required")

    beat_id = beat_meta.get("id", "")
    beat_type = beat_meta.get("type", "unknown")

    logger.info(f"Generating beat description for conflict {conflict_id}, beat {beat_id} ({beat_type})")

    # Run the slow LLM call
    prose = run_coro(_narrate_beat_async(conflict_id, beat_meta))

    # Cache the result
    key = cache_key("conflict", str(conflict_id), "beat", beat_id)
    set_json(key, {"prose": prose, "beat_meta": beat_meta}, ex=ttl)

    logger.info(f"Cached beat description at {key}")

    return {"status": "narrated", "conflict_id": conflict_id, "cache_key": key, "beat_id": beat_id}


__all__ = [
    "narrate_phase_transition",
    "generate_beat_description",
]
