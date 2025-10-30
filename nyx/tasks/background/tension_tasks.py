"""Background tasks for tension system (manifestations, escalations).

These tasks generate tension-related content in the background:
- Tension manifestations (environmental cues, NPC behaviors)
- Escalation narration
- Tension bundle updates

The hot path reads from cache; on miss, it dispatches these tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from infra.cache import cache_key, set_json
from nyx.tasks.utils import with_retry, run_coro
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _idempotency_key_tension_bundle(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for tension bundle generation."""
    scene_hash = payload.get("scene_hash", "")
    return f"tension_bundle:{scene_hash}"


def _idempotency_key_manifestation(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for manifestation generation."""
    scene_hash = payload.get("scene_context", {}).get("scene_hash", "")
    tension_level = payload.get("tension_level", "")
    return f"tension_manifestation:{scene_hash}:{tension_level}"


def _idempotency_key_escalation(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for escalation narration."""
    conflict_id = payload.get("conflict_id")
    event_type = payload.get("escalation_event", {}).get("type", "")
    return f"tension_escalation:{conflict_id}:{event_type}"


async def _generate_manifestations_async(
    scene_context: Dict[str, Any], tension_level: str
) -> list[str]:
    """Generate tension manifestations (slow LLM call)."""
    from logic.conflict_system.tension import manifestation_generator

    # This is the blocking LLM call
    manifestations = manifestation_generator(scene_context, tension_level)
    return manifestations


async def _narrate_escalation_async(
    conflict_id: int, escalation_event: Dict[str, Any]
) -> str:
    """Generate escalation narration (slow LLM call)."""
    from logic.conflict_system.tension import escalation_narrator

    # This is the blocking LLM call
    narration = escalation_narrator(conflict_id, escalation_event)
    return narration


async def _compute_tension_bundle_async(scene_hash: str) -> Dict[str, Any]:
    """Compute complete tension bundle for a scene (may involve LLM)."""
    from logic.conflict_system.tension import compute_tension_bundle

    # This may involve LLM or heavy computation
    bundle = compute_tension_bundle(scene_hash)
    return bundle


@shared_task(
    name="nyx.tasks.background.tension_tasks.update_tension_bundle_cache",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_tension_bundle)
def update_tension_bundle_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update tension bundle cache for a scene.

    This task is dispatched when the hot path detects a cache miss for the
    tension bundle. It computes tension level, manifestations, and atmosphere
    and caches the result in Redis.

    Args:
        payload: Dict with keys:
            - scene_hash: str
            - ttl: int (optional, cache TTL in seconds, default 1800)

    Returns:
        Dict with status and cache key
    """
    scene_hash = payload.get("scene_hash", "")
    ttl = payload.get("ttl", 1800)

    if not scene_hash:
        raise ValueError("scene_hash is required")

    logger.info(f"Updating tension bundle cache for scene {scene_hash}")

    # Compute the bundle (may involve LLM)
    bundle = run_coro(_compute_tension_bundle_async(scene_hash))

    # Cache the result
    key = cache_key("tension_bundle", scene_hash)
    set_json(key, bundle, ex=ttl)

    logger.info(f"Cached tension bundle at {key}")

    return {
        "status": "updated",
        "scene_hash": scene_hash,
        "cache_key": key,
        "tension_level": bundle.get("tension_level", "baseline"),
    }


@shared_task(
    name="nyx.tasks.background.tension_tasks.generate_tension_manifestations",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_manifestation)
def generate_tension_manifestations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate tension manifestations for a scene (slow LLM call).

    This task generates environmental cues and NPC behaviors that reflect
    the current tension level.

    Args:
        payload: Dict with keys:
            - scene_context: dict
            - tension_level: str
            - ttl: int (optional, cache TTL in seconds, default 1800)

    Returns:
        Dict with status and cache key
    """
    scene_context = payload.get("scene_context", {})
    tension_level = payload.get("tension_level", "baseline")
    ttl = payload.get("ttl", 1800)

    scene_hash = scene_context.get("scene_hash", "")
    if not scene_hash:
        raise ValueError("scene_context.scene_hash is required")

    logger.info(f"Generating tension manifestations for scene {scene_hash} (level={tension_level})")

    # Run the slow LLM call
    manifestations = run_coro(_generate_manifestations_async(scene_context, tension_level))

    # Cache the result
    key = cache_key("manifestations", scene_hash)
    set_json(key, manifestations, ex=ttl)

    logger.info(f"Cached {len(manifestations)} manifestations at {key}")

    return {
        "status": "generated",
        "scene_hash": scene_hash,
        "cache_key": key,
        "manifestation_count": len(manifestations),
    }


@shared_task(
    name="nyx.tasks.background.tension_tasks.narrate_escalation",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_escalation)
def narrate_escalation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate escalation narration (slow LLM call).

    This task narrates a tension escalation event for a conflict.

    Args:
        payload: Dict with keys:
            - conflict_id: int
            - escalation_event: dict
            - ttl: int (optional, cache TTL in seconds, default 3600)

    Returns:
        Dict with status and cache key
    """
    conflict_id = payload.get("conflict_id")
    escalation_event = payload.get("escalation_event", {})
    ttl = payload.get("ttl", 3600)

    if not conflict_id:
        raise ValueError("conflict_id is required")

    logger.info(f"Generating escalation narration for conflict {conflict_id}")

    # Run the slow LLM call
    narration = run_coro(_narrate_escalation_async(conflict_id, escalation_event))

    # Cache the result
    key = cache_key("conflict", conflict_id, "escalation_narrative")
    set_json(key, {"text": narration, "event": escalation_event}, ex=ttl)

    logger.info(f"Cached escalation narration at {key}")

    return {
        "status": "narrated",
        "conflict_id": conflict_id,
        "cache_key": key,
        "narration_length": len(narration),
    }


__all__ = [
    "update_tension_bundle_cache",
    "generate_tension_manifestations",
    "narrate_escalation",
]
