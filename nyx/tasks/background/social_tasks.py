"""Background tasks for social circle dynamics (gossip, reputation).

These tasks generate social dynamics data (gossip and reputation calculations)
in the background. The hot path reads from cache; on miss, it dispatches these
tasks and returns a fallback response.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from infra.cache import cache_key, set_json
from nyx.tasks.utils import with_retry, run_coro
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _idempotency_key_social_bundle(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for social bundle generation."""
    scene_hash = payload.get("scene_context", {}).get("hash", "")
    return f"social_bundle:{scene_hash}"


async def _generate_gossip_async(scene_context: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Generate gossip for the scene (slow LLM call)."""
    # Import dynamically to avoid circular deps
    from logic.conflict_system.social_circle import manager

    # This is the blocking LLM call
    gossip = manager.generate_gossip(scene_context)
    return gossip


async def _calculate_reputation_async(scene_context: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate reputation changes for the scene (slow LLM call)."""
    from logic.conflict_system.social_circle import manager

    # This may involve LLM or heavy computation
    reputations = manager.calculate_reputation(scene_context)
    return reputations


@shared_task(
    name="nyx.tasks.background.social_tasks.generate_social_bundle",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_social_bundle)
def generate_social_bundle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a complete social bundle (gossip + reputation) for a scene.

    This task is dispatched when the hot path detects a cache miss for the
    social bundle. It generates both gossip and reputation data and caches
    the result in Redis.

    Args:
        payload: Dict with keys:
            - scene_context: dict (scene hash, stakeholders, recent events, etc.)
            - ttl: int (optional, cache TTL in seconds, default 1800)

    Returns:
        Dict with status and cache key
    """
    scene_context = payload.get("scene_context", {})
    ttl = payload.get("ttl", 1800)

    scene_hash = scene_context.get("hash", "")
    if not scene_hash:
        raise ValueError("scene_context.hash is required")

    logger.info(f"Generating social bundle for scene {scene_hash}")

    # Run both slow operations
    gossip = run_coro(_generate_gossip_async(scene_context))
    reputations = run_coro(_calculate_reputation_async(scene_context))

    # Build the bundle
    bundle = {
        "gossip": gossip,
        "reputations": reputations,
        "scene_hash": scene_hash,
        "generated_at": None,  # Will be set by cache
    }

    # Cache the result
    key = cache_key("social_bundle", scene_hash)
    set_json(key, bundle, ex=ttl)

    logger.info(f"Cached social bundle at {key} (gossip={len(gossip)}, reputations={len(reputations)})")

    return {
        "status": "generated",
        "scene_hash": scene_hash,
        "cache_key": key,
        "gossip_count": len(gossip),
        "reputation_count": len(reputations),
    }


__all__ = [
    "generate_social_bundle",
]
