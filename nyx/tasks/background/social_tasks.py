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


def _idempotency_key_gossip(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for gossip generation."""
    scene_hash = payload.get("scene_context", {}).get("scene_hash", "")
    target_npcs = tuple(sorted(payload.get("target_npcs", [])))
    return f"generate_gossip:{scene_hash}:{target_npcs}"


def _idempotency_key_reputation_narration(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for reputation narration."""
    npc_id = payload.get("npc_id")
    timestamp = payload.get("timestamp", "")
    return f"reputation_narration:{npc_id}:{timestamp}"


def _idempotency_key_alliance(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for alliance formation."""
    initiator_id = payload.get("initiator_id")
    target_id = payload.get("target_id")
    return f"form_alliance:{initiator_id}:{target_id}"


async def _generate_single_gossip_async(
    scene_context: Dict[str, Any], target_npcs: list[int]
) -> Dict[str, Any]:
    """Generate a single gossip item (slow LLM call)."""
    from logic.conflict_system.social_circle import manager

    gossip_item = manager.generate_gossip(scene_context, target_npcs)
    return gossip_item


async def _narrate_reputation_change_async(
    npc_id: int, old_reputation: Dict[str, float], new_reputation: Dict[str, float]
) -> str:
    """Generate narration for reputation change (slow LLM call)."""
    from logic.conflict_system.social_circle import manager

    narration = manager.narrate_reputation_change(npc_id, old_reputation, new_reputation)
    return narration


async def _form_alliance_async(
    initiator_id: int, target_id: int, reason: str
) -> Dict[str, Any]:
    """Generate alliance details (slow LLM call)."""
    from logic.conflict_system.social_circle import manager

    alliance_data = manager.form_alliance(initiator_id, target_id, reason)
    return alliance_data


@shared_task(
    name="nyx.tasks.background.social_tasks.generate_gossip",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_gossip)
def generate_gossip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate gossip for a scene (slow LLM call).

    Args:
        payload: Dict with keys:
            - scene_context: dict
            - target_npcs: list[int] (optional)
            - ttl: int (optional, cache TTL in seconds, default 1800)

    Returns:
        Dict with status and gossip_id
    """
    scene_context = payload.get("scene_context", {})
    target_npcs = payload.get("target_npcs", [])
    ttl = payload.get("ttl", 1800)

    scene_hash = scene_context.get("scene_hash", "")
    if not scene_hash:
        raise ValueError("scene_context.scene_hash is required")

    logger.info(f"Generating gossip for scene {scene_hash}")

    # Run the slow LLM call
    gossip_item = run_coro(_generate_single_gossip_async(scene_context, target_npcs))

    # Cache the gossip
    from infra.cache import get_json
    key = cache_key("gossip", scene_hash)

    # Append to existing gossip list in cache
    existing_gossip = get_json(key, default=[])
    if isinstance(existing_gossip, list):
        existing_gossip.append(gossip_item)
        # Keep only recent 10 items
        existing_gossip = existing_gossip[-10:]
        set_json(key, existing_gossip, ex=ttl)

    logger.info(f"Cached gossip at {key}")

    return {
        "status": "generated",
        "scene_hash": scene_hash,
        "cache_key": key,
        "gossip_id": gossip_item.get("gossip_id"),
    }


@shared_task(
    name="nyx.tasks.background.social_tasks.narrate_reputation_change",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_reputation_narration)
def narrate_reputation_change(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate narration for reputation change (slow LLM call).

    Args:
        payload: Dict with keys:
            - npc_id: int
            - old_reputation: dict
            - new_reputation: dict
            - ttl: int (optional, cache TTL in seconds, default 3600)

    Returns:
        Dict with status and cache key
    """
    npc_id = payload.get("npc_id")
    old_reputation = payload.get("old_reputation", {})
    new_reputation = payload.get("new_reputation", {})
    ttl = payload.get("ttl", 3600)

    if not npc_id:
        raise ValueError("npc_id is required")

    logger.info(f"Generating reputation narration for NPC {npc_id}")

    # Run the slow LLM call
    narration = run_coro(
        _narrate_reputation_change_async(npc_id, old_reputation, new_reputation)
    )

    # Cache the narration
    key = cache_key("reputation", "narration", npc_id)
    set_json(key, {"text": narration, "timestamp": payload.get("timestamp")}, ex=ttl)

    logger.info(f"Cached reputation narration at {key}")

    return {
        "status": "narrated",
        "npc_id": npc_id,
        "cache_key": key,
        "narration_length": len(narration),
    }


@shared_task(
    name="nyx.tasks.background.social_tasks.form_alliance",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_alliance)
def form_alliance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Form an alliance with LLM-generated terms (slow LLM call).

    Args:
        payload: Dict with keys:
            - initiator_id: int
            - target_id: int
            - reason: str

    Returns:
        Dict with status and alliance_id
    """
    initiator_id = payload.get("initiator_id")
    target_id = payload.get("target_id")
    reason = payload.get("reason", "")

    if not initiator_id or not target_id:
        raise ValueError("initiator_id and target_id are required")

    logger.info(f"Forming alliance: {initiator_id} -> {target_id}")

    # Run the slow LLM call
    alliance_data = run_coro(_form_alliance_async(initiator_id, target_id, reason))

    logger.info(
        f"Formed alliance {alliance_data.get('alliance_id')} "
        f"between {initiator_id} and {target_id}"
    )

    return {
        "status": "formed",
        "alliance_id": alliance_data.get("alliance_id"),
        "initiator_id": initiator_id,
        "target_id": target_id,
    }


__all__ = [
    "generate_social_bundle",
    "generate_gossip",
    "narrate_reputation_change",
    "form_alliance",
]
