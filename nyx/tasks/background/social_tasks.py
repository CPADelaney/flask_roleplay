"""Background tasks for social circle dynamics (gossip, reputation).

These tasks generate social dynamics data (gossip and reputation calculations)
in the background. The hot path reads from cache; on miss, it dispatches these
tasks and returns a fallback response.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from nyx.tasks.base import NyxTask, app

from infra.cache import cache_key, get_json, set_json
from nyx.tasks.utils import with_retry, run_coro
from nyx.utils.idempotency import idempotent

from logic.conflict_system.social_circle import SocialCircle, SocialCircleManager

logger = logging.getLogger(__name__)


def _idempotency_key_social_bundle(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for social bundle generation."""
    scene_hash = payload.get("scene_context", {}).get("scene_hash") or payload.get("scene_context", {}).get("hash", "")
    return f"social_bundle:{scene_hash}"


def _idempotency_key_reputation(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for reputation calculation."""
    target_id = payload.get("target_id")
    scene_hash = payload.get("scene_context", {}).get("scene_hash", "")
    return f"reputation:{target_id}:{scene_hash}"


async def _generate_gossip_async(
    user_id: int,
    conversation_id: int,
    scene_context: Dict[str, Any],
    target_npcs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Generate gossip via the SocialCircleManager in the background."""

    manager = SocialCircleManager(user_id, conversation_id)
    return await manager.generate_gossip_background(scene_context, target_npcs or [])


async def _calculate_reputation_async(
    user_id: int,
    conversation_id: int,
    target_id: int,
    social_circle: Optional[Dict[str, Any]] = None,
    scene_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Calculate reputation using the manager's background helper."""

    manager = SocialCircleManager(user_id, conversation_id)
    circle = None
    if social_circle and social_circle.get("name"):
        circle = SocialCircle(
            circle_id=social_circle.get("circle_id", 0),
            name=social_circle.get("name", ""),
            description=social_circle.get("description", ""),
            members=social_circle.get("members", []),
            hierarchy=social_circle.get("hierarchy", {}),
            group_mood=social_circle.get("group_mood", "neutral"),
            shared_values=social_circle.get("shared_values", []),
            current_gossip=[],
            tension_points=social_circle.get("tension_points", {}),
        )

    reputation = await manager.calculate_reputation_background(target_id, circle)
    return reputation


@app.task(base=NyxTask, name="nyx.tasks.background.social_tasks.generate_social_bundle",
    bind=True,
    max_retries=2,
    acks_late=True,)
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
    scene_context = dict(payload.get("scene_context", {}))
    ttl = payload.get("ttl", 1800)
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")

    scene_hash = scene_context.get("scene_hash") or scene_context.get("hash", "")
    if not scene_hash:
        scene_hash = scene_context.get("scene_id") or "unknown"

    if not (user_id and conversation_id):
        logger.warning(
            "generate_social_bundle missing user/conversation context for scene %s", scene_hash
        )
        return {
            "status": "skipped",
            "scene_hash": scene_hash,
            "cache_key": cache_key("social_bundle", scene_hash),
            "gossip_count": 0,
            "reputation_count": 0,
        }

    target_npcs = payload.get("target_npcs") or scene_context.get("participants", [])
    reputation_targets = payload.get("reputation_targets") or target_npcs

    logger.info(
        "Generating social bundle for scene %s (targets=%s)",
        scene_hash,
        target_npcs,
    )

    gossip_item = run_coro(
        _generate_gossip_async(user_id, conversation_id, scene_context, target_npcs)
    )

    reputations: Dict[int, Dict[str, float]] = {}
    for npc_id in reputation_targets or []:
        rep_scores = run_coro(
            _calculate_reputation_async(
                user_id,
                conversation_id,
                npc_id,
                scene_context=scene_context,
            )
        )
        reputations[npc_id] = rep_scores

    bundle = {
        "gossip": [gossip_item] if gossip_item else [],
        "reputations": reputations,
        "scene_hash": scene_hash,
    }

    bundle_key = cache_key("social_bundle", scene_hash)
    set_json(bundle_key, bundle, ex=ttl)

    gossip_key = cache_key("gossip", scene_hash)
    existing_gossip = get_json(gossip_key, default=[])
    if isinstance(existing_gossip, list):
        existing_gossip.append(gossip_item)
        existing_gossip = existing_gossip[-10:]
        set_json(gossip_key, existing_gossip, ex=ttl)

    for npc_id, scores in reputations.items():
        for rep_name, score in scores.items():
            set_json(cache_key("reputation", npc_id, rep_name), score, ex=ttl)

    logger.info(
        "Cached social bundle at %s (gossip=%s, reputation_targets=%s)",
        bundle_key,
        len(bundle["gossip"]),
        len(reputations),
    )

    return {
        "status": "generated",
        "scene_hash": scene_hash,
        "cache_key": bundle_key,
        "gossip_count": len(bundle["gossip"]),
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


async def _narrate_reputation_change_async(
    user_id: int,
    conversation_id: int,
    npc_id: int,
    old_reputation: Dict[str, float],
    new_reputation: Dict[str, float],
) -> str:
    """Generate narration for reputation change (slow LLM call)."""

    manager = SocialCircleManager(user_id, conversation_id)
    return await manager.narrate_reputation_change(npc_id, old_reputation, new_reputation)


async def _form_alliance_async(
    user_id: int,
    conversation_id: int,
    initiator_id: int,
    target_id: int,
    reason: str,
) -> Dict[str, Any]:
    """Generate alliance details (slow LLM call)."""

    manager = SocialCircleManager(user_id, conversation_id)
    return await manager.form_alliance(initiator_id, target_id, reason)


@app.task(base=NyxTask, name="nyx.tasks.background.social_tasks.generate_gossip",
    bind=True,
    max_retries=2,
    acks_late=True,)
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
    scene_context = dict(payload.get("scene_context", {}))
    target_npcs = payload.get("target_npcs", [])
    ttl = payload.get("ttl", 1800)
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")

    scene_hash = scene_context.get("scene_hash") or scene_context.get("hash", "")
    if not scene_hash:
        scene_hash = scene_context.get("scene_id") or "unknown"

    if not (user_id and conversation_id):
        logger.warning("generate_gossip missing user/conversation for scene %s", scene_hash)
        return {
            "status": "skipped",
            "scene_hash": scene_hash,
            "cache_key": cache_key("gossip", scene_hash),
            "gossip_id": None,
        }

    logger.info("Generating gossip for scene %s", scene_hash)

    gossip_item = run_coro(
        _generate_gossip_async(user_id, conversation_id, scene_context, target_npcs)
    )

    key = cache_key("gossip", scene_hash)
    existing_gossip = get_json(key, default=[])
    if isinstance(existing_gossip, list):
        existing_gossip.append(gossip_item)
        existing_gossip = existing_gossip[-10:]
        set_json(key, existing_gossip, ex=ttl)

    logger.info("Cached gossip at %s", key)

    return {
        "status": "generated",
        "scene_hash": scene_hash,
        "cache_key": key,
        "gossip_id": gossip_item.get("gossip_id"),
    }


@app.task(base=NyxTask, name="nyx.tasks.background.social_tasks.calculate_reputation",
    bind=True,
    max_retries=2,
    acks_late=True,)
@with_retry
@idempotent(key_fn=_idempotency_key_reputation)
def calculate_reputation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate reputation scores and persist them to cache."""

    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    target_id = payload.get("target_id")
    ttl = payload.get("ttl", 1800)
    scene_context = payload.get("scene_context") or {}
    social_circle = payload.get("social_circle")

    if not (user_id and conversation_id and target_id):
        raise ValueError("user_id, conversation_id, and target_id are required")

    reputation = run_coro(
        _calculate_reputation_async(
            user_id,
            conversation_id,
            target_id,
            social_circle=social_circle,
            scene_context=scene_context,
        )
    )

    for rep_name, score in reputation.items():
        set_json(cache_key("reputation", target_id, rep_name), score, ex=ttl)

    logger.info(
        "Cached reputation for npc=%s (%s traits)",
        target_id,
        len(reputation),
    )

    return {
        "status": "calculated",
        "target_id": target_id,
        "reputation_count": len(reputation),
    }


@app.task(base=NyxTask, name="nyx.tasks.background.social_tasks.narrate_reputation_change",
    bind=True,
    max_retries=2,
    acks_late=True,)
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
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    old_reputation = payload.get("old_reputation", {})
    new_reputation = payload.get("new_reputation", {})
    ttl = payload.get("ttl", 3600)

    if not (npc_id and user_id and conversation_id):
        raise ValueError("npc_id, user_id, and conversation_id are required")

    logger.info(f"Generating reputation narration for NPC {npc_id}")

    # Run the slow LLM call
    narration = run_coro(
        _narrate_reputation_change_async(
            user_id,
            conversation_id,
            npc_id,
            old_reputation,
            new_reputation,
        )
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


@app.task(base=NyxTask, name="nyx.tasks.background.social_tasks.form_alliance",
    bind=True,
    max_retries=2,
    acks_late=True,)
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
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")

    if not (initiator_id and target_id and user_id and conversation_id):
        raise ValueError("initiator_id, target_id, user_id, and conversation_id are required")

    logger.info(f"Forming alliance: {initiator_id} -> {target_id}")

    # Run the slow LLM call
    alliance_data = run_coro(
        _form_alliance_async(
            user_id,
            conversation_id,
            initiator_id,
            target_id,
            reason,
        )
    )

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
    "calculate_reputation",
    "narrate_reputation_change",
    "form_alliance",
]
