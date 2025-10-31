"""Hot-path helpers for social circle dynamics (no blocking LLM calls).

This module provides fast, cache-first functions for social dynamics:
- Retrieve cached gossip and reputation from Redis
- Queue generation tasks for new social content
- Fast reputation updates (numeric only)

The slow LLM calls have been moved to nyx/tasks/background/social_tasks.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from infra.cache import redis_client, cache_key, get_json, set_json, redis_lock
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


def _compute_scene_hash(scene_context: Dict[str, Any]) -> str:
    """Compute stable hash for scene context."""
    stable_keys = ["scene_id", "location", "npcs_present"]
    stable_context = {k: scene_context.get(k) for k in stable_keys if k in scene_context}
    serialized = json.dumps(stable_context, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def get_scene_bundle(scene_hash: str) -> Dict[str, Any]:
    """Get cached social bundle for a scene (hot path).

    Returns cached gossip, reputation status, and alliance info.
    If cache miss, triggers background generation and returns fallback.

    Args:
        scene_hash: Scene hash identifier

    Returns:
        Social bundle dict with gossip, reputation_status, alliances
    """
    key = cache_key("social_bundle", scene_hash)
    cached = get_json(key)

    if cached:
        logger.debug(f"Cache hit for social bundle: {scene_hash}")
        return cached

    # Cache miss - trigger background generation with lock to prevent stampede
    lock_key = cache_key("social_bundle", scene_hash, "lock")
    try:
        with redis_lock(lock_key, ttl=15, blocking=False):
            from nyx.tasks.background.social_tasks import generate_social_bundle

            generate_social_bundle.delay({"scene_hash": scene_hash})
            logger.debug(f"Queued social bundle generation for scene {scene_hash}")
    except RuntimeError:
        # Lock already held, another request is generating
        logger.debug(f"Social bundle generation already in progress: {scene_hash}")

    # Return fallback while generation is in progress
    return {
        "gossip": [],
        "reputation_status": "pending",
        "alliances": [],
        "status": "generating"
    }


def apply_reputation_change(
    npc_id: int,
    reputation_type: str,
    delta: float
) -> None:
    """Apply numeric reputation change (fast, no LLM).

    Args:
        npc_id: NPC identifier
        reputation_type: Type of reputation to change
        delta: Change amount (-1.0 to +1.0)
    """
    key = cache_key("reputation", npc_id, reputation_type)

    try:
        current = get_json(key, default=0.5)
        new_value = max(0.0, min(1.0, current + delta))

        set_json(key, new_value, ex=3600)
        logger.debug(
            f"Applied reputation change: npc={npc_id}, "
            f"type={reputation_type}, delta={delta:.2f}"
        )
    except Exception as e:
        logger.warning(f"Failed to apply reputation change: {e}")


def queue_reputation_narration(
    npc_id: int,
    old_reputation: Dict[str, float],
    new_reputation: Dict[str, float]
) -> None:
    """Queue background task to narrate reputation change.

    Args:
        npc_id: NPC identifier
        old_reputation: Previous reputation scores
        new_reputation: New reputation scores
    """
    try:
        from nyx.tasks.background.social_tasks import narrate_reputation_change

        payload = {
            "npc_id": npc_id,
            "old_reputation": old_reputation,
            "new_reputation": new_reputation,
            "timestamp": datetime.utcnow().isoformat(),
        }

        narrate_reputation_change.delay(payload)
        logger.debug(f"Queued reputation narration for NPC {npc_id}")
    except Exception as e:
        logger.warning(f"Failed to queue reputation narration: {e}")


def queue_gossip_generation(
    scene_context: Dict[str, Any],
    target_npcs: Optional[List[int]] = None,
    *,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None,
    ttl: int = 1800,
) -> None:
    """Queue background task to generate gossip.

    Args:
        scene_context: Scene context
        target_npcs: Optional list of NPCs to gossip about
    """
    try:
        from nyx.tasks.background.social_tasks import generate_gossip

        context_payload = dict(scene_context or {})

        # Ensure we have a stable scene hash for caching
        scene_hash = context_payload.get("scene_hash") or context_payload.get("hash")
        if not scene_hash:
            try:
                scene_hash = _compute_scene_hash(context_payload)
            except Exception:
                serialized = json.dumps(context_payload, sort_keys=True)
                scene_hash = hashlib.sha256(serialized.encode()).hexdigest()[:16]
        context_payload["scene_hash"] = scene_hash

        payload = {
            "scene_context": context_payload,
            "target_npcs": target_npcs or [],
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id or context_payload.get("user_id"),
            "conversation_id": conversation_id or context_payload.get("conversation_id"),
            "ttl": ttl,
        }

        generate_gossip.delay(payload)
        logger.debug(
            "Queued gossip generation for scene %s (targets=%s)",
            scene_hash,
            target_npcs or [],
        )
    except Exception as e:
        logger.warning(f"Failed to queue gossip generation: {e}")


def queue_alliance_formation(
    initiator_id: int,
    target_id: int,
    reason: str
) -> None:
    """Queue background task to form alliance with LLM-generated terms.

    Args:
        initiator_id: Initiator NPC ID
        target_id: Target NPC ID
        reason: Reason for alliance
    """
    try:
        from nyx.tasks.background.social_tasks import form_alliance

        payload = {
            "initiator_id": initiator_id,
            "target_id": target_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }

        form_alliance.delay(payload)
        logger.debug(f"Queued alliance formation: {initiator_id} -> {target_id}")
    except Exception as e:
        logger.warning(f"Failed to queue alliance formation: {e}")


async def get_cached_gossip(scene_hash: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get cached gossip items for a scene (hot path).

    Args:
        scene_hash: Scene hash
        limit: Maximum gossip items to return

    Returns:
        List of gossip items
    """
    key = cache_key("gossip", scene_hash)
    cached = get_json(key, default=[])

    if isinstance(cached, list):
        return cached[:limit]

    return []


def queue_reputation_calculation(
    user_id: int,
    conversation_id: int,
    target_id: int,
    *,
    scene_context: Optional[Dict[str, Any]] = None,
    ttl: int = 1800,
) -> None:
    """Queue background task to calculate reputation for an NPC.

    Args:
        user_id: Owning user identifier
        conversation_id: Conversation identifier
        target_id: NPC identifier to calculate reputation for
        scene_context: Optional additional context for hashing/idempotency
        ttl: Cache TTL for stored reputation values
    """

    payload_context = dict(scene_context or {})

    if "scene_hash" not in payload_context and payload_context:
        try:
            payload_context["scene_hash"] = _compute_scene_hash(payload_context)
        except Exception:
            serialized = json.dumps(payload_context, sort_keys=True)
            payload_context["scene_hash"] = hashlib.sha256(serialized.encode()).hexdigest()[:16]

    try:
        from nyx.tasks.background.social_tasks import calculate_reputation

        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "target_id": target_id,
            "scene_context": payload_context,
            "timestamp": datetime.utcnow().isoformat(),
            "ttl": ttl,
        }

        calculate_reputation.delay(payload)
        logger.debug(
            "Queued reputation calculation for npc=%s (scene_hash=%s)",
            target_id,
            payload_context.get("scene_hash"),
        )
    except Exception as exc:
        logger.warning("Failed to queue reputation calculation: %s", exc)


async def get_cached_reputation(npc_id: int) -> Dict[str, float]:
    """Get cached reputation scores for NPC (hot path).

    Args:
        npc_id: NPC identifier

    Returns:
        Dict of reputation_type -> score
    """
    reputation_types = [
        "trustworthy", "submissive", "rebellious", "mysterious",
        "influential", "scandalous", "nurturing", "dangerous"
    ]

    reputation = {}
    for rep_type in reputation_types:
        key = cache_key("reputation", npc_id, rep_type)
        reputation[rep_type] = get_json(key, default=0.5)

    return reputation


def should_spread_gossip(
    gossip_truthfulness: float,
    spreader_gossip_tendency: float
) -> bool:
    """Fast rule-based check if gossip should spread (no LLM).

    Args:
        gossip_truthfulness: How true the gossip is (0.0-1.0)
        spreader_gossip_tendency: How likely the spreader is to gossip (0.0-1.0)

    Returns:
        True if gossip should spread
    """
    import random

    # More juicy (false) gossip spreads more
    spread_probability = (1.0 - gossip_truthfulness * 0.5) * spreader_gossip_tendency

    return random.random() < spread_probability


async def get_alliances(npc_id: int) -> List[Dict[str, Any]]:
    """Get active alliances for NPC (hot path - DB query only).

    Args:
        npc_id: NPC identifier

    Returns:
        List of alliance dicts
    """
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    alliance_id,
                    party1_id,
                    party2_id,
                    alliance_type,
                    terms,
                    is_secret,
                    created_at
                FROM social_alliances
                WHERE (party1_id = $1 OR party2_id = $1)
                  AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 10
                """,
                npc_id
            )

            alliances = []
            for row in rows:
                alliances.append({
                    "alliance_id": row["alliance_id"],
                    "party1_id": row["party1_id"],
                    "party2_id": row["party2_id"],
                    "alliance_type": row["alliance_type"],
                    "terms": json.loads(row["terms"]) if isinstance(row["terms"], str) else row["terms"],
                    "is_secret": row["is_secret"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                })

            return alliances
    except Exception as e:
        logger.error(f"Failed to get alliances for NPC {npc_id}: {e}")
        return []
