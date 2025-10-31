"""Hot-path helpers for autonomous stakeholder actions (no blocking LLM calls).

This module provides fast, cache-first functions for the hot path:
- Querying ready actions from planned_stakeholder_actions table
- Rule-based behavior hints (no LLM)
- Dispatching background tasks for action generation

The slow LLM calls have been moved to nyx/tasks/background/stakeholder_tasks.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


def _compute_context_hash(scene_context: Dict[str, Any]) -> str:
    """Compute a stable hash for scene context for deduplication."""
    stable_keys = ["scene_id", "conflict_id", "stakeholder_ids", "phase"]
    stable_context = {k: scene_context.get(k) for k in stable_keys if k in scene_context}
    serialized = json.dumps(stable_context, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


async def fetch_ready_actions_for_scene(
    scene_context: Dict[str, Any], limit: int = 10
) -> List[Dict[str, Any]]:
    """Fetch ready stakeholder actions for the current scene (hot path).

    Args:
        scene_context: Scene context with scene_id, scene_hash, stakeholder_ids
        limit: Max actions to fetch

    Returns:
        List of action payloads
    """
    scene_hash = scene_context.get("scene_hash")
    stakeholder_ids = scene_context.get("stakeholder_ids", [])

    if not (scene_hash or stakeholder_ids):
        return []

    async with get_db_connection_context() as conn:
        if scene_hash:
            rows = await conn.fetch(
                """
                SELECT id, stakeholder_id, kind, payload, created_at
                FROM planned_stakeholder_actions
                WHERE scene_hash = $1 AND status = 'ready' AND available_at <= NOW()
                ORDER BY priority DESC, available_at ASC
                LIMIT $2
                """,
                scene_hash,
                limit,
            )
        elif stakeholder_ids:
            rows = await conn.fetch(
                """
                SELECT id, stakeholder_id, kind, payload, created_at
                FROM planned_stakeholder_actions
                WHERE stakeholder_id = ANY($1) AND status = 'ready' AND available_at <= NOW()
                ORDER BY priority DESC, available_at ASC
                LIMIT $2
                """,
                stakeholder_ids,
                limit,
            )
        else:
            return []

        actions = []
        for row in rows:
            actions.append({
                "action_id": row["id"],
                "stakeholder_id": row["stakeholder_id"],
                "kind": row["kind"],
                "payload": json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })

        return actions


async def mark_action_consumed(action_id: int) -> bool:
    """Mark a planned action as consumed (hot path).

    Args:
        action_id: Action ID from planned_stakeholder_actions

    Returns:
        True if successful
    """
    try:
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE planned_stakeholder_actions
                SET status = 'consumed', consumed_at = NOW()
                WHERE id = $1
                """,
                action_id,
            )
        return True
    except Exception as e:
        logger.error(f"Failed to mark action {action_id} as consumed: {e}")
        return False


def determine_scene_behavior(stakeholder: Any, scene_context: Dict[str, Any]) -> str:
    """Fast rule-based behavior hint for a stakeholder (no LLM).

    This replaces the slow LLM-based decision logic with fast rules.
    Returns a behavioral hint like "agitated", "observant", "strategic", etc.

    Args:
        stakeholder: Stakeholder object
        scene_context: Scene context dict

    Returns:
        Behavior hint string
    """
    stress = getattr(stakeholder, "stress_level", 0.5)
    role = getattr(stakeholder, "current_role", None)

    # Fast rule-based heuristics
    if stress > 0.8:
        return "agitated"
    elif stress > 0.6:
        return "tense"
    elif hasattr(role, "value") and role.value in ["mediator", "peacemaker"]:
        return "diplomatic"
    elif hasattr(role, "value") and role.value in ["instigator", "escalator"]:
        return "provocative"
    elif hasattr(role, "value") and role.value == "defender":
        return "protective"
    elif hasattr(role, "value") and role.value == "opportunist":
        return "calculating"
    else:
        return "observant"


def should_dispatch_action_generation(
    stakeholder: Any, scene_context: Dict[str, Any]
) -> bool:
    """Fast check if we should dispatch background action generation.

    Args:
        stakeholder: Stakeholder object
        scene_context: Scene context

    Returns:
        True if background task should be dispatched
    """
    # Only dispatch for active stakeholders in high-stakes scenes
    stress = getattr(stakeholder, "stress_level", 0.5)
    role = getattr(stakeholder, "current_role", None)

    # Skip for passive roles with low stress
    if hasattr(role, "value") and role.value in ["bystander", "victim"] and stress < 0.4:
        return False

    # Dispatch for high-stress or active roles
    if stress > 0.6:
        return True
    if hasattr(role, "value") and role.value in ["instigator", "mediator", "escalator"]:
        return True

    # Random sampling for others (10% chance to keep task load manageable)
    import random
    return random.random() < 0.1


def dispatch_action_generation(stakeholder: Any, scene_context: Dict[str, Any]) -> None:
    """Dispatch background task to generate stakeholder action (non-blocking).

    Args:
        stakeholder: Stakeholder object
        scene_context: Scene context
    """
    try:
        from nyx.tasks.background.stakeholder_tasks import generate_stakeholder_action

        context_hash = _compute_context_hash(scene_context)
        payload = {
            "stakeholder_id": stakeholder.stakeholder_id,
            "scene_context": scene_context,
            "context_hash": context_hash,
            "priority": 7 if getattr(stakeholder, "stress_level", 0.5) > 0.7 else 5,
        }
        generate_stakeholder_action.delay(payload)
        logger.debug(f"Dispatched action generation for stakeholder {stakeholder.stakeholder_id}")
    except Exception as e:
        logger.warning(f"Failed to dispatch action generation: {e}")


def dispatch_reaction_generation(
    stakeholder: Any, event_data: Dict[str, Any], event_id: str
) -> None:
    """Dispatch background task to generate stakeholder reaction (non-blocking).

    Args:
        stakeholder: Stakeholder object
        event_data: Event data (player choice, etc.)
        event_id: Unique event identifier
    """
    try:
        from nyx.tasks.background.stakeholder_tasks import generate_stakeholder_reaction

        payload = {
            "stakeholder_id": stakeholder.stakeholder_id,
            "event_data": event_data,
            "event_id": event_id,
            "priority": 8,  # Reactions are higher priority
        }
        generate_stakeholder_reaction.delay(payload)
        logger.debug(f"Dispatched reaction generation for stakeholder {stakeholder.stakeholder_id}")
    except Exception as e:
        logger.warning(f"Failed to dispatch reaction generation: {e}")


def dispatch_stakeholder_creation(
    npc_id: int,
    conflict_id: int,
    conflict_type: str,
    user_id: int,
    conversation_id: int,
    suggested_role: Optional[str] = None
) -> None:
    """Dispatch background task to create stakeholder with LLM (non-blocking).

    Args:
        npc_id: NPC ID
        conflict_id: Conflict ID
        conflict_type: Type of conflict
        user_id: User ID
        conversation_id: Conversation ID
        suggested_role: Optional suggested role for the stakeholder
    """
    try:
        from nyx.tasks.background.stakeholder_tasks import create_stakeholder_background

        payload = {
            "npc_id": npc_id,
            "conflict_id": conflict_id,
            "conflict_type": conflict_type,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "suggested_role": suggested_role,
            "priority": 6,  # Stakeholder creation is medium priority
        }
        create_stakeholder_background.delay(payload)
        logger.debug(f"Dispatched stakeholder creation for NPC {npc_id} in conflict {conflict_id}")
    except Exception as e:
        logger.warning(f"Failed to dispatch stakeholder creation: {e}")


def dispatch_role_adaptation(
    stakeholder_id: int,
    adaptation_context: Dict[str, Any],
    user_id: int,
    conversation_id: int
) -> None:
    """Dispatch background task to adapt stakeholder role with LLM (non-blocking).

    Args:
        stakeholder_id: Stakeholder ID
        adaptation_context: Context for role adaptation (e.g., phase transition)
        user_id: User ID
        conversation_id: Conversation ID
    """
    try:
        from nyx.tasks.background.stakeholder_tasks import adapt_stakeholder_role_background

        payload = {
            "stakeholder_id": stakeholder_id,
            "adaptation_context": adaptation_context,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "priority": 5,  # Role adaptation is lower priority
        }
        adapt_stakeholder_role_background.delay(payload)
        logger.debug(f"Dispatched role adaptation for stakeholder {stakeholder_id}")
    except Exception as e:
        logger.warning(f"Failed to dispatch role adaptation: {e}")


async def cleanup_expired_actions(age_hours: int = 24) -> int:
    """Cleanup old consumed/expired actions (maintenance task).

    Args:
        age_hours: Age threshold in hours

    Returns:
        Number of actions deleted
    """
    cutoff = datetime.now() - timedelta(hours=age_hours)
    try:
        async with get_db_connection_context() as conn:
            result = await conn.execute(
                """
                DELETE FROM planned_stakeholder_actions
                WHERE status IN ('consumed', 'expired')
                  AND created_at < $1
                """,
                cutoff,
            )
            # Extract count from result string like "DELETE 42"
            count = int(result.split()[-1]) if result else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired stakeholder actions")
            return count
    except Exception as e:
        logger.error(f"Failed to cleanup expired actions: {e}")
        return 0
