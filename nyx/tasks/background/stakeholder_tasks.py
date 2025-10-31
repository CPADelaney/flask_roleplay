"""Background tasks for stakeholder autonomy (actions, reactions, decisions).

These tasks offload expensive LLM calls from the hot path (event handlers)
to background workers. Results are stored in planned_stakeholder_actions table
for fast hot-path consumption.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from celery import shared_task

from db.connection import get_db_connection_context
from nyx.tasks.utils import with_retry
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _compute_context_hash(context: Dict[str, Any]) -> str:
    """Compute a stable hash for the scene context for deduplication."""
    # Include only stable fields for context matching
    stable_keys = ["scene_id", "conflict_id", "stakeholder_ids", "phase"]
    stable_context = {k: context.get(k) for k in stable_keys if k in context}
    serialized = json.dumps(stable_context, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _idempotency_key_action(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for action generation."""
    stakeholder_id = payload.get("stakeholder_id")
    context_hash = payload.get("context_hash", "")
    return f"stakeholder_action:{stakeholder_id}:{context_hash}"


def _idempotency_key_reaction(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for reaction generation."""
    stakeholder_id = payload.get("stakeholder_id")
    event_id = payload.get("event_id", "")
    return f"stakeholder_reaction:{stakeholder_id}:{event_id}"


def _idempotency_key_populate(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for stakeholder population."""
    stakeholder_id = payload.get("stakeholder_id")
    return f"stakeholder_populate:{stakeholder_id}"


def _idempotency_key_create(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for stakeholder creation."""
    npc_id = payload.get("npc_id")
    conflict_id = payload.get("conflict_id")
    return f"stakeholder_create:{npc_id}:{conflict_id}"


def _idempotency_key_adapt(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for role adaptation."""
    stakeholder_id = payload.get("stakeholder_id")
    phase = payload.get("adaptation_context", {}).get("phase", "")
    return f"stakeholder_adapt:{stakeholder_id}:{phase}"


async def _make_autonomous_decision_async(
    stakeholder_id: int, scene_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Call the slow LLM-based decision logic (imported dynamically to avoid circular deps)."""
    # Import here to avoid circular dependencies and keep task module lightweight
    from logic.conflict_system.autonomous_stakeholder_actions import make_autonomous_decision

    # This is the blocking LLM call we're offloading from the hot path
    result = make_autonomous_decision(stakeholder_id, scene_context)
    return result


async def _generate_reaction_async(
    stakeholder_id: int, event_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Call the slow LLM-based reaction logic."""
    from logic.conflict_system.autonomous_stakeholder_actions import generate_reaction

    result = generate_reaction(stakeholder_id, event_data)
    return result


async def _populate_stakeholder_details_async(stakeholder_id: int) -> Dict[str, Any]:
    """Enrich a thin stakeholder record with LLM-generated details."""
    # This would call whatever slow LLM logic enriches stakeholder profiles
    # For now, this is a placeholder for the pattern
    logger.info(f"Populating details for stakeholder {stakeholder_id} (LLM call)")
    # TODO: Implement actual LLM enrichment logic
    return {"status": "enriched", "stakeholder_id": stakeholder_id}


@shared_task(
    name="nyx.tasks.background.stakeholder_tasks.generate_stakeholder_action",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_action)
def generate_stakeholder_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an autonomous action for a stakeholder (slow LLM call).

    This task is dispatched from hot-path event handlers when a stakeholder
    needs to make a decision. The result is written to planned_stakeholder_actions
    for fast consumption by the next hot-path query.

    Args:
        payload: Dict with keys:
            - stakeholder_id: int
            - scene_context: dict (scene data, other stakeholders, etc.)
            - context_hash: optional str for deduplication
            - priority: optional int (default 5)

    Returns:
        Dict with status and action_id
    """
    stakeholder_id = payload.get("stakeholder_id")
    scene_context = payload.get("scene_context", {})
    context_hash = payload.get("context_hash") or _compute_context_hash(scene_context)
    priority = payload.get("priority", 5)

    if not stakeholder_id:
        raise ValueError("stakeholder_id is required")

    logger.info(f"Generating autonomous action for stakeholder {stakeholder_id}")

    # Run the slow LLM call in an async context (if needed)
    from nyx.tasks.utils import run_coro

    action_result = run_coro(_make_autonomous_decision_async(stakeholder_id, scene_context))

    # Store the result in the DB for hot-path consumption
    async def _store_action():
        async with get_db_connection_context() as conn:
            action_id = await conn.fetchval(
                """
                INSERT INTO planned_stakeholder_actions
                (stakeholder_id, scene_id, scene_hash, conflict_id, kind, payload,
                 status, priority, context_hash, available_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                RETURNING id
                """,
                stakeholder_id,
                scene_context.get("scene_id"),
                scene_context.get("scene_hash"),
                scene_context.get("conflict_id"),
                "action",
                json.dumps(action_result),
                "ready",
                priority,
                context_hash,
            )
            logger.info(
                f"Stored action {action_id} for stakeholder {stakeholder_id} (context_hash={context_hash})"
            )
            return action_id

    action_id = run_coro(_store_action())

    return {"status": "generated", "action_id": action_id, "stakeholder_id": stakeholder_id}


@shared_task(
    name="nyx.tasks.background.stakeholder_tasks.generate_stakeholder_reaction",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_reaction)
def generate_stakeholder_reaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a reaction for a stakeholder to a specific event (slow LLM call).

    Args:
        payload: Dict with keys:
            - stakeholder_id: int
            - event_data: dict (event details, player choice, etc.)
            - event_id: str for deduplication
            - priority: optional int (default 5)

    Returns:
        Dict with status and action_id
    """
    stakeholder_id = payload.get("stakeholder_id")
    event_data = payload.get("event_data", {})
    event_id = payload.get("event_id", "")
    priority = payload.get("priority", 5)

    if not stakeholder_id:
        raise ValueError("stakeholder_id is required")

    logger.info(f"Generating reaction for stakeholder {stakeholder_id} to event {event_id}")

    from nyx.tasks.utils import run_coro

    reaction_result = run_coro(_generate_reaction_async(stakeholder_id, event_data))

    # Store the result
    async def _store_reaction():
        async with get_db_connection_context() as conn:
            action_id = await conn.fetchval(
                """
                INSERT INTO planned_stakeholder_actions
                (stakeholder_id, scene_id, conflict_id, kind, payload,
                 status, priority, context_hash, available_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                RETURNING id
                """,
                stakeholder_id,
                event_data.get("scene_id"),
                event_data.get("conflict_id"),
                "reaction",
                json.dumps(reaction_result),
                "ready",
                priority,
                event_id,
            )
            logger.info(f"Stored reaction {action_id} for stakeholder {stakeholder_id}")
            return action_id

    action_id = run_coro(_store_reaction())

    return {"status": "generated", "action_id": action_id, "stakeholder_id": stakeholder_id}


@shared_task(
    name="nyx.tasks.background.stakeholder_tasks.populate_stakeholder_details",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_populate)
def populate_stakeholder_details(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a thin stakeholder record with LLM-generated details (slow LLM call).

    This task is dispatched after creating a stakeholder with minimal data.
    The LLM enriches the record with personality, goals, relationships, etc.

    Args:
        payload: Dict with keys:
            - stakeholder_id: int

    Returns:
        Dict with status
    """
    stakeholder_id = payload.get("stakeholder_id")

    if not stakeholder_id:
        raise ValueError("stakeholder_id is required")

    logger.info(f"Populating details for stakeholder {stakeholder_id}")

    from nyx.tasks.utils import run_coro

    enrichment_result = run_coro(_populate_stakeholder_details_async(stakeholder_id))

    return {"status": "populated", "stakeholder_id": stakeholder_id, "result": enrichment_result}


@shared_task(
    name="nyx.tasks.background.stakeholder_tasks.create_stakeholder_background",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_create)
def create_stakeholder_background(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a stakeholder with LLM-generated profile (slow LLM call).

    This task is dispatched from event handlers when a new stakeholder needs to be created.
    The LLM generates personality, role, goals, etc. based on the NPC and conflict context.

    Args:
        payload: Dict with keys:
            - npc_id: int
            - conflict_id: int
            - conflict_type: str
            - user_id: int
            - conversation_id: int
            - suggested_role: optional str

    Returns:
        Dict with status and stakeholder_id
    """
    npc_id = payload.get("npc_id")
    conflict_id = payload.get("conflict_id")
    conflict_type = payload.get("conflict_type", "unknown")
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    suggested_role = payload.get("suggested_role")

    if not all([npc_id, conflict_id, user_id, conversation_id]):
        raise ValueError("npc_id, conflict_id, user_id, and conversation_id are required")

    logger.info(f"Creating stakeholder for NPC {npc_id} in conflict {conflict_id}")

    # Import the subsystem class and call the LLM method
    # Note: This is a workaround to avoid circular imports
    from logic.conflict_system.autonomous_stakeholder_actions import AutonomousStakeholderActions
    from nyx.tasks.utils import run_coro

    async def _create_stakeholder():
        # Instantiate the subsystem (lightweight, no heavy initialization)
        subsystem = AutonomousStakeholderActions(user_id=user_id, conversation_id=conversation_id)
        # Call the LLM method to create the stakeholder
        stakeholder = await subsystem.create_stakeholder(npc_id, conflict_id, suggested_role)
        return stakeholder

    stakeholder = run_coro(_create_stakeholder())

    if stakeholder:
        logger.info(f"Created stakeholder {stakeholder.stakeholder_id} for NPC {npc_id}")
        return {
            "status": "created",
            "stakeholder_id": stakeholder.stakeholder_id,
            "npc_id": npc_id,
            "conflict_id": conflict_id,
        }
    else:
        logger.warning(f"Failed to create stakeholder for NPC {npc_id}")
        return {"status": "failed", "npc_id": npc_id, "conflict_id": conflict_id}


@shared_task(
    name="nyx.tasks.background.stakeholder_tasks.adapt_stakeholder_role_background",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_adapt)
def adapt_stakeholder_role_background(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt stakeholder role based on context with LLM (slow LLM call).

    This task is dispatched when a stakeholder's role needs to adapt (e.g., phase transition).
    The LLM evaluates the new context and may change the stakeholder's role.

    Args:
        payload: Dict with keys:
            - stakeholder_id: int
            - adaptation_context: dict (e.g., {"phase": "climax"})
            - user_id: int
            - conversation_id: int

    Returns:
        Dict with status and adaptation result
    """
    stakeholder_id = payload.get("stakeholder_id")
    adaptation_context = payload.get("adaptation_context", {})
    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")

    if not all([stakeholder_id, user_id, conversation_id]):
        raise ValueError("stakeholder_id, user_id, and conversation_id are required")

    logger.info(f"Adapting role for stakeholder {stakeholder_id}")

    from logic.conflict_system.autonomous_stakeholder_actions import AutonomousStakeholderActions
    from nyx.tasks.utils import run_coro

    async def _adapt_role():
        subsystem = AutonomousStakeholderActions(user_id=user_id, conversation_id=conversation_id)
        # Get the stakeholder from the subsystem's active stakeholders
        stakeholder = subsystem._active_stakeholders.get(stakeholder_id)
        if not stakeholder:
            # Try to load from DB if not in memory
            logger.warning(f"Stakeholder {stakeholder_id} not in memory, skipping adaptation")
            return None
        # Call the LLM method to adapt the role
        result = await subsystem.adapt_stakeholder_role(stakeholder, adaptation_context)
        return result

    adaptation_result = run_coro(_adapt_role())

    if adaptation_result:
        logger.info(f"Adapted role for stakeholder {stakeholder_id}: {adaptation_result}")
        return {
            "status": "adapted",
            "stakeholder_id": stakeholder_id,
            "adaptation": adaptation_result,
        }
    else:
        logger.warning(f"Failed to adapt role for stakeholder {stakeholder_id}")
        return {"status": "failed", "stakeholder_id": stakeholder_id}


__all__ = [
    "generate_stakeholder_action",
    "generate_stakeholder_reaction",
    "populate_stakeholder_details",
    "create_stakeholder_background",
    "adapt_stakeholder_role_background",
]
