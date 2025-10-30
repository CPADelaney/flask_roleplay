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


__all__ = [
    "generate_stakeholder_action",
    "generate_stakeholder_reaction",
    "populate_stakeholder_details",
]
