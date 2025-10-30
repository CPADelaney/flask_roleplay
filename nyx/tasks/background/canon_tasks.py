"""Background tasks for conflict canon subsystem (canonization, lore compliance).

These tasks handle expensive LLM operations related to lore and canon:
- Canonizing conflict resolutions
- Generating canon references (NPC dialogue, world state updates)
- Checking lore compliance (when LLM analysis is needed)

The hot path uses fast vector similarity for initial compliance checks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from db.connection import get_db_connection_context
from nyx.tasks.utils import with_retry, run_coro
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _idempotency_key_canonize(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for canonization."""
    conflict_id = payload.get("conflict_id")
    resolution_hash = hash(str(payload.get("resolution", {})))
    return f"canonize:{conflict_id}:{resolution_hash}"


def _idempotency_key_references(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for canon reference generation."""
    conflict_id = payload.get("conflict_id")
    return f"canon_refs:{conflict_id}"


async def _build_canon_record_async(
    conflict_id: int, resolution_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a canon record from conflict resolution (may involve LLM)."""
    from logic.conflict_system.conflict_canon import build_canon_record

    # This may call LLM for summarization/analysis
    record = build_canon_record(conflict_id, resolution_data)
    return record


async def _generate_canon_dialogue_async(conflict_id: int) -> list[Dict[str, Any]]:
    """Generate NPC dialogue referencing the canonical event (LLM call)."""
    from logic.conflict_system.conflict_canon import generate_canon_dialogue

    # Slow LLM call to generate potential NPC dialogue
    dialogue_rows = generate_canon_dialogue(conflict_id)
    return dialogue_rows


@shared_task(
    name="nyx.tasks.background.canon_tasks.canonize_conflict",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_canonize)
def canonize_conflict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Canonize a conflict resolution in the background (slow LLM call).

    This task is dispatched when a conflict is resolved. It generates a
    canonical record of the conflict and stores it in the canon DB/cache.

    Args:
        payload: Dict with keys:
            - conflict_id: int
            - resolution: dict (winner, outcome, consequences, etc.)

    Returns:
        Dict with status and event_id
    """
    conflict_id = payload.get("conflict_id")
    resolution_data = payload.get("resolution", {})

    if not conflict_id:
        raise ValueError("conflict_id is required")

    logger.info(f"Canonizing conflict {conflict_id}")

    # Generate the canon record (may involve LLM)
    canon_record = run_coro(_build_canon_record_async(conflict_id, resolution_data))

    # Store in the canon.events table
    async def _store_canon_event():
        async with get_db_connection_context() as conn:
            import uuid
            request_id = uuid.uuid4()
            result = await conn.fetchrow(
                """
                SELECT canon.apply_event($1::jsonb) AS result
                """,
                {
                    "request_id": str(request_id),
                    "conflict_id": conflict_id,
                    "resolution": resolution_data,
                    "record": canon_record,
                },
            )
            event_id = result["result"]["event_id"] if result else None
            logger.info(f"Stored canon event {event_id} for conflict {conflict_id}")
            return event_id

    event_id = run_coro(_store_canon_event())

    # Dispatch follow-up task to generate canon references
    generate_canon_references.delay({"conflict_id": conflict_id})

    return {"status": "canonized", "conflict_id": conflict_id, "event_id": event_id}


@shared_task(
    name="nyx.tasks.background.canon_tasks.generate_canon_references",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_references)
def generate_canon_references(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate NPC dialogue and world references for a canonical event (slow LLM call).

    This task generates potential NPC dialogue that references the canonical
    event, allowing NPCs to naturally reference past conflicts in conversation.

    Args:
        payload: Dict with keys:
            - conflict_id: int

    Returns:
        Dict with status and reference_count
    """
    conflict_id = payload.get("conflict_id")

    if not conflict_id:
        raise ValueError("conflict_id is required")

    logger.info(f"Generating canon references for conflict {conflict_id}")

    # Generate dialogue (slow LLM call)
    dialogue_rows = run_coro(_generate_canon_dialogue_async(conflict_id))

    # Store in the DB (table TBD - could be canon.dialogue or similar)
    async def _store_references():
        async with get_db_connection_context() as conn:
            # For now, store as JSONB; in production you might have a dedicated table
            for i, row in enumerate(dialogue_rows):
                await conn.execute(
                    """
                    INSERT INTO canon.events (request_id, payload, applied_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (request_id) DO NOTHING
                    """,
                    f"dialogue:{conflict_id}:{i}",
                    row,
                )
        logger.info(f"Stored {len(dialogue_rows)} canon references for conflict {conflict_id}")
        return len(dialogue_rows)

    reference_count = run_coro(_store_references())

    return {"status": "generated", "conflict_id": conflict_id, "reference_count": reference_count}


def _idempotency_key_lore_check(payload: Dict[str, Any]) -> str:
    """Generate idempotency key for lore compliance check."""
    content_hash = payload.get("content_hash", "")
    category = payload.get("category", "general")
    return f"lore_check:{category}:{content_hash}"


async def _check_lore_compliance_async(
    content: str, category: str
) -> Dict[str, Any]:
    """Check lore compliance with detailed LLM analysis (slow)."""
    from logic.conflict_system.conflict_canon import check_lore_compliance

    # This involves vector search + LLM analysis
    compliance_result = check_lore_compliance(content, category)
    return compliance_result


@shared_task(
    name="nyx.tasks.background.canon_tasks.check_lore_compliance",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_lore_check)
def check_lore_compliance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Check content for lore compliance with detailed analysis (slow LLM call).

    This task performs a full semantic analysis of content against existing lore,
    using vector similarity + LLM analysis. The hot path uses a fast rule-based
    check and dispatches this for thorough review.

    Args:
        payload: Dict with keys:
            - content: str (content to check)
            - category: str (lore category)
            - content_hash: str (for caching)
            - ttl: int (optional, cache TTL in seconds, default 3600)

    Returns:
        Dict with status and compliance result
    """
    content = payload.get("content", "")
    category = payload.get("category", "general")
    content_hash = payload.get("content_hash", "")
    ttl = payload.get("ttl", 3600)

    if not content:
        raise ValueError("content is required")

    logger.info(f"Checking lore compliance for content (hash={content_hash}, category={category})")

    # Run the slow LLM analysis
    compliance_result = run_coro(_check_lore_compliance_async(content, category))

    # Cache the result
    from infra.cache import cache_key, set_json
    key = cache_key("lore_check", category, content_hash)
    set_json(key, compliance_result, ex=ttl)

    logger.info(
        f"Cached lore compliance result at {key}: "
        f"compliant={compliance_result.get('is_compliant')}"
    )

    return {
        "status": "checked",
        "content_hash": content_hash,
        "cache_key": key,
        "is_compliant": compliance_result.get("is_compliant", True),
        "conflicts": compliance_result.get("conflicts", []),
    }


__all__ = [
    "canonize_conflict",
    "generate_canon_references",
    "check_lore_compliance",
]
