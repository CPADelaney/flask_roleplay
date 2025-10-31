"""Background tasks for conflict canon slow-path operations."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from celery import shared_task

from infra.cache import cache_key, set_json
from logic.conflict_system.conflict_canon import ConflictCanonSubsystem
from nyx.tasks.utils import run_coro, with_retry
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _require_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer") from exc


def _idempotency_key_canonize(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    resolution_hash = hash(str(payload.get("resolution", {})))
    return f"canonize:{conflict_id}:{resolution_hash}"


def _idempotency_key_references(payload: Dict[str, Any]) -> str:
    cache_id = payload.get("cache_id")
    event_id = payload.get("event_id")
    return f"canon_refs:{cache_id}:{event_id}"


def _idempotency_key_suggestions(payload: Dict[str, Any]) -> str:
    cache_id = payload.get("cache_id")
    conflict_type = payload.get("conflict_type")
    return f"canon_suggestions:{cache_id}:{conflict_type}"


def _idempotency_key_mythology(payload: Dict[str, Any]) -> str:
    conflict_id = payload.get("conflict_id")
    return f"canon_myth:{conflict_id}"


async def _canonize_background(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
    resolution: Dict[str, Any],
    snapshot_id: int | None,
) -> Dict[str, Any]:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)

    try:
        result = await subsystem._perform_canon_evaluation_background(conflict_id, resolution)
        if snapshot_id:
            await subsystem._mark_snapshot_status(snapshot_id, 'completed', result=result)
        return result
    except Exception as exc:
        logger.exception("Canonization failed for conflict %s", conflict_id)
        if snapshot_id:
            await subsystem._mark_snapshot_status(
                snapshot_id,
                'failed',
                result={'became_canonical': False},
                error=str(exc),
            )
        raise


@shared_task(
    name="canon.canonize_conflict",
    bind=True,
    max_retries=3,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_canonize)
def canonize_conflict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the slow canon evaluation pipeline."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    conflict_id = _require_int(payload.get("conflict_id"), "conflict_id")
    snapshot_id = payload.get("snapshot_id")
    if snapshot_id is not None:
        snapshot_id = _require_int(snapshot_id, "snapshot_id")
    resolution = payload.get("resolution") or {}

    logger.info(
        "Canonizing conflict %s for user=%s conversation=%s", conflict_id, user_id, conversation_id
    )

    result = run_coro(
        _canonize_background(user_id, conversation_id, conflict_id, resolution, snapshot_id)
    )

    return {
        "status": "completed" if result.get("became_canonical") else "queued",
        "conflict_id": conflict_id,
        "result": result,
    }


async def _reference_background(
    user_id: int,
    conversation_id: int,
    cache_id: int,
    event_id: int,
    context: str,
) -> None:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)
    await subsystem.build_reference_cache_background(cache_id, event_id, context)


@shared_task(
    name="canon.generate_canon_references",
    bind=True,
    max_retries=3,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_references)
def generate_canon_references(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and persist canon reference cache entries."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    cache_id = _require_int(payload.get("cache_id"), "cache_id")
    event_id = _require_int(payload.get("event_id"), "event_id")
    context = str(payload.get("context", "casual"))

    logger.info(
        "Generating canon references for event %s (cache=%s, user=%s conv=%s)",
        event_id,
        cache_id,
        user_id,
        conversation_id,
    )

    run_coro(_reference_background(user_id, conversation_id, cache_id, event_id, context))

    return {
        "status": "queued",
        "event_id": event_id,
        "cache_id": cache_id,
    }


async def _suggestions_background(
    user_id: int,
    conversation_id: int,
    cache_id: int,
    conflict_type: str,
    conflict_context: Dict[str, Any],
    matching_event_ids: List[int],
) -> None:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)
    await subsystem.build_compliance_suggestions_background(
        cache_id,
        conflict_type,
        conflict_context,
        matching_event_ids,
    )


@shared_task(
    name="canon.generate_lore_suggestions",
    bind=True,
    max_retries=3,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_suggestions)
def generate_lore_suggestions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate lore compliance suggestions asynchronously."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    cache_id = _require_int(payload.get("cache_id"), "cache_id")
    conflict_type = str(payload.get("conflict_type", "unknown"))
    conflict_context = payload.get("conflict_context") or {}
    matching_event_ids = [int(e) for e in (payload.get("matching_event_ids") or [])]

    logger.info(
        "Generating lore suggestions for cache %s (user=%s conv=%s type=%s)",
        cache_id,
        user_id,
        conversation_id,
        conflict_type,
    )

    run_coro(
        _suggestions_background(
            user_id,
            conversation_id,
            cache_id,
            conflict_type,
            conflict_context,
            matching_event_ids,
        )
    )

    return {"status": "queued", "cache_id": cache_id}


async def _mythology_background(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
) -> str:
    subsystem = ConflictCanonSubsystem(user_id, conversation_id)
    mythology_text = await subsystem._generate_mythology_text_background(conflict_id)

    cache_payload = {
        "text": mythology_text,
        "created_at": datetime.utcnow().isoformat(),
    }
    set_json(cache_key("canon", "mythology", conflict_id), cache_payload, ex=3600)
    return mythology_text


@shared_task(
    name="canon.generate_mythology",
    bind=True,
    max_retries=3,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_idempotency_key_mythology)
def generate_mythology_reinterpretation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate mythological reinterpretation text for a conflict."""

    user_id = _require_int(payload.get("user_id"), "user_id")
    conversation_id = _require_int(payload.get("conversation_id"), "conversation_id")
    conflict_id = _require_int(payload.get("conflict_id"), "conflict_id")

    logger.info(
        "Generating mythology reinterpretation for conflict %s (user=%s conv=%s)",
        conflict_id,
        user_id,
        conversation_id,
    )

    mythology_text = run_coro(_mythology_background(user_id, conversation_id, conflict_id))

    return {
        "status": "generated",
        "conflict_id": conflict_id,
        "mythology": mythology_text,
    }


__all__ = [
    "canonize_conflict",
    "generate_canon_references",
    "generate_lore_suggestions",
    "generate_mythology_reinterpretation",
]
