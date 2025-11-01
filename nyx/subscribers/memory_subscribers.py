"""Celery subscribers reacting to domain events for memory orchestration."""

from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from nyx.tasks.base import NyxTask, app
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _payload_from_args(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if "payload" in kwargs and isinstance(kwargs["payload"], dict):
        return kwargs["payload"]
    if args:
        candidate = args[-1]
        if isinstance(candidate, dict):
            return candidate
    return {}


def _event_key(expected_type: str, field: str):
    def _key(*args: Any, **kwargs: Any) -> str:
        payload = _payload_from_args(args, kwargs)
        if payload.get("type") != expected_type:
            return ""
        data = payload.get("data")
        if not isinstance(data, dict):
            return ""
        value = data.get(field)
        return f"memory:{expected_type}:{value}" if value else ""

    return _key


def _extract_event(payload: Dict[str, Any], expected_type: str) -> Dict[str, Any] | None:
    if payload.get("type") != expected_type:
        logger.debug("Ignoring payload for %s due to mismatched type", expected_type)
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        logger.warning("Event %s missing data payload", expected_type)
        return None
    return data


def _parse_uuid(value: Any, *, field: str) -> UUID | None:
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        logger.warning("Event field %s must be a UUID-compatible value", field)
        return None


def create_conflict_memories(conflict_id: UUID, outcome: Dict[str, Any], trace_id: str | None) -> None:
    """Hook invoked when a conflict reaches the integrated state."""

    logger.info(
        "Creating conflict memories",
        extra={"conflict_id": str(conflict_id), "trace_id": trace_id, "outcome": outcome},
    )


def store_npc_short_term_memory(
    npc_id: UUID, action: str, payload: Dict[str, Any], trace_id: str | None
) -> None:
    """Hook storing NPC short-term memories for recent actions."""

    logger.info(
        "Persisting NPC action memory",
        extra={
            "npc_id": str(npc_id),
            "action": action,
            "trace_id": trace_id,
            "payload": payload,
        },
    )


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.subscribers.memory.on_conflict_resolved",
    queue="nyx-memory",
)
@idempotent(key_fn=_event_key("ConflictResolved", "conflict_id"))
def on_conflict_resolved(
    self, payload: Dict[str, Any], trace_id: str | None = None
) -> Dict[str, Any] | None:
    event = _extract_event(payload, "ConflictResolved")
    if event is None:
        return None

    conflict_uuid = _parse_uuid(event.get("conflict_id"), field="conflict_id")
    if conflict_uuid is None:
        return None

    outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
    effective_trace = event.get("trace_id") or trace_id
    create_conflict_memories(conflict_uuid, outcome, effective_trace)
    return {"status": "applied", "conflict_id": str(conflict_uuid)}


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.subscribers.memory.on_npc_action_taken",
    queue="nyx-memory",
)
@idempotent(key_fn=_event_key("NPCActionTaken", "npc_id"))
def on_npc_action_taken(
    self, payload: Dict[str, Any], trace_id: str | None = None
) -> Dict[str, Any] | None:
    event = _extract_event(payload, "NPCActionTaken")
    if event is None:
        return None

    npc_uuid = _parse_uuid(event.get("npc_id"), field="npc_id")
    if npc_uuid is None:
        return None

    action = event.get("action")
    if not isinstance(action, str) or not action:
        logger.warning("NPCActionTaken event missing action string")
        return None

    memory_payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    effective_trace = event.get("trace_id") or trace_id
    store_npc_short_term_memory(npc_uuid, action, memory_payload, effective_trace)
    return {"status": "applied", "npc_id": str(npc_uuid), "action": action}


__all__ = [
    "create_conflict_memories",
    "on_conflict_resolved",
    "on_npc_action_taken",
    "store_npc_short_term_memory",
]

