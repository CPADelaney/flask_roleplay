"""NPC domain reactions to Nyx domain events."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List
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


def _event_key(prefix: str, expected_type: str, *fields: str):
    def _key(*args: Any, **kwargs: Any) -> str:
        payload = _payload_from_args(args, kwargs)
        if payload.get("type") != expected_type:
            return ""
        data = payload.get("data")
        if not isinstance(data, dict):
            return ""
        values: List[str] = []
        for field in fields:
            value = data.get(field)
            if value is None:
                return ""
            values.append(str(value))
        joined = ":".join(values)
        return f"{prefix}:{expected_type}:{joined}" if joined else ""

    return _key


def _extract_event(payload: Dict[str, Any], expected_type: str) -> Dict[str, Any] | None:
    if payload.get("type") != expected_type:
        logger.debug("Ignoring %s payload due to mismatched type", expected_type)
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


def adjust_relationship_stats(conflict_id: UUID, outcome: Dict[str, Any], trace_id: str | None) -> None:
    """Placeholder hook adjusting NPC relationship stats post-conflict."""

    logger.info(
        "Adjusting NPC relationships",
        extra={"conflict_id": str(conflict_id), "trace_id": trace_id, "outcome": outcome},
    )


def adjust_npc_mood(owner_id: UUID, memory_id: UUID, tags: Iterable[str], trace_id: str | None) -> None:
    """Placeholder hook mutating NPC mood based on new memories."""

    logger.info(
        "Updating NPC mood from memory",
        extra={
            "owner_id": str(owner_id),
            "memory_id": str(memory_id),
            "tags": list(tags),
            "trace_id": trace_id,
        },
    )


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.subscribers.npc.on_conflict_resolved",
    queue="nyx-npc",
)
@idempotent(key_fn=_event_key("npc", "ConflictResolved", "conflict_id"))
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
    adjust_relationship_stats(conflict_uuid, outcome, effective_trace)
    return {"status": "applied", "conflict_id": str(conflict_uuid)}


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.subscribers.npc.on_memory_created",
    queue="nyx-npc",
)
@idempotent(key_fn=_event_key("npc", "MemoryCreated", "owner_id", "memory_id"))
def on_memory_created(
    self, payload: Dict[str, Any], trace_id: str | None = None
) -> Dict[str, Any] | None:
    event = _extract_event(payload, "MemoryCreated")
    if event is None:
        return None

    owner_uuid = _parse_uuid(event.get("owner_id"), field="owner_id")
    memory_uuid = _parse_uuid(event.get("memory_id"), field="memory_id")
    if owner_uuid is None or memory_uuid is None:
        return None

    tags_value = event.get("tags")
    tags: Iterable[str]
    if isinstance(tags_value, list) and all(isinstance(tag, str) for tag in tags_value):
        tags = tags_value
    else:
        tags = []
    effective_trace = event.get("trace_id") or trace_id
    adjust_npc_mood(owner_uuid, memory_uuid, tags, effective_trace)
    return {
        "status": "applied",
        "owner_id": str(owner_uuid),
        "memory_id": str(memory_uuid),
        "tags": list(tags),
    }


__all__ = [
    "adjust_npc_mood",
    "adjust_relationship_stats",
    "on_conflict_resolved",
    "on_memory_created",
]

