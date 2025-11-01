"""Domain event definitions and helpers for the Nyx platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from nyx.common.outbox import append_event


@dataclass(slots=True)
class ConflictResolved:
    """Event emitted when a conflict pipeline reaches the integrated state."""

    conflict_id: UUID
    outcome: Dict[str, Any]
    trace_id: str | None = None


@dataclass(slots=True)
class NPCActionTaken:
    """Event describing an action executed by an NPC orchestrator."""

    npc_id: UUID
    action: str
    payload: Dict[str, Any]
    trace_id: str | None = None


@dataclass(slots=True)
class MemoryCreated:
    """Event emitted whenever a memory record is persisted."""

    owner_id: UUID
    memory_id: UUID
    tags: list[str]
    trace_id: str | None = None


def _coerce_payload(value: Any) -> Any:
    """Recursively coerce UUID objects into serialisable representations."""

    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {key: _coerce_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_payload(item) for item in value]
    return value


def publish(session: Session, event_obj: Any) -> None:
    """Persist the supplied domain event into the Nyx outbox."""

    event_type = type(event_obj).__name__
    data = _coerce_payload(asdict(event_obj))
    payload = {"type": event_type, "data": data}
    append_event(
        session,
        topic=event_type,
        payload=payload,
        dedupe_key=str(uuid4()),
    )


__all__ = [
    "ConflictResolved",
    "MemoryCreated",
    "NPCActionTaken",
    "publish",
]

