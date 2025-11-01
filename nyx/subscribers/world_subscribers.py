"""World state subscribers for Nyx domain events."""

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


def _conflict_key(*args: Any, **kwargs: Any) -> str:
    payload = _payload_from_args(args, kwargs)
    if payload.get("type") != "ConflictResolved":
        return ""
    data = payload.get("data")
    if not isinstance(data, dict):
        return ""
    conflict_id = data.get("conflict_id")
    return f"world:ConflictResolved:{conflict_id}" if conflict_id else ""


def _extract_event(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if payload.get("type") != "ConflictResolved":
        logger.debug("Ignoring payload that is not ConflictResolved")
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        logger.warning("ConflictResolved payload missing data")
        return None
    return data


def _parse_uuid(value: Any) -> UUID | None:
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        logger.warning("ConflictResolved event includes invalid conflict_id")
        return None


def update_world_tension_counters(conflict_id: UUID, outcome: Dict[str, Any], trace_id: str | None) -> None:
    """Placeholder hook updating aggregated world state metrics."""

    logger.info(
        "Updating world tension counters",
        extra={"conflict_id": str(conflict_id), "trace_id": trace_id, "outcome": outcome},
    )


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.subscribers.world.on_conflict_resolved",
    queue="nyx-world",
)
@idempotent(key_fn=_conflict_key)
def on_conflict_resolved(
    self, payload: Dict[str, Any], trace_id: str | None = None
) -> Dict[str, Any] | None:
    event = _extract_event(payload)
    if event is None:
        return None

    conflict_uuid = _parse_uuid(event.get("conflict_id"))
    if conflict_uuid is None:
        return None

    outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
    effective_trace = event.get("trace_id") or trace_id
    update_world_tension_counters(conflict_uuid, outcome, effective_trace)
    return {"status": "applied", "conflict_id": str(conflict_uuid)}


__all__ = ["on_conflict_resolved", "update_world_tension_counters"]

