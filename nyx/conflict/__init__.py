"""Helpers for orchestrating conflict resolution pipelines."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict

from sqlalchemy import select

from nyx.common.outbox import DuplicateEventError, append_event, get_session_factory
from nyx.conflict.fsm import Status
from nyx.conflict.models import ConflictResolution, get_session_factory as get_conflict_session_factory

logger = logging.getLogger(__name__)


def _coerce_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid conflict_id: {value!r}") from exc


def _append_outbox_event(payload: Dict[str, Any]) -> bool:
    """Persist a ConflictResolutionRequested outbox event synchronously."""

    factory = get_session_factory()
    session = factory()
    try:
        with session.begin():
            append_event(
                session,
                topic="ConflictResolutionRequested",
                payload=payload,
            )
    except DuplicateEventError:
        logger.debug("Conflict resolution request already enqueued", extra={"conflict_id": payload.get("conflict_id")})
        return False
    finally:
        session.close()
    return True


def _lookup_existing_status(conflict_id: uuid.UUID) -> Status | None:
    """Return the latest recorded pipeline status for a conflict if available."""

    factory = get_conflict_session_factory()
    session = factory()
    try:
        stmt = (
            select(ConflictResolution)
            .where(ConflictResolution.conflict_id == conflict_id)
            .order_by(ConflictResolution.created_at.desc())
        )
        row = session.scalars(stmt).first()
        if row is None:
            return None
        try:
            return Status(row.status)
        except ValueError:  # pragma: no cover - defensive guard for legacy rows
            logger.warning("Unknown conflict resolution status encountered: %s", row.status)
            return None
    finally:
        session.close()


async def request_conflict_resolution(
    user_id: int,
    conversation_id: int,
    conflict_id: Any,
    resolution_type: str,
    context: Dict[str, Any] | None = None,
    *,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Persist an outbox event that kicks off the conflict resolution pipeline."""

    conflict_uuid = _coerce_uuid(conflict_id)
    payload: Dict[str, Any] = {
        "conflict_id": str(conflict_uuid),
        "user_id": int(user_id),
        "conversation_id": int(conversation_id),
        "resolution_type": resolution_type,
        "context": dict(context or {}),
        "request_id": uuid.uuid4().hex,
    }
    if metadata:
        payload["metadata"] = dict(metadata)

    queued = await asyncio.to_thread(_append_outbox_event, payload)
    status = await asyncio.to_thread(_lookup_existing_status, conflict_uuid)

    return {
        "conflict_id": str(conflict_uuid),
        "queued": queued,
        "pipeline_status": status.value if status else None,
    }


__all__ = ["request_conflict_resolution"]
