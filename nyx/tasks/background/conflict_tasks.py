"""Conflict synthesizer background tasks."""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"conflict:{payload.get('conversation_id')}:{payload.get('turn_id')}"


@shared_task(name="nyx.tasks.background.conflict_tasks.process_events", acks_late=True)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def process_events(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Process deferred conflict computations."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")

    snapshot = _SNAPSHOTS.get(user_id, conversation_id)
    history = snapshot.setdefault("conflict_history", [])
    history.append({"turn_id": turn_id, "payload": payload.get("payload", {})})
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)

    logger.debug("Processed conflict events for turn=%s conversation=%s", turn_id, conversation_id)
    return {"status": "queued", "turn_id": turn_id}


__all__ = ["process_events"]
