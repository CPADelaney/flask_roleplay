"""World-state post-turn tasks."""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.utils.idempotency import idempotent
from nyx.utils.versioning import reject_if_stale

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"world:{payload.get('conversation_id')}:{payload.get('turn_id')}"


@shared_task(name="nyx.tasks.background.world_tasks.apply_universal", acks_late=True)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def apply_universal(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Apply normalized world deltas with optimistic concurrency."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")
    incoming_world_version = payload.get("incoming_world_version", 0)
    deltas = payload.get("deltas") or {}

    snapshot = _SNAPSHOTS.get(user_id, conversation_id)
    current_version = snapshot.get("world_version", 0)

    if not reject_if_stale(current_version, incoming_world_version):
        logger.debug(
            "Skipping stale world apply (turn=%s current=%s incoming=%s)",
            turn_id,
            current_version,
            incoming_world_version,
        )
        return {"status": "stale", "current": current_version, "incoming": incoming_world_version}

    if not deltas:
        logger.debug("No world deltas for turn %s", turn_id)
    else:
        logger.info(
            "Queuing world deltas turn=%s conversation=%s keys=%s",
            turn_id,
            conversation_id,
            list(deltas.keys()),
        )
        # Placeholder for integration with the actual universal updater.  The real
        # implementation should apply the deltas via DAO/async functions.

    snapshot["world_version"] = incoming_world_version
    if deltas:
        snapshot.setdefault("pending_world_deltas", []).append({"turn_id": turn_id, "deltas": deltas})
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)
    return {"status": "queued", "version": incoming_world_version}


__all__ = ["apply_universal"]
