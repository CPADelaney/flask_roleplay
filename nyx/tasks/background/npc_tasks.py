"""NPC adaptation tasks executed post-turn."""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"npc:{payload.get('conversation_id')}:{payload.get('turn_id')}"


@shared_task(name="nyx.tasks.background.npc_tasks.run_adaptation_cycle", acks_late=True)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def run_adaptation_cycle(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Update NPC state in the background."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")

    snapshot = _SNAPSHOTS.get(user_id, conversation_id)
    npc_log = snapshot.setdefault("npc_events", [])
    npc_log.append({
        "turn_id": turn_id,
        "npcs": payload.get("npcs", []),
        "payload": payload.get("payload", {}),
    })
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)

    logger.debug(
        "NPC adaptation queued turn=%s conversation=%s count=%s",
        turn_id,
        conversation_id,
        len(payload.get("npcs", [])),
    )
    return {"status": "queued", "turn_id": turn_id, "npcs": payload.get("npcs", [])}


__all__ = ["run_adaptation_cycle"]
