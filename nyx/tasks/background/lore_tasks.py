"""Lore precomputation tasks."""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"lore:{payload.get('scene_id')}:{payload.get('region_id')}:{payload.get('turn_id')}"


@shared_task(name="nyx.tasks.background.lore_tasks.precompute_scene_bundle", acks_late=True)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def precompute_scene_bundle(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Warm the lore cache for the referenced scene."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")

    snapshot = _SNAPSHOTS.get(user_id, conversation_id)
    lore_log = snapshot.setdefault("lore_requests", [])
    lore_log.append({
        "turn_id": turn_id,
        "scene_id": payload.get("scene_id"),
        "region_id": payload.get("region_id"),
    })
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)

    logger.debug(
        "Lore precompute queued turn=%s scene=%s region=%s",
        turn_id,
        payload.get("scene_id"),
        payload.get("region_id"),
    )
    return {"status": "queued", "turn_id": turn_id}


__all__ = ["precompute_scene_bundle"]
