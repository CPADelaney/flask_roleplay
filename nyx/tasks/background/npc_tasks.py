"""NPC adaptation tasks executed post-turn."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from celery import shared_task

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.nyx_agent.context import (
    build_canonical_snapshot_payload,
    fetch_canonical_snapshot,
    persist_canonical_snapshot,
)
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"npc:{payload.get('conversation_id')}:{payload.get('turn_id')}"


def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _coerce_ids(user_id: str, conversation_id: str) -> Optional[Tuple[int, int]]:
    try:
        return int(user_id), int(conversation_id)
    except (TypeError, ValueError):
        return None


def _hydrate_snapshot(user_id: str, conversation_id: str) -> Dict[str, Any]:
    snapshot = _SNAPSHOTS.get(user_id, conversation_id)
    if snapshot:
        return snapshot
    ids = _coerce_ids(user_id, conversation_id)
    if not ids:
        return snapshot
    canonical = _run_coro(fetch_canonical_snapshot(*ids))
    if canonical:
        hydrated = dict(canonical)
        _SNAPSHOTS.put(user_id, conversation_id, hydrated)
        return hydrated
    return snapshot


def _persist_snapshot(user_id: str, conversation_id: str, snapshot: Dict[str, Any]) -> None:
    ids = _coerce_ids(user_id, conversation_id)
    if not ids:
        return
    payload = build_canonical_snapshot_payload(snapshot)
    if not payload:
        return
    _run_coro(persist_canonical_snapshot(ids[0], ids[1], payload))


@shared_task(name="nyx.tasks.background.npc_tasks.run_adaptation_cycle", acks_late=True)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def run_adaptation_cycle(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Update NPC state in the background."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")

    snapshot = _hydrate_snapshot(user_id, conversation_id)
    npc_log = snapshot.setdefault("npc_events", [])
    npc_log.append({
        "turn_id": turn_id,
        "npcs": payload.get("npcs", []),
        "payload": payload.get("payload", {}),
    })
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)
    _persist_snapshot(user_id, conversation_id, snapshot)

    logger.debug(
        "NPC adaptation queued turn=%s conversation=%s count=%s",
        turn_id,
        conversation_id,
        len(payload.get("npcs", [])),
    )
    return {"status": "queued", "turn_id": turn_id, "npcs": payload.get("npcs", [])}


__all__ = ["run_adaptation_cycle"]
