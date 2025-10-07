"""World-state post-turn tasks."""

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
from nyx.utils.versioning import reject_if_stale

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"world:{payload.get('conversation_id')}:{payload.get('turn_id')}"


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

    snapshot = _hydrate_snapshot(user_id, conversation_id)
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
    _persist_snapshot(user_id, conversation_id, snapshot)
    return {"status": "queued", "version": incoming_world_version}


__all__ = ["apply_universal"]
