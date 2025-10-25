"""NPC adaptation tasks executed post-turn."""

from __future__ import annotations

import asyncio
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
from npcs.npc_learning_adaptation import NPCLearningManager

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


async def _execute_adaptation_async(
    user_id: int, conversation_id: int, npc_ids: Tuple[int, ...], payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the NPC adaptation pipeline for the provided identifiers."""

    manager = NPCLearningManager(user_id, conversation_id)
    await manager.initialize()

    event_payload = payload.get("payload") or {}
    event_result: Dict[str, Any] | None = None
    if event_payload:
        event_text = (
            event_payload.get("event_text")
            or event_payload.get("summary")
            or event_payload.get("description")
            or "NPC interaction"
        )
        event_type = event_payload.get("event_type") or event_payload.get("type") or "interaction"
        player_response = event_payload.get("player_response")
        event_result = await manager.process_event_for_learning(
            event_text,
            event_type,
            list(npc_ids),
            player_response=player_response,
        )
        if event_result and not event_result.get("event_processed", True):
            error_detail = event_result.get("error") or "NPC learning event failed"
            raise RuntimeError(error_detail)

    cycle_result = await manager.run_regular_adaptation_cycle(list(npc_ids))
    if cycle_result is None:
        raise RuntimeError("NPC adaptation cycle returned no result")
    if event_result:
        cycle_result = dict(cycle_result)
        cycle_result["event_learning"] = event_result
    return cycle_result


@shared_task(name="nyx.tasks.background.npc_tasks.run_adaptation_cycle", acks_late=True)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def run_adaptation_cycle(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Update NPC state in the background."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")

    raw_npc_ids = payload.get("npcs") or []
    try:
        normalized_npc_ids = tuple(int(npc_id) for npc_id in raw_npc_ids)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("NPC identifiers must be integers") from exc

    adaptation_result: Dict[str, Any] | None = None
    if normalized_npc_ids:
        ids = _coerce_ids(user_id, conversation_id)
        if not ids:
            raise RuntimeError("Cannot run NPC adaptation without numeric identifiers")
        try:
            adaptation_result = _run_coro(
                _execute_adaptation_async(ids[0], ids[1], normalized_npc_ids, payload)
            )
        except Exception:
            logger.exception(
                "NPC adaptation failed turn=%s conversation=%s", turn_id, conversation_id
            )
            raise

        if adaptation_result.get("error"):
            raise RuntimeError(adaptation_result["error"])
        if not adaptation_result.get("cycle_completed", True):
            raise RuntimeError("NPC adaptation cycle did not complete")
    else:
        logger.debug(
            "No NPC identifiers provided for adaptation turn=%s conversation=%s", turn_id, conversation_id
        )

    snapshot = _hydrate_snapshot(user_id, conversation_id)
    if adaptation_result is not None:
        npc_log = snapshot.setdefault("npc_events", [])
        npc_log.append({
            "turn_id": turn_id,
            "npcs": list(normalized_npc_ids),
            "payload": payload.get("payload", {}),
            "result": adaptation_result,
        })
        snapshot["last_npc_adaptation"] = {
            "turn_id": turn_id,
            "npc_count": len(normalized_npc_ids),
        }
        _SNAPSHOTS.put(user_id, conversation_id, snapshot)
        _persist_snapshot(user_id, conversation_id, snapshot)

    logger.debug(
        "NPC adaptation processed turn=%s conversation=%s count=%s",
        turn_id,
        conversation_id,
        len(normalized_npc_ids),
    )

    response: Dict[str, Any] = {
        "status": "applied" if adaptation_result is not None else "noop",
        "turn_id": turn_id,
        "npcs": list(normalized_npc_ids),
    }
    if adaptation_result is not None:
        response["result"] = adaptation_result
    return response


__all__ = ["run_adaptation_cycle"]
