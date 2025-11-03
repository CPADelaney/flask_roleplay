"""World-state post-turn tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from context.unified_cache import invalidate_prefixes
from db.connection import get_db_connection_context
from logic.universal_updater_agent import (
    UniversalUpdaterContext,
    apply_universal_updates_async,
    convert_updates_for_database,
)
from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.nyx_agent.context import (
    build_canonical_snapshot_payload,
    fetch_canonical_snapshot,
    persist_canonical_snapshot,
)
from nyx.config.flags import context_warmers_enabled
from nyx.tasks.base import NyxTask, app
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


async def _apply_world_deltas_async(
    user_id: int, conversation_id: int, deltas: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute the universal updater pipeline for the provided deltas."""

    ctx = UniversalUpdaterContext(user_id, conversation_id)
    await ctx.initialize()

    try:
        db_ready_updates = convert_updates_for_database(dict(deltas))
    except Exception:
        logger.exception(
            "Failed to normalize world deltas for conversation=%s", conversation_id
        )
        raise

    async with get_db_connection_context() as conn:
        return await apply_universal_updates_async(
            ctx, user_id, conversation_id, db_ready_updates, conn
        )


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.world_tasks.apply_universal",
    acks_late=True,
    queue="background",
    priority=3,
)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def apply_universal(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Apply normalized world deltas with optimistic concurrency."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")
    incoming_world_version = payload.get("incoming_world_version", 0)
    deltas = payload.get("deltas") or {}

    ids = _coerce_ids(user_id, conversation_id)

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

    apply_result: Dict[str, Any] | None = None

    if not deltas:
        logger.debug("No world deltas for turn %s", turn_id)
    else:
        logger.info(
            "Applying world deltas turn=%s conversation=%s keys=%s",
            turn_id,
            conversation_id,
            list(deltas.keys()),
        )
        if not ids:
            raise RuntimeError("Cannot apply world deltas without numeric identifiers")
        try:
            apply_result = _run_coro(_apply_world_deltas_async(ids[0], ids[1], deltas))
        except Exception:
            logger.exception(
                "Universal updater failed turn=%s conversation=%s", turn_id, conversation_id
            )
            raise

        if apply_result is None:
            raise RuntimeError("Universal updater returned no result")

        if not apply_result.get("success", True):
            error_message = apply_result.get("error") or apply_result.get("reason") or "Unknown error"
            logger.error(
                "Universal updater rejected deltas turn=%s conversation=%s error=%s",
                turn_id,
                conversation_id,
                error_message,
            )
            raise RuntimeError(error_message)

    snapshot["world_version"] = incoming_world_version
    if deltas:
        snapshot.setdefault("pending_world_deltas", []).append({"turn_id": turn_id, "deltas": deltas})
        if apply_result is not None:
            snapshot["last_world_apply"] = {"turn_id": turn_id, "result": apply_result}
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)
    _persist_snapshot(user_id, conversation_id, snapshot)

    response: Dict[str, Any] = {"status": "applied" if apply_result else "noop", "version": incoming_world_version}
    if apply_result is not None:
        response["result"] = apply_result

    if context_warmers_enabled() and ids:
        prefixes = [
            f"context:{ids[0]}:{ids[1]}",
            f"context:lore:{ids[0]}:{ids[1]}",
        ]
        try:
            _run_coro(invalidate_prefixes(prefixes))
        except Exception:  # pragma: no cover - invalidation is best-effort
            logger.debug(
                "Context cache invalidation failed for prefixes=%s", prefixes, exc_info=True
            )

    return response


__all__ = ["apply_universal"]
