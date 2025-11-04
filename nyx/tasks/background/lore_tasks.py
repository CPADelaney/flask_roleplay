"""Lore precomputation tasks."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from context.unified_cache import CacheOperationRequest, context_cache
from db.connection import get_db_connection_context
from lore.cache_version import bump_lore_version
from lore.lore_orchestrator import get_lore_orchestrator
from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.nyx_agent.context import (
    build_canonical_snapshot_payload,
    fetch_canonical_snapshot,
    persist_canonical_snapshot,
)
from nyx.config.flags import context_warmers_enabled
from nyx.tasks.base import NyxTask, app
from nyx.tasks.utils import run_coro
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)

_SNAPSHOTS = ConversationSnapshotStore()

_SUMMARY_PLACEHOLDER = "Culture summary pending refresh"
_SUMMARY_TTL = timedelta(hours=24)
_MAX_SUMMARY_REFRESH = 3


def _idempotency_key(payload: Dict[str, Any]) -> str:
    return f"lore:{payload.get('scene_id')}:{payload.get('region_id')}:{payload.get('turn_id')}"


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
    canonical = run_coro(fetch_canonical_snapshot(*ids))
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
    run_coro(persist_canonical_snapshot(ids[0], ids[1], payload))


async def _load_existing_culture_summaries(nation_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    if not nation_ids:
        return {}
    async with get_db_connection_context() as conn:
        rows = await conn.fetch(
            """
            SELECT COALESCE(id, nation_id) AS id,
                   culture_summary,
                   culture_summary_updated_at
            FROM Nations
            WHERE COALESCE(id, nation_id) = ANY($1::int[])
            """,
            list(nation_ids),
        )
    return {int(row["id"]): dict(row) for row in rows}


def _should_refresh_summary(row: Dict[str, Any]) -> bool:
    summary = (row.get("culture_summary") or "").strip()
    if not summary or summary == _SUMMARY_PLACEHOLDER:
        return True
    updated_at = row.get("culture_summary_updated_at")
    if not updated_at:
        return True
    if not isinstance(updated_at, datetime):
        return True
    current = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return (current - updated_at) >= _SUMMARY_TTL


async def _refresh_culture_summaries_async(
    user_id: int,
    conversation_id: int,
    nation_ids: Sequence[int],
) -> None:
    if not nation_ids:
        return

    existing = await _load_existing_culture_summaries(nation_ids)
    ordered_ids: List[int] = []
    for nid in nation_ids:
        if nid not in ordered_ids:
            ordered_ids.append(nid)

    candidates: List[int] = []
    for nid in ordered_ids:
        row = existing.get(nid)
        if row is None or _should_refresh_summary(row):
            candidates.append(nid)

    if not candidates:
        return

    orchestrator = await get_lore_orchestrator(user_id, conversation_id)

    updates: List[Tuple[int, str]] = []
    for nid in candidates[:_MAX_SUMMARY_REFRESH]:
        try:
            summary = await orchestrator.rc_summarize_culture(nid, format_type="brief")
        except Exception:  # pragma: no cover - best-effort refresh
            logger.debug("Culture summary refresh failed for nation_id=%s", nid, exc_info=True)
            continue
        if not summary:
            continue
        rendered = summary.strip()
        if not rendered:
            continue
        if len(rendered) > 600:
            rendered = rendered[:600].rstrip()
        updates.append((nid, rendered))

    if not updates:
        return

    async with get_db_connection_context() as conn:
        await conn.executemany(
            """
            UPDATE Nations
               SET culture_summary = $2,
                   culture_summary_updated_at = NOW()
             WHERE COALESCE(id, nation_id) = $1
            """,
            updates,
        )

    bump_lore_version()


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.background.lore_tasks.precompute_scene_bundle",
    acks_late=True,
    queue="background",
    priority=6,
)
@idempotent(key_fn=lambda payload: _idempotency_key(payload))
def precompute_scene_bundle(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Warm the lore cache for the referenced scene."""

    if not payload:
        return None

    conversation_id = str(payload.get("conversation_id", ""))
    user_id = str(payload.get("user_id", ""))
    turn_id = payload.get("turn_id")

    snapshot = _hydrate_snapshot(user_id, conversation_id)
    lore_log = snapshot.setdefault("lore_requests", [])
    lore_log.append({
        "turn_id": turn_id,
        "scene_id": payload.get("scene_id"),
        "region_id": payload.get("region_id"),
    })
    _SNAPSHOTS.put(user_id, conversation_id, snapshot)
    _persist_snapshot(user_id, conversation_id, snapshot)

    ids = _coerce_ids(user_id, conversation_id)
    if ids:
        nation_candidates: Sequence[int] = []
        raw_nation_payloads: List[Any] = []
        payload_meta = payload.get("payload")
        if isinstance(payload_meta, dict) and payload_meta.get("nation_ids") is not None:
            raw_nation_payloads.append(payload_meta.get("nation_ids"))
        if payload.get("nation_ids") is not None:
            raw_nation_payloads.append(payload.get("nation_ids"))

        for raw_entry in raw_nation_payloads:
            if isinstance(raw_entry, (list, tuple, set)):
                for candidate in raw_entry:
                    if isinstance(candidate, int):
                        nation_candidates.append(candidate)
                    elif isinstance(candidate, str) and candidate.strip().isdigit():
                        nation_candidates.append(int(candidate.strip()))
            elif isinstance(raw_entry, int):
                nation_candidates.append(raw_entry)
            elif isinstance(raw_entry, str) and raw_entry.strip().isdigit():
                nation_candidates.append(int(raw_entry.strip()))

        if nation_candidates:
            nation_candidates = list(dict.fromkeys(nation_candidates))
        if nation_candidates:
            try:
                run_coro(
                    _refresh_culture_summaries_async(
                        ids[0],
                        ids[1],
                        list(nation_candidates)[:5],
                    )
                )
            except Exception:  # pragma: no cover - refresh best effort
                logger.debug(
                    "Failed to refresh culture summaries for turn=%s", turn_id,
                    exc_info=True,
                )

        if context_warmers_enabled():
            cache_key = f"context:lore:{ids[0]}:{ids[1]}:{turn_id}"
            bundle = {
                "turn_id": turn_id,
                "scene_id": payload.get("scene_id"),
                "region_id": payload.get("region_id"),
                "payload": payload,
            }
            try:
                run_coro(
                    context_cache.set_request(
                        CacheOperationRequest(
                            key=cache_key,
                            cache_level=2,
                            importance=0.4,
                        ),
                        bundle,
                    )
                )
            except Exception:  # pragma: no cover - cache warm best effort
                logger.debug(
                    "Failed to warm lore context cache for key=%s", cache_key, exc_info=True
                )

    logger.debug(
        "Lore precompute queued turn=%s scene=%s region=%s",
        turn_id,
        payload.get("scene_id"),
        payload.get("region_id"),
    )
    return {"status": "queued", "turn_id": turn_id}


__all__ = ["precompute_scene_bundle"]
