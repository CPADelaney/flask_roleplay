"""Post-turn outbox dispatcher and worker."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping

from celery import shared_task

from db.connection import get_db_connection_context, run_async_in_worker_loop

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from celery import Celery

logger = logging.getLogger(__name__)

_OUTBOX_SIDE_EFFECTS: Mapping[str, Dict[str, Any]] = {
    "world": {
        "task": "nyx.tasks.background.world_tasks.apply_universal",
        "queue": "background",
    },
    "memory": {
        "task": "nyx.tasks.heavy.memory_tasks.add_and_embed",
        "queue": "heavy",
    },
    "conflict": {
        "task": "nyx.tasks.background.conflict_tasks.process_events",
        "queue": "background",
    },
    "npc": {
        "task": "nyx.tasks.background.npc_tasks.run_adaptation_cycle",
        "queue": "background",
    },
    "lore": {
        "task": "nyx.tasks.background.lore_tasks.precompute_scene_bundle",
        "queue": "background",
    },
}

_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF_SECONDS = 30
_MAX_BACKOFF_SECONDS = 30 * 60
_ERROR_TRUNCATE_LENGTH = 512

_SELECT_OUTBOX_SQL = """
SELECT id, payload, attempts
  FROM canon.outbox
 WHERE dead_lettered = FALSE
   AND next_run_at <= $1
 ORDER BY next_run_at ASC, id ASC
 FOR UPDATE SKIP LOCKED
 LIMIT 1
"""

_DELETE_OUTBOX_SQL = "DELETE FROM canon.outbox WHERE id = $1"

_UPDATE_OUTBOX_SQL = """
UPDATE canon.outbox
   SET attempts = $2,
       last_error = $3,
       next_run_at = $4,
       dead_lettered = $5
 WHERE id = $1
"""

INSERT_OUTBOX_SQL = """
INSERT INTO canon.outbox (payload, next_run_at)
VALUES ($1::jsonb, $2)
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _outbox_entries_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    side_effects: Dict[str, Dict[str, Any]] = payload.get("side_effects") or {}
    turn_id = payload.get("turn_id")
    entries: List[Dict[str, Any]] = []

    for key, config in _OUTBOX_SIDE_EFFECTS.items():
        effect_payload = side_effects.get(key)
        if not effect_payload:
            continue
        entry: Dict[str, Any] = {
            "effect": key,
            "task_name": config["task"],
            "queue": config.get("queue"),
            "priority": config.get("priority"),
            "kwargs": {"payload": effect_payload},
        }
        if turn_id is not None:
            entry["turn_id"] = turn_id
        entries.append(entry)

    return entries


async def _insert_outbox_entries(entries: Iterable[Dict[str, Any]]) -> int:
    entries = list(entries)
    if not entries:
        return 0

    now = _utcnow()
    records = [(entry, now) for entry in entries]

    async with get_db_connection_context() as conn:
        async with conn.transaction():
            await conn.executemany(INSERT_OUTBOX_SQL, records)
    return len(entries)


def _get_celery_app() -> "Celery":
    from celery_config import celery_app

    return celery_app


def _enqueue_task(entry_payload: Dict[str, Any]) -> None:
    task_name = entry_payload.get("task_name")
    if not task_name:
        raise RuntimeError("Outbox entry missing task name")

    kwargs = entry_payload.get("kwargs") or {}
    options: Dict[str, Any] = entry_payload.get("options") or {}

    queue = entry_payload.get("queue")
    if queue:
        options.setdefault("queue", queue)
        options.setdefault("routing_key", queue)
    priority = entry_payload.get("priority")
    if priority is not None:
        options.setdefault("priority", priority)

    _get_celery_app().send_task(task_name, kwargs=kwargs, **options)


def _backoff_seconds(attempts: int) -> int:
    attempts = max(attempts, 1)
    delay = _INITIAL_BACKOFF_SECONDS * (2 ** (attempts - 1))
    return min(delay, _MAX_BACKOFF_SECONDS)


def _format_error(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        message = f"{error.__class__.__name__}: {error}"
    else:
        message = str(error)
    if len(message) > _ERROR_TRUNCATE_LENGTH:
        return message[: _ERROR_TRUNCATE_LENGTH - 3] + "..."
    return message


async def _record_failure(conn, entry_id: int, attempts: int, error: BaseException | str) -> None:
    next_attempts = attempts + 1
    dead_lettered = next_attempts >= _MAX_ATTEMPTS
    next_run_at = _utcnow()
    if not dead_lettered:
        next_run_at = next_run_at + timedelta(seconds=_backoff_seconds(next_attempts))

    await conn.execute(
        _UPDATE_OUTBOX_SQL,
        entry_id,
        next_attempts,
        _format_error(error),
        next_run_at,
        dead_lettered,
    )


async def _drain_outbox(limit: int = 10) -> int:
    if limit <= 0:
        return 0

    processed = 0
    async with get_db_connection_context() as conn:
        for _ in range(limit):
            async with conn.transaction():
                now = _utcnow()
                row = await conn.fetchrow(_SELECT_OUTBOX_SQL, now)
                if not row:
                    break

                entry_id = row["id"]
                payload = row["payload"] or {}
                attempts = int(row.get("attempts", 0))

                try:
                    _enqueue_task(payload)
                except Exception as exc:
                    logger.exception("Failed to enqueue outbox entry id=%s", entry_id)
                    await _record_failure(conn, entry_id, attempts, exc)
                else:
                    await conn.execute(_DELETE_OUTBOX_SQL, entry_id)
                    processed += 1
    return processed


@shared_task(name="nyx.tasks.realtime.post_turn.dispatch", acks_late=True, autoretry_for=(), retry_backoff=False)
def dispatch(payload: Dict[str, Any] | None = None) -> str:
    """Persist side effects into the outbox and trigger the worker."""

    payload = payload or {}
    entries = _outbox_entries_from_payload(payload)

    if not entries:
        logger.debug("TurnPostProcessor no-op (turn_id=%s)", payload.get("turn_id"))
        return "no-op"

    inserted = run_async_in_worker_loop(_insert_outbox_entries(entries))
    drain_outbox.apply_async(kwargs={"limit": inserted}, queue="realtime", priority=0)
    logger.debug("TurnPostProcessor enqueued %s side effects via outbox", inserted)
    return f"outbox:{inserted}"


@shared_task(name="nyx.tasks.realtime.post_turn.drain_outbox", acks_late=True, autoretry_for=(), retry_backoff=False)
def drain_outbox(limit: int = 10) -> int:
    """Process pending outbox entries."""

    return run_async_in_worker_loop(_drain_outbox(limit=limit))


__all__ = ["dispatch", "drain_outbox"]
