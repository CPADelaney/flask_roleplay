# nyx/tasks/realtime/post_turn.py
"""Post-turn outbox dispatcher and worker."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, TypedDict

from celery import Celery
from db.connection import AsyncpgConnection, get_db_connection_context, run_async_in_worker_loop
from nyx.tasks.base import NyxTask, app

logger = logging.getLogger(__name__)

# --- Configuration Constants ---

_OUTBOX_SIDE_EFFECTS: Mapping[str, Dict[str, Any]] = {
    "world": {
        "task": "nyx.tasks.background.world_tasks.apply_universal",
        "queue": "background",
        "priority": 3,
    },
    "memory": {
        "task": "nyx.tasks.heavy.memory_tasks.add_and_embed",
        "queue": "heavy",
        "priority": 6,
    },
    "conflict": {
        "task": "nyx.tasks.background.conflict_tasks.process_events",
        "queue": "background",
        "priority": 4,
    },
    "npc": {
        "task": "nyx.tasks.background.npc_tasks.run_adaptation_cycle",
        "queue": "background",
        "priority": 4,
    },
    "lore": {
        "task": "nyx.tasks.background.lore_tasks.precompute_scene_bundle",
        "queue": "background",
        "priority": 6,
    },
}

_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF_SECONDS = 30
_MAX_BACKOFF_SECONDS = 30 * 60
_ERROR_TRUNCATE_LENGTH = 512


# --- Typed Dictionaries for Clarity ---

class OutboxEntryPayload(TypedDict, total=False):
    """Defines the structure of the JSON payload in the outbox."""
    effect: str
    task_name: str
    queue: str | None
    priority: int | None
    kwargs: Dict[str, Any]
    turn_id: str | int | None
    options: Dict[str, Any]

class TurnPayload(TypedDict, total=False):
    """Defines the structure of the input payload to the dispatch task."""
    side_effects: Dict[str, Any]
    turn_id: str | int


# --- Service Class for Outbox Logic ---

class OutboxService:
    """Encapsulates all database and business logic for the transactional outbox."""

    _SELECT_SQL = """
        SELECT id, payload, attempts FROM canon.outbox
         WHERE dead_lettered = FALSE AND next_run_at <= $1
         ORDER BY next_run_at ASC, id ASC
         FOR UPDATE SKIP LOCKED LIMIT 1
    """
    _DELETE_SQL = "DELETE FROM canon.outbox WHERE id = $1"
    _UPDATE_SQL = """
        UPDATE canon.outbox SET attempts = $2, last_error = $3, next_run_at = $4, dead_lettered = $5
         WHERE id = $1
    """
    _INSERT_SQL = "INSERT INTO canon.outbox (payload, next_run_at) VALUES ($1, $2)"

    def __init__(self, celery_app: Celery | None = None):
        self._celery_app = celery_app

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _backoff_seconds(attempts: int) -> int:
        attempts = max(attempts, 1)
        delay = _INITIAL_BACKOFF_SECONDS * (2 ** (attempts - 1))
        return min(delay, _MAX_BACKOFF_SECONDS)

    @staticmethod
    def _format_error(error: BaseException | str) -> str:
        message = f"{error.__class__.__name__}: {error}" if isinstance(error, BaseException) else str(error)
        return message[:_ERROR_TRUNCATE_LENGTH - 3] + "..." if len(message) > _ERROR_TRUNCATE_LENGTH else message

    def _get_celery_app(self) -> Celery:
        if self._celery_app:
            return self._celery_app
        from nyx.tasks.celery_app import app as celery_app

        self._celery_app = celery_app
        return self._celery_app

    def _enqueue_task(self, entry_payload: OutboxEntryPayload) -> None:
        task_name = entry_payload.get("task_name")
        if not task_name:
            raise RuntimeError("Outbox entry missing task name")

        kwargs = entry_payload.get("kwargs", {})
        options = entry_payload.get("options", {})
        
        if queue := entry_payload.get("queue"):
            options.setdefault("queue", queue)
            options.setdefault("routing_key", queue)
        if priority := entry_payload.get("priority"):
            options.setdefault("priority", priority)

        self._get_celery_app().send_task(task_name, kwargs=kwargs, **options)

    async def _record_failure(self, conn: AsyncpgConnection, entry_id: int, attempts: int, error: BaseException) -> None:
        next_attempts = attempts + 1
        dead_lettered = next_attempts >= _MAX_ATTEMPTS
        
        next_run_at = self._utcnow()
        if not dead_lettered:
            next_run_at += timedelta(seconds=self._backoff_seconds(next_attempts))

        await conn.execute(self._UPDATE_SQL, entry_id, next_attempts, self._format_error(error), next_run_at, dead_lettered)

    async def insert_entries(self, entries: Iterable[OutboxEntryPayload]) -> int:
        """Serializes and inserts entries into the outbox."""
        entry_list = list(entries)
        if not entry_list:
            return 0

        now = self._utcnow()
        
        # --- CORE FIX: Serialize the dictionary to a JSON string using json.dumps() ---
        records = [(json.dumps(entry), now) for entry in entry_list]

        async with get_db_connection_context() as conn:
            await conn.executemany(self._INSERT_SQL, records)
        return len(entry_list)

    async def drain(self, limit: int = 10) -> int:
        """Selects and processes a batch of entries from the outbox."""
        if limit <= 0:
            return 0
            
        processed_count = 0
        async with get_db_connection_context() as conn:
            for _ in range(limit):
                async with conn.transaction():
                    row = await conn.fetchrow(self._SELECT_SQL, self._utcnow())
                    if not row:
                        break # No more work to do

                    entry_id, payload, attempts = row["id"], row["payload"], row["attempts"]
                    
                    try:
                        self._enqueue_task(payload)
                        await conn.execute(self._DELETE_SQL, entry_id)
                        processed_count += 1
                    except Exception as exc:
                        logger.exception("Failed to process outbox entry id=%s", entry_id)
                        await self._record_failure(conn, entry_id, attempts, exc)
        return processed_count

def _outbox_entries_from_payload(payload: TurnPayload) -> List[OutboxEntryPayload]:
    """Transforms turn side effects into a list of structured outbox entries."""
    side_effects = payload.get("side_effects", {})
    turn_id = payload.get("turn_id")
    entries: List[OutboxEntryPayload] = []

    for key, config in _OUTBOX_SIDE_EFFECTS.items():
        if effect_payload := side_effects.get(key):
            entry: OutboxEntryPayload = {
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


# --- Celery Tasks ---

@app.task(bind=True, base=NyxTask, name="nyx.tasks.realtime.post_turn.dispatch")
def dispatch(self, payload: TurnPayload | None = None) -> str:
    """Persists side effects into the outbox and triggers the drain worker."""
    payload = payload or {}
    entries = _outbox_entries_from_payload(payload)

    if not entries:
        logger.debug("TurnPostProcessor no-op (turn_id=%s)", payload.get("turn_id"))
        return "no-op"

    service = OutboxService()
    inserted = run_async_in_worker_loop(service.insert_entries(entries))
    
    # Trigger the drain task to process the items we just inserted.
    drain_outbox.apply_async(kwargs={"limit": inserted}, queue="realtime", priority=0)
    
    logger.debug("TurnPostProcessor enqueued %s side effects via outbox", inserted)
    return f"outbox:{inserted}"


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.realtime.post_turn.drain_outbox",
    acks_late=True,
)
def drain_outbox(self, limit: int = 10) -> int:
    """Processes a batch of pending entries from the outbox table."""
    service = OutboxService()
    return run_async_in_worker_loop(service.drain(limit=limit))
