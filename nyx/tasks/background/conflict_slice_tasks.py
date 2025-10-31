"""Background tasks for slice-of-life conflict slow-path work."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from celery import shared_task

from db.connection import get_db_connection_context
from logic.conflict_system.slice_of_life_conflicts import (
    ConflictDailyIntegration,
    EmergentConflictDetector,
    PatternBasedResolution,
    SliceOfLifeConflictManager,
)
from nyx.tasks.utils import run_coro, with_retry
from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


TENSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slice_conflict_tension_cache (
    user_id INT NOT NULL,
    conversation_id INT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB,
    last_error TEXT,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (user_id, conversation_id)
);
"""

ACTIVITY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slice_conflict_activity_cache (
    conflict_id INT NOT NULL,
    activity_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB,
    last_error TEXT,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (conflict_id, activity_hash)
);
"""

RESOLUTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slice_conflict_resolution_cache (
    conflict_id INT PRIMARY KEY,
    status TEXT NOT NULL,
    payload JSONB,
    last_error TEXT,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);
"""

TIME_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slice_conflict_time_cache (
    conflict_id INT NOT NULL,
    time_of_day TEXT NOT NULL,
    status TEXT NOT NULL,
    result BOOLEAN,
    last_error TEXT,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (conflict_id, time_of_day)
);
"""


async def _exec(sql: str, *args: Any) -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(sql, *args)


async def _upsert_tension(
    user_id: int,
    conversation_id: int,
    status: str,
    payload: Optional[Any],
    error: Optional[str],
) -> None:
    await _exec(TENSION_TABLE_SQL)
    await _exec(
        """
        INSERT INTO slice_conflict_tension_cache
            (user_id, conversation_id, status, payload, last_error, updated_at)
        VALUES ($1, $2, $3, $4::jsonb, $5, NOW())
        ON CONFLICT (user_id, conversation_id)
        DO UPDATE SET
            status = EXCLUDED.status,
            payload = EXCLUDED.payload,
            last_error = EXCLUDED.last_error,
            updated_at = NOW()
        """,
        int(user_id),
        int(conversation_id),
        status,
        json.dumps(payload) if payload is not None else None,
        error,
    )


async def _upsert_activity(
    conflict_id: int,
    activity_hash: str,
    status: str,
    payload: Optional[Any],
    error: Optional[str],
) -> None:
    await _exec(ACTIVITY_TABLE_SQL)
    await _exec(
        """
        INSERT INTO slice_conflict_activity_cache
            (conflict_id, activity_hash, status, payload, last_error, updated_at)
        VALUES ($1, $2, $3, $4::jsonb, $5, NOW())
        ON CONFLICT (conflict_id, activity_hash)
        DO UPDATE SET
            status = EXCLUDED.status,
            payload = EXCLUDED.payload,
            last_error = EXCLUDED.last_error,
            updated_at = NOW()
        """,
        int(conflict_id),
        activity_hash,
        status,
        json.dumps(payload) if payload is not None else None,
        error,
    )


async def _upsert_resolution(
    conflict_id: int,
    status: str,
    payload: Optional[Any],
    error: Optional[str],
) -> None:
    await _exec(RESOLUTION_TABLE_SQL)
    await _exec(
        """
        INSERT INTO slice_conflict_resolution_cache
            (conflict_id, status, payload, last_error, updated_at)
        VALUES ($1, $2, $3::jsonb, $4, NOW())
        ON CONFLICT (conflict_id)
        DO UPDATE SET
            status = EXCLUDED.status,
            payload = EXCLUDED.payload,
            last_error = EXCLUDED.last_error,
            updated_at = NOW()
        """,
        int(conflict_id),
        status,
        json.dumps(payload) if payload is not None else None,
        error,
    )


async def _upsert_time(
    conflict_id: int,
    time_of_day: str,
    status: str,
    result: Optional[bool],
    error: Optional[str],
) -> None:
    await _exec(TIME_TABLE_SQL)
    await _exec(
        """
        INSERT INTO slice_conflict_time_cache
            (conflict_id, time_of_day, status, result, last_error, updated_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (conflict_id, time_of_day)
        DO UPDATE SET
            status = EXCLUDED.status,
            result = EXCLUDED.result,
            last_error = EXCLUDED.last_error,
            updated_at = NOW()
        """,
        int(conflict_id),
        time_of_day,
        status,
        result,
        error,
    )


def _tension_key(payload: Dict[str, Any]) -> str:
    return f"slice_tension:{payload.get('user_id')}:{payload.get('conversation_id')}"


def _activity_key(payload: Dict[str, Any]) -> str:
    return f"slice_activity:{payload.get('conflict_id')}:{payload.get('activity_hash')}"


def _resolution_key(payload: Dict[str, Any]) -> str:
    return f"slice_resolution:{payload.get('conflict_id')}"


def _time_key(payload: Dict[str, Any]) -> str:
    return f"slice_time:{payload.get('conflict_id')}:{payload.get('time_of_day')}"


@shared_task(
    name="nyx.tasks.background.conflict_slice_tasks.refresh_tension_cache",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_tension_key)
def refresh_tension_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute emerging tensions and persist them in the cache table."""

    user_id = payload.get("user_id")
    conversation_id = payload.get("conversation_id")
    if user_id is None or conversation_id is None:
        raise ValueError("user_id and conversation_id are required")

    run_coro(_upsert_tension(user_id, conversation_id, "pending", None, None))

    detector = EmergentConflictDetector(int(user_id), int(conversation_id))
    try:
        tensions = run_coro(detector._detect_tensions())
        run_coro(_upsert_tension(user_id, conversation_id, "ready", tensions, None))
        return {
            "status": "ready",
            "user_id": int(user_id),
            "conversation_id": int(conversation_id),
            "items": len(tensions or []),
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "Failed to refresh slice-of-life tensions for user=%s conversation=%s",
            user_id,
            conversation_id,
        )
        run_coro(_upsert_tension(user_id, conversation_id, "failed", None, str(exc)))
        raise


@shared_task(
    name="nyx.tasks.background.conflict_slice_tasks.generate_conflict_activity",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_activity_key)
def generate_conflict_activity(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an activity manifestation for a conflict."""

    conflict_id = payload.get("conflict_id")
    activity_hash = payload.get("activity_hash")
    if conflict_id is None or not activity_hash:
        raise ValueError("conflict_id and activity_hash are required")

    run_coro(
        _upsert_activity(conflict_id, activity_hash, "pending", None, None)
    )

    manager = SliceOfLifeConflictManager(
        int(payload.get("user_id")), int(payload.get("conversation_id"))
    )

    try:
        event = run_coro(
            manager._embed_conflict_in_activity(
                int(conflict_id),
                payload.get("activity_type", "daily_routine"),
                list(payload.get("participating_npcs", [])),
            )
        )
        serialized = asdict(event)
        run_coro(
            _upsert_activity(conflict_id, activity_hash, "ready", serialized, None)
        )
        return {
            "status": "ready",
            "conflict_id": int(conflict_id),
            "activity_hash": activity_hash,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "Failed to generate conflict activity for conflict=%s", conflict_id
        )
        run_coro(
            _upsert_activity(conflict_id, activity_hash, "failed", None, str(exc))
        )
        raise


@shared_task(
    name="nyx.tasks.background.conflict_slice_tasks.evaluate_conflict_resolution",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_resolution_key)
def evaluate_conflict_resolution(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate whether a conflict should resolve based on accumulated patterns."""

    conflict_id = payload.get("conflict_id")
    if conflict_id is None:
        raise ValueError("conflict_id is required")

    run_coro(_upsert_resolution(conflict_id, "pending", None, None))

    resolver = PatternBasedResolution(
        int(payload.get("user_id")), int(payload.get("conversation_id"))
    )

    try:
        result = run_coro(resolver._evaluate_resolution(int(conflict_id)))
        run_coro(_upsert_resolution(conflict_id, "ready", result, None))
        return {
            "status": "ready",
            "conflict_id": int(conflict_id),
            "resolved": bool(result),
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "Failed to evaluate conflict resolution for conflict=%s", conflict_id
        )
        run_coro(_upsert_resolution(conflict_id, "failed", None, str(exc)))
        raise


async def _load_conflict(conflict_id: int) -> Optional[Dict[str, Any]]:
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM Conflicts WHERE id = $1",
            int(conflict_id),
        )
    return dict(row) if row else None


@shared_task(
    name="nyx.tasks.background.conflict_slice_tasks.evaluate_time_appropriateness",
    bind=True,
    max_retries=2,
    acks_late=True,
)
@with_retry
@idempotent(key_fn=_time_key)
def evaluate_time_appropriateness(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate if a conflict beat is appropriate for the given time of day."""

    conflict_id = payload.get("conflict_id")
    time_of_day = payload.get("time_of_day")
    if conflict_id is None or not time_of_day:
        raise ValueError("conflict_id and time_of_day are required")

    run_coro(_upsert_time(conflict_id, time_of_day, "pending", None, None))

    integration = ConflictDailyIntegration(
        int(payload.get("user_id")), int(payload.get("conversation_id"))
    )

    try:
        conflict_data = run_coro(_load_conflict(int(conflict_id)))
        if not conflict_data:
            conflict_data = payload.get("conflict_snapshot") or {}
        result = run_coro(
            integration._evaluate_time_appropriateness(conflict_data or {}, time_of_day)
        )
        run_coro(_upsert_time(conflict_id, time_of_day, "ready", bool(result), None))
        return {
            "status": "ready",
            "conflict_id": int(conflict_id),
            "time_of_day": time_of_day,
            "appropriate": bool(result),
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "Failed to evaluate time appropriateness for conflict=%s", conflict_id
        )
        run_coro(
            _upsert_time(conflict_id, time_of_day, "failed", None, str(exc))
        )
        raise


__all__ = [
    "refresh_tension_cache",
    "generate_conflict_activity",
    "evaluate_conflict_resolution",
    "evaluate_time_appropriateness",
]

