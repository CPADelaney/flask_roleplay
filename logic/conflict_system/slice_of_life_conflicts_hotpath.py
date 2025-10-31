"""Hot-path helpers for slice-of-life conflict subsystem.

These helpers provide cache-first accessors that avoid blocking LLM calls on the
request path. They read from lightweight cache tables and, on misses, enqueue
background Celery tasks that recompute the slow outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from asyncpg.exceptions import UndefinedTableError

from db.connection import get_db_connection_context
from logic.conflict_system.slice_of_life_conflicts import (
    ConflictIntensity,
    DailyConflictEvent,
    ResolutionApproach,
    SliceOfLifeConflictType,
)

logger = logging.getLogger(__name__)


TENSION_CACHE_TTL = timedelta(minutes=15)
ACTIVITY_CACHE_TTL = timedelta(minutes=30)
RESOLUTION_CACHE_TTL = timedelta(minutes=45)
TIME_CACHE_TTL = timedelta(minutes=20)
PENDING_GRACE = timedelta(minutes=5)


def _now() -> datetime:
    return datetime.utcnow()


def _should_refresh(updated_at: Optional[datetime], ttl: timedelta) -> bool:
    if not updated_at:
        return True
    return updated_at < _now() - ttl


def _pending_expired(updated_at: Optional[datetime]) -> bool:
    if not updated_at:
        return True
    return updated_at < _now() - PENDING_GRACE


def _queue_task(task_name: str, payload: Dict[str, Any]) -> None:
    try:
        module = __import__(
            "nyx.tasks.background.conflict_slice_tasks",
            fromlist=[task_name],
        )
        task = getattr(module, task_name)
        task.delay(payload)
        logger.debug("Queued %s with payload %s", task_name, payload)
    except Exception as exc:  # pragma: no cover - best-effort dispatch
        logger.warning("Failed to queue %s: %s", task_name, exc)


def _activity_hash(conflict_id: int, activity_type: str, npc_ids: Iterable[int]) -> str:
    serialized = json.dumps(
        {
            "conflict_id": int(conflict_id),
            "activity": activity_type,
            "npcs": sorted(int(n) for n in npc_ids),
        },
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _coerce_tension_item(item: Dict[str, Any]) -> Dict[str, Any]:
    tension = dict(item)
    ctype = str(tension.get("type", "subtle_rivalry"))
    intensity = str(tension.get("intensity", "tension"))
    try:
        tension["type"] = SliceOfLifeConflictType[ctype.upper()]
    except Exception:
        tension["type"] = SliceOfLifeConflictType.SUBTLE_RIVALRY
    try:
        tension["intensity"] = ConflictIntensity[intensity.upper()]
    except Exception:
        tension["intensity"] = ConflictIntensity.TENSION
    return tension


def _deserialize_event(payload: Dict[str, Any], fallback_activity: str) -> DailyConflictEvent:
    try:
        return DailyConflictEvent(
            activity_type=str(payload.get("activity_type", fallback_activity)),
            conflict_manifestation=str(
                payload.get(
                    "conflict_manifestation",
                    "A gentle ripple of tension threads through the routine",
                )
            ),
            choice_presented=bool(payload.get("choice_presented", False)),
            accumulation_impact=float(payload.get("accumulation_impact", 0.1)),
            npc_reactions={
                int(k): str(v)
                for k, v in dict(payload.get("npc_reactions", {})).items()
            },
        )
    except Exception:
        return DailyConflictEvent(
            activity_type=fallback_activity,
            conflict_manifestation="Tension simmers quietly in the background",
            choice_presented=False,
            accumulation_impact=0.05,
            npc_reactions={},
        )


async def get_detected_tensions(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
    """Fetch cached tensions and dispatch regeneration when stale."""

    async with get_db_connection_context() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT status, payload, updated_at
                  FROM slice_conflict_tension_cache
                 WHERE user_id = $1 AND conversation_id = $2
                """,
                int(user_id),
                int(conversation_id),
            )
        except UndefinedTableError:
            row = None

    payload: List[Dict[str, Any]] = []
    status = None
    updated_at: Optional[datetime] = None

    if row:
        status = row.get("status")
        updated_at = row.get("updated_at")
        payload = list(row.get("payload") or [])

    needs_refresh = (status != "pending" and _should_refresh(updated_at, TENSION_CACHE_TTL)) or (
        status == "pending" and _pending_expired(updated_at)
    )

    if needs_refresh:
        _queue_task(
            "refresh_tension_cache",
            {"user_id": int(user_id), "conversation_id": int(conversation_id)},
        )

    if status == "ready" and payload:
        return [_coerce_tension_item(item) for item in payload if isinstance(item, dict)]

    return []


async def get_activity_manifestation(
    user_id: int,
    conversation_id: int,
    conflict_id: int,
    activity_type: str,
    participating_npcs: List[int],
) -> DailyConflictEvent:
    """Return cached manifestation for a conflict activity or dispatch background job."""

    npc_ids = [int(n) for n in participating_npcs]
    activity_key = _activity_hash(conflict_id, activity_type, npc_ids)

    async with get_db_connection_context() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT status, payload, updated_at
                  FROM slice_conflict_activity_cache
                 WHERE conflict_id = $1 AND activity_hash = $2
                """,
                int(conflict_id),
                activity_key,
            )
        except UndefinedTableError:
            row = None

    status = row.get("status") if row else None
    updated_at = row.get("updated_at") if row else None
    payload = row.get("payload") if row else None

    needs_refresh = (status != "pending" and _should_refresh(updated_at, ACTIVITY_CACHE_TTL)) or (
        status == "pending" and _pending_expired(updated_at)
    )

    if needs_refresh:
        _queue_task(
            "generate_conflict_activity",
            {
                "user_id": int(user_id),
                "conversation_id": int(conversation_id),
                "conflict_id": int(conflict_id),
                "activity_type": activity_type,
                "participating_npcs": npc_ids,
                "activity_hash": activity_key,
            },
        )

    if status == "ready" and isinstance(payload, dict):
        return _deserialize_event(payload, activity_type)

    # Deterministic fallback while background job runs
    npc_display = ", ".join(str(npc_id) for npc_id in npc_ids[:2])
    return DailyConflictEvent(
        activity_type=activity_type,
        conflict_manifestation=(
            f"Routine tension lingers during {activity_type}"
            + (f" with {npc_display}" if npc_display else "")
        ),
        choice_presented=False,
        accumulation_impact=0.05,
        npc_reactions={},
    )


async def get_resolution_recommendation(
    user_id: int, conversation_id: int, conflict_id: int
) -> Optional[Dict[str, Any]]:
    """Return cached resolution recommendation for a conflict."""

    async with get_db_connection_context() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT status, payload, updated_at
                  FROM slice_conflict_resolution_cache
                 WHERE conflict_id = $1
                """,
                int(conflict_id),
            )
        except UndefinedTableError:
            row = None

    status = row.get("status") if row else None
    updated_at = row.get("updated_at") if row else None
    payload = row.get("payload") if row else None

    needs_refresh = (status != "pending" and _should_refresh(updated_at, RESOLUTION_CACHE_TTL)) or (
        status == "pending" and _pending_expired(updated_at)
    )

    if needs_refresh:
        _queue_task(
            "evaluate_conflict_resolution",
            {
                "user_id": int(user_id),
                "conversation_id": int(conversation_id),
                "conflict_id": int(conflict_id),
            },
        )

    if status == "ready" and isinstance(payload, dict):
        result = dict(payload)
        rtype = str(result.get("resolution_type", "time_erosion"))
        try:
            result["resolution_type"] = ResolutionApproach[rtype.upper()]
        except Exception:
            result["resolution_type"] = ResolutionApproach.TIME_EROSION
        return result

    return None


async def is_conflict_appropriate_for_time(
    user_id: int,
    conversation_id: int,
    conflict: Dict[str, Any],
    time_of_day: str,
) -> bool:
    """Return cached time-of-day suitability or queue evaluation task."""

    conflict_id = int(conflict.get("id") or conflict.get("conflict_id") or 0)
    async with get_db_connection_context() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT status, result, updated_at
                  FROM slice_conflict_time_cache
                 WHERE conflict_id = $1 AND time_of_day = $2
                """,
                conflict_id,
                time_of_day,
            )
        except UndefinedTableError:
            row = None

    status = row.get("status") if row else None
    updated_at = row.get("updated_at") if row else None
    result_value = row.get("result") if row else None

    needs_refresh = (status != "pending" and _should_refresh(updated_at, TIME_CACHE_TTL)) or (
        status == "pending" and _pending_expired(updated_at)
    )

    if needs_refresh and conflict_id:
        _queue_task(
            "evaluate_time_appropriateness",
            {
                "user_id": int(user_id),
                "conversation_id": int(conversation_id),
                "conflict_id": conflict_id,
                "time_of_day": time_of_day,
                "conflict_snapshot": {
                    "conflict_type": conflict.get("conflict_type"),
                    "intensity": conflict.get("intensity"),
                    "phase": conflict.get("phase"),
                },
            },
        )

    if status == "ready" and isinstance(result_value, bool):
        return result_value

    # Deterministic fallback heuristic
    intensity = str(conflict.get("intensity", "tension")).lower()
    if intensity in {"confrontation", "direct"}:
        return time_of_day.lower() not in {"late_night", "midnight"}
    if intensity == "subtext":
        return True
    return time_of_day.lower() not in {"sleep", "rest"}


__all__ = [
    "get_detected_tensions",
    "get_activity_manifestation",
    "get_resolution_recommendation",
    "is_conflict_appropriate_for_time",
]

