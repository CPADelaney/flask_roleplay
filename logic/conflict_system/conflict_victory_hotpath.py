"""Utilities for caching and deterministic fallbacks in the victory subsystem."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from celery.result import AsyncResult

from celery_config import celery_app
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


class TaskKey(str, Enum):
    """Keys used inside the task metadata blob on victory condition rows."""

    GENERATOR = "generator"
    SUMMARY = "summary"
    CONSEQUENCES = "consequences"
    CONSOLATION = "consolation"
    EPILOGUE = "epilogue"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def get_condition_metadata(condition: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the metadata dictionary stored on a victory condition row."""

    raw = condition.get("task_metadata") if isinstance(condition, Mapping) else None
    return _coerce_json(raw)


def get_task_entry(metadata: Mapping[str, Any], key: TaskKey) -> Dict[str, Any]:
    entry = metadata.get(key.value)
    if isinstance(entry, dict):
        return dict(entry)
    return {}


def resolved_result(entry: Mapping[str, Any]) -> Any:
    if entry.get("status") == "ready":
        return entry.get("result")
    return None


def fallback_result(entry: Mapping[str, Any]) -> Any:
    if entry.get("result") is not None:
        return entry.get("result")
    return None


def should_queue_task(entry: Mapping[str, Any]) -> bool:
    if entry.get("status") == "ready":
        return False
    if entry.get("task_id"):
        # Already queued or running
        return False
    return True


def build_entry(status: str, *, task_id: Optional[str], result: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"status": status, "updated_at": _now_iso()}
    if task_id:
        entry["task_id"] = task_id
    if result is not None:
        entry["result"] = result
    return entry


def enqueue_task(task_name: str, payload: Dict[str, Any], queue: str = "background") -> Optional[str]:
    try:
        async_result: AsyncResult = celery_app.send_task(task_name, kwargs=payload, queue=queue)
    except Exception:
        logger.exception("Failed to enqueue task %s", task_name)
        return None
    return async_result.id


async def write_condition_metadata(condition_id: int, key: TaskKey, entry: Dict[str, Any]) -> None:
    async with get_db_connection_context() as conn:
        try:
            await conn.execute(
                """
                UPDATE victory_conditions
                   SET task_metadata = jsonb_set(
                           COALESCE(task_metadata, '{}'::jsonb),
                           $2,
                           $3::jsonb,
                           true
                       ),
                       updated_at = CURRENT_TIMESTAMP
                 WHERE condition_id = $1
                """,
                int(condition_id),
                [key.value],
                json.dumps(entry),
            )
        except Exception:
            logger.exception(
                "Failed to persist task metadata for condition_id=%s", condition_id
            )


async def write_many_condition_metadata(
    updates: Sequence[Tuple[int, Dict[str, Any]]], key: TaskKey
) -> None:
    if not updates:
        return
    serialized = [
        (int(condition_id), [key.value], json.dumps(entry))
        for condition_id, entry in updates
    ]
    async with get_db_connection_context() as conn:
        try:
            await conn.executemany(
                """
                UPDATE victory_conditions
                   SET task_metadata = jsonb_set(
                           COALESCE(task_metadata, '{}'::jsonb),
                           $2,
                           $3::jsonb,
                           true
                       ),
                       updated_at = CURRENT_TIMESTAMP
                 WHERE condition_id = $1
                """,
                serialized,
            )
        except Exception:
            logger.exception(
                "Failed to update task metadata for %s conditions", len(serialized)
            )


def fallback_victory_conditions(conflict_type: str, stakeholder: Mapping[str, Any]) -> List[Dict[str, Any]]:
    role = str(stakeholder.get("role") or "participant").lower()
    involvement = str(stakeholder.get("involvement") or "primary").lower()
    stakeholder_id = stakeholder.get("id")
    description = (
        f"Secure a {conflict_type} breakthrough that affirms the {role} role "
        f"for stakeholder {stakeholder_id}."
    )
    if involvement != "primary":
        description = (
            f"Create leverage in the {conflict_type} dispute so the {role} "
            f"aligned with stakeholder {stakeholder_id} can dictate terms."
        )

    requirements = {
        "specific": {
            "momentum": 1.0,
            "influence": 0.75 if involvement == "support" else 1.0,
        },
        "narrative": {
            "milestone": f"Showcase a defining moment for stakeholder {stakeholder_id}"
        },
    }

    impact = {
        "relationship": 0.2,
        "power": 0.3 if involvement == "primary" else 0.15,
        "satisfaction": 0.25,
    }

    return [
        {
            "victory_type": "narrative",
            "description": description,
            "requirements": requirements,
            "impact": impact,
        }
    ]


def fallback_achievement_summary(
    condition: Mapping[str, Any], current_state: Mapping[str, Any]
) -> str:
    victory_type = str(condition.get("victory_type") or "narrative").replace("_", " ").title()
    description = str(condition.get("description") or "fulfils their ambition")
    stakeholder_id = condition.get("stakeholder_id")
    momentum = current_state.get("momentum")
    momentum_clause = ""
    if isinstance(momentum, (int, float)):
        momentum_clause = f" Momentum now stands at {float(momentum):.0%}."
    return (
        f"Stakeholder {stakeholder_id} claims a {victory_type} victory: {description}.{momentum_clause}"
    ).strip()


def fallback_victory_consequences(
    condition: Mapping[str, Any], current_state: Mapping[str, Any]
) -> Dict[str, Any]:
    stakeholder_id = condition.get("stakeholder_id")
    victory_type = str(condition.get("victory_type") or "narrative").replace("_", " ")
    conflict_pulse = current_state.get("conflict_pulse")
    immediate = (
        f"Stakeholder {stakeholder_id} consolidates influence as the conflict shifts toward a {victory_type} outcome."
    )
    if conflict_pulse:
        immediate += f" The conflict pulse settles around {conflict_pulse}."

    long_term = (
        "Allies recalibrate around the new balance, while rivals quietly search for counter-moves."
    )

    hidden = [
        f"Long dormant grudges stir in the wake of the {victory_type} triumph."
    ]

    return {
        "immediate": {"summary": immediate},
        "long_term": {"summary": long_term},
        "hidden_consequences": hidden,
    }


def fallback_conflict_epilogue(
    conflict: Mapping[str, Any],
    achievements: Sequence[Mapping[str, Any]],
    resolution_data: Mapping[str, Any],
) -> str:
    conflict_name = conflict.get("conflict_name") or "The conflict"
    conflict_type = conflict.get("conflict_type") or "struggle"
    achieved = len(achievements)
    resolution = resolution_data.get("resolution_type") or "victory"
    summary_bits = [
        f"{conflict_name} closes with a {resolution} rooted in {conflict_type} tensions.",
        f"{achieved} victory condition{'s' if achieved != 1 else ''} shape the aftermath.",
    ]
    return " ".join(summary_bits)


def fallback_consolation(condition: Mapping[str, Any]) -> str:
    victory_type = str(condition.get("victory_type") or "narrative").replace("_", " ")
    progress = float(condition.get("progress") or 0.0)
    description = str(condition.get("description") or "the objective")
    pct = int(progress * 100)
    return (
        f"The {victory_type} path nearly crystallizes—{description} reaches {pct}% completion,"
        " leaving allies inspired to try again."
    )


async def mark_task_ready(condition_ids: Iterable[int], key: TaskKey, result: Any, task_id: str) -> None:
    updates = [
        (int(condition_id), build_entry("ready", task_id=task_id, result=result))
        for condition_id in condition_ids
    ]
    await write_many_condition_metadata(updates, key)


__all__ = [
    "TaskKey",
    "build_entry",
    "enqueue_task",
    "fallback_achievement_summary",
    "fallback_conflict_epilogue",
    "fallback_consolation",
    "fallback_result",
    "fallback_victory_conditions",
    "fallback_victory_consequences",
    "get_condition_metadata",
    "get_task_entry",
    "mark_task_ready",
    "resolved_result",
    "should_queue_task",
    "write_condition_metadata",
    "write_many_condition_metadata",
]
