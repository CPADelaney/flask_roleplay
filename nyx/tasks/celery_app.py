"""Central Celery application for Nyx tasks."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from celery import Celery

from .beat import build_beat_schedule
from .queues import QUEUES, ROUTES as DEFAULT_ROUTES

_LOGGER = logging.getLogger(__name__)


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    return default


def _to_int(value: str | None, fallback: int, *, setting: str) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        _LOGGER.warning("Invalid integer for Celery setting %s: %r", setting, value)
        return fallback


def _load_task_routes() -> Dict[str, Dict[str, Any]]:
    routes: Dict[str, Dict[str, Any]] = dict(DEFAULT_ROUTES)

    legacy_routes: Dict[str, Dict[str, Any]] = {
        "nyx.tasks.background.conflict.*": {"queue": "nyx-conflict"},
        "nyx.tasks.background.npc.*": {"queue": "nyx-npc"},
        "nyx.tasks.background.lore.*": {"queue": "nyx-lore"},
    }

    routes.update(legacy_routes)

    raw = os.getenv("CELERY_TASK_ROUTES")
    if not raw:
        return routes

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        _LOGGER.warning("Failed to parse CELERY_TASK_ROUTES JSON: %s", exc)
        return routes

    if not isinstance(parsed, dict):
        _LOGGER.warning("CELERY_TASK_ROUTES must be a JSON object, got %r", type(parsed))
        return routes

    routes.update(parsed)
    return routes


app = Celery("nyx")

app.conf.update(
    broker_url=_get_env("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0")),
    result_backend=_get_env("CELERY_RESULT_BACKEND", "rpc://"),
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="realtime",
    task_queues=QUEUES,
    task_routes=_load_task_routes(),
    task_time_limit=_to_int(os.getenv("CELERY_TASK_TIME_LIMIT"), 120, setting="CELERY_TASK_TIME_LIMIT"),
    task_soft_time_limit=_to_int(os.getenv("CELERY_TASK_SOFT_LIMIT"), 90, setting="CELERY_TASK_SOFT_LIMIT"),
    imports=(
        "nyx.tasks.realtime.post_turn",
        "nyx.tasks.background.conflict_pipeline",
        "nyx.tasks.background.npc_tasks",
        "nyx.tasks.background.world_tasks",
        "nyx.subscribers.memory_subscribers",
        "nyx.subscribers.npc_subscribers",
        "nyx.subscribers.world_subscribers",
        "tasks",
        "memory.tasks.schema_tasks",
    ),
)

app.conf.beat_schedule = build_beat_schedule()

app.autodiscover_tasks(["nyx.tasks", "nyx.subscribers", "memory.tasks"])

try:  # pragma: no cover - optional legacy shim
    import celery_config  # noqa: F401  # side-effect import to merge legacy task routes
except ImportError:  # pragma: no cover - environments without shim
    pass

__all__ = ["app"]
