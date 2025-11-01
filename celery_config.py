"""Compatibility shim exposing the shared Nyx Celery application."""

from __future__ import annotations

import logging
from typing import Any, Dict

from nyx.tasks.celery_app import app
from nyx.tasks.queues import ROUTES as NYX_TASK_ROUTES

logger = logging.getLogger(__name__)

celery_app = app


# Ensure legacy imports that expect routes to be registered via this module see
# the consolidated mapping.
existing_routes: Dict[str, Dict[str, Any]] = celery_app.conf.task_routes or {}
if existing_routes is celery_app.conf.task_routes:
    existing_routes = dict(existing_routes)
existing_routes.update(NYX_TASK_ROUTES)
celery_app.conf.task_routes = existing_routes


def get_queue_stats() -> Dict[str, Any]:
    """Return lightweight queue stats for compatibility callers."""

    try:
        inspector = celery_app.control.inspect()
    except Exception:  # pragma: no cover - optional broker connectivity
        logger.debug("Celery inspect unavailable", exc_info=True)
        return {}
    if not inspector:
        return {}
    stats: Dict[str, Any] = {}
    try:
        stats["active"] = inspector.active() or {}
    except Exception:  # pragma: no cover
        logger.debug("Failed to fetch active tasks", exc_info=True)
    try:
        stats["reserved"] = inspector.reserved() or {}
    except Exception:  # pragma: no cover
        logger.debug("Failed to fetch reserved tasks", exc_info=True)
    return stats


QUEUE_PRIORITIES = {
    route["queue"]: {"routing_key": route.get("routing_key", route["queue"])}
    for route in NYX_TASK_ROUTES.values()
}


__all__ = ["celery_app", "get_queue_stats", "QUEUE_PRIORITIES"]
