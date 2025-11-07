"""Lightweight Nyx Celery tasks."""

from .location_tasks import notify_canon_of_location_task

__all__ = [
    "notify_canon_of_location_task",
]
