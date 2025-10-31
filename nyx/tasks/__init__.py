"""Nyx Celery task package.

This package is intentionally lightweight; importing it registers the
post-turn dispatcher along with background and heavy-weight handlers.
The actual task functions live in submodules which can be imported on
Demand by Celery when autodiscovering tasks.
"""

from __future__ import annotations

# Import submodules for side effects so Celery discovers tasks when this
# package is imported from the worker entrypoint.
# We keep imports local and tolerant of errors so that environments
# missing optional dependencies do not fail during import.

try:  # pragma: no cover - defensive import
    from .realtime import post_turn  # noqa: F401
except Exception:  # pragma: no cover
    # Log lazily to avoid configuring logging during import.  Workers will
    # surface the failure when the task is first used.
    import logging

    logging.getLogger(__name__).exception("Failed to import realtime.post_turn")

for _module in ("world_tasks", "conflict_tasks", "conflict_template_tasks", "npc_tasks", "lore_tasks"):
    try:  # pragma: no cover - defensive import
        __import__(f"nyx.tasks.background.{_module}")
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to import background task %s", _module)

for _module in ("memory_tasks",):
    try:  # pragma: no cover - defensive import
        __import__(f"nyx.tasks.heavy.{_module}")
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to import heavy task %s", _module)

try:  # pragma: no cover
    __import__("nyx.tasks.beat.periodic")
except Exception:
    import logging

    logging.getLogger(__name__).exception("Failed to import beat.periodic")

__all__ = [
    "post_turn",
]
