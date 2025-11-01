"""Celery beat schedule factory for Nyx."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict

from celery.schedules import crontab, schedule

LOGGER = logging.getLogger(__name__)


def build_beat_schedule() -> Dict[str, Dict[str, Any]]:
    """Build the Celery beat schedule for Nyx tasks."""

    beat: Dict[str, Dict[str, Any]] = {
        "nyx-outbox-dispatcher": {
            "task": "nyx.tasks.maintenance.dispatch_outbox",
            "schedule": schedule(run_every=5.0),
        },
        "nyx-memory-consolidation": {
            "task": "nyx.tasks.heavy.memory_tasks.consolidate_decay",
            "schedule": timedelta(hours=1),
        },
    }

    try:
        import nyx.tasks.background.npc_tasks  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.debug("NPC periodic task not available: %s", exc)
    else:
        beat["nyx-npc-sweep"] = {
            "task": "nyx.tasks.background.npc_tasks.run_adaptation_cycle",
            "schedule": timedelta(minutes=5),
            "args": (
                {
                    "turn_id": "beat",
                    "npcs": [],
                    "payload": {"periodic": True},
                },
            ),
        }

    beat.update(
        {
            "nyx-lore-refresh": {
                "task": "nyx.tasks.background.lore_tasks.precompute_scene_bundle",
                "schedule": crontab(hour="*/4"),
                "args": (
                    {
                        "turn_id": "beat",
                        "scene_id": None,
                        "region_id": None,
                        "payload": {"periodic": True},
                    },
                ),
            },
            "nyx-conflict-template-warmup": {
                "task": "nyx.tasks.background.conflict_template_tasks.warm_template_cache",
                "schedule": crontab(minute="*/15"),
                "args": (25,),
            },
        }
    )

    return beat


__all__ = ["build_beat_schedule"]
