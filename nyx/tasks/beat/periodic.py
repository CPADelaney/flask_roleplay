"""Celery beat schedule helpers for Nyx tasks."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Any

from celery.schedules import crontab

BEAT_SCHEDULE: Dict[str, Dict[str, Any]] = {
    "nyx-memory-consolidation": {
        "task": "nyx.tasks.heavy.memory_tasks.consolidate_decay",
        "schedule": timedelta(minutes=10),
    },
    "nyx-npc-global-adaptation": {
        "task": "nyx.tasks.background.npc_tasks.run_adaptation_cycle",
        "schedule": crontab(minute="*/30"),
        "args": ({"turn_id": "beat", "npcs": [], "payload": {"periodic": True}},),
    },
    "nyx-lore-refresh": {
        "task": "nyx.tasks.background.lore_tasks.precompute_scene_bundle",
        "schedule": crontab(hour="*/4"),
        "args": ({"turn_id": "beat", "scene_id": None, "region_id": None, "payload": {"periodic": True}},),
    },
    "nyx-conflict-template-warmup": {
        "task": "nyx.tasks.background.conflict_template_tasks.warm_template_cache",
        "schedule": crontab(minute="*/15"),
        "args": (25,),
    },
}

__all__ = ["BEAT_SCHEDULE"]
