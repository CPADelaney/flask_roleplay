"""Celery queue configuration for Nyx async fan-out."""

from __future__ import annotations

from kombu import Exchange, Queue

EXCHANGE = Exchange("nyx", type="direct")

QUEUES = (
    Queue("realtime", EXCHANGE, routing_key="realtime"),
    Queue("background", EXCHANGE, routing_key="background"),
    Queue("heavy", EXCHANGE, routing_key="heavy"),
)

ROUTES = {
    "nyx.tasks.realtime.post_turn.dispatch": {"queue": "realtime", "routing_key": "realtime"},
    "nyx.tasks.realtime.post_turn.drain_outbox": {"queue": "realtime", "routing_key": "realtime"},
    "nyx.tasks.background.world_tasks.apply_universal": {"queue": "background", "routing_key": "background"},
    "nyx.tasks.background.conflict_tasks.process_events": {"queue": "background", "routing_key": "background"},
    "nyx.tasks.background.npc_tasks.run_adaptation_cycle": {"queue": "background", "routing_key": "background"},
    "nyx.tasks.background.lore_tasks.precompute_scene_bundle": {"queue": "background", "routing_key": "background"},
    "nyx.tasks.heavy.memory_tasks.add_and_embed": {"queue": "heavy", "routing_key": "heavy"},
    "nyx.tasks.heavy.memory_tasks.consolidate_decay": {"queue": "heavy", "routing_key": "heavy"},
}

__all__ = ["EXCHANGE", "QUEUES", "ROUTES"]
