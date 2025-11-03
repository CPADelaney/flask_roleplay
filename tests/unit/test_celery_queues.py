"""Integration checks for Celery queue routing."""

import importlib
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nyx.tasks.celery_app import app
from nyx.tasks.queues import QUEUES as NYX_QUEUES


def _get_queue(route):
    """Extract the queue name from a route entry."""
    if isinstance(route, dict):
        return route.get("queue")
    return getattr(route, "queue", None)


def test_new_game_tasks_use_registered_queues():
    """Ensure new game flow tasks map to queues declared by Nyx."""
    known_queues = {queue.name for queue in NYX_QUEUES}
    assert known_queues, "Expected Nyx queue list to be populated"

    routes = app.conf.task_routes or {}
    tasks_to_verify = {
        "tasks.process_new_game_task",
        "tasks.create_npcs_task",
        "tasks.ensure_npc_pool_task",
        "tasks.background_chat_task_with_memory",
        "tasks.generate_lore_background_task",
        "tasks.generate_initial_conflict_task",
    }

    missing_routes = tasks_to_verify.difference(routes.keys())
    assert not missing_routes, f"Missing task route definitions: {sorted(missing_routes)}"

    for task_name in tasks_to_verify:
        queue = _get_queue(routes[task_name])
        assert queue in known_queues, f"{task_name} routed to unknown queue '{queue}'"


def test_background_chat_task_is_registered(monkeypatch):
    """Celery should load the legacy top-level tasks module."""
    assert "tasks" in app.conf.imports

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    try:
        from pydantic._internal import _generate_schema as _pydantic_generate_schema

        monkeypatch.setattr(
            _pydantic_generate_schema, "_SUPPORTS_TYPEDDICT", True, raising=False
        )
    except ImportError:  # pragma: no cover
        pass
    importlib.import_module("tasks")

    task_name = "tasks.background_chat_task_with_memory"
    assert task_name in app.tasks, f"{task_name} not registered with Celery"

    queue = _get_queue(app.conf.task_routes.get(task_name))
    assert (
        queue == "realtime"
    ), f"{task_name} expected to route to realtime queue, got {queue!r}"
