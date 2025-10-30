import asyncio
import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyRedisClient:
    def __init__(self):
        self._locks = set()

    async def get(self, key: str) -> Optional[str]:
        return None

    async def set(self, key: str, value: str, ex: int | None = None, nx: bool = False):
        if nx:
            if key in self._locks:
                return False
            self._locks.add(key)
            return True
        return True


class DummyCeleryApp:
    def __init__(self):
        self.sent_tasks: List[tuple[str, List[Any], Optional[Dict[str, Any]]]] = []

    def send_task(self, name: str, args: List[Any] | None = None, kwargs: Dict[str, Any] | None = None):
        self.sent_tasks.append((name, args, kwargs))


class _StubSubsystemType(Enum):
    TENSION = "tension"


class _StubEventType(Enum):
    STATE_SYNC = "state_sync"


@dataclass
class _StubSystemEvent:
    event_id: str
    event_type: _StubEventType
    source_subsystem: _StubSubsystemType
    payload: Dict[str, Any]
    timestamp: Any = None
    target_subsystems: Any = None
    requires_response: bool = False
    priority: int = 5


@dataclass
class _StubSubsystemResponse:
    subsystem: _StubSubsystemType
    event_id: str
    success: bool
    data: Dict[str, Any]
    side_effects: List[Any] = field(default_factory=list)


def test_state_sync_dispatches_celery_task(monkeypatch):
    stub_dynamic_template = types.ModuleType(
        "logic.conflict_system.dynamic_conflict_template"
    )
    stub_dynamic_template.extract_runner_response = lambda *args, **kwargs: {}
    stub_dynamic_template.__spec__ = ModuleSpec(
        name="logic.conflict_system.dynamic_conflict_template",
        loader=None,
    )
    import logic.conflict_system  # noqa: F401
    monkeypatch.setitem(
        sys.modules,
        "logic.conflict_system.dynamic_conflict_template",
        stub_dynamic_template,
    )

    stub_agents = types.ModuleType("agents")
    stub_agents.Agent = object
    stub_agents.Runner = object
    stub_agents.RunContextWrapper = object

    def passthrough(func=None, **_kwargs):
        if func is None:
            def decorator(inner):
                return inner
            return decorator
        return func

    stub_agents.function_tool = passthrough
    monkeypatch.setitem(sys.modules, "agents", stub_agents)

    stub_db = types.ModuleType("db")
    stub_db_connection = types.ModuleType("db.connection")

    class _DummyAsyncContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    stub_db_connection.get_db_connection_context = lambda: _DummyAsyncContext()
    stub_db.connection = stub_db_connection
    monkeypatch.setitem(sys.modules, "db", stub_db)
    monkeypatch.setitem(sys.modules, "db.connection", stub_db_connection)

    stub_celery_config = types.ModuleType("celery_config")
    stub_celery_config.celery_app = DummyCeleryApp()
    monkeypatch.setitem(sys.modules, "celery_config", stub_celery_config)

    tension_path = PROJECT_ROOT / "logic/conflict_system/tension.py"
    assert tension_path.exists()
    module_name = "logic.conflict_system.tension"
    tension_spec = importlib.util.spec_from_file_location(
        module_name,
        tension_path,
    )
    assert tension_spec and tension_spec.loader is not None
    tension_module = importlib.util.module_from_spec(tension_spec)
    tension_module.__package__ = "logic.conflict_system"
    sys.modules[module_name] = tension_module
    tension_spec.loader.exec_module(tension_module)

    def fake_orch():
        return (
            _StubSubsystemType,
            _StubEventType,
            _StubSystemEvent,
            _StubSubsystemResponse,
        )

    monkeypatch.setattr(tension_module, "_orch", fake_orch)

    subsystem = tension_module.TensionSubsystem(user_id=1, conversation_id=42)
    subsystem._current_tensions = {tension_module.TensionType.EMOTIONAL: 0.5}

    dummy_redis = DummyRedisClient()

    async def fake_get_redis_client():
        return dummy_redis

    monkeypatch.setattr(tension_module, "get_redis_client", fake_get_redis_client)

    dummy_celery = DummyCeleryApp()
    monkeypatch.setattr(tension_module, "celery_app", dummy_celery)

    event = _StubSystemEvent(
        event_id="evt-123",
        event_type=_StubEventType.STATE_SYNC,
        source_subsystem=_StubSubsystemType.TENSION,
        payload={
            "scene_context": {
                "location_id": 5,
                "npcs": [1, 2],
            }
        },
    )

    async def invoke():
        return await subsystem._on_state_sync_non_blocking(event)

    response = asyncio.run(invoke())

    assert dummy_celery.sent_tasks == [
        (
            "tasks.update_tension_bundle_cache",
            [1, 42, {"location_id": 5, "npcs": [1, 2]}],
            None,
        )
    ]
    assert response.success is True
    assert response.data["status"] == "generation_in_progress"
