import asyncio
import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
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


def test_state_sync_dispatches_background_task(monkeypatch):
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

    queued_payloads: List[Dict[str, Any]] = []

    def fake_queue_manifestation_generation(**kwargs):
        queued_payloads.append(kwargs)
        return "hash123"

    def fake_get_tension_bundle(user_id: int, conversation_id: int, scene_hash: str) -> Dict[str, Any]:
        return {"status": "generating"}

    hotpath_module = importlib.import_module("logic.conflict_system.tension_hotpath")
    monkeypatch.setattr(
        hotpath_module,
        "queue_manifestation_generation",
        fake_queue_manifestation_generation,
    )
    monkeypatch.setattr(
        hotpath_module,
        "get_tension_bundle",
        fake_get_tension_bundle,
    )

    subsystem = tension_module.TensionSubsystem(user_id=1, conversation_id=42)
    subsystem._current_tensions = {tension_module.TensionType.EMOTIONAL: 0.5}

    event = _StubSystemEvent(
        event_id="evt-123",
        event_type=_StubEventType.STATE_SYNC,
        source_subsystem=_StubSubsystemType.TENSION,
        payload={
            "scene_context": {
                "location": "lounge",
                "npcs": [1, 2],
            }
        },
    )

    async def invoke():
        return await subsystem._on_state_sync_non_blocking(event)

    response = asyncio.run(invoke())

    assert queued_payloads, "queue_manifestation_generation should be invoked"
    queued = queued_payloads[0]
    assert queued["user_id"] == 1
    assert queued["conversation_id"] == 42
    assert queued["scene_context"]["npcs"] == [1, 2]
    assert queued["dominant_type"] == "emotional"
    assert response.success is True
    assert response.data["status"] == "generating"


