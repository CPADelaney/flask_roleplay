import asyncio
import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _StubSubsystemType(Enum):
    TENSION = "tension"


class _StubEventType(Enum):
    STATE_SYNC = "state_sync"
    CONFLICT_CREATED = "conflict_created"
    TENSION_CHANGED = "tension_changed"


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


def _load_tension_module(monkeypatch: pytest.MonkeyPatch) -> Tuple[Any, List[Dict[str, Any]]]:
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

        async def execute(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    stub_db_connection.get_db_connection_context = lambda: _DummyAsyncContext()
    stub_db.connection = stub_db_connection
    monkeypatch.setitem(sys.modules, "db", stub_db)
    monkeypatch.setitem(sys.modules, "db.connection", stub_db_connection)

    tension_path = PROJECT_ROOT / "logic/conflict_system/tension.py"
    assert tension_path.exists()
    module_name = "logic.conflict_system.tension"
    sys.modules.pop(module_name, None)
    tension_spec = importlib.util.spec_from_file_location(module_name, tension_path)
    assert tension_spec and tension_spec.loader is not None
    tension_module = importlib.util.module_from_spec(tension_spec)
    tension_module.__package__ = "logic.conflict_system"
    monkeypatch.setitem(sys.modules, module_name, tension_module)
    tension_spec.loader.exec_module(tension_module)

    def fake_orch():
        return (
            _StubSubsystemType,
            _StubEventType,
            _StubSystemEvent,
            _StubSubsystemResponse,
        )

    monkeypatch.setattr(tension_module, "_orch", fake_orch)

    hotpath_module = importlib.import_module("logic.conflict_system.tension_hotpath")
    queued_payloads: List[Dict[str, Any]] = []

    def fake_queue_manifestation_generation(**kwargs):
        queued_payloads.append(kwargs)
        return "hash123"

    def fake_get_tension_bundle(user_id: int, conversation_id: int, scene_hash: str) -> Dict[str, Any]:
        return {"status": "generating"}

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

    return tension_module, queued_payloads


def test_state_sync_skips_manifestation_when_warmup_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENSION_EAGER_WARMUP", raising=False)
    tension_module, queued_payloads = _load_tension_module(monkeypatch)

    subsystem = tension_module.TensionSubsystem(user_id=1, conversation_id=42)
    subsystem._current_tensions = {tension_module.TensionType.EMOTIONAL: 0.5}

    event = _StubSystemEvent(
        event_id="evt-123",
        event_type=_StubEventType.STATE_SYNC,
        source_subsystem=_StubSubsystemType.TENSION,
        payload={"scene_context": {"location": "lounge", "npcs": [1, 2]}},
    )

    async def invoke():
        return await subsystem._on_state_sync_non_blocking(event)

    response = asyncio.run(invoke())

    assert response.success is True
    assert response.data["status"] == "generating"
    assert queued_payloads == []


def test_state_sync_honors_eager_warmup_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENSION_EAGER_WARMUP", "1")
    tension_module, queued_payloads = _load_tension_module(monkeypatch)

    subsystem = tension_module.TensionSubsystem(user_id=1, conversation_id=42)
    subsystem._current_tensions = {tension_module.TensionType.EMOTIONAL: 0.5}

    event = _StubSystemEvent(
        event_id="evt-456",
        event_type=_StubEventType.STATE_SYNC,
        source_subsystem=_StubSubsystemType.TENSION,
        payload={"scene_context": {"location": "lounge", "npcs": [3]}},
    )

    async def invoke():
        return await subsystem._on_state_sync_non_blocking(event)

    asyncio.run(invoke())

    assert queued_payloads, "Warm-up should enqueue when toggle is enabled"
    queued = queued_payloads[0]
    assert queued["user_id"] == 1
    assert queued["conversation_id"] == 42
    assert queued["scene_context"]["npcs"] == [3]


def test_conflict_created_triggers_manifestation_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENSION_EAGER_WARMUP", raising=False)
    tension_module, queued_payloads = _load_tension_module(monkeypatch)

    subsystem = tension_module.TensionSubsystem(user_id=7, conversation_id=99)

    event = _StubSystemEvent(
        event_id="evt-conflict",
        event_type=_StubEventType.CONFLICT_CREATED,
        source_subsystem=_StubSubsystemType.TENSION,
        payload={
            "conflict_type": "power_struggle",
            "context": {"location": "study", "npcs": [5, 6]},
        },
    )

    async def invoke():
        return await subsystem._on_conflict_created(event)

    asyncio.run(invoke())

    assert queued_payloads, "Conflict creation should eagerly queue manifestation generation"
    queued = queued_payloads[0]
    assert queued["user_id"] == 7
    assert queued["conversation_id"] == 99
    assert queued["scene_context"]["location"] == "study"


