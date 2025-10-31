import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
import typing
import typing_extensions
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment shim
    sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 12):  # pragma: no cover - compatibility shim
    typing.TypedDict = typing_extensions.TypedDict  # type: ignore[attr-defined]

try:  # pragma: no cover - optional metrics stub for broken dependency
    import monitoring.metrics  # type: ignore
except Exception:  # pragma: no cover - fallback stub for tests
    metrics_stub = types.ModuleType("monitoring.metrics")

    async def _noop(*_args, **_kwargs):
        return None

    metrics_stub.record_cache_operation = _noop  # type: ignore[attr-defined]
    monitoring_pkg = sys.modules.get("monitoring")
    if monitoring_pkg is None:
        monitoring_pkg = types.ModuleType("monitoring")
        sys.modules["monitoring"] = monitoring_pkg
    setattr(monitoring_pkg, "metrics", metrics_stub)
    sys.modules["monitoring.metrics"] = metrics_stub

from logic.conflict_system.enhanced_conflict_integration import (  # noqa: E402
    EnhancedIntegrationSubsystem,
)


@pytest.fixture
def anyio_backend():  # pragma: no cover - test backend shim
    return "asyncio"


@pytest.mark.anyio
async def test_get_scene_bundle_read_only_on_cache_miss(monkeypatch):
    subsystem = EnhancedIntegrationSubsystem(user_id=1, conversation_id=2)

    remember_calls = []
    monkeypatch.setattr(subsystem, "_remember_scope", lambda *args, **kwargs: remember_calls.append(args))

    schedule_called = False

    def fail_schedule(*_args, **_kwargs):
        nonlocal schedule_called
        schedule_called = True

    monkeypatch.setattr(subsystem, "_schedule_refresh", fail_schedule)

    cached_mock = AsyncMock(return_value={})
    monkeypatch.setattr(subsystem, "_get_cached_tension_summary", cached_mock)

    monkeypatch.setattr(
        "monitoring.metrics.record_cache_operation",
        AsyncMock(return_value=None),
    )

    scope = SimpleNamespace(
        location_id=11,
        npc_ids=[7, 8],
        topics=["market"],
        scene_type="bazaar",
        activity="browsing",
        current_activity=None,
    )

    bundle = await subsystem.get_scene_bundle(scope)

    assert bundle["summary_source"] == "pending"
    assert bundle["manifestations"] == []
    assert not schedule_called
    assert not remember_calls
    cached_mock.assert_awaited_once()
    assert cached_mock.await_args.kwargs["mutate_cache"] is False


@pytest.mark.anyio
async def test_get_scene_bundle_read_only_on_cache_hit(monkeypatch):
    subsystem = EnhancedIntegrationSubsystem(user_id=3, conversation_id=4)

    remember_calls = []
    monkeypatch.setattr(subsystem, "_remember_scope", lambda *args, **kwargs: remember_calls.append(args))

    schedule_called = False

    def fail_schedule(*_args, **_kwargs):
        nonlocal schedule_called
        schedule_called = True

    monkeypatch.setattr(subsystem, "_schedule_refresh", fail_schedule)

    cached_summary = {
        "manifestation": ["side-eye"],
        "should_generate_conflict": True,
        "suggested_type": "slice_of_life",
        "cached_at": "2024-05-22T10:00:00",
    }

    cached_mock = AsyncMock(return_value=cached_summary)
    monkeypatch.setattr(subsystem, "_get_cached_tension_summary", cached_mock)

    monkeypatch.setattr(
        "monitoring.metrics.record_cache_operation",
        AsyncMock(return_value=None),
    )

    scope = SimpleNamespace(
        location_id=21,
        npc_ids=[9],
        topics=["tension"],
        scene_type="dining",
        activity="meal",
        current_activity=None,
    )

    bundle = await subsystem.get_scene_bundle(scope)

    assert bundle["summary_source"] == "cached"
    assert bundle["manifestations"] == ["side-eye"]
    assert bundle["opportunities"]
    assert not schedule_called
    assert not remember_calls
    cached_mock.assert_awaited_once()
    assert cached_mock.await_args.kwargs["mutate_cache"] is False


@pytest.mark.anyio
async def test_register_scene_context_schedules_refresh(monkeypatch):
    subsystem = EnhancedIntegrationSubsystem(user_id=5, conversation_id=6)

    cached_mock = AsyncMock(return_value={})
    monkeypatch.setattr(subsystem, "_get_cached_tension_summary", cached_mock)

    scheduled = {}

    def record_schedule(contexts, reason):
        scheduled["contexts"] = contexts
        scheduled["reason"] = reason

    monkeypatch.setattr(subsystem, "_schedule_refresh", record_schedule)

    scene_context = {
        "location": 31,
        "present_npcs": [1, 2],
        "topics": ["harvest"],
        "scene_type": "festival",
    }

    await subsystem._register_scene_context_for_refresh(scene_context, reason="test_event")

    assert scheduled["reason"] == "test_event"
    assert scheduled["contexts"]
    cached_mock.assert_awaited()


@pytest.mark.anyio
async def test_register_scene_context_skips_refresh_when_cached(monkeypatch):
    subsystem = EnhancedIntegrationSubsystem(user_id=7, conversation_id=8)

    cached_mock = AsyncMock(return_value={"cached": True})
    monkeypatch.setattr(subsystem, "_get_cached_tension_summary", cached_mock)

    schedule_mock = AsyncMock()
    monkeypatch.setattr(subsystem, "_schedule_refresh", schedule_mock)

    scene_context = {
        "location": 41,
        "present_npcs": [3],
        "topics": [],
        "scene_type": "library",
    }

    await subsystem._register_scene_context_for_refresh(scene_context, reason="test_event")

    schedule_mock.assert_not_called()
    cached_mock.assert_awaited()
