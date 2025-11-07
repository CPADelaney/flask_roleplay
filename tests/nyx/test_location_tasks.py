from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from nyx.tasks.light import location_tasks


class _DummyConnectionContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - no cleanup
        return None


class _DummyMemoryOrchestrator:
    def __init__(self, calls: Dict[str, Any]):
        self._calls = calls

    async def store_memory(self, **kwargs: Any) -> None:
        self._calls["store_memory"] = kwargs

    async def add_to_vector_store(self, **kwargs: Any) -> None:
        self._calls["add_to_vector_store"] = kwargs


def test_notify_canon_of_location_task_accepts_trace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    async def fake_log_canonical_event(*args: Any, **kwargs: Any) -> None:
        calls["log_canonical_event"] = {
            "args": args,
            "kwargs": kwargs,
        }

    async def fake_get_canon_memory_orchestrator(user_id: int, conversation_id: int):
        calls["memory_orchestrator_args"] = (user_id, conversation_id)
        return _DummyMemoryOrchestrator(calls)

    monkeypatch.setattr(location_tasks, "get_db_connection_context", lambda: _DummyConnectionContext())
    monkeypatch.setattr(location_tasks, "log_canonical_event", fake_log_canonical_event)
    monkeypatch.setattr(location_tasks, "get_canon_memory_orchestrator", fake_get_canon_memory_orchestrator)
    monkeypatch.setattr(location_tasks, "run_coro", lambda coro: asyncio.run(coro))

    monkeypatch.setitem(location_tasks.app.conf, "task_always_eager", True)
    monkeypatch.setitem(location_tasks.app.conf, "task_eager_propagates", True)

    result = location_tasks.notify_canon_of_location_task.delay(
        101,
        202,
        {"location_name": "Arcadia", "city": "Eryndor"},
        trace_id="trace-123",
    )

    assert result.get() is None

    assert "log_canonical_event" in calls
    assert calls.get("memory_orchestrator_args") == (101, 202)
    assert "store_memory" in calls
    assert "add_to_vector_store" in calls
