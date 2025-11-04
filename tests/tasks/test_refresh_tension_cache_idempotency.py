from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


def _load_clear_cache():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "nyx_utils_idempotency", repo_root / "nyx" / "utils" / "idempotency.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.clear_cache


clear_cache = _load_clear_cache()

class DummyTask:
    def __init__(self) -> None:
        self.request = object()


def _load_conflict_slice_tasks(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "nyx" / "tasks" / "background" / "conflict_slice_tasks.py"

    stub_nyx = types.ModuleType("nyx")
    stub_nyx.__path__ = [str(repo_root / "nyx")]
    monkeypatch.setitem(sys.modules, "nyx", stub_nyx)

    stub_nyx_utils = types.ModuleType("nyx.utils")
    stub_nyx_utils.__path__ = [str(repo_root / "nyx" / "utils")]
    monkeypatch.setitem(sys.modules, "nyx.utils", stub_nyx_utils)

    # Create lightweight stubs for the Celery task package to avoid expensive imports.
    stub_tasks = types.ModuleType("nyx.tasks")
    stub_tasks.__path__ = [str(module_path.parents[1])]

    stub_background = types.ModuleType("nyx.tasks.background")
    stub_background.__path__ = [str(module_path.parent)]
    stub_tasks.background = stub_background  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "nyx.tasks", stub_tasks)
    monkeypatch.setitem(sys.modules, "nyx.tasks.background", stub_background)

    stub_base = types.ModuleType("nyx.tasks.base")

    class _NyxTask:  # pragma: no cover - minimal stand-in
        abstract = True

    class _App:  # pragma: no cover - minimal Celery stub
        def task(self, *args: Any, **kwargs: Any):
            def _decorator(func):
                return func

            return _decorator

    stub_base.NyxTask = _NyxTask  # type: ignore[attr-defined]
    stub_base.app = _App()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "nyx.tasks.base", stub_base)

    stub_utils = types.ModuleType("nyx.tasks.utils")

    def run_coro_sync(coro):
        return asyncio.run(coro)

    def with_retry(func):
        return func

    stub_utils.run_coro = run_coro_sync  # type: ignore[attr-defined]
    stub_utils.with_retry = with_retry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.tasks.utils", stub_utils)

    stub_conflict_llm = types.ModuleType("nyx.tasks.background.conflict_llm_helpers")

    class _FakeEnum:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:  # pragma: no cover - debug friendly
            return f"FakeEnum({self.value})"

    async def fake_analyze_patterns_async(
        memories: List[str], relationships: List[str]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "type": _FakeEnum("subtle_rivalry"),
                "intensity": _FakeEnum("tension"),
                "description": "A simmering everyday disagreement.",
                "evidence": [*memories, *relationships],
                "tension_level": 0.42,
            }
        ]

    stub_conflict_llm.analyze_patterns_async = fake_analyze_patterns_async  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.tasks.background.conflict_llm_helpers", stub_conflict_llm)

    stub_slice = types.ModuleType("logic.conflict_system.slice_of_life_conflicts")

    class EmergentConflictDetector:  # pragma: no cover - simple helper for the test
        def __init__(self, user_id: int, conversation_id: int) -> None:
            self.user_id = user_id
            self.conversation_id = conversation_id

        async def collect_tension_inputs(self) -> Tuple[List[str], List[str]]:
            return ["memory"], ["relationship"]

        async def _analyze_patterns_with_llm(
            self, memories: List[str], relationships: List[str]
        ) -> List[Dict[str, Any]]:
            return [{"fallback": True}]

    stub_slice.EmergentConflictDetector = EmergentConflictDetector  # type: ignore[attr-defined]
    stub_slice.ConflictDailyIntegration = object  # type: ignore[attr-defined]
    stub_slice.PatternBasedResolution = object  # type: ignore[attr-defined]
    stub_slice.SliceOfLifeConflictManager = object  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "logic.conflict_system.slice_of_life_conflicts", stub_slice)

    stub_db_connection = types.ModuleType("db.connection")

    class _DummyConnection:
        async def execute(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _DummyContext:
        async def __aenter__(self) -> _DummyConnection:
            return _DummyConnection()

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    def get_db_connection_context() -> _DummyContext:
        return _DummyContext()

    stub_db_connection.get_db_connection_context = get_db_connection_context  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "db.connection", stub_db_connection)

    spec = importlib.util.spec_from_file_location(
        "nyx.tasks.background.conflict_slice_tasks", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_refresh_tension_cache_accepts_celery_self(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()

    module = _load_conflict_slice_tasks(monkeypatch)

    calls: Dict[str, List[Tuple[Tuple[Any, ...], Dict[str, Any]]]] = {"upserts": []}

    async def fake_upsert_tension(*args: Any, **kwargs: Any) -> None:
        calls["upserts"].append((args, kwargs))

    monkeypatch.setattr(module, "_upsert_tension", fake_upsert_tension)

    dummy_payload = {"user_id": 101, "conversation_id": 202}

    result = module.refresh_tension_cache(DummyTask(), dummy_payload)

    assert result == {
        "status": "ready",
        "user_id": 101,
        "conversation_id": 202,
        "items": 1,
    }
    assert len(calls["upserts"]) == 2

    ready_payload = calls["upserts"][1][0][3]
    assert isinstance(ready_payload, list)
    assert ready_payload[0]["type"] == "subtle_rivalry"
    assert ready_payload[0]["intensity"] == "tension"
    # Ensure the payload mirrors the real upsert behavior of json.dumps(...)
    json.dumps(ready_payload)

    clear_cache()
