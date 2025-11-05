from __future__ import annotations

"""Unit tests for social task metadata sanitization."""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def social_tasks(monkeypatch):
    """Load the social_tasks module with lightweight stubs for heavy deps."""
    # Ensure we start with a clean slate for nyx.tasks.* imports
    for name in list(sys.modules):
        if name.startswith("nyx.tasks") or name.startswith("logic.conflict_system"):
            sys.modules.pop(name)
    repo_root = Path(__file__).resolve().parents[2]

    # Keep the real nyx package so utils (sanitize helper) are available
    import nyx as nyx_pkg

    # Stub nyx.tasks packages to avoid executing heavy __init__ logic
    tasks_pkg = types.ModuleType("nyx.tasks")
    tasks_pkg.__path__ = [str(repo_root / "nyx" / "tasks")]
    monkeypatch.setitem(sys.modules, "nyx.tasks", tasks_pkg)
    monkeypatch.setattr(nyx_pkg, "tasks", tasks_pkg, raising=False)

    background_pkg = types.ModuleType("nyx.tasks.background")
    background_pkg.__path__ = [str(repo_root / "nyx" / "tasks" / "background")]
    monkeypatch.setitem(sys.modules, "nyx.tasks.background", background_pkg)
    monkeypatch.setattr(tasks_pkg, "background", background_pkg, raising=False)

    # Stub nyx.tasks.base so Celery isn't required
    base_module = types.ModuleType("nyx.tasks.base")

    class _DummyNyxTask:
        abstract = True

    class _DummyApp:
        def task(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    base_module.NyxTask = _DummyNyxTask
    base_module.app = _DummyApp()
    monkeypatch.setitem(sys.modules, "nyx.tasks.base", base_module)

    # Stub nyx.tasks.utils to bypass async worker helpers
    utils_module = types.ModuleType("nyx.tasks.utils")

    def _with_retry(func):
        return func

    def _run_coro(coro):
        return asyncio.run(coro)

    utils_module.with_retry = _with_retry
    utils_module.run_coro = _run_coro
    monkeypatch.setitem(sys.modules, "nyx.tasks.utils", utils_module)

    # Stub idempotency decorator to no-op during tests
    utils_pkg = types.ModuleType("nyx.utils")
    utils_pkg.__path__ = [str(repo_root / "nyx" / "utils")]
    monkeypatch.setitem(sys.modules, "nyx.utils", utils_pkg)
    monkeypatch.setattr(nyx_pkg, "utils", utils_pkg, raising=False)

    idempotency_module = types.ModuleType("nyx.utils.idempotency")

    def _idempotent(*args, **kwargs):  # noqa: D401 - signature shim
        def decorator(func):
            return func

        return decorator

    idempotency_module.idempotent = _idempotent
    monkeypatch.setitem(sys.modules, "nyx.utils.idempotency", idempotency_module)

    # Lightweight cache helpers
    infra_pkg = types.ModuleType("infra")
    infra_pkg.__path__ = [str(repo_root / "infra")]
    monkeypatch.setitem(sys.modules, "infra", infra_pkg)

    cache_module = types.ModuleType("infra.cache")
    cache_module.cache_key = lambda *parts: ":".join(str(part) for part in parts)
    cache_module.get_json = lambda *args, **kwargs: []
    cache_module.set_json = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "infra.cache", cache_module)
    monkeypatch.setattr(infra_pkg, "cache", cache_module, raising=False)

    # Provide a minimal SocialCircleManager implementation
    logic_pkg = types.ModuleType("logic")
    logic_pkg.__path__ = [str(repo_root / "logic")]
    monkeypatch.setitem(sys.modules, "logic", logic_pkg)

    conflict_pkg = types.ModuleType("logic.conflict_system")
    conflict_pkg.__path__ = [str(repo_root / "logic" / "conflict_system")]
    monkeypatch.setitem(sys.modules, "logic.conflict_system", conflict_pkg)
    monkeypatch.setattr(logic_pkg, "conflict_system", conflict_pkg, raising=False)

    social_module = types.ModuleType("logic.conflict_system.social_circle")

    class _DummySocialCircle:  # noqa: D401 - simple placeholder
        """Placeholder to satisfy imports."""

    class _DummyManager:
        def __init__(self, user_id: int, conversation_id: int) -> None:  # noqa: D401
            self.user_id = user_id
            self.conversation_id = conversation_id

        async def generate_gossip_background(self, scene_context, target_npcs):
            return {"user": self.user_id, "targets": target_npcs}

        async def calculate_reputation_background(self, target_id, circle=None):  # noqa: ARG002
            return {"influence": 0.7}

    social_module.SocialCircle = _DummySocialCircle
    social_module.SocialCircleManager = _DummyManager
    monkeypatch.setitem(sys.modules, "logic.conflict_system.social_circle", social_module)

    module = importlib.import_module("nyx.tasks.background.social_tasks")
    return module


def test_generate_social_bundle_traces_use_string_ids(social_tasks):
    captured = {}

    class _TraceRecorder:
        def __init__(self, metadata):
            self.metadata = metadata

        def __enter__(self):
            captured["metadata"] = self.metadata
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_trace(*, metadata=None, **kwargs):  # noqa: ARG001
        return _TraceRecorder(metadata)

    # Patch the trace context manager on the loaded module
    social_tasks.trace = _fake_trace

    dummy_self = types.SimpleNamespace(request=types.SimpleNamespace(retries=0))
    dummy_self.retry = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("retry"))

    payload = {
        "scene_context": {"scene_hash": 9876},
        "user_id": 42,
        "conversation_id": 314,
        "target_npcs": [1],
        "reputation_targets": [1],
    }

    result = social_tasks.generate_social_bundle(dummy_self, payload)
    assert result["status"] == "generated"

    metadata = captured["metadata"]
    assert metadata["scene_hash"] == "9876"
    assert metadata["conversation_id"] == "314"
    assert metadata["user_id"] == "42"
