import sys
import types

import pytest

from logic.conflict_system import social_circle_hotpath as hotpath


class _DummyLock:
    def __enter__(self):
        return True

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyTask:
    def __init__(self):
        self.calls = []

    def delay(self, payload):
        self.calls.append(payload)


@pytest.fixture(autouse=True)
def _reset_conflict_flag(monkeypatch):
    monkeypatch.setattr(hotpath.settings, "CONFLICT_EAGER_WARMUP", False)


def _configure_hotpath(monkeypatch):
    monkeypatch.setattr(hotpath, "get_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(hotpath, "redis_lock", lambda *args, **kwargs: _DummyLock())
    monkeypatch.setattr(hotpath, "cache_key", lambda *parts: ":".join(str(part) for part in parts))

    task = _DummyTask()
    fake_module = types.SimpleNamespace(generate_social_bundle=task)
    monkeypatch.setitem(sys.modules, "nyx.tasks.background.social_tasks", fake_module)

    return task


def test_get_scene_bundle_skips_dispatch_when_not_eager(monkeypatch):
    task = _configure_hotpath(monkeypatch)

    scene_hash = "hash123"
    scene_context = {"scene_id": 99, "location": "market"}

    bundle = hotpath.get_scene_bundle(scene_hash, scene_context=scene_context)

    assert bundle["status"] == "generating"
    assert task.calls == []


def test_get_scene_bundle_dispatches_when_eager_flag(monkeypatch):
    task = _configure_hotpath(monkeypatch)

    scene_hash = "hash123"
    scene_context = {"scene_id": 99, "location": "market"}
    user_id = 42
    conversation_id = 314
    targets = [1, 2, 3]

    bundle = hotpath.get_scene_bundle(
        scene_hash,
        scene_context=scene_context,
        user_id=user_id,
        conversation_id=conversation_id,
        target_npcs=targets,
        eager=True,
    )

    assert bundle["status"] == "generating"
    assert task.calls, "generate_social_bundle.delay should be invoked"

    payload = task.calls[0]
    assert payload["scene_hash"] == scene_hash
    assert payload["scene_context"] == scene_context
    assert payload["user_id"] == user_id
    assert payload["conversation_id"] == conversation_id
    assert payload["target_npcs"] == targets


def test_get_scene_bundle_dispatches_when_config_opt_in(monkeypatch):
    task = _configure_hotpath(monkeypatch)
    monkeypatch.setattr(hotpath.settings, "CONFLICT_EAGER_WARMUP", True)

    scene_hash = "envhash"
    scene_context = {"scene_id": 7, "location": "plaza"}

    hotpath.get_scene_bundle(scene_hash, scene_context=scene_context)

    assert task.calls, "generate_social_bundle.delay should respect config opt-in"
