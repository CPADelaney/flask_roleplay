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


def test_get_scene_bundle_queues_enriched_payload(monkeypatch):
    monkeypatch.setattr(hotpath, "get_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(hotpath, "redis_lock", lambda *args, **kwargs: _DummyLock())
    monkeypatch.setattr(hotpath, "cache_key", lambda *parts: ":".join(str(part) for part in parts))

    task = _DummyTask()
    fake_module = types.SimpleNamespace(generate_social_bundle=task)
    monkeypatch.setitem(sys.modules, "nyx.tasks.background.social_tasks", fake_module)

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
    )

    assert bundle["status"] == "generating"
    assert task.calls, "generate_social_bundle.delay should be invoked"

    payload = task.calls[0]
    assert payload["scene_hash"] == scene_hash
    assert payload["scene_context"] == scene_context
    assert payload["user_id"] == user_id
    assert payload["conversation_id"] == conversation_id
    assert payload["target_npcs"] == targets
