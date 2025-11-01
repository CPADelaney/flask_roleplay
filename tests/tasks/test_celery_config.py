from __future__ import annotations

import pytest

from nyx.tasks.base import NyxTask, app
from nyx.tasks.beat import build_beat_schedule
from nyx.tasks.maintenance import dispatch_outbox


def test_celery_app_configuration():
    routes = app.conf.task_routes or {}
    assert "nyx.tasks.background.conflict.*" in routes
    assert routes["nyx.tasks.background.conflict.*"]["queue"] == "nyx-conflict"

    beat_schedule = build_beat_schedule()
    assert "nyx-outbox-dispatcher" in beat_schedule
    assert (
        beat_schedule["nyx-outbox-dispatcher"]["task"]
        == "nyx.tasks.maintenance.dispatch_outbox"
    )


def test_nyx_task_defaults():
    @app.task(bind=True, base=NyxTask, name="tests.nyx_task_defaults")
    def sample(self):  # pragma: no cover - not executed
        return "ok"

    try:
        assert TimeoutError in sample.autoretry_for
        assert ConnectionError in sample.autoretry_for
        assert sample.retry_kwargs["max_retries"] == 3
        assert sample.retry_kwargs["countdown"] == 2
    finally:
        app.tasks.pop(sample.name, None)


def test_dispatch_outbox_calls_dispatcher(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, Any] = {}

    def fake_run_once():
        calls["count"] = calls.get("count", 0) + 1
        return 5

    monkeypatch.setattr("nyx.tasks.maintenance.run_once", fake_run_once)

    result = dispatch_outbox._orig_run()

    assert calls["count"] == 1
    assert result == "processed:5"
