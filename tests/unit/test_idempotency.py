from __future__ import annotations

from nyx.utils.idempotency import clear_cache, idempotent


class DummyTask:
    """Minimal stand-in for a Celery bound task."""

    def __init__(self) -> None:
        self.request = object()


def test_idempotent_strips_celery_self_for_key_fn() -> None:
    clear_cache()

    captured_args: list[tuple[str, ...]] = []

    def key_fn(*key_args: str) -> str:
        captured_args.append(tuple(key_args))
        return "::".join(key_args) or "noop"

    calls: list[tuple[DummyTask, str]] = []

    @idempotent(key_fn)
    def sample_task(task_self: DummyTask, value: str) -> str:
        calls.append((task_self, value))
        return value

    dummy = DummyTask()

    assert sample_task(dummy, "alpha") == "alpha"
    # The key function should not receive the Celery task instance.
    assert captured_args == [("alpha",)]
    assert calls == [(dummy, "alpha")]

    clear_cache()


def test_in_memory_idempotency_respects_ttl(monkeypatch) -> None:
    clear_cache()

    timestamp = {"value": 1000.0}

    def fake_monotonic() -> float:
        return timestamp["value"]

    monkeypatch.setattr("nyx.utils.idempotency.time.monotonic", fake_monotonic)

    executions: list[str] = []

    @idempotent(lambda value: value, ttl_sec=5)
    def sample(value: str) -> str:
        executions.append(value)
        return value

    assert sample("beta") == "beta"
    # Within the TTL window the cached key should block re-execution.
    assert sample("beta") is None

    # Advance time past the TTL window and ensure the entry is pruned.
    timestamp["value"] += 6
    assert sample("beta") == "beta"

    assert executions == ["beta", "beta"]

    clear_cache()
