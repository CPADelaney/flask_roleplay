import asyncio
import os
import sys
from datetime import datetime
from types import SimpleNamespace
import types

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("SENTENCE_TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(os.getcwd(), ".st_cache"))
os.makedirs(os.environ["SENTENCE_TRANSFORMERS_HOME"], exist_ok=True)
os.environ.setdefault("OPENAI_API_KEY", "test-key")

if "nyx.core.memory.vector_store" not in sys.modules:
    vector_store_stub = types.ModuleType("nyx.core.memory.vector_store")

    async def _stub_add(text, meta):
        return "stub"

    async def _stub_query(ctx, k: int = 5):
        return []

    vector_store_stub.add = _stub_add  # type: ignore[attr-defined]
    vector_store_stub.query = _stub_query  # type: ignore[attr-defined]
    sys.modules["nyx.core.memory.vector_store"] = vector_store_stub

from lore.core.canon import log_canonical_event
from lore.core import canon as canon_module


class DummyConnection:
    def __init__(self, fetch_return=None, fetchval_result=1):
        self.fetch_return = fetch_return or []
        self.fetchval_result = fetchval_result
        self.fetch_calls = []
        self.fetchval_calls = []
        self.execute_calls = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return self.fetch_return

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return self.fetchval_result

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))

    def is_in_transaction(self):
        return False


@pytest.mark.asyncio
async def test_log_canonical_event_defers_memory_persistence(monkeypatch):
    ctx = SimpleNamespace(user_id=7, conversation_id=12)
    conn = DummyConnection(fetch_return=[{"id": 3}], fetchval_result=99)

    store_calls = []
    store_started = asyncio.Event()
    allow_store = asyncio.Event()

    async def fake_store_memory(**kwargs):
        store_started.set()
        await allow_store.wait()
        store_calls.append(kwargs)

    orchestrator = SimpleNamespace(store_memory=fake_store_memory)

    async def fake_get_canon_memory_orchestrator(user_id, conversation_id):
        assert (user_id, conversation_id) == (ctx.user_id, ctx.conversation_id)
        return orchestrator

    monkeypatch.setattr(
        "lore.core.canon.get_canon_memory_orchestrator",
        fake_get_canon_memory_orchestrator,
    )

    event_id = await log_canonical_event(
        ctx,
        conn,
        "Test canonical memory deferral",
        tags=["lore"],
        significance=5,
    )

    assert event_id == 99

    # Memory write should not block the caller even though it is waiting.
    assert not allow_store.is_set()

    allow_store.set()
    await store_started.wait()
    await asyncio.sleep(0)

    assert store_calls
    stored = store_calls[0]
    assert stored["memory_text"] == "Test canonical memory deferral"
    assert "canonical_event" in stored["tags"]


@pytest.mark.asyncio
async def test_log_canonical_event_memory_errors_logged(monkeypatch, caplog):
    ctx = SimpleNamespace(user_id=2, conversation_id=4)
    conn = DummyConnection()

    async def failing_store_memory(**kwargs):
        raise RuntimeError("boom")

    orchestrator = SimpleNamespace(store_memory=failing_store_memory)

    async def fake_get_canon_memory_orchestrator(user_id, conversation_id):
        return orchestrator

    monkeypatch.setattr(
        "lore.core.canon.get_canon_memory_orchestrator",
        fake_get_canon_memory_orchestrator,
    )
    monkeypatch.setattr(canon_module, "_TRACK_OPERATION_AVAILABLE", False)

    with caplog.at_level("ERROR"):
        event_id = await log_canonical_event(
            ctx,
            conn,
            "Test canonical memory failure logging",
            tags=["lore"],
            significance=6,
        )

        # Let the background task run.
        await asyncio.sleep(0)

    assert event_id == 1
    assert any(
        "Failed to persist canonical event" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_schedule_canonical_memory_persist_skips_when_shutting_down(monkeypatch):
    ctx = SimpleNamespace(user_id=1, conversation_id=2)

    async def fail_persist(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("Persistence coroutine should not run while shutting down")

    def fail_create_task(coro):  # pragma: no cover - defensive
        raise AssertionError("create_task should not be invoked during shutdown")

    monkeypatch.setattr(canon_module, "is_shutting_down", lambda: True)
    monkeypatch.setattr(canon_module, "_persist_canonical_event_memory", fail_persist)
    monkeypatch.setattr(asyncio, "create_task", fail_create_task)

    canon_module._schedule_canonical_memory_persist(
        ctx,
        event_id=123,
        event_text="shutdown test",
        tags=["test"],
        significance=1,
        parent_ids=[],
        event_timestamp=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_schedule_canonical_memory_persist_handles_track_operation_shutdown(
    monkeypatch, caplog
):
    ctx = SimpleNamespace(user_id=3, conversation_id=4)
    created_tasks = []

    async def noop_persist(*args, **kwargs):
        return None

    async def failing_track_operation(coro):
        raise ConnectionError("worker shutting down")

    original_create_task = asyncio.create_task

    def tracking_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(canon_module, "is_shutting_down", lambda: False)
    monkeypatch.setattr(canon_module, "_persist_canonical_event_memory", noop_persist)
    monkeypatch.setattr(canon_module, "track_operation", failing_track_operation)
    monkeypatch.setattr(canon_module, "_TRACK_OPERATION_AVAILABLE", True)
    monkeypatch.setattr(asyncio, "create_task", tracking_create_task)

    with caplog.at_level("INFO"):
        canon_module._schedule_canonical_memory_persist(
            ctx,
            event_id=456,
            event_text="track op shutdown",
            tags=["test"],
            significance=2,
            parent_ids=[],
            event_timestamp=datetime.utcnow(),
        )

        # Allow the scheduled task to run and handle the ConnectionError.
        await asyncio.sleep(0)

    assert created_tasks, "Expected the shutdown-safe wrapper task to be scheduled"
    await asyncio.gather(*created_tasks)

    assert any(
        "Skipping canonical memory persistence due to shutdown" in record.getMessage()
        for record in caplog.records
    )
