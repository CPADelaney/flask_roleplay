import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lore.core.canon import log_canonical_event


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
