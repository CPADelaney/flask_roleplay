from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Mapping, Optional

import pytest

from nyx.conversation.store import ConversationStore, ThreadBinding


@dataclass
class _FakeConnection:
    fetchrow_results: List[Optional[Mapping[str, Any]]]
    fetch_results: List[List[Mapping[str, Any]]] | None = None

    def __post_init__(self) -> None:
        self.fetch_calls: List[Any] = []
        self.fetchrow_calls: List[Any] = []
        self.execute_calls: List[Any] = []

    async def fetchrow(self, query: str, *args: Any) -> Optional[Mapping[str, Any]]:
        self.fetchrow_calls.append((query, args))
        if self.fetchrow_results:
            return self.fetchrow_results.pop(0)
        return None

    async def fetch(self, query: str, *args: Any) -> List[Mapping[str, Any]]:
        self.fetch_calls.append((query, args))
        if self.fetch_results:
            return self.fetch_results.pop(0)
        return []

    async def execute(self, query: str, *args: Any) -> None:
        self.execute_calls.append((query, args))


class _ConnectionManagerFactory:
    def __init__(self, connections: List[_FakeConnection]):
        self._connections = connections
        self._index = 0

    def __call__(self):
        factory = self

        class _Manager:
            async def __aenter__(self_inner):
                if factory._index >= len(factory._connections):
                    raise AssertionError("No more fake connections available")
                conn = factory._connections[factory._index]
                factory._index += 1
                return conn

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Manager()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_get_or_create_thread_id_returns_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = {
        "user_id": 1,
        "conversation_id": 9,
        "channel": "openai",
        "remote_id": "thread-123",
        "created_at": datetime.utcnow(),
    }
    connections = [_FakeConnection(fetchrow_results=[existing])]
    monkeypatch.setattr(
        "nyx.conversation.store.get_db_connection_context",
        _ConnectionManagerFactory(connections),
    )
    monkeypatch.setattr(
        "nyx.conversation.store.ConversationStore._create_remote_thread",
        lambda self, user_id, conversation_id: pytest.fail("remote creation should not occur"),
    )

    store = ConversationStore()
    binding = await store.get_or_create_thread_id(user_id=1, conversation_id=9)

    assert isinstance(binding, ThreadBinding)
    assert binding.remote_id == "thread-123"
    assert connections[0].fetchrow_calls  # the existing row was fetched


@pytest.mark.anyio("asyncio")
async def test_get_or_create_thread_id_creates_new(monkeypatch: pytest.MonkeyPatch) -> None:
    inserted = {
        "user_id": 1,
        "conversation_id": 2,
        "channel": "openai",
        "remote_id": "thread-new",
        "created_at": datetime.utcnow(),
    }

    connections = [
        _FakeConnection(fetchrow_results=[None]),
        _FakeConnection(fetchrow_results=[inserted]),
    ]

    monkeypatch.setattr(
        "nyx.conversation.store.get_db_connection_context",
        _ConnectionManagerFactory(connections),
    )

    async def _fake_create_remote_thread(self, *, user_id: int, conversation_id: int) -> str:
        return "thread-new"

    monkeypatch.setattr(
        "nyx.conversation.store.ConversationStore._create_remote_thread",
        _fake_create_remote_thread,
    )

    store = ConversationStore()
    binding = await store.get_or_create_thread_id(user_id=1, conversation_id=2)

    assert binding.remote_id == "thread-new"
    assert connections[1].fetchrow_calls  # insertion returning row used


@pytest.mark.anyio("asyncio")
async def test_append_turn_inserts_normalized_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(fetchrow_results=[])
    monkeypatch.setattr(
        "nyx.conversation.store.get_db_connection_context",
        _ConnectionManagerFactory([connection]),
    )

    store = ConversationStore()
    await store.append_turn(
        user_id=1,
        conversation_id=11,
        turn={"sender": "npc", "content": "hello", "ignored": "value"},
    )

    assert connection.execute_calls
    query, params = connection.execute_calls[0]
    assert "INSERT INTO messages" in query
    assert params == (11, "npc", "hello")


@pytest.mark.anyio("asyncio")
async def test_fetch_recent_turns_returns_chronological(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"sender": "assistant", "content": "second"},
        {"sender": "user", "content": "first"},
    ]
    connection = _FakeConnection(fetchrow_results=[], fetch_results=[rows])
    monkeypatch.setattr(
        "nyx.conversation.store.get_db_connection_context",
        _ConnectionManagerFactory([connection]),
    )

    store = ConversationStore()
    history = await store.fetch_recent_turns(user_id=1, conversation_id=2, limit=2)

    assert history == [
        {"sender": "user", "content": "first"},
        {"sender": "assistant", "content": "second"},
    ]


@pytest.mark.anyio("asyncio")
async def test_fetch_recent_turns_handles_invalid_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ConversationStore()
    result = await store.fetch_recent_turns(user_id="bad", conversation_id=1)
    assert result == []
