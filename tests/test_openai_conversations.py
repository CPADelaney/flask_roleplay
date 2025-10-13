import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai_integration import ConversationManager, conversations
from logic.universal_updater_agent import apply_universal_updates_async
from openai_integration.conversations import ConversationStreamError


@pytest.fixture
def anyio_backend():
    return "asyncio"


class StubConnection:
    def __init__(self, fetchrow_results):
        self.fetchrow_results = list(fetchrow_results)
        self.queries = []
        self.execute_queries = []

    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        if self.fetchrow_results:
            return self.fetchrow_results.pop(0)
        return None

    async def execute(self, query, *args):
        self.execute_queries.append((query, args))
        return "EXECUTE"


class _SealTransaction:
    def __init__(self):
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class SealAwareConnection:
    def __init__(self, result):
        self._result = result
        self.calls = []
        self.transaction_manager = _SealTransaction()

    def transaction(self):
        self.calls.append("transaction")
        return self.transaction_manager

    async def fetchrow(self, query, payload):
        self.calls.append((query, payload))
        return {"result": self._result}


class DummyEvent:
    def __init__(self, event_type: str, **payload: Any) -> None:
        self.type = event_type
        for key, value in payload.items():
            setattr(self, key, value)


class DummyResponsesStream:
    def __init__(self, events: Iterable[DummyEvent], final_response: Dict[str, Any]):
        self._events = list(events)
        self._final_response = final_response
        self._iter: Iterable[DummyEvent] | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        self._iter = iter(self._events)
        return self

    async def __anext__(self):
        assert self._iter is not None
        try:
            return next(self._iter)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise StopAsyncIteration from exc

    def get_final_response(self):
        return self._final_response


class DummyResponsesClient:
    def __init__(self, events: Iterable[DummyEvent], final_response: Dict[str, Any]):
        self._events = list(events)
        self._final_response = final_response
        self.calls: List[Dict[str, Any]] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return DummyResponsesStream(self._events, self._final_response)


class DummyOpenAIClient:
    def __init__(self, events: Iterable[DummyEvent], final_response: Dict[str, Any]):
        self.responses = DummyResponsesClient(events, final_response)


@pytest.mark.anyio("asyncio")
async def test_create_conversation_uses_defined_columns():
    expected_row = {
        "id": 1,
        "user_id": 7,
        "conversation_id": 99,
        "openai_assistant_id": "asst",
        "openai_thread_id": "thread-1",
        "openai_run_id": "run-1",
        "openai_response_id": "resp-1",
        "status": "active",
        "last_error": None,
        "metadata": {"foo": "bar", "openai_conversation_id": "conv-remote-1"},
        "openai_conversation_id": "conv-remote-1",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([expected_row])

    result = await conversations.create_conversation(
        user_id=7,
        conversation_id=99,
        openai_assistant_id="asst",
        openai_thread_id="thread-1",
        openai_run_id="run-1",
        openai_response_id="resp-1",
        status="active",
        metadata={"foo": "bar"},
        openai_conversation_id="conv-remote-1",
        conn=conn,
    )

    assert result == expected_row
    assert len(conn.queries) == 1
    insert_query, params = conn.queries[0]
    assert "INSERT INTO openai_conversations" in insert_query
    assert "openai_assistant_id" in insert_query
    assert "openai_thread_id" in insert_query
    assert "ON CONFLICT (conversation_id)" in insert_query
    assert params == (
        7,
        99,
        "asst",
        "thread-1",
        "run-1",
        "resp-1",
        "active",
        None,
        {"foo": "bar", "openai_conversation_id": "conv-remote-1"},
    )


@pytest.mark.anyio("asyncio")
async def test_get_or_create_returns_existing_row_without_insert():
    existing_row = {
        "id": 10,
        "user_id": 3,
        "conversation_id": 55,
        "openai_assistant_id": "existing",
        "openai_thread_id": "thread-existing",
        "openai_run_id": None,
        "openai_response_id": None,
        "status": "ready",
        "last_error": None,
        "metadata": {"openai_conversation_id": "existing-remote"},
        "openai_conversation_id": "existing-remote",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([existing_row])

    result = await conversations.get_or_create_conversation(
        user_id=3,
        conversation_id=55,
        openai_assistant_id="ignored",
        openai_thread_id="ignored",
        conn=conn,
    )

    assert result == existing_row
    assert len(conn.queries) == 1
    select_query, params = conn.queries[0]
    assert select_query.strip().startswith("SELECT")
    assert params == (55,)


@pytest.mark.anyio("asyncio")
async def test_get_or_create_inserts_when_missing():
    inserted_row = {
        "id": 11,
        "user_id": 4,
        "conversation_id": 77,
        "openai_assistant_id": "new",
        "openai_thread_id": "thread-new",
        "openai_run_id": None,
        "openai_response_id": None,
        "status": "pending",
        "last_error": None,
        "metadata": {"openai_conversation_id": "conv-new"},
        "openai_conversation_id": "conv-new",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([None, inserted_row])

    result = await conversations.get_or_create_conversation(
        user_id=4,
        conversation_id=77,
        openai_assistant_id="new",
        openai_thread_id="thread-new",
        openai_conversation_id="conv-new",
        conn=conn,
    )

    assert result == inserted_row
    assert len(conn.queries) == 2
    select_query, select_params = conn.queries[0]
    insert_query, insert_params = conn.queries[1]
    assert select_query.strip().startswith("SELECT")
    assert insert_query.strip().startswith("INSERT INTO openai_conversations")
    assert insert_params[0] == 4
    assert insert_params[1] == 77
    assert insert_params[-1] == {"openai_conversation_id": "conv-new"}


@pytest.mark.anyio("asyncio")
async def test_rotate_scene_updates_previous_scene_and_inserts_active_scene():
    closing_scene = {
        "scene_summary": "Curtain falls",
        "scene_state": {"mood": "somber"},
        "metadata": {"score": "minor"},
    }

    new_scene_row = {
        "id": 22,
        "conversation_id": 123,
        "scene_number": 6,
        "scene_title": "New Dawn",
        "scene_summary": "Sunrise breaks",
        "scene_state": {"weather": "clear"},
        "active_npc_ids": [1, 2],
        "location_reference": None,
        "tension_level": 0,
        "tags": ["hope"],
        "metadata": {"music": "calm"},
        "is_active": True,
        "started_at": "2024-01-01T00:00:00Z",
        "ended_at": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }

    conn = StubConnection([
        {"id": 10, "scene_number": 5},
        new_scene_row,
    ])

    result = await conversations.rotate_conversation_scene(
        conversation_id=123,
        closing_scene=closing_scene,
        new_scene={
            "scene_title": "New Dawn",
            "scene_summary": "Sunrise breaks",
            "scene_state": {"weather": "clear"},
            "metadata": {"music": "calm"},
            "active_npc_ids": [1, 2],
            "tags": ["hope"],
        },
        conn=conn,
    )

    assert result == new_scene_row
    assert len(conn.execute_queries) == 1
    update_query, update_params = conn.execute_queries[0]
    assert "scene_summary" in update_query
    assert "scene_state" in update_query
    assert update_params[1] == closing_scene["scene_summary"]
    assert update_params[2] == closing_scene["scene_state"]
    assert update_params[3] == closing_scene["metadata"]

    assert len(conn.queries) == 3
    insert_query, insert_params = conn.queries[1]
    assert "is_active" in insert_query
    assert insert_params[1] == 6
    assert insert_params[3] == "Sunrise breaks"
    assert insert_params[4] == {"weather": "clear"}
    assert insert_params[9] == {"music": "calm"}

    seal_query, seal_params = conn.queries[2]
    assert "conversations.items" in seal_query
    assert seal_params[0] == 123


@pytest.mark.anyio("asyncio")
async def test_rotate_scene_persists_scene_seal_system_message():
    new_scene_row = {
        "id": 44,
        "conversation_id": 321,
        "scene_number": 1,
        "scene_title": "Opening Night",
        "scene_summary": "Curtains part",
        "scene_state": {},
        "active_npc_ids": [],
        "location_reference": None,
        "tension_level": 0,
        "tags": [],
        "metadata": {},
        "is_active": True,
        "started_at": "2024-01-01T00:00:00Z",
        "ended_at": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }

    conn = StubConnection([
        None,
        new_scene_row,
        {
            "id": 999,
            "conversation_id": 321,
            "role": "system",
            "item_type": "scene_seal",
            "content": {},
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        },
    ])

    new_scene = {
        "scene_title": "Opening Night",
        "metadata": {
            "scene_seal": {
                "venue": "Velvet Sanctum",
                "date": "2177-03-07",
                "non_negotiables": [
                    "Obey her whispered command",
                    "No safe word tonight",
                ],
            }
        },
    }

    await conversations.rotate_conversation_scene(
        conversation_id=321,
        new_scene=new_scene,
        conn=conn,
    )

    assert len(conn.queries) == 3
    seal_query, seal_params = conn.queries[2]
    assert "conversations.items" in seal_query

    payload = seal_params[1]
    metadata = seal_params[2]

    assert payload["role"] == "system"
    assert payload["content"][0]["type"] == "input_text"
    text = payload["content"][0]["text"]
    assert "Velvet Sanctum" in text
    assert "2177-03-07" in text
    assert "Obey her whispered command" in text
    assert metadata["venue"] == "Velvet Sanctum"
    assert metadata["date"] == "2177-03-07"
    assert metadata["non_negotiables"] == [
        "Obey her whispered command",
        "No safe word tonight",
    ]


@pytest.mark.anyio("asyncio")
async def test_get_active_scene_filters_on_is_active():
    active_scene = {
        "id": 200,
        "conversation_id": 7,
        "scene_number": 3,
        "scene_title": "Midnight Chase",
        "scene_summary": "High stakes pursuit",
        "scene_state": {"pursuit": True},
        "active_npc_ids": [],
        "location_reference": "Downtown",
        "tension_level": 8,
        "tags": [],
        "metadata": {},
        "is_active": True,
        "started_at": "2024-01-01T01:00:00Z",
        "ended_at": None,
        "created_at": "2024-01-01T01:00:00Z",
        "updated_at": "2024-01-01T01:00:00Z",
    }

    conn = StubConnection([active_scene])

    result = await conversations.get_active_scene(
        conversation_id=7,
        conn=conn,
    )

    assert result == active_scene
    assert len(conn.queries) == 1
    select_query, params = conn.queries[0]
    assert "is_active = TRUE" in select_query
    assert params == (7,)


@pytest.mark.anyio("asyncio")
async def test_conversation_manager_proxies_crud_helpers():
    manager = ConversationManager(client=DummyOpenAIClient([], {}))
    expected_row = {
        "id": 1,
        "user_id": 5,
        "conversation_id": 123,
        "openai_assistant_id": "assistant",
        "openai_thread_id": "thread",
        "openai_run_id": None,
        "openai_response_id": None,
        "status": "pending",
        "last_error": None,
        "metadata": {},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([expected_row])

    row = await manager.create_conversation(
        user_id=5,
        conversation_id=123,
        openai_assistant_id="assistant",
        openai_thread_id="thread",
        conn=conn,
    )

    assert row == expected_row


@pytest.mark.anyio("asyncio")
async def test_conversation_manager_streams_events_and_final_response():
    events = [
        DummyEvent("response.output_text.delta", delta="Hello"),
        DummyEvent("response.output_text.delta", delta=" world"),
        DummyEvent("response.completed"),
    ]
    final_response = {"output": "Hello world"}
    dummy_client = DummyOpenAIClient(events, final_response)
    manager = ConversationManager(client=dummy_client)

    collected = []
    async for chunk in manager.send_message(model="gpt-5-nano", input="Hi there"):
        collected.append(chunk)

    assert [c["type"] for c in collected] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert collected[0]["delta"] == "Hello"
    assert collected[1]["delta"] == " world"
    assert collected[2]["response"] == final_response
    assert dummy_client.responses.calls == [
        {"model": "gpt-5-nano", "input": "Hi there"}
    ]


@pytest.mark.anyio("asyncio")
async def test_conversation_manager_stream_raises_on_error_event():
    events = [
        DummyEvent("response.output_text.delta", delta="Hello"),
        DummyEvent("response.error", error={"message": "boom"}),
    ]
    dummy_client = DummyOpenAIClient(events, {})
    manager = ConversationManager(client=dummy_client)

    with pytest.raises(ConversationStreamError) as exc_info:
        async for _ in manager.send_message(model="gpt-4.1", input="Hello"):
            pass

    assert "boom" in str(exc_info.value)


@pytest.mark.anyio("asyncio")
async def test_metadata_payload_passed_to_roleplay_chat_server():
    from chatkit_server.metadata import build_metadata_payload
    from chatkit_server.server import RoleplayChatServer
    from chatkit_server.streaming import stream_chatkit_tokens

    class DummyStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class DummyResponsesClient:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def stream(self, **kwargs):
            self.calls.append(kwargs)
            return DummyStream()

    class DummyClient:
        def __init__(self) -> None:
            self.responses = DummyResponsesClient()

    async def noop(_: str) -> None:
        return None

    client = DummyClient()
    server = RoleplayChatServer(client)

    metadata = build_metadata_payload(
        conversation_id=321,
        user_id=654,
        request_id=789,
        assistant_id="asst-1",
        openai_conversation_id="conv-remote",
        thread_id="thread-1",
    )

    await stream_chatkit_tokens(
        server,
        model="gpt-test",
        input_data=[],
        metadata=metadata,
        on_delta=noop,
    )

    assert client.responses.calls, "RoleplayChatServer.respond should invoke the responses client"
    call = client.responses.calls[0]
    recorded_metadata = call["metadata"]
    assert recorded_metadata["conversation_id"] == "321"
    assert isinstance(recorded_metadata["conversation_id"], str)
    assert recorded_metadata["user_id"] == "654"
    assert isinstance(recorded_metadata["user_id"], str)


@pytest.mark.anyio("asyncio")
async def test_slow_path_scene_seal_updates_on_location_change(monkeypatch):
    conn = SealAwareConnection({"event_id": 10, "applied": True, "replayed": False})
    captured = []

    async def _capture_scene_seal(
        _conn,
        *,
        conversation_id: int,
        venue: Optional[str],
        date: Optional[str],
        non_negotiables,
        source: str,
    ):
        captured.append(
            {
                "conversation_id": conversation_id,
                "venue": venue,
                "date": date,
                "non_negotiables": list(non_negotiables or []),
                "source": source,
            }
        )
        return None

    monkeypatch.setattr(
        "logic.universal_updater_agent.ensure_scene_seal_item",
        _capture_scene_seal,
    )

    updates = {
        "npc_updates": [{"npc_id": 7, "current_location": "Velvet Sanctum"}],
        "roleplay_updates": {
            "CurrentLocation": "Velvet Sanctum",
            "CurrentTime": "Dawn",
        },
        "session_constraints": {"non_negotiables": ["Obey", "Trust"]},
    }

    result = await apply_universal_updates_async(
        ctx=None,
        user_id=5,
        conversation_id=9,
        updates=updates,
        conn=conn,
    )

    assert result["success"] is True
    assert captured
    call = captured[0]
    assert call["conversation_id"] == 9
    assert call["venue"] == "Velvet Sanctum"
    assert call["date"] == "Dawn"
    assert call["non_negotiables"] == ["Obey", "Trust"]
    assert call["source"] == "universal_updates"


@pytest.mark.anyio("asyncio")
async def test_slow_path_scene_seal_updates_on_time_jump(monkeypatch):
    conn = SealAwareConnection({"event_id": 11, "applied": True, "replayed": False})
    captured = []

    async def _capture_scene_seal(
        _conn,
        *,
        conversation_id: int,
        venue: Optional[str],
        date: Optional[str],
        non_negotiables,
        source: str,
    ):
        captured.append(
            {
                "conversation_id": conversation_id,
                "venue": venue,
                "date": date,
                "non_negotiables": list(non_negotiables or []),
                "source": source,
            }
        )
        return None

    monkeypatch.setattr(
        "logic.universal_updater_agent.ensure_scene_seal_item",
        _capture_scene_seal,
    )

    updates = {
        "npc_updates": [{"npc_id": 3, "current_location": "Velvet Sanctum"}],
        "roleplay_updates": {"CurrentTime": "Midnight"},
        "metadata": {
            "scene_seal": {
                "venue": "Velvet Sanctum",
                "non_negotiables": ["No safe word"],
            }
        },
    }

    result = await apply_universal_updates_async(
        ctx=None,
        user_id=2,
        conversation_id=6,
        updates=updates,
        conn=conn,
    )

    assert result["success"] is True
    assert captured
    call = captured[0]
    assert call["conversation_id"] == 6
    assert call["venue"] == "Velvet Sanctum"
    assert call["date"] == "Midnight"
    assert call["non_negotiables"] == ["No safe word"]
    assert call["source"] == "universal_updates"
