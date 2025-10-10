import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai_integration import conversations


@pytest.fixture
def anyio_backend():
    return "asyncio"


class StubConnection:
    def __init__(self, fetchrow_results):
        self.fetchrow_results = list(fetchrow_results)
        self.queries = []

    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        if self.fetchrow_results:
            return self.fetchrow_results.pop(0)
        return None


@pytest.mark.anyio("asyncio")
async def test_create_chatkit_thread_inserts_expected_columns():
    expected_row = {
        "id": 1,
        "conversation_id": 99,
        "chatkit_assistant_id": "asst-1",
        "chatkit_thread_id": "thread-1",
        "chatkit_run_id": "run-1",
        "status": "completed",
        "last_error": None,
        "metadata": {"response_id": "resp-1"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([expected_row])

    result = await conversations.create_chatkit_thread(
        conversation_id=99,
        chatkit_assistant_id="asst-1",
        chatkit_thread_id="thread-1",
        chatkit_run_id="run-1",
        status="completed",
        metadata={"response_id": "resp-1"},
        conn=conn,
    )

    assert result == expected_row
    assert len(conn.queries) == 1
    insert_query, params = conn.queries[0]
    assert "INSERT INTO chatkit_threads" in insert_query
    assert "ON CONFLICT (conversation_id, chatkit_thread_id)" in insert_query
    assert params == (
        99,
        "asst-1",
        "thread-1",
        "run-1",
        "completed",
        None,
        {"response_id": "resp-1"},
    )


@pytest.mark.anyio("asyncio")
async def test_get_or_create_chatkit_thread_returns_existing_row():
    existing_row = {
        "id": 5,
        "conversation_id": 42,
        "chatkit_assistant_id": "asst-existing",
        "chatkit_thread_id": "thread-existing",
        "chatkit_run_id": "run-existing",
        "status": "ready",
        "last_error": None,
        "metadata": {},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([existing_row])

    result = await conversations.get_or_create_chatkit_thread(
        conversation_id=42,
        chatkit_assistant_id="ignored",
        chatkit_thread_id="thread-existing",
        conn=conn,
    )

    assert result == existing_row
    assert len(conn.queries) == 1
    select_query, params = conn.queries[0]
    assert select_query.strip().startswith("SELECT")
    assert params == (42, "thread-existing")


@pytest.mark.anyio("asyncio")
async def test_get_or_create_chatkit_thread_inserts_when_missing():
    inserted_row = {
        "id": 6,
        "conversation_id": 77,
        "chatkit_assistant_id": "asst-new",
        "chatkit_thread_id": "thread-new",
        "chatkit_run_id": "run-new",
        "status": "pending",
        "last_error": None,
        "metadata": {},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([None, inserted_row])

    result = await conversations.get_or_create_chatkit_thread(
        conversation_id=77,
        chatkit_assistant_id="asst-new",
        chatkit_thread_id="thread-new",
        chatkit_run_id="run-new",
        conn=conn,
    )

    assert result == inserted_row
    assert len(conn.queries) == 2
    select_query, select_params = conn.queries[0]
    insert_query, insert_params = conn.queries[1]
    assert select_query.strip().startswith("SELECT")
    assert "INSERT INTO chatkit_threads" in insert_query
    assert insert_params == (
        77,
        "asst-new",
        "thread-new",
        "run-new",
        "pending",
        None,
        {},
    )


@pytest.mark.anyio("asyncio")
async def test_get_latest_chatkit_thread_returns_most_recent():
    latest_row = {
        "id": 10,
        "conversation_id": 11,
        "chatkit_assistant_id": "assistant",
        "chatkit_thread_id": "thread-latest",
        "chatkit_run_id": "run-latest",
        "status": "completed",
        "last_error": None,
        "metadata": {},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
    }
    conn = StubConnection([latest_row])

    result = await conversations.get_latest_chatkit_thread(
        conversation_id=11,
        conn=conn,
    )

    assert result == latest_row
    assert len(conn.queries) == 1
    query, params = conn.queries[0]
    assert "ORDER BY updated_at DESC" in query
    assert params == (11,)
