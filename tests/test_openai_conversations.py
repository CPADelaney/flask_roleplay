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
        "metadata": {"foo": "bar"},
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
        {"foo": "bar"},
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
        "metadata": {},
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
        "metadata": {},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    conn = StubConnection([None, inserted_row])

    result = await conversations.get_or_create_conversation(
        user_id=4,
        conversation_id=77,
        openai_assistant_id="new",
        openai_thread_id="thread-new",
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
    assert insert_params[-1] == {}
