from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import sys

import pytest
from quart import Quart

sys.path.append(str(Path(__file__).resolve().parents[2]))

import routes.multiuser_routes as multiuser_routes


class StubConnection:
    def __init__(self, fetchrow_results=None, fetch_results=None):
        self.fetchrow_results = deque(fetchrow_results or [])
        self.fetch_results = deque(fetch_results or [])
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetch_calls: list[tuple[str, tuple]] = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if self.fetchrow_results:
            return self.fetchrow_results.popleft()
        return None

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        if self.fetch_results:
            return self.fetch_results.popleft()
        return []

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return None


class StubConnectionContext:
    def __init__(self, connection: StubConnection):
        self._connection = connection

    async def __aenter__(self) -> StubConnection:
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture
def app():
    app = Quart(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(multiuser_routes.multiuser_bp, url_prefix="/multiuser")
    return app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_get_messages_allows_string_user_id(monkeypatch, app):
    message_time = datetime(2024, 1, 1, 12, 0, 0)
    connection = StubConnection(
        fetchrow_results=deque([
            {"user_id": 123},
        ]),
        fetch_results=deque([
            [
                {"sender": "user", "content": "Hello", "created_at": message_time},
            ]
        ]),
    )

    monkeypatch.setattr(
        multiuser_routes,
        "get_db_connection_context",
        lambda: StubConnectionContext(connection),
    )

    test_client = app.test_client()
    async with test_client.session_transaction() as session:
        session["user_id"] = "123"

    response = await test_client.get("/multiuser/conversations/77/messages")
    assert response.status_code == 200
    payload = await response.get_json()
    assert payload["messages"] == [
        {"sender": "user", "content": "Hello", "created_at": message_time.isoformat()}
    ]
    assert payload["has_more"] is False
    # Default pagination should request offset 0 and default limit + 1 for the sentinel row
    assert connection.fetch_calls[0][1] == (77, 0, 51)


@pytest.mark.anyio
async def test_add_message_accepts_string_user_id(monkeypatch, app):
    connection = StubConnection(
        fetchrow_results=deque([
            {"user_id": 123},
        ])
    )

    monkeypatch.setattr(
        multiuser_routes,
        "get_db_connection_context",
        lambda: StubConnectionContext(connection),
    )

    test_client = app.test_client()
    async with test_client.session_transaction() as session:
        session["user_id"] = "123"

    response = await test_client.post(
        "/multiuser/conversations/5/messages",
        json={"sender": "user", "content": "Howdy"},
    )

    assert response.status_code == 200
    assert connection.execute_calls, "Message insert should be executed"


@pytest.mark.anyio
async def test_move_folder_auto_create_uses_normalized_user_id(monkeypatch, app):
    connection = StubConnection(
        fetchrow_results=deque([
            {"user_id": 123},
            {"id": 9},
        ])
    )

    monkeypatch.setattr(
        multiuser_routes,
        "get_db_connection_context",
        lambda: StubConnectionContext(connection),
    )

    test_client = app.test_client()
    async with test_client.session_transaction() as session:
        session["user_id"] = "123"

    response = await test_client.post(
        "/multiuser/conversations/3/move_folder",
        json={"folder_name": "Quests"},
    )

    assert response.status_code == 200
    # The folder lookup should receive an integer user id
    assert connection.fetchrow_calls[1][1][0] == 123
    # Conversation update should write the folder id from the lookup
    assert connection.execute_calls
    assert connection.execute_calls[-1][1][0] == 9


@pytest.mark.anyio
async def test_get_messages_paginates_results(monkeypatch, app):
    times = [
        datetime(2024, 1, 1, 12, 0, 0),
        datetime(2024, 1, 1, 12, 1, 0),
        datetime(2024, 1, 1, 12, 2, 0),
    ]
    connection = StubConnection(
        fetchrow_results=deque([
            {"user_id": 123},
            {"user_id": 123},
        ]),
        fetch_results=deque([
            [
                {"sender": "user", "content": "Hello", "created_at": times[0]},
                {"sender": "npc", "content": "Hi there", "created_at": times[1]},
                {"sender": "user", "content": "Follow-up", "created_at": times[2]},
            ],
            [
                {"sender": "user", "content": "Follow-up", "created_at": times[2]},
            ],
        ]),
    )

    monkeypatch.setattr(
        multiuser_routes,
        "get_db_connection_context",
        lambda: StubConnectionContext(connection),
    )

    test_client = app.test_client()
    async with test_client.session_transaction() as session:
        session["user_id"] = 123

    first_response = await test_client.get(
        "/multiuser/conversations/77/messages",
        query_string={"limit": 2},
    )
    assert first_response.status_code == 200
    first_payload = await first_response.get_json()
    assert [m["content"] for m in first_payload["messages"]] == ["Hello", "Hi there"]
    assert first_payload["has_more"] is True
    assert connection.fetch_calls[0][1] == (77, 0, 3)

    second_response = await test_client.get(
        "/multiuser/conversations/77/messages",
        query_string={"offset": 2, "limit": 2},
    )
    assert second_response.status_code == 200
    second_payload = await second_response.get_json()
    assert [m["content"] for m in second_payload["messages"]] == ["Follow-up"]
    assert second_payload["has_more"] is False
    assert connection.fetch_calls[1][1] == (77, 2, 3)
