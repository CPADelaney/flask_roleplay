import datetime as dt
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from quart import Quart

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODULE_PATH = _PROJECT_ROOT / "routes" / "multiuser_routes.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("routes.multiuser_routes", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
multiuser_routes = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = multiuser_routes
_MODULE_SPEC.loader.exec_module(multiuser_routes)
multiuser_bp = multiuser_routes.multiuser_bp


class _DummyConnection:
    def __init__(self, owner_id: int) -> None:
        self.owner_id = owner_id
        self.execute_calls: List[Tuple[str, Tuple[Any, ...]]] = []
        self.messages: List[Dict[str, Any]] = [
            {
                "sender": "user",
                "content": "Hello from the owner",
                "created_at": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            }
        ]

    async def fetchrow(self, query: str, *args: Any) -> Dict[str, Any]:
        query_upper = query.upper()
        if "FROM CONVERSATIONS" in query_upper:
            return {"user_id": self.owner_id}
        if "FROM FOLDERS" in query_upper:
            return {"user_id": self.owner_id}
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        return self.messages

    async def execute(self, query: str, *args: Any) -> None:
        self.execute_calls.append((query, args))


class _DummyConnectionContext:
    def __init__(self, connection: _DummyConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _DummyConnection:
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        return False


def _setup_app(monkeypatch: pytest.MonkeyPatch, connection: _DummyConnection) -> Quart:
    app = Quart(__name__)
    app.secret_key = "testing-secret"
    app.register_blueprint(multiuser_bp, url_prefix="/multiuser")

    monkeypatch.setattr(
        multiuser_routes,
        "get_db_connection_context",
        lambda: _DummyConnectionContext(connection),
    )

    return app


@pytest.mark.asyncio
async def test_get_messages_accepts_string_user_session(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _DummyConnection(owner_id=7)
    app = _setup_app(monkeypatch, connection)

    test_client = app.test_client()
    async with test_client.session_transaction() as test_session:
        test_session["user_id"] = "7"

    response = await test_client.get("/multiuser/conversations/42/messages")
    assert response.status_code == 200

    payload = await response.get_json()
    assert payload == {
        "messages": [
            {
                "sender": "user",
                "content": "Hello from the owner",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]
    }


@pytest.mark.asyncio
async def test_add_message_accepts_string_user_session(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _DummyConnection(owner_id=11)
    app = _setup_app(monkeypatch, connection)

    test_client = app.test_client()
    async with test_client.session_transaction() as test_session:
        test_session["user_id"] = "11"

    response = await test_client.post(
        "/multiuser/conversations/99/messages",
        json={"sender": "user", "content": "A new thought"},
    )
    assert response.status_code == 200

    payload = await response.get_json()
    assert payload == {"status": "ok"}

    assert len(connection.execute_calls) == 1
    executed_query, executed_args = connection.execute_calls[0]
    assert "INSERT INTO messages" in executed_query
    assert executed_args == (99, "user", "A new thought")
