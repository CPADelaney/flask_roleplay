from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
from quart import Quart

sys.path.append(str(Path(__file__).resolve().parents[2]))

universal_updater_stub = types.ModuleType("logic.universal_updater_agent")
aggregator_stub = types.ModuleType("logic.aggregator_sdk")


class _StubUpdaterContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id


async def _stub_apply(ctx, user_id, conversation_id, data, conn):
    return {"success": True}


async def _stub_get_aggregated_roleplay_context(*args, **kwargs):
    return {"stub": True}


setattr(universal_updater_stub, "UniversalUpdaterContext", _StubUpdaterContext)
setattr(universal_updater_stub, "apply_universal_updates_async", _stub_apply)
setattr(aggregator_stub, "get_aggregated_roleplay_context", _stub_get_aggregated_roleplay_context)
sys.modules.setdefault("logic.universal_updater_agent", universal_updater_stub)
sys.modules.setdefault("logic.aggregator_sdk", aggregator_stub)

import routes.universal_update as universal_update_route


class StubConnection:
    def __init__(self):
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []


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
    app.register_blueprint(universal_update_route.universal_bp)
    return app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_universal_update_passes_context(monkeypatch, app):
    connection = StubConnection()
    monkeypatch.setattr(
        universal_update_route,
        "get_db_connection_context",
        lambda: StubConnectionContext(connection),
    )

    captured_call = {}

    async def fake_apply(ctx, user_id, conversation_id, data, conn):
        captured_call["args"] = (ctx, user_id, conversation_id, data, conn)
        return {"success": True}

    monkeypatch.setattr(
        universal_update_route,
        "apply_universal_updates_async",
        fake_apply,
    )

    test_client = app.test_client()
    async with test_client.session_transaction() as session:
        session["user_id"] = 7

    response = await test_client.post(
        "/universal_update",
        json={"conversation_id": "11", "updates": {"foo": "bar"}},
    )

    assert response.status_code == 200
    payload = await response.get_json()
    assert payload == {"success": True}

    assert "args" in captured_call
    ctx, user_id, conversation_id, data, conn = captured_call["args"]
    from logic.universal_updater_agent import UniversalUpdaterContext

    assert isinstance(ctx, UniversalUpdaterContext)
    assert user_id == 7
    assert conversation_id == 11
    assert data["conversation_id"] == "11"
    assert conn is connection
