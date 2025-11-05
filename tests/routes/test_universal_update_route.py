from __future__ import annotations

from typing import Any, Dict

import pytest
from quart import Quart

import routes.universal_update as universal_update


class _StubConnectionContext:
    def __init__(self, conn: Any):
        self._conn = conn
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> Any:
        self.entered = True
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.exited = True
        return False


@pytest.fixture
def app():
    app = Quart(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(universal_update.universal_bp)
    return app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_universal_update_route_initializes_context(monkeypatch, app):
    calls: Dict[str, Any] = {}

    class DummyContext:
        def __init__(self, user_id: int, conversation_id: int) -> None:
            calls["ctx_args"] = (user_id, conversation_id)

        async def initialize(self) -> None:
            calls["ctx_initialized"] = True

    async def fake_apply(ctx, user_id, conversation_id, updates, conn):
        calls["apply_args"] = (ctx, user_id, conversation_id, updates, conn)
        return {"success": True, "updates_applied": 1}

    stub_conn = object()
    conn_ctx = _StubConnectionContext(stub_conn)

    monkeypatch.setattr(
        universal_update,
        "UniversalUpdaterContext",
        DummyContext,
    )
    monkeypatch.setattr(
        universal_update,
        "apply_universal_updates_async",
        fake_apply,
    )
    monkeypatch.setattr(
        universal_update,
        "get_db_connection_context",
        lambda: conn_ctx,
    )

    test_client = app.test_client()
    async with test_client.session_transaction() as session:
        session["user_id"] = 7

    response = await test_client.post(
        "/universal_update",
        json={"conversation_id": 9, "payload": {"foo": "bar"}},
    )

    assert response.status_code == 200
    payload = await response.get_json()
    assert payload == {"success": True, "updates_applied": 1}

    assert calls["ctx_args"] == (7, 9)
    assert calls["ctx_initialized"] is True
    ctx, user_id, conversation_id, updates, conn = calls["apply_args"]
    assert isinstance(ctx, DummyContext)
    assert user_id == 7
    assert conversation_id == 9
    assert updates == {"conversation_id": 9, "payload": {"foo": "bar"}}
    assert conn is stub_conn
    assert conn_ctx.entered is True
    assert conn_ctx.exited is True

