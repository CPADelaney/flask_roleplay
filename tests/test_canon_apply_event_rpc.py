from uuid import UUID

import pytest

from db.rpc import CanonEventError, write_event
from logic.universal_delta import DeltaBuildError, build_delta_from_legacy_payload


pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeTransaction:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class _FakeConnection:
    def __init__(self, result):
        self._result = result
        self.calls = []
        self.transaction_manager = _FakeTransaction()

    def transaction(self):
        self.calls.append("transaction")
        return self.transaction_manager

    async def fetchrow(self, query, payload):
        self.calls.append((query, payload))
        return {"result": self._result}


async def test_write_event_happy_path():
    delta = build_delta_from_legacy_payload(
        user_id=1,
        conversation_id=2,
        payload={
            "narrative": "NPC moved to the plaza.",
            "npc_updates": [{"npc_id": 99, "current_location": "Plaza"}],
            "social_links": [
                {
                    "entity1_type": "npc",
                    "entity1_id": 99,
                    "entity2_type": "player",
                    "entity2_id": 1,
                    "level_change": 2,
                }
            ],
        },
    )

    conn = _FakeConnection({"event_id": 5, "applied": True, "replayed": False})
    result = await write_event(conn, delta)

    assert result["event_id"] == 5
    assert conn.transaction_manager.entered and conn.transaction_manager.exited
    assert any(isinstance(call, tuple) for call in conn.calls)


async def test_write_event_rejects_invalid_response():
    delta = build_delta_from_legacy_payload(
        user_id=1,
        conversation_id=2,
        payload={
            "narrative": "Narrative text",
            "npc_updates": [{"npc_id": 7, "current_location": "Garden"}],
        },
    )

    class _BadConnection(_FakeConnection):
        async def fetchrow(self, query, payload):
            return None

    conn = _BadConnection(None)
    with pytest.raises(CanonEventError):
        await write_event(conn, delta)


async def test_write_event_idempotent_replay():
    delta = build_delta_from_legacy_payload(
        user_id=3,
        conversation_id=4,
        payload={
            "npc_updates": [{"npc_id": 3, "current_location": "Cafe"}],
        },
        request_id=UUID("11111111-1111-1111-1111-111111111111"),
    )

    conn = _FakeConnection({"event_id": 42, "applied": False, "replayed": True})
    result = await write_event(conn, delta)

    assert result["replayed"] is True
    assert result["event_id"] == 42


def test_build_delta_rejects_empty_payload():
    with pytest.raises(DeltaBuildError):
        build_delta_from_legacy_payload(user_id=1, conversation_id=2, payload={})
