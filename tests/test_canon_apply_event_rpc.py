from contextlib import asynccontextmanager
from uuid import UUID

import json
import pytest

from db.rpc import CanonEventError, write_event
from logic.universal_delta import DeltaBuildError, build_delta_from_legacy_payload
from logic import universal_updater_agent


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
    def __init__(self, result, *, expected_payload=None):
        self._result = result
        self.calls = []
        self.transaction_manager = _FakeTransaction()
        self.expected_payload = expected_payload
        self.decoded_payload = None

    def transaction(self):
        self.calls.append("transaction")
        return self.transaction_manager

    async def fetchrow(self, query, payload):
        assert isinstance(payload, str)
        decoded_payload = json.loads(payload)
        if self.expected_payload is not None:
            assert decoded_payload == self.expected_payload
        self.decoded_payload = decoded_payload
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

    conn = _FakeConnection(
        {"event_id": 5, "applied": True, "replayed": False},
        expected_payload=delta.model_dump(mode="json", by_alias=True),
    )
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
            await super().fetchrow(query, payload)
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

    conn = _FakeConnection(
        {"event_id": 42, "applied": False, "replayed": True},
        expected_payload=delta.model_dump(mode="json", by_alias=True),
    )
    result = await write_event(conn, delta)

    assert result["replayed"] is True
    assert result["event_id"] == 42


def test_build_delta_rejects_empty_payload():
    with pytest.raises(DeltaBuildError):
        build_delta_from_legacy_payload(user_id=1, conversation_id=2, payload={})


def test_build_delta_extracts_player_location_from_roleplay_updates():
    delta = build_delta_from_legacy_payload(
        user_id=11,
        conversation_id=22,
        payload={"roleplay_updates": {"CurrentLocation": "Velvet Sanctum"}},
    )

    assert delta.operation_count == 1
    op = delta.operations[0]
    assert op.type == "player.move"
    assert op.location_slug == "Velvet Sanctum"
    assert op.player_id == 11


def test_build_delta_handles_roleplay_updates_array_payload():
    delta = build_delta_from_legacy_payload(
        user_id=5,
        conversation_id=6,
        payload={
            "roleplay_updates": [
                {"key": "current_location", "value": "Obsidian Parlor"},
                {"key": "current_location_id", "value": 77},
            ]
        },
    )

    assert delta.operation_count == 1
    op = delta.operations[0]
    assert op.type == "player.move"
    assert op.location_slug == "Obsidian Parlor"
    assert op.location_id == 77


def test_build_delta_handles_roleplay_updates_array_field_payload():
    delta = build_delta_from_legacy_payload(
        user_id=9,
        conversation_id=10,
        payload={
            "roleplay_updates": [
                {"field": "CurrentLocation", "value": "New Scene"},
            ]
        },
    )

    assert delta.operation_count == 1
    op = delta.operations[0]
    assert op.type == "player.move"
    assert op.location_slug == "New Scene"
    assert op.player_id == 9


@pytest.mark.parametrize(
    "location_payload, expected_slug, expected_id",
    [
        ("Moonlit Plaza", "Moonlit Plaza", None),
        ({"slug": "Hidden Alcove", "id": "77"}, "Hidden Alcove", 77),
    ],
)
def test_build_delta_accepts_location_field(location_payload, expected_slug, expected_id):
    delta = build_delta_from_legacy_payload(
        user_id=21,
        conversation_id=22,
        payload={
            "npc_updates": [
                {
                    "npc_id": 303,
                    "location": location_payload,
                }
            ]
        },
    )

    assert delta.operation_count == 1
    op = delta.operations[0]
    assert op.type == "npc.move"
    assert op.npc_id == 303
    assert op.location_slug == expected_slug
    assert op.location_id == expected_id


def test_build_delta_relationship_updates_only_level_change():
    delta = build_delta_from_legacy_payload(
        user_id=13,
        conversation_id=14,
        payload={
            "relationship_updates": [
                {
                    "entity1_type": "npc",
                    "entity1_id": 7,
                    "entity2_type": "player",
                    "entity2_id": 13,
                    "level_change": -2,
                }
            ]
        },
    )

    assert delta.operation_count == 1
    op = delta.operations[0]
    assert op.type == "relationship.bump"
    assert op.delta == -2
    assert op.source_id == 7
    assert op.target_id == 13


def test_build_delta_relationship_updates_dimension_changes_only():
    delta = build_delta_from_legacy_payload(
        user_id=15,
        conversation_id=16,
        payload={
            "relationship_updates": [
                {
                    "entity1_type": "npc",
                    "entity1_id": 5,
                    "entity2_type": "npc",
                    "entity2_id": 6,
                    "dimension_changes": {"trust": 1.2, "respect": 0.8},
                }
            ]
        },
    )

    assert delta.operation_count == 1
    op = delta.operations[0]
    assert op.type == "relationship.bump"
    assert op.delta == 2
    assert op.source_id == 5
    assert op.target_id == 6


async def test_apply_universal_updates_impl_handles_field_payload(monkeypatch):
    class _Governor:
        async def check_action_permission(self, *args, **kwargs):
            return {"approved": True}

        async def process_agent_action_report(self, *args, **kwargs):
            return None

    class _Ctx:
        def __init__(self):
            self.user_id = 1
            self.conversation_id = 2
            self.governor = _Governor()

    class _RunCtx:
        def __init__(self):
            self.context = _Ctx()

    captured_delta = {}

    async def _fake_write_event(conn, delta):
        captured_delta["delta"] = delta
        return {"applied": True, "event_id": 5, "messages": []}

    @asynccontextmanager
    async def _fake_connection_ctx():
        yield object()

    async def _fake_ensure_scene_seal(*args, **kwargs):
        return None

    monkeypatch.setattr(universal_updater_agent.db_rpc, "write_event", _fake_write_event)
    monkeypatch.setattr(
        universal_updater_agent,
        "get_db_connection_context",
        _fake_connection_ctx,
    )
    monkeypatch.setattr(
        universal_updater_agent,
        "ensure_scene_seal_item",
        _fake_ensure_scene_seal,
    )

    result = await universal_updater_agent._apply_universal_updates_impl(
        _RunCtx(),
        {"roleplay_updates": [{"field": "CurrentLocation", "value": "New Scene"}]},
    )

    assert result["success"] is True
    assert result["updates_applied"] == 1
    assert "delta" in captured_delta
    delta = captured_delta["delta"]
    assert delta.operation_count == 1
    op = delta.operations[0]
    assert op.type == "player.move"
    assert op.location_slug == "New Scene"
    assert op.player_id == 1
