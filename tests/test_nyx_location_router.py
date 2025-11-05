import asyncio
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from nyx.conversation.snapshot_store import ConversationSnapshotStore
from nyx.location import router as location_router
from nyx.location.types import ResolveResult, STATUS_NOT_FOUND, STATUS_ASK


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_resolve_place_or_travel_uses_disneyland_shortcut(monkeypatch):
    store = ConversationSnapshotStore()
    user_id = "42"
    conversation_id = "7"
    store.put(
        user_id,
        conversation_id,
        {
            "location_name": location_router.DISNEYLAND_PARK_SHORTCUT["name"],
            "place_id": location_router.DISNEYLAND_PARK_SHORTCUT["place_id"],
        },
    )

    meta = {
        "world": {
            "type": "real",
            "primary_city": location_router.DISNEYLAND_PARK_SHORTCUT["city"],
            "region": location_router.DISNEYLAND_PARK_SHORTCUT["region"],
            "country": location_router.DISNEYLAND_PARK_SHORTCUT["country"],
        },
        "locationInfo": {
            "geo": {
                "lat": location_router.DISNEYLAND_PARK_SHORTCUT["lat"],
                "lon": location_router.DISNEYLAND_PARK_SHORTCUT["lon"],
                "city": location_router.DISNEYLAND_PARK_SHORTCUT["city"],
                "region": location_router.DISNEYLAND_PARK_SHORTCUT["region"],
                "country": location_router.DISNEYLAND_PARK_SHORTCUT["country"],
            }
        },
        "currentRoleplay": {
            "CurrentLocation": {
                "name": location_router.DISNEYLAND_PARK_SHORTCUT["name"],
                "city": location_router.DISNEYLAND_PARK_SHORTCUT["city"],
                "region": location_router.DISNEYLAND_PARK_SHORTCUT["region"],
                "country": location_router.DISNEYLAND_PARK_SHORTCUT["country"],
                "place_id": location_router.DISNEYLAND_PARK_SHORTCUT["place_id"],
            }
        },
    }

    gemini_called = False

    async def fake_resolve_location_with_gemini(*_args, **_kwargs):
        nonlocal gemini_called
        gemini_called = True
        return ResolveResult(status=STATUS_NOT_FOUND)

    class DummyContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_get_or_create_location(*_args, **_kwargs):
        class DummyLocation:
            id = 101
            location_name = location_router.DISNEYLAND_PARK_SHORTCUT["name"]
            location_type = "venue"

        return DummyLocation()

    async def fake_track_player_movement(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        location_router,
        "resolve_location_with_gemini",
        fake_resolve_location_with_gemini,
    )
    monkeypatch.setattr(
        location_router,
        "get_db_connection_context",
        lambda: DummyContext(),
    )
    monkeypatch.setattr(
        location_router,
        "get_or_create_location",
        fake_get_or_create_location,
    )
    monkeypatch.setattr(
        location_router,
        "_track_player_movement",
        fake_track_player_movement,
    )

    result = await location_router.resolve_place_or_travel(
        "I go to Disneyland Park",
        meta,
        store,
        user_id,
        conversation_id,
    )

    await asyncio.sleep(0)

    assert result.status == "exact"
    assert result.candidates
    top_candidate = result.candidates[0]
    assert top_candidate.place.key == location_router.DISNEYLAND_PARK_SHORTCUT["place_id"]
    assert top_candidate.place.address["city"] == location_router.DISNEYLAND_PARK_SHORTCUT["city"]
    assert result.metadata["router"]["disneyland_shortcut"] is True
    assert any(op.get("op") == "poi.navigate" for op in result.operations)
    assert gemini_called is False


@pytest.mark.anyio("asyncio")
async def test_real_chain_preserves_ask_payload(monkeypatch):
    store = ConversationSnapshotStore()
    user_id = "101"
    conversation_id = "202"
    meta = {"world": {"type": "real"}}

    class DummyGeoAnchor:
        lat = 37.0
        lon = -122.0
        city = "Testville"
        region = "Test Region"
        country = "Testland"
        label = "Testville"

    async def fake_derive_geo_anchor(meta_arg, user_id_arg, conversation_id_arg):
        return DummyGeoAnchor()

    async def fake_resolve_location_with_gemini(*_args, **_kwargs):
        return ResolveResult(status=STATUS_NOT_FOUND)

    ask_message = "Need a specific terminal before I can confirm."
    ask_operations = [{"op": "clarify", "prompt": "Which terminal are you heading to?"}]
    ask_result = ResolveResult(
        status=STATUS_ASK,
        message=ask_message,
        operations=list(ask_operations),
        scope="real",
        anchor=location_router.Anchor(scope="real"),
    )

    async def fake_resolve_real(*_args, **_kwargs):
        return ask_result

    enqueue_calls = []

    def fake_enqueue_fictional_fallback(**kwargs):
        enqueue_calls.append(kwargs)

    monkeypatch.setattr(location_router, "derive_geo_anchor", fake_derive_geo_anchor)
    monkeypatch.setattr(
        location_router,
        "resolve_location_with_gemini",
        fake_resolve_location_with_gemini,
    )
    monkeypatch.setattr(location_router, "resolve_real", fake_resolve_real)
    monkeypatch.setattr(
        location_router.place_enrichment,
        "enqueue_fictional_fallback",
        fake_enqueue_fictional_fallback,
    )

    result = await location_router.resolve_place_or_travel(
        "Head to the international terminal",
        meta,
        store,
        user_id,
        conversation_id,
    )

    assert result is ask_result
    assert result.status == STATUS_ASK
    assert result.message == ask_message
    assert result.operations == ask_operations
    assert enqueue_calls
