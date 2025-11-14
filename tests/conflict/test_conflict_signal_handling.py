from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from conflict.signals import ConflictSignal, ConflictSignalType
from logic.conflict_system import conflict_synthesizer


pytestmark = pytest.mark.anyio


@pytest.fixture
def synthesizer(monkeypatch):
    class DummyProcessor:
        def __init__(self) -> None:
            self._processing_queue: list[dict] = []

        async def process_queued_items(self, max_items: int = 5):  # pragma: no cover - fallback safety
            return []

        async def process_daily_updates(self):  # pragma: no cover - fallback safety
            return {}

    class DummyScheduler:
        def __init__(self) -> None:
            self.processor = DummyProcessor()

        def get_processor(self, user_id: int, conversation_id: int):
            return self.processor

    scheduler = DummyScheduler()
    monkeypatch.setattr(
        conflict_synthesizer,
        "get_conflict_scheduler",
        lambda: scheduler,
    )
    return conflict_synthesizer.ConflictSynthesizer(user_id=1, conversation_id=5)


@pytest.fixture
def anyio_backend():  # pragma: no cover - harness hint
    return "asyncio"


async def test_handle_signal_validates_target(synthesizer):
    signal = ConflictSignal(
        type=ConflictSignalType.PLAYER_ACTION,
        user_id=999,
        conversation_id=5,
        scene_scope=None,
        payload={},
    )

    with pytest.raises(ValueError):
        await synthesizer.handle_signal(signal)


async def test_handle_signal_scene_entered_tracks_last_scene(synthesizer):
    synthesizer._last_scene = {"location": "old"}
    synthesizer._handle_scene_transition = AsyncMock()

    scene_scope = {"location": "new"}
    signal = ConflictSignal(
        type=ConflictSignalType.SCENE_ENTERED,
        user_id=1,
        conversation_id=5,
        scene_scope=scene_scope,
    )

    await synthesizer.handle_signal(signal)

    synthesizer._handle_scene_transition.assert_awaited_once_with({"location": "old"}, scene_scope)
    assert synthesizer._last_scene == scene_scope


async def test_handle_signal_time_tick_with_day_calls_day_transition(synthesizer):
    synthesizer.handle_day_transition = AsyncMock()
    synthesizer.emit_event = AsyncMock()

    signal = ConflictSignal(
        type=ConflictSignalType.TIME_TICK,
        user_id=1,
        conversation_id=5,
        scene_scope=None,
        payload={"day": "7"},
    )

    await synthesizer.handle_signal(signal)

    synthesizer.handle_day_transition.assert_awaited_once_with(7)
    synthesizer.emit_event.assert_not_called()


async def test_handle_signal_time_tick_without_day_emits_state_sync(synthesizer):
    synthesizer.emit_event = AsyncMock()

    scene_scope = {"location": "guild"}
    signal = ConflictSignal(
        type=ConflictSignalType.TIME_TICK,
        user_id=1,
        conversation_id=5,
        scene_scope=scene_scope,
        payload={"phase": "dusk"},
    )

    await synthesizer.handle_signal(signal)

    synthesizer.emit_event.assert_awaited()
    event = synthesizer.emit_event.await_args[0][0]
    assert event.event_type == conflict_synthesizer.EventType.STATE_SYNC
    assert event.target_subsystems == {conflict_synthesizer.SubsystemType.BACKGROUND}
    assert event.payload["tick"]["phase"] == "dusk"
    assert event.payload["tick"]["user_id"] == 1
    assert "timestamp" in event.payload["tick"]
    assert event.payload.get("scene_context", {}).get("location") == "guild"


@pytest.mark.parametrize(
    ("signal_type", "expected_event", "expected_targets"),
    [
        (
            ConflictSignalType.PLAYER_ACTION,
            conflict_synthesizer.EventType.PLAYER_CHOICE,
            {
                conflict_synthesizer.SubsystemType.TENSION,
                conflict_synthesizer.SubsystemType.STAKEHOLDER,
                conflict_synthesizer.SubsystemType.SOCIAL,
            },
        ),
        (
            ConflictSignalType.FACT_BECAME_PUBLIC,
            conflict_synthesizer.EventType.CANON_ESTABLISHED,
            {
                conflict_synthesizer.SubsystemType.CANON,
                conflict_synthesizer.SubsystemType.BACKGROUND,
                conflict_synthesizer.SubsystemType.SOCIAL,
                conflict_synthesizer.SubsystemType.LEVERAGE,
            },
        ),
        (
            ConflictSignalType.RELATIONSHIP_CHANGE,
            conflict_synthesizer.EventType.STAKEHOLDER_ACTION,
            {
                conflict_synthesizer.SubsystemType.STAKEHOLDER,
                conflict_synthesizer.SubsystemType.SOCIAL,
            },
        ),
        (
            ConflictSignalType.CONFLICT_CREATED,
            conflict_synthesizer.EventType.CONFLICT_CREATED,
            None,
        ),
        (
            ConflictSignalType.CONFLICT_UPDATED,
            conflict_synthesizer.EventType.CONFLICT_UPDATED,
            None,
        ),
        (
            ConflictSignalType.CONFLICT_RESOLVED,
            conflict_synthesizer.EventType.CONFLICT_RESOLVED,
            None,
        ),
    ],
)
async def test_handle_signal_emits_expected_system_event(
    synthesizer,
    signal_type,
    expected_event,
    expected_targets,
):
    synthesizer.emit_event = AsyncMock()

    signal = ConflictSignal(
        type=signal_type,
        user_id=1,
        conversation_id=5,
        scene_scope=None,
        payload={"key": "value"},
    )

    await synthesizer.handle_signal(signal)

    synthesizer.emit_event.assert_awaited_once()
    event = synthesizer.emit_event.await_args[0][0]
    assert event.event_type == expected_event
    assert event.payload["key"] == "value"
    assert event.payload["user_id"] == 1
    assert event.payload["conversation_id"] == 5
    assert "timestamp" in event.payload

    if expected_targets is None:
        assert event.target_subsystems is None
    else:
        assert event.target_subsystems == expected_targets
