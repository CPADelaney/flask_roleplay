import asyncio
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from logic.conflict_system.conflict_flow import (
    ConflictFlow,
    ConflictFlowSubsystem,
    ConflictPhase,
    DramaticBeat,
    PacingStyle,
)


class _DummyConn:
    async def execute(self, *args: Any, **kwargs: Any) -> None:
        return None


class _DummyContext:
    async def __aenter__(self) -> _DummyConn:
        return _DummyConn()

    async def __aexit__(self, exc_type, exc: Any, tb: Any) -> None:
        return None


def test_initialize_conflict_flow_uses_cached_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        subsystem = ConflictFlowSubsystem(user_id=7, conversation_id=9)

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow.get_db_connection_context",
            lambda: _DummyContext(),
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow._orch_types",
            lambda: (
                SimpleNamespace(FLOW="flow"),
                SimpleNamespace(PHASE_TRANSITION="phase_transition", INTENSITY_CHANGED="intensity_changed"),
                lambda **kwargs: SimpleNamespace(**kwargs),
                lambda **kwargs: SimpleNamespace(**kwargs),
            ),
        )

        recorded_payload: Dict[str, Any] = {}

        def _queue_init(conflict_id: int, user_id: int, conversation_id: int, conflict_type: str, context: Dict[str, Any], **_: Any) -> None:
            recorded_payload.update(
                {
                    "conflict_id": conflict_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "conflict_type": conflict_type,
                    "context": context,
                }
            )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.queue_flow_initialization",
            _queue_init,
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.get_cached_flow_bootstrap",
            lambda conflict_id: {
                "phase": "rising",
                "pacing": "rapid_escalation",
                "intensity": 0.65,
                "momentum": 0.4,
                "conditions": ["crowd unrest"],
            },
        )

        flow = await subsystem.initialize_conflict_flow(11, "urgent", {"location": "plaza"})

        assert flow.current_phase == ConflictPhase.RISING
        assert flow.pacing_style == PacingStyle.RAPID_ESCALATION
        assert pytest.approx(flow.intensity, rel=1e-3) == 0.65
        assert pytest.approx(flow.momentum, rel=1e-3) == 0.4
        assert recorded_payload["conflict_id"] == 11
        assert recorded_payload["user_id"] == 7
        assert recorded_payload["conversation_id"] == 9

    asyncio.run(_run())


def test_handle_conflict_updated_merges_cached_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        subsystem = ConflictFlowSubsystem(user_id=3, conversation_id=4)

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow.get_db_connection_context",
            lambda: _DummyContext(),
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow._orch_types",
            lambda: (
                SimpleNamespace(FLOW="flow"),
                SimpleNamespace(PHASE_TRANSITION="phase_transition", INTENSITY_CHANGED="intensity_changed"),
                lambda **kwargs: SimpleNamespace(**kwargs),
                lambda **kwargs: SimpleNamespace(**kwargs),
            ),
        )

        flow = ConflictFlow(
            conflict_id=21,
            current_phase=ConflictPhase.EMERGING,
            pacing_style=PacingStyle.STEADY,
            intensity=0.3,
            momentum=0.1,
            phase_progress=0.2,
            transitions_history=[],
            dramatic_beats=[],
            next_transition_conditions=[],
        )
        subsystem._flow_states[21] = flow

        event_payload = {
            "conflict_id": 21,
            "event_id": "evt-123",
            "intensity_delta": 0.05,
            "momentum_delta": 0.02,
            "progress_delta": 0.15,
        }

        queued_events: Dict[str, Any] = {}

        def _queue_event(conflict_id: int, *_args: Any, **kwargs: Any) -> None:
            queued_events["conflict_id"] = conflict_id
            queued_events["payload"] = kwargs

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.queue_flow_event_analysis",
            _queue_event,
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.get_cached_event_analysis",
            lambda _conflict_id, _event_id: {
                "intensity": 0.55,
                "momentum": 0.3,
                "progress_change": 0.6,
                "should_transition": True,
                "narrative_impact": "dramatic surge",
            },
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.get_cached_transition_payload",
            lambda _cid: {
                "to_phase": "rising",
                "transition_type": "triggered",
                "text": "Energy spikes throughout the crowd.",
            },
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.queue_phase_narration",
            lambda *args, **kwargs: None,
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.get_cached_transition_text",
            lambda _cid: "Energy spikes throughout the crowd.",
        )

        monkeypatch.setattr(
            "logic.conflict_system.conflict_flow_hotpath.should_trigger_beat",
            lambda _flow: "twist",
        )

        generated_beats: Dict[str, DramaticBeat] = {}

        async def _fake_generate(flow_obj: ConflictFlow, ctx: Dict[str, Any]) -> DramaticBeat:
            beat = DramaticBeat(
                beat_type=ctx.get("beat_type", "twist"),
                description="A twist takes everyone by surprise",
                impact_on_flow=0.25,
                characters_involved=[],
                timestamp=datetime.now(),
            )
            generated_beats["beat"] = beat
            return beat

        monkeypatch.setattr(subsystem, "generate_dramatic_beat", _fake_generate)
        monkeypatch.setattr(subsystem, "_store_transition", lambda *_args, **_kwargs: asyncio.sleep(0))
        monkeypatch.setattr(subsystem, "_save_flow_state", lambda *_args, **_kwargs: asyncio.sleep(0))
        async def _fake_next_phase(*_args: Any, **_kwargs: Any) -> ConflictPhase:
            return ConflictPhase.RISING

        monkeypatch.setattr(subsystem, "_determine_next_phase", _fake_next_phase)

        response = await subsystem._handle_conflict_updated(
            SimpleNamespace(payload=event_payload, event_id="evt-123", event_type=None)
        )

        assert response.data["transition_occurred"] is True
        assert response.data["beat_generated"] is True
        assert response.data["narrative_impact"] == "dramatic surge"
        assert queued_events["conflict_id"] == 21
        assert "beat" in generated_beats

    asyncio.run(_run())
