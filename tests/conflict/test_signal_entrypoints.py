import os
import asyncio
from contextlib import asynccontextmanager

import pytest
from quart import Quart

os.environ.setdefault("OPENAI_API_KEY", "test")

from conflict.signals import ConflictSignalType
from routes import conflict_routes
from routes.conflict_routes import conflict_bp
from logic.conflict_system.signal_tasks import (
    dispatch_tension_update_signal,
    generate_scene_conflict_context,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"
from routes.conflict_signal_helpers import (
    emit_daily_update_signal,
    emit_end_of_day_signal,
    emit_player_action_signal,
)


class _StubSynthesizer:
    def __init__(self):
        self.signals = []
        self.invalidated_location = None

    async def handle_signal(self, signal):
        self.signals.append(signal)

    async def _invalidate_caches_for_scene(self, location_id):
        self.invalidated_location = location_id

    async def get_scene_bundle(self, scope):
        return {
            'conflicts': [{'id': 1}],
            'active_tensions': {'npc': 0.2},
            'opportunities': ['hook'],
            'ambient_effects': ['tense air'],
            'world_tension': 0.5,
        }


async def test_daily_update_signal_emitted_when_time_advances():
    synthesizer = _StubSynthesizer()
    time_result = {
        'time_advanced': True,
        'new_time': {'day': 2, 'time_of_day': 'morning'},
    }

    update = await emit_daily_update_signal(synthesizer, 1, 2, time_result)

    assert update == {'daily_update_triggered': True}
    assert len(synthesizer.signals) == 1
    signal = synthesizer.signals[0]
    assert signal.type is ConflictSignalType.TIME_TICK
    assert signal.payload['type'] == 'daily_update'
    assert signal.payload['new_day'] == 2


async def test_player_action_signal_payload_contains_activity_context():
    synthesizer = _StubSynthesizer()
    impact = await emit_player_action_signal(
        synthesizer,
        user_id=4,
        conversation_id=7,
        activity_type='conversation',
        user_input='hello there',
        context={'location': 'lounge'},
        npc_responses=[{'npc_id': 9}, {'npc_id': None}],
    )

    assert impact == {'processed': True, 'responses': 0}
    assert len(synthesizer.signals) == 1
    signal = synthesizer.signals[0]
    assert signal.type is ConflictSignalType.PLAYER_ACTION
    assert signal.payload['activity_type'] == 'conversation'
    assert signal.payload['involved_npcs'] == [9]


async def test_end_of_day_signal_uses_time_tick():
    synthesizer = _StubSynthesizer()
    await emit_end_of_day_signal(synthesizer, 5, 11, 2045, 3, 17)

    assert len(synthesizer.signals) == 1
    signal = synthesizer.signals[0]
    assert signal.type is ConflictSignalType.TIME_TICK
    assert signal.payload['type'] == 'end_of_day'
    assert signal.payload['day'] == 17


async def test_scene_transition_route_emits_signal(monkeypatch):
    app = Quart(__name__)
    app.secret_key = 'testing'
    app.register_blueprint(conflict_bp)

    synthesizer = _StubSynthesizer()

    async def fake_get_synthesizer(user_id, conversation_id):
        return synthesizer

    async def fake_check_permission(**kwargs):
        return {'approved': True}

    async def fake_on_scene_transition(user_id, conversation_id, old_scene, new_scene):
        return {'context': True}

    monkeypatch.setattr(conflict_routes, 'get_synthesizer', fake_get_synthesizer)
    monkeypatch.setattr(conflict_routes, 'check_conflict_permission', fake_check_permission)
    from logic.conflict_system import integration_hooks

    monkeypatch.setattr(
        integration_hooks.ConflictEventHooks,
        'on_scene_transition',
        staticmethod(fake_on_scene_transition),
    )

    async with app.test_client() as client:
        async with client.session_transaction() as sess:
            sess['user_id'] = 1

        response = await client.post(
            '/api/conflict/scene-transition',
            json={
                'conversation_id': 42,
                'old_scene': {'location_id': 1},
                'new_scene': {'location_id': 2},
            },
        )

        assert response.status_code == 200

    assert len(synthesizer.signals) == 1
    signal = synthesizer.signals[0]
    assert signal.type is ConflictSignalType.SCENE_ENTERED
    assert signal.scene_scope == {'location_id': 2}


async def test_generate_scene_conflict_context_uses_scene_signal():
    synthesizer = _StubSynthesizer()

    scene_info = {'location_id': 3, 'npcs': [5, 6]}

    context = await generate_scene_conflict_context(
        synthesizer,
        user_id=1,
        conversation_id=2,
        scene_info=scene_info,
    )

    assert len(synthesizer.signals) == 1
    signal = synthesizer.signals[0]
    assert signal.type is ConflictSignalType.SCENE_ENTERED
    assert context['conflicts'] == [{'id': 1}]
    assert context['tensions'] == {'npc': 0.2}


async def test_dispatch_tension_update_signal_emits_time_tick():
    synthesizer = _StubSynthesizer()

    await dispatch_tension_update_signal(synthesizer, 7, 8)

    assert len(synthesizer.signals) == 1
    signal = synthesizer.signals[0]
    assert signal.type is ConflictSignalType.TIME_TICK
    assert signal.payload['update_tensions'] is True
