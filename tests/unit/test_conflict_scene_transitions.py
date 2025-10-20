import os
import sys
import types
import typing
from pathlib import Path

import pytest
from typing_extensions import TypedDict as _CompatTypedDict

os.environ.setdefault("OPENAI_API_KEY", "test-key")

stub_sentence_transformers = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):  # pragma: no cover - simple shim
        return []

    def get_sentence_embedding_dimension(self):  # pragma: no cover - shim
        return 384


stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
stub_util_module = types.ModuleType("sentence_transformers.util")
stub_util_module.cos_sim = lambda *args, **kwargs: 0.0  # pragma: no cover - shim
stub_util_module.pairwise_cos_sim = lambda *args, **kwargs: 0.0  # pragma: no cover - shim
stub_sentence_transformers.util = stub_util_module
stub_models_module = types.ModuleType("sentence_transformers.models")
stub_models_module.Transformer = lambda *args, **kwargs: None  # pragma: no cover - shim
stub_models_module.Pooling = lambda *args, **kwargs: None  # pragma: no cover - shim
stub_sentence_transformers.models = stub_models_module

sys.modules['sentence_transformers'] = stub_sentence_transformers
sys.modules['sentence_transformers.util'] = stub_util_module
sys.modules['sentence_transformers.models'] = stub_models_module

stub_dynamic_template = types.ModuleType("logic.conflict_system.dynamic_conflict_template")
stub_dynamic_template.extract_runner_response = lambda *args, **kwargs: {}
sys.modules['logic.conflict_system.dynamic_conflict_template'] = stub_dynamic_template

stub_vector_store = types.ModuleType("nyx.core.memory.vector_store")
sys.modules['nyx.core.memory.vector_store'] = stub_vector_store

sys.path.append(str(Path(__file__).resolve().parents[2]))

typing.TypedDict = _CompatTypedDict

from logic.conflict_system.background_grand_conflicts import BackgroundConflictSubsystem
from logic.conflict_system.conflict_synthesizer import ConflictSynthesizer
from logic.conflict_system.integration_hooks import ConflictEventHooks
from nyx.nyx_agent.context import SceneScope


@pytest.mark.asyncio
async def test_handle_scene_transition_normalizes_scene_scope(monkeypatch):
    synthesizer = ConflictSynthesizer(user_id=1, conversation_id=2)

    captured = {}

    async def fake_on_scene_transition(user_id, conversation_id, old_scene, new_scene):
        captured['args'] = (user_id, conversation_id, old_scene, new_scene)
        return {'ok': True}

    monkeypatch.setattr(ConflictEventHooks, 'on_scene_transition', fake_on_scene_transition)

    enqueued = []

    def fake_put_nowait(event):
        enqueued.append(event)

    monkeypatch.setattr(synthesizer._event_queue, 'put_nowait', fake_put_nowait)

    old_scope = SceneScope(location_id=1, npc_ids={1, 2}, topics={'gossip'})
    new_scope = SceneScope(location_id=2, npc_ids={3}, topics={'mystery'})

    await synthesizer._handle_scene_transition(old_scope, new_scope)

    assert captured['args'][2] == old_scope.to_dict()
    assert captured['args'][3] == new_scope.to_dict()

    assert enqueued, "scene transition event should be enqueued"
    payload = enqueued[0].payload
    assert payload['new_scene'] == new_scope.to_dict()
    assert isinstance(payload['scene_context'], dict)
    assert payload['scene_context'] == new_scope.to_dict()
    assert payload['context'] == {'ok': True}


@pytest.mark.asyncio
async def test_background_subsystem_handles_scene_enter_payload(monkeypatch):
    synthesizer = ConflictSynthesizer(user_id=7, conversation_id=9)

    async def fake_on_scene_transition(user_id, conversation_id, old_scene, new_scene):
        return {'ok': True}

    monkeypatch.setattr(ConflictEventHooks, 'on_scene_transition', fake_on_scene_transition)

    enqueued = []

    def fake_put_nowait(event):
        enqueued.append(event)

    monkeypatch.setattr(synthesizer._event_queue, 'put_nowait', fake_put_nowait)

    new_scope = SceneScope(location_id=5, npc_ids={10}, topics={'update'})

    await synthesizer._handle_scene_transition(None, new_scope)

    assert enqueued, "scene transition event should be enqueued"
    event = enqueued[0]

    background = BackgroundConflictSubsystem(user_id=7, conversation_id=9)

    seen = {}

    async def fake_is_relevant(scene_context):
        seen['relevance_ctx'] = scene_context
        return True

    async def fake_get_scene_context(scene_context):
        seen['context_ctx'] = scene_context
        return {
            'active_conflicts': [],
            'ambient_atmosphere': [],
            'world_tension': 0.0,
            'last_changed_at': 0.0,
        }

    monkeypatch.setattr(background, 'is_relevant_to_scene', fake_is_relevant)
    monkeypatch.setattr(background, 'get_scene_context', fake_get_scene_context)

    response = await background.handle_event(event)

    assert response.success is True
    assert isinstance(seen['relevance_ctx'], dict)
    assert isinstance(seen['context_ctx'], dict)
    assert seen['relevance_ctx'] == event.payload['scene_context']
    assert seen['context_ctx'] == event.payload['scene_context']
