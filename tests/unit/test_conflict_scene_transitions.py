import os
import sys
import types
import typing
from typing import Any, List, Tuple
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


class _FakeNationConnection:
    def __init__(self, location_id: int, nation_ids: List[int]):
        self._location_id = location_id
        self._nation_ids = nation_ids
        self.calls: List[Tuple[str, str, Tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any):
        self.calls.append(("fetchrow", query, args))
        if "from locations" in query.lower() and args and int(args[0]) == self._location_id:
            return {"id": self._location_id}
        return None

    async def fetch(self, query: str, *args: Any):
        self.calls.append(("fetch", query, args))
        lowered = query.lower()
        if "from conflict_stakeholders" in lowered:
            return []
        if "from conflict_locations" in lowered:
            return []
        if "from conflicts" in lowered:
            return []
        if "from loreconnections" in lowered:
            if "source_type = 'nations'" in lowered:
                return [{"source_id": nid} for nid in self._nation_ids]
            if "target_type = 'nations'" in lowered:
                return []
        if "from nationalconflicts" in lowered:
            assert args and list(args[0]) == self._nation_ids
            return [
                {"id": 501, "name": "Border Standoff", "type": "geopolitical", "intensity": 0.7}
            ]
        if "from domesticissues" in lowered:
            assert args and list(args[0]) == self._nation_ids
            return [
                {"id": 601, "name": "Labor Unrest", "type": "domestic", "intensity": 0.4}
            ]
        return []

    async def fetchval(self, query: str, *args: Any):
        self.calls.append(("fetchval", query, args))
        return None


class _FakeNationContext:
    def __init__(self, connection: _FakeNationConnection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_conflict_fast_path_uses_loreconnections_for_nations(monkeypatch):
    synthesizer = ConflictSynthesizer(user_id=11, conversation_id=22)

    fake_connection = _FakeNationConnection(location_id=303, nation_ids=[7, 9])

    def fake_get_db_connection_context():
        return _FakeNationContext(fake_connection)

    monkeypatch.setattr(
        'logic.conflict_system.conflict_synthesizer.get_db_connection_context',
        fake_get_db_connection_context,
    )

    scene_info = {"location_id": 303}

    result = await synthesizer.conflict_context_for_scene(scene_info)

    names = {conflict['name'] for conflict in result['conflicts']}
    assert "Border Standoff" in names
    assert "Labor Unrest" in names

    # Ensure we derived nation IDs via lore connections instead of querying a missing column
    assert not any(
        "nation_id" in query.lower() and "from locations" in query.lower()
        for _kind, query, _ in fake_connection.calls
    )
    assert any("from loreconnections" in query.lower() for _kind, query, _ in fake_connection.calls)
