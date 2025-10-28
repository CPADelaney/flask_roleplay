import asyncio
import importlib
import json
import os
import sys
import types
import typing

import pytest
from unittest.mock import AsyncMock

from pathlib import Path

from typing_extensions import TypedDict as _CompatTypedDict

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DB_DSN", "postgresql://user:pass@localhost/testdb")
typing.TypedDict = _CompatTypedDict  # type: ignore[attr-defined]


class _StubSentenceTransformer:  # pragma: no cover - import shim
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return [[0.0] for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 768


stub_sentence_transformers = types.ModuleType("sentence_transformers")
stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
stub_sentence_transformers.models = types.SimpleNamespace(Transformer=lambda *args, **kwargs: None)
sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)


class _StubIndex:  # pragma: no cover - minimal FAISS shim
    def __init__(self, *args, **kwargs):
        self._ntotal = 0

    def add_with_ids(self, *args, **kwargs):
        return None

    def remove_ids(self, *args, **kwargs):
        return None

    def search(self, vec, k):
        import numpy as np

        return np.zeros((1, k), dtype="float32"), np.full((1, k), -1, dtype="int64")

    @property
    def ntotal(self):
        return self._ntotal


stub_faiss = types.ModuleType("faiss")
stub_faiss.IndexFlatIP = lambda dim: _StubIndex()
stub_faiss.IndexIDMap = lambda base: _StubIndex()
stub_faiss.write_index = lambda *args, **kwargs: None
stub_faiss.read_index = lambda *args, **kwargs: _StubIndex()
sys.modules.setdefault("faiss", stub_faiss)


stub_conflict_synthesizer = types.ModuleType("logic.conflict_system.conflict_synthesizer")
stub_conflict_synthesizer.get_synthesizer = lambda *args, **kwargs: None
stub_conflict_synthesizer.ConflictSynthesizer = object
stub_conflict_synthesizer.ConflictContext = object
stub_conflict_synthesizer.SubsystemType = object
stub_conflict_synthesizer.EventType = object
stub_conflict_synthesizer.SystemEvent = object
sys.modules.setdefault("logic.conflict_system.conflict_synthesizer", stub_conflict_synthesizer)


class _StubScheduler:  # pragma: no cover - conflict scheduler shim
    def get_processor(self, *args, **kwargs):
        return None


stub_conflict_background = types.ModuleType("logic.conflict_system.background_processor")
stub_conflict_background.get_conflict_scheduler = lambda *args, **kwargs: _StubScheduler()
sys.modules.setdefault("logic.conflict_system.background_processor", stub_conflict_background)


stub_memory_orchestrator = types.ModuleType("memory.memory_orchestrator")


class _StubMemoryOrchestrator:  # pragma: no cover - orchestrator shim
    async def initialize(self):
        return None


stub_memory_orchestrator.MemoryOrchestrator = _StubMemoryOrchestrator
stub_memory_orchestrator.EntityType = object
stub_memory_orchestrator.get_memory_orchestrator = (
    lambda *args, **kwargs: _StubMemoryOrchestrator()
)
sys.modules.setdefault("memory.memory_orchestrator", stub_memory_orchestrator)


stub_lore_orchestrator = types.ModuleType("lore.lore_orchestrator")


class _StubLoreOrchestrator:  # pragma: no cover - orchestrator shim
    async def initialize(self):
        return None


class _StubOrchestratorConfig:  # pragma: no cover - config shim
    def __init__(self, *args, **kwargs):
        pass


stub_lore_orchestrator.LoreOrchestrator = _StubLoreOrchestrator
stub_lore_orchestrator.OrchestratorConfig = _StubOrchestratorConfig
stub_lore_orchestrator.get_lore_orchestrator = (
    lambda *args, **kwargs: _StubLoreOrchestrator()
)
sys.modules.setdefault("lore.lore_orchestrator", stub_lore_orchestrator)


stub_npc_orchestrator = types.ModuleType("npcs.npc_orchestrator")


class _StubNPCOrchestrator:  # pragma: no cover - orchestrator shim
    async def initialize(self):
        return None

    async def get_all_npcs(self):
        return []

    def invalidate_npc_bundles_for_scene_key(self, *args, **kwargs):
        return None


stub_npc_orchestrator.NPCOrchestrator = _StubNPCOrchestrator
stub_npc_orchestrator.NPCSnapshot = object
stub_npc_orchestrator.NPCStatus = object
sys.modules.setdefault("npcs.npc_orchestrator", stub_npc_orchestrator)


async def _stub_assess_action_feasibility_fast(*_args, **_kwargs):
    return {}


async def _stub_assess_action_feasibility(*_args, **_kwargs):
    return {}


async def _stub_record_impossibility(*_args, **_kwargs):
    return None


async def _stub_record_possibility(*_args, **_kwargs):
    return None


def _stub_detect_setting_type(*_args, **_kwargs):
    return "default"


stub_feasibility = types.ModuleType("nyx.nyx_agent.feasibility")
stub_feasibility.assess_action_feasibility_fast = _stub_assess_action_feasibility_fast
stub_feasibility.assess_action_feasibility = _stub_assess_action_feasibility
stub_feasibility.record_impossibility = _stub_record_impossibility
stub_feasibility.record_possibility = _stub_record_possibility
stub_feasibility.detect_setting_type = _stub_detect_setting_type
sys.modules.setdefault("nyx.nyx_agent.feasibility", stub_feasibility)


stub_story_models = types.ModuleType("story_agent.world_simulation_models")


class _StubAgentSafeModel:  # pragma: no cover - typed shim
    pass


class _StubKVList:  # pragma: no cover - typed shim
    def __init__(self, *args, **kwargs):
        self.items = []


class _StubPydanticLike:  # pragma: no cover - typed shim
    pass


for _name, _value in [
    ("AgentSafeModel", _StubAgentSafeModel),
    ("KVList", _StubKVList),
    ("NarrativeResponse", _StubPydanticLike),
    ("MemoryItem", _StubPydanticLike),
    ("SliceOfLifeEvent", _StubPydanticLike),
    ("NPCDialogue", _StubPydanticLike),
]:
    setattr(stub_story_models, _name, _value)

for _name in [
    "CompleteWorldState",
    "WorldState",
    "WorldMood",
    "TimeOfDay",
    "ActivityType",
    "PowerDynamicType",
    "PowerExchange",
    "WorldTension",
    "RelationshipDynamics",
    "NPCRoutine",
    "CurrentTimeData",
    "VitalsData",
    "AddictionCravingData",
    "DreamData",
    "RevelationData",
    "ChoiceData",
    "ChoiceProcessingResult",
]:
    setattr(stub_story_models, _name, _StubPydanticLike)
sys.modules.setdefault("story_agent.world_simulation_models", stub_story_models)


stub_story_director = types.ModuleType("story_agent.world_director_agent")


class _StubCompleteWorldDirector:  # pragma: no cover - director shim
    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        return None


stub_story_director.CompleteWorldDirector = _StubCompleteWorldDirector
stub_story_director.WorldDirector = _StubCompleteWorldDirector
stub_story_director.CompleteWorldDirectorContext = object
stub_story_director.WorldDirectorContext = object
sys.modules.setdefault("story_agent.world_director_agent", stub_story_director)


stub_slice_of_life = types.ModuleType("story_agent.slice_of_life_narrator")


class _StubSliceOfLifeNarrator:  # pragma: no cover - narrator shim
    def __init__(self, *args, **kwargs):
        pass

    async def initialize(self):
        return None


stub_slice_of_life.SliceOfLifeNarrator = _StubSliceOfLifeNarrator
sys.modules.setdefault("story_agent.slice_of_life_narrator", stub_slice_of_life)


stub_db_connection = types.ModuleType("db.connection")


class _AsyncNullConnection:  # pragma: no cover - db shim
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def fetchrow(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        return None

    async def fetch(self, *args, **kwargs):
        return []

    async def fetchval(self, *args, **kwargs):
        return 1


stub_db_connection.get_db_connection_context = lambda: _AsyncNullConnection()
stub_db_connection.is_shutting_down = lambda: False
sys.modules.setdefault("db.connection", stub_db_connection)

stub_asyncpg = types.ModuleType("asyncpg")
stub_asyncpg.Pool = object
stub_asyncpg.Connection = object
sys.modules.setdefault("asyncpg", stub_asyncpg)

stub_user_model_sdk = types.ModuleType("nyx.user_model_sdk")
stub_user_model_sdk.UserModelManager = object
sys.modules.setdefault("nyx.user_model_sdk", stub_user_model_sdk)


stub_task_integration = types.ModuleType("nyx.nyx_task_integration")
stub_task_integration.NyxTaskIntegration = object
sys.modules.setdefault("nyx.nyx_task_integration", stub_task_integration)


stub_response_filter = types.ModuleType("nyx.response_filter")
stub_response_filter.ResponseFilter = object
sys.modules.setdefault("nyx.response_filter", stub_response_filter)


stub_emotional_core = types.ModuleType("nyx.core.emotions.emotional_core")
stub_emotional_core.EmotionalCore = object
sys.modules.setdefault("nyx.core.emotions.emotional_core", stub_emotional_core)


from logic.universal_delta import build_delta_from_legacy_payload

from nyx.nyx_agent.context import (
    BundleSection,
    ContextBundle,
    ContextBroker,
    NyxContext,
    SceneScope,
    _SNAPSHOT_STORE,
)

from context.context_service import (
    BaseContextData,
    ContextService,
    LocationData,
    PlayerStats,
    RoleplayData,
    TimeInfo,
)
from context.projection_helpers import SceneProjection


class _StubContextBroker(ContextBroker):
    def __init__(self, ctx: NyxContext) -> None:
        super().__init__(ctx)
        self.last_scope: SceneScope | None = None

    async def initialize(self) -> None:  # pragma: no cover - simple stub
        return None

    async def compute_scene_scope(self, user_input, current_context):
        scope = await super().compute_scene_scope(user_input, current_context)
        self.last_scope = scope
        return scope

    async def load_or_fetch_bundle(self, scene_scope: SceneScope) -> ContextBundle:
        return ContextBundle(
            scene_scope=scene_scope,
            npcs=BundleSection(data={}, canonical=True),
            memories=BundleSection(data={}, canonical=True),
            lore=BundleSection(data={}, canonical=True),
            conflicts=BundleSection(data={}, canonical=True),
            world=BundleSection(data={}, canonical=True),
            narrative=BundleSection(data={}, canonical=True),
            metadata={},
        )

    def log_metrics_line(self, scene_key, packed_context):  # pragma: no cover - noop
        return None


def test_initialize_seeds_location_from_context_fallback(monkeypatch):
    async def fake_fetch_canonical_snapshot(*_args, **_kwargs):
        return None

    async def fake_get_comprehensive_context(*_args, **_kwargs):
        return {
            "currentRoleplay": {
                "CurrentLocation": {
                    "id": "azure_library",
                    "name": "Azure Library",
                }
            },
            "location_name": "Azure Library",
        }

    class _InitOnlyBroker:
        def __init__(self, ctx: NyxContext) -> None:
            self.ctx = ctx

        async def initialize(self) -> None:  # pragma: no cover - simple stub
            return None

    monkeypatch.setattr(
        sys.modules["nyx.nyx_agent.context"],
        "fetch_canonical_snapshot",
        fake_fetch_canonical_snapshot,
    )
    monkeypatch.setattr(
        sys.modules["nyx.nyx_agent.context"],
        "get_comprehensive_context",
        fake_get_comprehensive_context,
    )
    monkeypatch.setattr(
        sys.modules["nyx.nyx_agent.context"],
        "ContextBroker",
        _InitOnlyBroker,
    )

    context = NyxContext(user_id=31, conversation_id=37)
    asyncio.run(context.initialize())

    assert context.current_location == "Azure Library"
    assert context.current_context["location_name"] == "Azure Library"
    assert context.current_context["current_location"] == "Azure Library"
    assert context.current_context["location_id"] == "azure_library"

    snapshot = _SNAPSHOT_STORE.get(str(context.user_id), str(context.conversation_id))
    assert snapshot["location_name"] == "Azure Library"
    assert snapshot["scene_id"] == "Azure Library"


def test_location_refresh_persists_canonical_location():
    context = NyxContext(user_id=7, conversation_id=9)
    broker = _StubContextBroker(context)
    context.context_broker = broker

    user_key = str(context.user_id)
    convo_key = str(context.conversation_id)
    _SNAPSHOT_STORE.put(user_key, convo_key, {})

    context_data = {
        "currentRoleplay": {
            "CurrentLocation": {
                "id": "frostpeak_tavern",
                "name": "Frostpeak Tavern",
            }
        }
    }

    asyncio.run(context.build_context_for_input("where am I?", context_data))

    assert context.current_location == "Frostpeak Tavern"
    assert context.current_context["location_name"] == "Frostpeak Tavern"
    assert context.current_context["current_location"] == "Frostpeak Tavern"
    assert context.current_context["location_id"] == "frostpeak_tavern"
    assert broker.last_scope is not None
    assert broker.last_scope.location_id == "frostpeak_tavern"
    assert broker.last_scope.location_name == "Frostpeak Tavern"

    snapshot = _SNAPSHOT_STORE.get(user_key, convo_key)
    assert snapshot["location_name"] == "Frostpeak Tavern"
    assert snapshot["scene_id"] == "Frostpeak Tavern"

    asyncio.run(context.build_context_for_input("where am I now?", {}))

    assert context.current_location == "Frostpeak Tavern"
    assert broker.last_scope is not None


def test_collect_and_apply_location_updates(monkeypatch):
    context = NyxContext(user_id=101, conversation_id=202)
    broker = ContextBroker(context)
    context.current_location = "Old Port"
    context.current_context = {
        "currentRoleplay": {
            "CurrentLocation": {"id": "old_port", "name": "Old Port"}
        },
        "location_name": "Old Port",
        "location_id": "old_port",
    }

    scope = SceneScope(location_id="radiant_bazaar", location_name="Radiant Bazaar")
    bundle = ContextBundle(
        scene_scope=scope,
        npcs=BundleSection(data={}, canonical=True),
        memories=BundleSection(data={}, canonical=True),
        lore=BundleSection(data={}, canonical=True),
        conflicts=BundleSection(data={}, canonical=True),
        world=BundleSection(data={}, canonical=True),
        narrative=BundleSection(data={}, canonical=True),
        metadata={},
    )

    updates = asyncio.run(broker.collect_universal_updates(bundle))
    assert updates
    assert updates["roleplay_updates"]["CurrentLocation"] == "Radiant Bazaar"
    scene_payload = json.loads(updates["roleplay_updates"]["CurrentScene"])
    assert scene_payload["location"]["name"] == "Radiant Bazaar"
    assert scene_payload["location"]["id"] == "radiant_bazaar"

    recorded: dict[str, typing.Any] = {}

    async def fake_apply(ctx_obj, user_id, conversation_id, db_updates, conn):
        recorded["user_id"] = user_id
        recorded["conversation_id"] = conversation_id
        recorded["updates"] = db_updates
        return {"success": True, "updates_applied": 1}

    class _StubUpdaterContext:
        def __init__(self, user_id: int, conversation_id: int) -> None:
            self.user_id = user_id
            self.conversation_id = conversation_id

        async def initialize(self) -> None:  # pragma: no cover - simple stub
            return None

    monkeypatch.setattr(
        sys.modules["nyx.nyx_agent.context"],
        "_APPLY_UNIVERSAL_UPDATES_ASYNC",
        fake_apply,
    )
    monkeypatch.setattr(
        sys.modules["nyx.nyx_agent.context"],
        "UniversalUpdaterContext",
        _StubUpdaterContext,
    )

    user_key = str(context.user_id)
    convo_key = str(context.conversation_id)
    _SNAPSHOT_STORE.put(user_key, convo_key, {})

    result = asyncio.run(broker.apply_updates_async(updates))

    assert recorded["user_id"] == context.user_id
    assert recorded["conversation_id"] == context.conversation_id
    roleplay_applied = recorded["updates"]["roleplay_updates"]
    assert roleplay_applied["CurrentLocation"] == "Radiant Bazaar"
    assert json.loads(roleplay_applied["CurrentScene"])["location"]["name"] == "Radiant Bazaar"
    assert result["success"] is True

    assert context.current_location == "Radiant Bazaar"
    assert context.current_context["location_name"] == "Radiant Bazaar"
    assert context.current_context["location_id"] == "radiant_bazaar"

    snapshot = _SNAPSHOT_STORE.get(user_key, convo_key)
    assert snapshot["location_name"] == "Radiant Bazaar"
    assert snapshot["scene_id"] == "radiant_bazaar"


def test_collect_updates_from_intent_emits_player_move():
    context = NyxContext(user_id=404, conversation_id=505)
    broker = ContextBroker(context)
    context.current_location = "Old Port"
    context.current_context = {
        "location_name": "Old Port",
        "location_id": "old_port",
        "feasibility": {
            "overall": {"feasible": True, "strategy": "allow"},
            "per_intent": [
                {
                    "destination": {
                        "name": "Radiant Bazaar",
                        "id": 777,
                    }
                }
            ],
        },
    }

    scope = SceneScope(location_id="old_port", location_name="Old Port")
    bundle = ContextBundle(
        scene_scope=scope,
        npcs=BundleSection(data={}, canonical=True),
        memories=BundleSection(data={}, canonical=True),
        lore=BundleSection(data={}, canonical=True),
        conflicts=BundleSection(data={}, canonical=True),
        world=BundleSection(data={}, canonical=True),
        narrative=BundleSection(data={}, canonical=True),
        metadata={},
    )

    updates = asyncio.run(broker.collect_universal_updates(bundle))

    assert updates
    roleplay_updates = updates["roleplay_updates"]
    assert roleplay_updates["CurrentLocation"] == "Radiant Bazaar"

    scene_payload = json.loads(roleplay_updates["CurrentScene"])
    assert scene_payload["location"]["name"] == "Radiant Bazaar"
    assert scene_payload["location"]["id"] == "777"

    delta = build_delta_from_legacy_payload(
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        payload=updates,
    )

    assert delta.operations, "expected at least one delta operation"
    move = delta.operations[0]
    assert move.type == "player.move"
    assert move.player_id == context.user_id
    assert move.location_slug == "Radiant Bazaar"
    assert move.location_id == 777


def test_collect_updates_uses_last_user_input_for_movement():
    context = NyxContext(user_id=414, conversation_id=515)
    broker = _StubContextBroker(context)
    context.context_broker = broker
    context.current_location = "Old Port"
    context.current_context = {
        "currentRoleplay": {"CurrentLocation": {"id": "old_port", "name": "Old Port"}},
        "location_name": "Old Port",
        "location_id": "old_port",
    }

    asyncio.run(context.build_context_for_input("Walk to Radiant Bazaar", {}))

    assert context.last_user_input == "Walk to Radiant Bazaar"
    assert context.current_context["last_user_input"] == "Walk to Radiant Bazaar"

    scope = broker.last_scope
    assert scope is not None
    assert scope.location_id == "old_port"

    bundle = asyncio.run(broker.load_or_fetch_bundle(scope))
    updates = asyncio.run(broker.collect_universal_updates(bundle))

    assert updates
    roleplay_updates = updates["roleplay_updates"]
    assert roleplay_updates["CurrentLocation"] == "Radiant Bazaar"

    scene_payload = json.loads(roleplay_updates["CurrentScene"])
    assert scene_payload["location"]["name"] == "Radiant Bazaar"
    assert scene_payload["location"]["id"] == "Radiant Bazaar"


def test_process_user_input_persists_location(monkeypatch):
    from nyx.nyx_agent import context as context_module
    import nyx.nyx_agent.orchestrator as orchestrator_module

    monkeypatch.setattr(context_module, "ContextBroker", _StubContextBroker)

    async def fake_initialize(self):
        self.context_broker = _StubContextBroker(self)
        self.current_context = {}
        self.current_location = None
        self.last_packed_context = None

    monkeypatch.setattr(context_module.NyxContext, "initialize", fake_initialize)

    original_build = context_module.NyxContext.build_context_for_input
    build_calls: dict[str, typing.Any] = {}

    async def tracking_build(self, user_input_arg, context_payload_arg=None):
        build_calls["called"] = True
        build_calls["context"] = self
        build_calls["payload"] = context_payload_arg
        return await original_build(self, user_input_arg, context_payload_arg or {})

    monkeypatch.setattr(context_module.NyxContext, "build_context_for_input", tracking_build)

    recorded_updates: list[tuple[str, str]] = []

    async def fake_update_current_roleplay(cctx, conn, key, value):
        recorded_updates.append((key, value))

    monkeypatch.setattr(context_module.canon, "update_current_roleplay", fake_update_current_roleplay)

    class _StubRunner:
        @staticmethod
        async def run(agent, prompt, context=None, run_config=None, **_kwargs):
            return types.SimpleNamespace(
                messages=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "narrative"}
                        ],
                    }
                ]
            )

    monkeypatch.setattr(orchestrator_module, "Runner", _StubRunner)

    class _StubRunContextWrapper:
        def __init__(self, ctx):
            self.context = ctx

    monkeypatch.setattr(orchestrator_module, "RunContextWrapper", _StubRunContextWrapper)

    class _StubModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(orchestrator_module, "ModelSettings", _StubModelSettings)

    class _StubRunConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(orchestrator_module, "RunConfig", _StubRunConfig)

    monkeypatch.setattr(
        orchestrator_module,
        "nyx_main_agent",
        types.SimpleNamespace(name="Nyx", tools=[]),
    )

    class _StubUpdatesResult:
        def __init__(self):
            self.success = False
            self.updates_generated = False

    async def fake_generate_universal_updates_impl(ctx, narrative):
        return _StubUpdatesResult()

    monkeypatch.setattr(
        orchestrator_module,
        "generate_universal_updates_impl",
        fake_generate_universal_updates_impl,
    )

    async def fake_resolve_scene_requests(resp_stream, ctx):
        return resp_stream

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_scene_requests",
        fake_resolve_scene_requests,
    )

    async def fake_assemble_nyx_response(**kwargs):
        return types.SimpleNamespace(
            narrative="narrative",
            metadata={"performance": {}},
            world_state={},
            choices=[],
            emergent_events=[],
            image=None,
        )

    monkeypatch.setattr(
        orchestrator_module,
        "assemble_nyx_response",
        fake_assemble_nyx_response,
    )

    async def fake_decide_image_generation_standalone(ctx, narrative):
        return "{}"

    monkeypatch.setattr(
        orchestrator_module,
        "decide_image_generation_standalone",
        fake_decide_image_generation_standalone,
    )

    monkeypatch.setattr(orchestrator_module, "enforce_all_rules_on_player", None)
    monkeypatch.setattr(
        orchestrator_module,
        "assess_action_feasibility_fast",
        _stub_assess_action_feasibility_fast,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "assess_action_feasibility",
        _stub_assess_action_feasibility,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "record_impossibility",
        _stub_record_impossibility,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "record_possibility",
        _stub_record_possibility,
    )

    response_holder: dict[str, typing.Any] = {}

    async def _run():
        response_holder["value"] = await orchestrator_module.process_user_input(
            user_id=5,
            conversation_id=6,
            user_input="Look around",
            context_data={
                "currentRoleplay": {
                    "CurrentLocation": {
                        "id": "gilded_grove",
                        "name": "Gilded Grove",
                    }
                }
            },
        )

    asyncio.run(_run())

    response = response_holder["value"]

    assert response["success"] is True
    assert recorded_updates, "Expected canonical update call"
    key, value = recorded_updates[0]
    assert key == "CurrentLocation"
    assert value == "Gilded Grove"
    assert build_calls.get("called") is True
    assert build_calls["context"].last_packed_context is not None


def test_location_refresh_uses_aggregator_current_roleplay():
    context = NyxContext(user_id=11, conversation_id=13)
    broker = _StubContextBroker(context)
    context.context_broker = broker

    user_key = str(context.user_id)
    convo_key = str(context.conversation_id)
    _SNAPSHOT_STORE.put(user_key, convo_key, {})

    context_data = {
        "aggregator_data": {
            "currentRoleplay": {
                "CurrentLocation": {
                    "id": "sunspire_plaza",
                    "name": "Sunspire Plaza",
                }
            }
        }
    }

    asyncio.run(context.build_context_for_input("look around", context_data))

    assert context.current_location == "Sunspire Plaza"
    assert context.current_context["location_name"] == "Sunspire Plaza"
    assert context.current_context["current_location"] == "Sunspire Plaza"
    assert context.current_context["location_id"] == "sunspire_plaza"
    assert broker.last_scope is not None
    assert broker.last_scope.location_id == "sunspire_plaza"
    assert broker.last_scope.location_name == "Sunspire Plaza"

    snapshot = _SNAPSHOT_STORE.get(user_key, convo_key)
    assert snapshot["location_name"] == "Sunspire Plaza"
    assert snapshot["scene_id"] == "Sunspire Plaza"


def test_build_context_for_input_schedules_persist(monkeypatch):
    from nyx.nyx_agent import context as context_module

    context = context_module.NyxContext(user_id=501, conversation_id=502)
    broker = _StubContextBroker(context)
    context.context_broker = broker

    start_event = asyncio.Event()
    release_event = asyncio.Event()

    async def fake_persist(self, canonical_location):
        start_event.set()
        await release_event.wait()

    monkeypatch.setattr(
        context_module.NyxContext,
        "_persist_location_to_db",
        fake_persist,
    )

    context_data = {
        "currentRoleplay": {
            "CurrentLocation": {
                "id": "async_square",
                "name": "Async Square",
            }
        }
    }

    async def _run() -> None:
        await asyncio.wait_for(
            context.build_context_for_input("look around", context_data),
            timeout=0.5,
        )

        assert start_event.is_set(), "persist task should start asynchronously"
        assert context.background_tasks, "background task tracking should retain the task"
        assert any(
            not task.done() for task in context.background_tasks
        ), "persist task should still be pending"

        release_event.set()
        await asyncio.gather(
            *list(context.background_tasks),
            return_exceptions=True,
        )
        assert not context.background_tasks, "task set should be cleared after completion"

    asyncio.run(_run())


def test_location_refresh_ignores_placeholder_overrides():
    context = NyxContext(user_id=21, conversation_id=23)
    broker = _StubContextBroker(context)
    context.context_broker = broker

    user_key = str(context.user_id)
    convo_key = str(context.conversation_id)
    _SNAPSHOT_STORE.put(user_key, convo_key, {})

    initial_context = {
        "currentRoleplay": {
            "CurrentLocation": {
                "id": "frostpeak_tavern",
                "name": "Frostpeak Tavern",
            }
        }
    }

    asyncio.run(context.build_context_for_input("set scene", initial_context))

    assert context.current_location == "Frostpeak Tavern"

    placeholder_context = {
        "currentRoleplay": {"CurrentLocation": "Unknown"},
        "location_name": "Unknown",
        "location": "Unknown",
        "location_id": "unknown",
    }

    asyncio.run(context.build_context_for_input("placeholder scene", placeholder_context))

    assert context.current_location == "Frostpeak Tavern"
    assert context.current_context["location_name"] == "Frostpeak Tavern"
    assert context.current_context["location_id"] == "frostpeak_tavern"

    snapshot = _SNAPSHOT_STORE.get(user_key, convo_key)
    assert snapshot["location_name"] == "Frostpeak Tavern"
    assert snapshot["scene_id"] == "Frostpeak Tavern"


def test_location_persist_falls_back_when_canon_update_fails(monkeypatch):
    executed: list[tuple[str, tuple]] = []
    contexts: list[object] = []

    class _StubConn:
        async def execute(self, query, *params):
            executed.append((query.strip(), params))

    class _StubManager:
        def __init__(self) -> None:
            contexts.append(self)
            self._conn = _StubConn()

        async def __aenter__(self):
            return self._conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    import nyx.nyx_agent.context as context_module

    monkeypatch.setattr(
        context_module,
        "get_db_connection_context",
        lambda: _StubManager(),
    )

    async def _fail_update(*_args, **_kwargs):  # pragma: no cover - explicit failure path
        raise RuntimeError("boom")

    monkeypatch.setattr(context_module.canon, "update_current_roleplay", _fail_update)

    ctx = NyxContext(user_id=31, conversation_id=42)

    asyncio.run(ctx._persist_location_to_db("Azure Library"))

    assert len(contexts) == 2  # first attempt + fallback
    assert executed, "Fallback insert was not executed"
    query, params = executed[-1]
    assert "INSERT INTO CurrentRoleplay" in query
    assert params == (31, 42, "CurrentLocation", "Azure Library")


def test_context_service_filters_npcs_by_resolved_location(monkeypatch):
    calls: dict[str, object] = {}

    projection = SceneProjection(
        current_roleplay={},
        player_stats={},
        npcs=[
            {
                "npc_id": "npc-1",
                "npc_name": "Aria",
                "dominance": 0.1,
                "cruelty": 0.2,
                "closeness": 0.9,
                "trust": 0.8,
                "respect": 0.7,
                "intensity": 0.6,
                "current_location": "Frostpeak Tavern",
                "physical_description": "A bard with a quick smile.",
                "introduced": True,
            }
        ],
        events=[],
        quests=[],
    )

    async def fake_ensure_projection(self):
        calls["projection"] = True
        return projection

    monkeypatch.setattr(ContextService, "_ensure_projection", fake_ensure_projection)

    base_context = BaseContextData(
        time_info=TimeInfo(year="1040", month="6", day="15", time_of_day="Morning"),
        player_stats=PlayerStats(),
        current_roleplay=RoleplayData(CurrentLocation="Frostpeak Tavern"),
        current_location="Frostpeak Tavern",
        relationship_overview=None,
    )

    async def fake_get_base_context(self, requested_location):
        return base_context

    observed_location: dict[str, str | None] = {}

    async def fake_get_location_details(self, location):
        observed_location["location"] = location
        return LocationData(location_name=location or "Unknown")

    async def fake_get_quest_information(self):
        return []

    async def fake_trim(self, context, budget):
        return context

    monkeypatch.setattr(ContextService, "_get_base_context", fake_get_base_context)
    monkeypatch.setattr(ContextService, "_get_location_details", fake_get_location_details)
    monkeypatch.setattr(ContextService, "_get_quest_information", fake_get_quest_information)
    monkeypatch.setattr(ContextService, "_trim_to_budget", fake_trim)

    service = ContextService(user_id=1, conversation_id=2)
    service.initialized = True
    service.config = types.SimpleNamespace(is_enabled=lambda _: False)
    service.context_manager = types.SimpleNamespace(version=7)

    result = asyncio.run(service.get_context(input_text="hello there"))

    assert calls.get("projection") is True
    assert result["npcs"]
    assert result["npcs"][0]["npc_name"] == "Aria"
    assert result["npcs"][0]["current_location"] == "Frostpeak Tavern"
    assert result["npcs"][0]["relevance"] == 0.7
    assert observed_location["location"] == "Frostpeak Tavern"


def test_memory_orchestrator_normalizes_location_slug(monkeypatch):
    """Regression test for resolving location slugs before memory retrieval."""

    stub_module = sys.modules.get("memory.memory_orchestrator")
    if stub_module is not None:
        sys.modules.pop("memory.memory_orchestrator")

    from db import connection as db_connection

    if not hasattr(db_connection, "run_async_in_worker_loop"):
        monkeypatch.setattr(
            db_connection,
            "run_async_in_worker_loop",
            lambda func, *args, **kwargs: None,
            raising=False,
        )

    real_memory_module = importlib.import_module("memory.memory_orchestrator")

    if stub_module is not None:
        sys.modules["memory.memory_orchestrator"] = stub_module

    resolved_location_id = 101
    db_calls = {"count": 0}

    class _FakeConnCtx:
        async def __aenter__(self):
            db_calls["count"] += 1
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def fetchrow(self, query, user_id, conversation_id, location_name):
            assert "FROM Locations" in query
            if str(location_name).lower() == "shadow-market":
                return {"id": resolved_location_id}
            return None

    def _fake_get_db_connection_context(*_args, **_kwargs):
        return _FakeConnCtx()

    monkeypatch.setattr(
        db_connection,
        "get_db_connection_context",
        _fake_get_db_connection_context,
    )

    class _StubCache:
        def __init__(self):
            self.store: dict[str, dict[str, typing.Any]] = {}
            self.ops: list[tuple[str, str]] = []

        async def get(self, key):
            self.ops.append(("get", key))
            return self.store.get(key)

        async def set(self, key, value):
            self.ops.append(("set", key))
            self.store[key] = value

        async def clear(self):
            self.store.clear()

    class _FakeUnifiedMemoryManager:
        instances: list["_FakeUnifiedMemoryManager"] = []

        def __init__(self, *, entity_type, entity_id, user_id, conversation_id):
            assert isinstance(entity_id, int)
            self.entity_type = entity_type
            self.entity_id = entity_id
            self.user_id = user_id
            self.conversation_id = conversation_id
            _FakeUnifiedMemoryManager.instances.append(self)

        async def query_memories(self, **kwargs):
            return []

    stub_core = types.ModuleType("memory.core")
    stub_core.UnifiedMemoryManager = _FakeUnifiedMemoryManager
    monkeypatch.setitem(sys.modules, "memory.core", stub_core)

    monkeypatch.setattr(real_memory_module, "MemoryTelemetry", None, raising=False)

    async def _exercise():
        orchestrator = real_memory_module.MemoryOrchestrator(user_id=7, conversation_id=8)
        orchestrator.initialized = True
        orchestrator.cache = _StubCache()
        orchestrator._telemetry = AsyncMock(return_value=None)
        orchestrator.search_vector_store = AsyncMock(return_value=[])

        slug_result = await orchestrator.retrieve_memories(
            real_memory_module.EntityType.LOCATION.value,
            "Shadow-Market",
        )

        assert slug_result["entity_id"] == resolved_location_id
        assert slug_result["memories"] == []
        assert slug_result["count"] == 0
        assert _FakeUnifiedMemoryManager.instances
        assert _FakeUnifiedMemoryManager.instances[0].entity_id == resolved_location_id

        cached_result = await orchestrator.retrieve_memories(
            real_memory_module.EntityType.LOCATION.value,
            resolved_location_id,
        )

        assert cached_result == slug_result
        assert len(_FakeUnifiedMemoryManager.instances) == 1
        assert db_calls["count"] == 1
        assert orchestrator.memory_managers.get("location_101") is _FakeUnifiedMemoryManager.instances[0]
        assert len(orchestrator.cache.ops) == 3
        assert orchestrator.cache.ops[0][0] == "get"
        assert orchestrator.cache.ops[1][0] == "set"
        assert orchestrator.cache.ops[2][0] == "get"
        assert orchestrator.cache.ops[0][1] == orchestrator.cache.ops[2][1]

    asyncio.run(_exercise())
