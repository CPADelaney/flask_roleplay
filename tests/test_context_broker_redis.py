import asyncio
import os
import sys
import types
from types import SimpleNamespace

import pytest
import typing
from typing_extensions import TypedDict as _CompatTypedDict

# Ensure repository modules are importable
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
typing.TypedDict = _CompatTypedDict  # type: ignore[attr-defined]

# Stub heavy dependencies before importing the Nyx context module


class _StubSentenceTransformer:  # pragma: no cover - import shim
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return [[0.0] for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 768


stub_sentence_transformers = types.ModuleType("sentence_transformers")
stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
stub_sentence_transformers.models = types.SimpleNamespace(Transformer=lambda *a, **k: None)
sys.modules["sentence_transformers"] = stub_sentence_transformers


stub_faiss = types.ModuleType("faiss")
stub_faiss.IndexFlatIP = lambda dim: object()
stub_faiss.IndexIDMap = lambda base: object()
stub_faiss.write_index = lambda *args, **kwargs: None
stub_faiss.read_index = lambda *args, **kwargs: object()
sys.modules["faiss"] = stub_faiss


stub_conflict_synthesizer = types.ModuleType("logic.conflict_system.conflict_synthesizer")
stub_conflict_synthesizer.ConflictSynthesizer = object
stub_conflict_synthesizer.ConflictContext = object
stub_conflict_synthesizer.SubsystemType = object
stub_conflict_synthesizer.EventType = object
stub_conflict_synthesizer.SystemEvent = object
stub_conflict_synthesizer.get_synthesizer = lambda *args, **kwargs: None
sys.modules["logic.conflict_system.conflict_synthesizer"] = stub_conflict_synthesizer


class _StubScheduler:  # pragma: no cover - conflict scheduler shim
    def get_processor(self, *args, **kwargs):
        return None


stub_conflict_background = types.ModuleType("logic.conflict_system.background_processor")
stub_conflict_background.get_conflict_scheduler = lambda *args, **kwargs: _StubScheduler()
sys.modules["logic.conflict_system.background_processor"] = stub_conflict_background


stub_memory_orchestrator = types.ModuleType("memory.memory_orchestrator")


class _StubMemoryOrchestrator:  # pragma: no cover - orchestrator shim
    async def initialize(self):
        return None


stub_memory_orchestrator.MemoryOrchestrator = _StubMemoryOrchestrator
stub_memory_orchestrator.EntityType = object
stub_memory_orchestrator.get_memory_orchestrator = lambda *a, **k: _StubMemoryOrchestrator()
sys.modules["memory.memory_orchestrator"] = stub_memory_orchestrator


stub_npc_orchestrator = types.ModuleType("npcs.npc_orchestrator")
stub_npc_orchestrator.NPCOrchestrator = object
stub_npc_orchestrator.NPCSnapshot = object
stub_npc_orchestrator.NPCStatus = object
sys.modules["npcs.npc_orchestrator"] = stub_npc_orchestrator


stub_lore_orchestrator = types.ModuleType("lore.lore_orchestrator")
stub_lore_orchestrator.LoreOrchestrator = object
stub_lore_orchestrator.OrchestratorConfig = object
stub_lore_orchestrator.get_lore_orchestrator = lambda *a, **k: object()
sys.modules["lore.lore_orchestrator"] = stub_lore_orchestrator


stub_world_models = types.ModuleType("story_agent.world_simulation_models")
for attr in [
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
    "AgentSafeModel",
    "KVList",
    "NarrativeResponse",
    "MemoryItem",
    "SliceOfLifeEvent",
    "NPCDialogue",
]:
    setattr(stub_world_models, attr, object)
sys.modules["story_agent.world_simulation_models"] = stub_world_models


stub_world_director = types.ModuleType("story_agent.world_director_agent")


class _StubCompleteWorldDirector:
    def __init__(self, *args, **kwargs):
        async def _bundle(*args, **kwargs):
            return {}

        self.context = types.SimpleNamespace(
            get_world_bundle=_bundle
        )

    async def initialize(self, *args, **kwargs):
        return None

    async def get_world_state(self):  # pragma: no cover - compatibility shim
        return {}


stub_world_director.CompleteWorldDirector = _StubCompleteWorldDirector
stub_world_director.WorldDirector = _StubCompleteWorldDirector
stub_world_director.CompleteWorldDirectorContext = object
stub_world_director.WorldDirectorContext = object
sys.modules["story_agent.world_director_agent"] = stub_world_director


if "nyx" not in sys.modules:
    nyx_pkg = types.ModuleType("nyx")
    nyx_pkg.__path__ = [os.path.abspath("nyx")]
    sys.modules["nyx"] = nyx_pkg

if "nyx.nyx_agent" not in sys.modules:
    nyx_agent_pkg = types.ModuleType("nyx.nyx_agent")
    nyx_agent_pkg.__path__ = [os.path.abspath("nyx/nyx_agent")]
    sys.modules["nyx.nyx_agent"] = nyx_agent_pkg


stub_nyx_vector_store = types.ModuleType("nyx.core.memory.vector_store")


async def _async_stub(*args, **kwargs):  # pragma: no cover - simple async shim
    return []


stub_nyx_vector_store.add = _async_stub
stub_nyx_vector_store.query = _async_stub
sys.modules["nyx.core.memory.vector_store"] = stub_nyx_vector_store

from nyx.nyx_agent import context as context_module
from nyx.nyx_agent.context import ContextBroker


class _StubRedisClient:
    def __init__(self):
        self.pings = 0

    async def ping(self):
        self.pings += 1
        raise RuntimeError("transient failure")


def test_context_broker_redis_connection_handles_non_typeerror(monkeypatch):
    async def _run():
        broker = ContextBroker.__new__(ContextBroker)
        broker.ctx = SimpleNamespace()
        broker.redis_client = None
        broker._redis_backoff_until = 0
        broker._redis_failures = 0

        captured_kwargs = {}
        stub_client = _StubRedisClient()

        def fake_from_url(url, **kwargs):
            captured_kwargs.update(kwargs)
            if kwargs.get("encoding", "__missing__") is None:
                raise TypeError("encoding cannot be None")
            return stub_client

        monkeypatch.setattr(context_module.redis, "from_url", fake_from_url)

        await broker._try_connect_redis()

        # The runtime error from ping should be handled without surfacing a TypeError.
        assert broker.redis_client is None
        assert broker._redis_failures == 1
        assert captured_kwargs["decode_responses"] is False

    asyncio.run(_run())
