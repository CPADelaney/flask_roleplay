import asyncio
import importlib
import os
import sys
import time
import types
import asyncio
import importlib
import os
import sys
import time
import types
import typing
from enum import Enum
from types import SimpleNamespace

import pytest
from typing_extensions import TypedDict as _CompatTypedDict

# Ensure repository modules are importable
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
typing.TypedDict = _CompatTypedDict  # type: ignore[attr-defined]

# Stub heavy dependencies before importing the Nyx context module


class _StubSentenceTransformer:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple shim
        pass

    def encode(self, *args, **kwargs):  # pragma: no cover - simple shim
        return []


stub_sentence_transformers = types.ModuleType("sentence_transformers")
stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)

stub_conflict_synthesizer = types.ModuleType("logic.conflict_system.conflict_synthesizer")
stub_conflict_synthesizer.get_synthesizer = lambda *args, **kwargs: None
stub_conflict_synthesizer.ConflictSynthesizer = object
stub_conflict_synthesizer.ConflictContext = object
stub_conflict_synthesizer.SubsystemType = object
stub_conflict_synthesizer.EventType = object
stub_conflict_synthesizer.SystemEvent = object

stub_conflict_background = types.ModuleType("logic.conflict_system.background_processor")
stub_conflict_background.get_conflict_scheduler = lambda *args, **kwargs: types.SimpleNamespace(
    get_processor=lambda *a, **k: None
)

logic_pkg = importlib.import_module("logic")
conflict_pkg = types.ModuleType("logic.conflict_system")
conflict_pkg.__dict__["conflict_synthesizer"] = stub_conflict_synthesizer
conflict_pkg.__dict__["background_processor"] = stub_conflict_background
conflict_pkg.__path__ = []  # type: ignore[attr-defined]
setattr(logic_pkg, "conflict_system", conflict_pkg)
sys.modules["logic.conflict_system"] = conflict_pkg
sys.modules["logic.conflict_system.conflict_synthesizer"] = stub_conflict_synthesizer
sys.modules["logic.conflict_system.background_processor"] = stub_conflict_background


class _StubEntityType(Enum):
    NPC = "npc"
    LOCATION = "location"
    PLAYER = "player"


stub_memory_module = types.ModuleType("memory.memory_orchestrator")
stub_memory_module.EntityType = _StubEntityType
stub_memory_module.MemoryOrchestrator = object
stub_memory_module.get_memory_orchestrator = lambda *args, **kwargs: None

memory_pkg = importlib.import_module("memory")
setattr(memory_pkg, "memory_orchestrator", stub_memory_module)
sys.modules["memory.memory_orchestrator"] = stub_memory_module


from nyx.nyx_agent.context import ContextBroker, EntityType, SceneScope


class _StubMemoryOrchestrator:
    def __init__(self, responses):
        self._responses = responses

    async def retrieve_memories(
        self,
        *,
        entity_type,
        entity_id,
        query="",
        limit=5,
        use_llm_analysis=False,
    ):
        delay, payload = self._responses.get((entity_type, entity_id), (0, None))
        await asyncio.sleep(delay)
        return payload


async def _run_memory_section_timeout_check():
    responses = {
        (EntityType.NPC, 1): (0.01, {"memories": [{"id": "npc-1"}]}),
        (EntityType.NPC, 2): (0.2, {"memories": [{"id": "npc-2"}]}),
        (EntityType.NPC, 3): (0.01, {"memories": [{"id": "npc-3"}]}),
        (EntityType.LOCATION, 999): (0.2, {"memories": [{"id": "loc"}]}),
    }

    orchestrator = _StubMemoryOrchestrator(responses)

    async def analyze_memory_patterns(*, topic: str):
        await asyncio.sleep(0.01)
        return {"predictions": [f"pattern:{topic}"]}

    ctx = SimpleNamespace(
        memory_orchestrator=orchestrator,
        analyze_memory_patterns=analyze_memory_patterns,
        memory_fetch_timeout=0.05,
    )

    broker = ContextBroker.__new__(ContextBroker)
    broker.ctx = ctx

    scope = SceneScope()
    scope.npc_ids.add(1)
    scope.npc_ids.add(2)
    scope.npc_ids.add(3)
    scope.location_id = 999
    scope.topics.update({"trade", "magic"})

    start = time.perf_counter()
    section = await broker._fetch_memory_section(scope)
    duration = time.perf_counter() - start

    # Slow NPC and location calls should time out individually, keeping total under the slow delay.
    assert duration < 0.15

    assert [m["id"] for m in section.data.relevant] == ["npc-1", "npc-3"]
    assert section.data.recent == []
    assert section.data.patterns == ["pattern:magic, trade"] or section.data.patterns == ["pattern:trade, magic"]


def test_memory_section_times_out_slow_calls():
    asyncio.run(_run_memory_section_timeout_check())
