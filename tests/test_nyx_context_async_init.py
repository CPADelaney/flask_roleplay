import asyncio
import os
import sys
import time
import types
import typing
import typing_extensions
from enum import Enum

import pytest

typing.TypedDict = typing_extensions.TypedDict
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_dummy_models = types.ModuleType("sentence_transformers.models")
_dummy_models.Transformer = lambda *args, **kwargs: None
_dummy_models.Pooling = lambda *args, **kwargs: None


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):  # pragma: no cover - defensive stub
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):  # pragma: no cover - defensive stub
        return 3


_dummy_sentence_transformers = types.ModuleType("sentence_transformers")
_dummy_sentence_transformers.SentenceTransformer = _DummySentenceTransformer
_dummy_sentence_transformers.models = _dummy_models

sys.modules["sentence_transformers"] = _dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = _dummy_models

from nyx.nyx_agent import context as context_module


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_nyx_context_initialize_non_blocking(monkeypatch):
    memory_delay = 0.1
    broker_delay = 0.05
    sentinel_memory = object()

    class DummyStore:
        def __init__(self) -> None:
            self._data = {}

        def get(self, user_key, conversation_key):
            return self._data.get((user_key, conversation_key))

        def put(self, user_key, conversation_key, value):
            self._data[(user_key, conversation_key)] = value

    class StubBroker:
        def __init__(self, ctx):
            self.ctx = ctx
            self.initialized = False

        async def initialize(self):
            await asyncio.sleep(broker_delay)
            self.initialized = True

    async def _fake_fetch_snapshot(*_args, **_kwargs):
        return None

    async def _fake_get_context(*_args, **_kwargs):
        return {}

    async def _noop_hydrate(self):
        return None

    async def _noop_refresh(self, previous_location_id=None):
        return None

    async def _slow_memory(self):
        await asyncio.sleep(memory_delay)
        self.memory_orchestrator = sentinel_memory

    async def _fast_noop(self):
        return None

    monkeypatch.setattr(context_module, "_SNAPSHOT_STORE", DummyStore())
    monkeypatch.setattr(context_module, "ContextBroker", StubBroker)
    monkeypatch.setattr(context_module, "fetch_canonical_snapshot", _fake_fetch_snapshot)
    monkeypatch.setattr(context_module, "get_comprehensive_context", _fake_get_context)
    monkeypatch.setattr(context_module, "WORLD_SIMULATION_AVAILABLE", False)
    monkeypatch.setattr(context_module.NyxContext, "_hydrate_location_from_db", _noop_hydrate)
    monkeypatch.setattr(context_module.NyxContext, "_refresh_location_from_context", _noop_refresh)
    monkeypatch.setattr(context_module.NyxContext, "_init_memory_orchestrator", _slow_memory)
    monkeypatch.setattr(context_module.NyxContext, "_init_lore_orchestrator", _fast_noop)
    monkeypatch.setattr(context_module.NyxContext, "_init_npc_orchestrator", _fast_noop)
    monkeypatch.setattr(context_module.NyxContext, "_init_conflict_synthesizer", _fast_noop)

    ctx = context_module.NyxContext(user_id=1, conversation_id=2)
    start = time.perf_counter()
    await ctx.initialize()
    elapsed = time.perf_counter() - start

    assert elapsed < memory_delay / 2
    mem_task = ctx.get_init_task("memory")
    broker_task = ctx.get_init_task("context_broker")
    assert mem_task is not None and not mem_task.done()
    assert broker_task is not None and not broker_task.done()
    assert ctx.memory_orchestrator is None

    mem_start = time.perf_counter()
    await ctx.await_orchestrator("memory")
    mem_elapsed = time.perf_counter() - mem_start

    assert mem_elapsed >= memory_delay - 0.02
    assert ctx.memory_orchestrator is sentinel_memory

    await ctx.await_orchestrator("context_broker")
    assert isinstance(ctx.context_broker, StubBroker)
    assert ctx.context_broker.initialized is True


@pytest.mark.anyio
async def test_context_broker_waits_for_memory(monkeypatch):
    memory_delay = 0.05

    class DummyRedisClient:
        async def ping(self):
            return None

        async def get(self, *_args, **_kwargs):
            return None

        async def delete(self, *_args, **_kwargs):
            return None

        async def setex(self, *_args, **_kwargs):
            return None

    class DummyRedisModule:
        Redis = type("Redis", (), {})

        @staticmethod
        def from_url(*_args, **_kwargs):
            return DummyRedisClient()

    class DummyScheduler:
        def get_processor(self, *_args, **_kwargs):
            return None

    class MemoryStub:
        def __init__(self):
            self.retrievals = 0

        async def retrieve_memories(self, *args, **kwargs):
            self.retrievals += 1
            return {"memories": []}

        async def retrieve_recent_memories(self, *args, **kwargs):
            return {"memories": []}

        async def analyze_patterns(self, *args, **kwargs):
            return {"patterns": []}

    class BrokerCtx:
        def __init__(self):
            self.user_id = 7
            self.conversation_id = 9
            self.current_context = {}
            self.current_scene_npcs = []
            self.memory_fetch_timeout = memory_delay * 2
            self.memory_orchestrator = None
            self.npc_orchestrator = None
            self._init_tasks = {
                "memory": asyncio.create_task(self._finish_memory())
            }

        async def _finish_memory(self):
            await asyncio.sleep(memory_delay)
            self.memory_orchestrator = MemoryStub()

        async def await_orchestrator(self, name: str):
            task = self._init_tasks.get(name)
            if task:
                await task

    monkeypatch.setattr(context_module, "redis", DummyRedisModule)
    monkeypatch.setattr(context_module, "get_conflict_scheduler", lambda: DummyScheduler())

    ctx = BrokerCtx()
    broker = context_module.ContextBroker(ctx)
    scope = context_module.SceneScope()

    start = time.perf_counter()
    section = await broker._fetch_memory_section(scope)
    elapsed = time.perf_counter() - start

    assert elapsed >= memory_delay - 0.01
    assert isinstance(section, context_module.BundleSection)
    assert isinstance(ctx.memory_orchestrator, MemoryStub)
    assert ctx.memory_orchestrator.retrievals >= 0

    # Ensure the initialization task completed cleanly
    await ctx.await_orchestrator("memory")


@pytest.mark.anyio
async def test_context_broker_initialize_is_idempotent(monkeypatch):
    class DummyRedisClient:
        def __init__(self) -> None:
            self.ping_calls = 0

        async def ping(self):
            self.ping_calls += 1
            return True

    dummy_redis = DummyRedisClient()

    def fake_from_url(*_args, **_kwargs):
        return dummy_redis

    class DummyScheduler:
        def get_processor(self, *_args, **_kwargs):
            return None

    class DummyNPCOrchestrator:
        async def get_all_npcs(self):
            return [{"name": "Alice", "id": 1}]

    class DummyContext:
        def __init__(self) -> None:
            self.user_id = 123
            self.conversation_id = 456
            self.npc_orchestrator = DummyNPCOrchestrator()
            self.awaited = []

        async def await_orchestrator(self, name: str):
            self.awaited.append(name)

    monkeypatch.setattr(context_module.redis, "from_url", fake_from_url)
    monkeypatch.setattr(context_module, "get_conflict_scheduler", lambda: DummyScheduler())

    ctx = DummyContext()
    broker = context_module.ContextBroker(ctx)

    await broker.initialize()

    assert broker._is_initialized is True
    assert broker.redis_client is dummy_redis
    assert dummy_redis.ping_calls == 1
    assert broker._npc_alias_cache == {"alice": 1}
    assert ctx.awaited == ["npc"]

    await broker.initialize()

    assert dummy_redis.ping_calls == 1


@pytest.mark.anyio
async def test_world_section_fetch_after_world_ready():
    class Mood(Enum):
        CALM = "calm"

    class Weather(Enum):
        RAINY = "rainy"

    class DummyWorldState:
        def __init__(self) -> None:
            self.current_time = "dusk"
            self.world_mood = Mood.CALM
            self.weather = Weather.RAINY
            self.active_events = ["storm"]
            self.player_vitals = types.SimpleNamespace(
                fatigue=5,
                hunger=15,
                thirst=25,
            )

    class DummyWorldDirector:
        def __init__(self, state: DummyWorldState) -> None:
            self._state = state

        async def get_world_state(self) -> DummyWorldState:
            return self._state

    class BrokerCtx:
        def __init__(self) -> None:
            self.user_id = 1
            self.conversation_id = 2
            self._state = DummyWorldState()
            self.world_director = None
            self._init_tasks = {
                "world": asyncio.create_task(self._initialize_world())
            }

        async def _initialize_world(self) -> None:
            await asyncio.sleep(0)
            self.world_director = DummyWorldDirector(self._state)

        async def await_orchestrator(self, name: str) -> None:
            task = self._init_tasks.get(name)
            if task:
                await task

    ctx = BrokerCtx()
    broker = context_module.ContextBroker(ctx)
    scope = context_module.SceneScope()

    section = await broker._fetch_world_section(scope)

    assert section.data["time"] == ctx._state.current_time
    assert section.data["mood"] == ctx._state.world_mood.value
    assert section.data["weather"] == ctx._state.weather.value
    assert section.data["events"] == ctx._state.active_events
    assert section.data["vitals"]["fatigue"] == ctx._state.player_vitals.fatigue
    assert broker._world_section_cache is section

    cached_section = await broker._fetch_world_section(scope)
    assert cached_section is section
