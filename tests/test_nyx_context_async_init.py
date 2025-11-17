import asyncio
import os
import sys
import time
import types
import typing
import typing_extensions
from enum import Enum
from typing import Any, Dict

import pytest

typing.TypedDict = typing_extensions.TypedDict
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import context.cache_warmup as cache_warmup

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

    async def _slow_memory(self, *, warm_minimal: bool = False):
        if warm_minimal:
            return
        await asyncio.sleep(memory_delay)
        self.memory_orchestrator = sentinel_memory

    async def _fast_noop(self, *, warm_minimal: bool = False):
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
        def __init__(self) -> None:
            self.retrievals = 0

        async def retrieve_memories(self, *args, **kwargs):
            self.retrievals += 1
            entity_id = kwargs.get("entity_id")
            return {
                "memories": [
                    {
                        "id": entity_id,
                        "text": f"memory-{entity_id}",
                    }
                ]
            }

        async def retrieve_recent_memories(self, *args, **kwargs):
            return {
                "memories": [
                    {
                        "id": "recent-1",
                        "text": "recent memory",
                    }
                ]
            }

        async def analyze_patterns(self, *args, **kwargs):
            return {
                "predictions": [
                    {
                        "pattern": "arc",
                        "confidence": 0.9,
                    }
                ]
            }

    class BrokerCtx:
        def __init__(self) -> None:
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

        async def _finish_memory(self) -> None:
            await asyncio.sleep(memory_delay)
            self.memory_orchestrator = MemoryStub()

        async def analyze_memory_patterns(self, *args, **kwargs):
            return {
                "predictions": [
                    {
                        "pattern": "arc",
                        "confidence": 0.9,
                    }
                ]
            }

        def is_orchestrator_ready(self, name: str) -> bool:
            task = self._init_tasks.get(name)
            return bool(task and task.done())

        async def await_orchestrator(self, name: str) -> bool:
            task = self._init_tasks.get(name)
            if not task:
                return False
            await task
            return True

    monkeypatch.setattr(context_module, "redis", DummyRedisModule)
    monkeypatch.setattr(context_module, "get_conflict_scheduler", lambda: DummyScheduler())

    ctx = BrokerCtx()
    broker = context_module.ContextBroker(ctx)
    scope = context_module.SceneScope()
    scope.npc_ids = {1}
    scope.topics = {"mystery"}
    scope.location_id = "loc-1"

    start = time.perf_counter()
    fallback_section = await broker._fetch_memory_section(scope)
    elapsed = time.perf_counter() - start

    assert elapsed < memory_delay / 2
    assert isinstance(fallback_section, context_module.BundleSection)
    assert fallback_section.data.relevant == []
    assert fallback_section.data.recent == []
    assert fallback_section.data.patterns == []

    # Ensure the initialization task completed cleanly and readiness flips
    assert await ctx.await_orchestrator("memory") is True
    assert ctx.is_orchestrator_ready("memory") is True
    assert isinstance(ctx.memory_orchestrator, MemoryStub)

    ready_section = await broker._fetch_memory_section(scope)

    assert ready_section.data.relevant  # orchestrator data populated
    assert ready_section.data.recent
    assert ready_section.data.patterns
    assert ctx.memory_orchestrator.retrievals > 0


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

        def is_orchestrator_ready(self, name: str) -> bool:
            return True

        async def await_orchestrator(self, name: str):
            self.awaited.append(name)
            return True

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

    class DummyWorldOrchestrator:
        def __init__(self, state: DummyWorldState) -> None:
            self._state = state

        @property
        def director(self):  # pragma: no cover - compatibility shim
            return None

        async def get_scene_bundle(self, scope=None) -> typing.Dict[str, typing.Any]:
            vitals = self._state.player_vitals
            return {
                "summary": {
                    "time": self._state.current_time,
                    "mood": self._state.world_mood.value,
                    "weather": self._state.weather.value,
                    "events": list(self._state.active_events),
                    "vitals": {
                        "fatigue": vitals.fatigue,
                        "hunger": vitals.hunger,
                        "thirst": vitals.thirst,
                    },
                }
            }

        async def expand_state(self, **_kwargs):  # pragma: no cover - unused in test
            bundle = await self.get_scene_bundle()
            return {"bundle": bundle, "world_state": None}

    class BrokerCtx:
        def __init__(self) -> None:
            self.user_id = 1
            self.conversation_id = 2
            self._state = DummyWorldState()
            self.world_orchestrator = None
            self._init_tasks = {
                "world": asyncio.create_task(self._initialize_world())
            }

        async def _initialize_world(self) -> None:
            await asyncio.sleep(0)
            self.world_orchestrator = DummyWorldOrchestrator(self._state)

        def is_orchestrator_ready(self, name: str) -> bool:
            task = self._init_tasks.get(name)
            return bool(task and task.done())

        async def await_orchestrator(self, name: str) -> bool:
            task = self._init_tasks.get(name)
            if not task:
                return False
            await task
            return True

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


@pytest.mark.anyio
async def test_lore_initialization_shielding_allows_retry(monkeypatch):
    lore_ready = asyncio.Event()

    class DummyStore:
        def __init__(self) -> None:
            self._data = {}

        def get(self, user_key, conversation_key):
            return self._data.get((user_key, conversation_key))

        def put(self, user_key, conversation_key, value):
            self._data[(user_key, conversation_key)] = value

    class DummyLoreOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        async def get_scene_bundle(self, scope):
            self.calls += 1
            return {
                "data": {
                    "location": {"name": "Arcadia"},
                    "world": {"factions": []},
                    "canonical_rules": ["Rule of Dawn"],
                },
                "canonical": True,
                "last_changed_at": time.time(),
                "version": "lore_stub",
            }

    async def _fake_fetch_snapshot(*_args, **_kwargs):
        return None

    async def _fake_get_context(*_args, **_kwargs):
        return {}

    async def _noop_hydrate(self):
        return None

    async def _noop_refresh(self, previous_location_id=None):
        return None

    async def _fast_memory(self, *, warm_minimal: bool = False):
        if warm_minimal:
            return
        self.memory_orchestrator = object()

    async def _fast_npc(self, *, warm_minimal: bool = False):
        if warm_minimal:
            return
        self.npc_orchestrator = object()

    async def _fast_conflict(self, *, warm_minimal: bool = False):
        if warm_minimal:
            return
        self.conflict_synthesizer = object()

    async def _slow_lore(self, *, warm_minimal: bool = False):
        if warm_minimal:
            return
        await lore_ready.wait()
        self.lore_orchestrator = DummyLoreOrchestrator()

    async def _fast_broker_init(self):
        self._is_initialized = True

    monkeypatch.setattr(context_module, "_SNAPSHOT_STORE", DummyStore())
    monkeypatch.setattr(context_module, "fetch_canonical_snapshot", _fake_fetch_snapshot)
    monkeypatch.setattr(context_module, "get_comprehensive_context", _fake_get_context)
    monkeypatch.setattr(context_module, "WORLD_SIMULATION_AVAILABLE", False)
    monkeypatch.setattr(context_module.NyxContext, "_hydrate_location_from_db", _noop_hydrate)
    monkeypatch.setattr(context_module.NyxContext, "_refresh_location_from_context", _noop_refresh)
    monkeypatch.setattr(context_module.NyxContext, "_init_memory_orchestrator", _fast_memory)
    monkeypatch.setattr(context_module.NyxContext, "_init_npc_orchestrator", _fast_npc)
    monkeypatch.setattr(context_module.NyxContext, "_init_conflict_synthesizer", _fast_conflict)
    monkeypatch.setattr(context_module.NyxContext, "_init_lore_orchestrator", _slow_lore)
    monkeypatch.setattr(context_module.ContextBroker, "initialize", _fast_broker_init)

    ctx = context_module.NyxContext(user_id=11, conversation_id=22)
    await ctx.initialize()

    lore_task = ctx.get_init_task("lore")
    assert lore_task is not None and not lore_task.done()
    assert not ctx.is_orchestrator_ready("lore")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            ctx.context_broker.await_orchestrator("lore"),
            timeout=0.05,
        )

    assert lore_task.cancelled() is False
    assert "lore" not in ctx._init_failures

    lore_ready.set()
    await ctx.await_orchestrator("lore")
    assert ctx.is_orchestrator_ready("lore")
    assert isinstance(ctx.lore_orchestrator, DummyLoreOrchestrator)

    scope = context_module.SceneScope()
    section = await ctx.context_broker._fetch_lore_section(scope)

    assert isinstance(section.data, context_module.LoreSectionData)
    assert section.data.location.get("name") == "Arcadia"
    assert section.canonical is True


@pytest.mark.anyio
async def test_warm_user_context_minimal_mode(monkeypatch):
    monkeypatch.setattr(cache_warmup, "_context_warm_promises", {})
    monkeypatch.setattr(cache_warmup.settings, "CONFLICT_EAGER_WARMUP", False)

    memory_event = asyncio.Event()
    lore_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, memory_event.set)
    loop.call_later(0.1, lore_event.set)

    class DummyRedis:
        def __init__(self) -> None:
            self.get_calls = []
            self.set_calls = []
            self.delete_calls = []

        def get(self, key):
            self.get_calls.append(key)
            return None

        def setex(self, key, ttl, value):
            self.set_calls.append((key, ttl, value))

        def delete(self, key):
            self.delete_calls.append(key)

    dummy_redis = DummyRedis()

    class StubNyxContext:
        instances = []

        def __init__(self, user_id: int, conversation_id: int) -> None:
            self.user_id = user_id
            self.conversation_id = conversation_id
            self.initialized = False
            self.memory_ready = False
            self.lore_ready = False
            self.awaited = []
            self.build_called = False
            self.memory_orchestrator = None
            self.lore_bundle = None
            StubNyxContext.instances.append(self)

        async def initialize(self, *, warm_minimal: bool = False) -> None:
            self.initialized = True
            self.warm_minimal = warm_minimal

        async def await_orchestrator(self, name: str) -> bool:
            self.awaited.append(name)
            if name == "memory":
                await memory_event.wait()
                self.memory_ready = True
                self.memory_orchestrator = {"status": "ready"}
                return True
            if name == "lore":
                assert self.memory_ready is True
                await lore_event.wait()
                self.lore_ready = True
                self.lore_bundle = {"status": "ready"}
                return True
            if name == "context_broker":
                return True
            return False

        async def build_context_for_input(self, *_args):
            if not (self.memory_ready and self.lore_ready):
                raise AssertionError("Context built before orchestrators became ready")
            self.build_called = True
            return None

        async def warm_minimal_context(self) -> Dict[str, Any]:
            self.build_called = False
            await self.await_orchestrator("context_broker")
            return {
                "status": "warmed",
                "mode": "minimal",
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
            }

    monkeypatch.setattr(cache_warmup, "NyxContext", StubNyxContext)

    result = await cache_warmup.warm_user_context_cache(
        user_id=5,
        conversation_id=7,
        redis_client=dummy_redis,
    )

    assert result == {
        "status": "warmed",
        "user_id": 5,
        "conversation_id": 7,
        "mode": "minimal",
    }
    assert dummy_redis.get_calls == ["ctx:warmed:5:7"]
    assert dummy_redis.set_calls == [("ctx:warmed:5:7", 600, "1")]
    assert dummy_redis.delete_calls == []

    context = StubNyxContext.instances[-1]
    assert context.initialized is True
    assert context.warm_minimal is True
    assert context.build_called is False
    assert context.memory_orchestrator is None
    assert context.lore_bundle is None
    assert context.awaited.count("context_broker") >= 1
    assert context.awaited.count("memory") == 0
    assert context.awaited.count("lore") == 0


@pytest.mark.anyio
async def test_warm_user_context_eager_opt_in(monkeypatch):
    monkeypatch.setattr(cache_warmup, "_context_warm_promises", {})
    monkeypatch.setattr(cache_warmup.settings, "CONFLICT_EAGER_WARMUP", True)

    memory_event = asyncio.Event()
    lore_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, memory_event.set)
    loop.call_later(0.1, lore_event.set)

    class DummyRedis:
        def __init__(self) -> None:
            self.get_calls = []
            self.set_calls = []
            self.delete_calls = []

        def get(self, key):
            self.get_calls.append(key)
            return None

        def setex(self, key, ttl, value):
            context = StubNyxContext.instances[-1]
            assert context.build_called is True
            assert context.memory_ready is True
            assert context.lore_ready is True
            self.set_calls.append((key, ttl, value))

        def delete(self, key):
            self.delete_calls.append(key)

    dummy_redis = DummyRedis()

    class StubNyxContext:
        instances = []

        def __init__(self, user_id: int, conversation_id: int) -> None:
            self.user_id = user_id
            self.conversation_id = conversation_id
            self.initialized = False
            self.memory_ready = False
            self.lore_ready = False
            self.awaited = []
            self.build_called = False
            self.memory_orchestrator = None
            self.lore_bundle = None
            StubNyxContext.instances.append(self)

        async def initialize(self, *, warm_minimal: bool = False) -> None:
            self.initialized = True
            self.warm_minimal = warm_minimal

        async def await_orchestrator(self, name: str) -> bool:
            self.awaited.append(name)
            if name == "memory":
                await memory_event.wait()
                self.memory_ready = True
                self.memory_orchestrator = {"status": "ready"}
                return True
            if name == "lore":
                assert self.memory_ready is True
                await lore_event.wait()
                self.lore_ready = True
                self.lore_bundle = {"status": "ready"}
                return True
            if name == "context_broker":
                return True
            return False

        async def build_context_for_input(self, *_args):
            if not (self.memory_ready and self.lore_ready):
                raise AssertionError("Context built before orchestrators became ready")
            self.build_called = True
            return None

        async def warm_minimal_context(self) -> Dict[str, Any]:  # pragma: no cover - should not run
            raise AssertionError("Minimal warm helper should not be called when eager flag is set")

    monkeypatch.setattr(cache_warmup, "NyxContext", StubNyxContext)

    result = await cache_warmup.warm_user_context_cache(
        user_id=9,
        conversation_id=11,
        redis_client=dummy_redis,
    )

    assert result == {
        "status": "warmed",
        "user_id": 9,
        "conversation_id": 11,
        "mode": "full",
    }
    assert dummy_redis.get_calls == ["ctx:warmed:9:11"]
    assert dummy_redis.set_calls == [("ctx:warmed:9:11", 600, "1")]

    context = StubNyxContext.instances[-1]
    assert context.initialized is True
    assert context.warm_minimal is False
    assert context.build_called is True
    assert context.memory_orchestrator == {"status": "ready"}
    assert context.lore_bundle == {"status": "ready"}
    assert context.awaited.count("memory") >= 1
    assert context.awaited.count("lore") >= 1
