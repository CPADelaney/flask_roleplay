import asyncio
import importlib
import os
import sys
import time
import types
import typing
from types import SimpleNamespace

from typing_extensions import TypedDict as _CompatTypedDict

# Ensure repository modules are importable
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
typing.TypedDict = _CompatTypedDict  # type: ignore[attr-defined]


class _StubSentenceTransformer:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - shim
        pass

    def encode(self, *args, **kwargs):  # pragma: no cover - shim
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


from nyx.nyx_agent.context import BundleSection, ContextBroker, SceneScope


class _StubConflictSynthesizer:
    def __init__(self, responses):
        self._responses = responses

    async def get_conflict_state(self, conflict_id):
        delay, payload = self._responses.get(conflict_id, (0, None))
        await asyncio.sleep(delay)
        if isinstance(payload, Exception):
            raise payload
        return payload


async def _run_conflict_section_timeout_check():
    responses = {
        1: (
            0.01,
            {
                "conflict_type": "trade dispute",
                "subsystem_data": {
                    "tension": {"level": 0.8},
                    "stakeholder": {"stakeholders": ["Guild"]},
                },
            },
        ),
        2: (
            0.2,
            {
                "conflict_type": "border clash",
                "subsystem_data": {
                    "tension": {"level": 0.9},
                    "stakeholder": {"stakeholders": ["Legion"]},
                },
            },
        ),
        3: (
            0.01,
            {
                "conflict_type": "mystery",
                "subsystem_data": {
                    "tension": {"level": 0.4},
                    "stakeholder": {"stakeholders": ["Detectives"]},
                },
            },
        ),
    }

    synthesizer = _StubConflictSynthesizer(responses)

    async def calculate_conflict_tensions():
        await asyncio.sleep(0.2)
        return {"npc:101|npc:202": 0.95}

    async def _await_orchestrator(_: str) -> bool:
        return True

    ctx = SimpleNamespace(
        conflict_synthesizer=synthesizer,
        calculate_conflict_tensions=calculate_conflict_tensions,
        conflict_state_timeout=0.05,
        conflict_tension_timeout=0.05,
        await_orchestrator=_await_orchestrator,
    )

    broker = ContextBroker.__new__(ContextBroker)
    broker.ctx = ctx

    scope = SceneScope()
    scope.conflict_ids.update({1, 2, 3})
    scope.npc_ids.update({101})
    scope.link_hints["related_npcs"] = [202]

    start = time.perf_counter()
    section = await broker._fetch_conflict_section(scope)
    duration = time.perf_counter() - start

    # Even though one conflict and the tension call take 0.2s, per-task timeouts keep total low.
    assert duration < 0.15

    active_ids = {item["id"] for item in section.data["active"]}
    assert active_ids == {1, 3}

    # Tension calculation should have timed out, yielding an empty mapping.
    assert section.data["tensions"] == {}


def test_conflict_section_times_out_slow_calls():
    asyncio.run(_run_conflict_section_timeout_check())


async def _run_fetch_returns_stale(monkeypatch):
    async def _await_orchestrator(_: str) -> bool:
        return True

    broker = ContextBroker.__new__(ContextBroker)
    broker.ctx = SimpleNamespace(
        await_orchestrator=_await_orchestrator,
        user_id=1,
        conversation_id=2,
    )
    broker.section_ttls = {"test": 0.5}

    stale_section = BundleSection(data={"value": "stale"})
    setattr(broker, "test_cache", stale_section)
    setattr(broker, "test_cache_expires_at", time.time() - 10)

    loop = asyncio.get_running_loop()
    created_tasks = []

    def _capture_task(coro):
        task = loop.create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", _capture_task)

    refresh_event = asyncio.Event()
    call_counter = 0

    async def _fetcher() -> BundleSection:
        nonlocal call_counter
        call_counter += 1
        if call_counter == 1:
            raise asyncio.TimeoutError()
        await asyncio.sleep(0)
        section = BundleSection(data={"value": "fresh"})
        refresh_event.set()
        return section

    def _fallback() -> BundleSection:
        return BundleSection(data={"value": "fallback"})

    result = await broker.perform_fetch_and_cache(
        orchestrator_name="test",
        cache_attribute="test_cache",
        expires_attribute="test_cache_expires_at",
        fetcher=_fetcher,
        fallback_factory=_fallback,
        ttl=0.5,
    )

    assert result is stale_section
    assert len(created_tasks) == 1

    await asyncio.wait_for(refresh_event.wait(), timeout=0.1)
    await created_tasks[0]

    refreshed_section = getattr(broker, "test_cache")
    assert refreshed_section.data == {"value": "fresh"}
    assert getattr(broker, "test_cache_expires_at") > time.time()
    assert call_counter == 2


def test_perform_fetch_returns_stale_on_timeout_and_refreshes_cache(monkeypatch):
    asyncio.run(_run_fetch_returns_stale(monkeypatch))


async def _run_fetch_timeout_without_cache(monkeypatch):
    async def _await_orchestrator(_: str) -> bool:
        return True

    broker = ContextBroker.__new__(ContextBroker)
    broker.ctx = SimpleNamespace(
        await_orchestrator=_await_orchestrator,
        user_id=3,
        conversation_id=4,
    )
    broker.section_ttls = {"test": 0.5}

    def _fallback() -> BundleSection:
        return BundleSection(data={"value": "fallback"})

    async def _fetcher() -> BundleSection:
        raise asyncio.TimeoutError()

    def _no_task(_: typing.Any) -> None:
        raise AssertionError("Background refresh should not be scheduled without cache")

    monkeypatch.setattr(asyncio, "create_task", _no_task)

    result = await broker.perform_fetch_and_cache(
        orchestrator_name="test",
        cache_attribute="test_cache",
        expires_attribute="test_cache_expires_at",
        fetcher=_fetcher,
        fallback_factory=_fallback,
        ttl=0.5,
    )

    assert result.data == {"value": "fallback"}
    assert getattr(broker, "test_cache", None) is None


def test_perform_fetch_timeout_without_cache_returns_fallback(monkeypatch):
    asyncio.run(_run_fetch_timeout_without_cache(monkeypatch))
