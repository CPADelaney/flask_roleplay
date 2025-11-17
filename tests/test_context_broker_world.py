import asyncio
import sys
from types import SimpleNamespace, ModuleType

# Stub nyx.tasks before importing heavy Nyx modules
stub_tasks_pkg = ModuleType("nyx.tasks")
stub_tasks_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("nyx.tasks", stub_tasks_pkg)
background_stub = ModuleType("nyx.tasks.background")
background_stub.place_enrichment = ModuleType("nyx.tasks.background.place_enrichment")
sys.modules.setdefault("nyx.tasks.background", background_stub)
sys.modules.setdefault("nyx.tasks.heavy", ModuleType("nyx.tasks.heavy"))
sys.modules.setdefault("nyx.tasks.light", ModuleType("nyx.tasks.light"))

from nyx.nyx_agent.context import ContextBroker


class _StubScope:
    pass


class _StubWorldOrchestrator:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def expand_state(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        if callable(self.result):
            return await self.result()
        return self.result


def test_expand_world_passes_arguments_and_returns_payload():
    async def _run():
        orchestrator = _StubWorldOrchestrator({"state": {"time": "noon"}})
        scope = _StubScope()

        ctx = SimpleNamespace(
            world_orchestrator=orchestrator,
            world_fetch_timeout=0.5,
            is_orchestrator_ready=lambda name: name == "world",
        )

        async def _await_orchestrator(name):
            return name == "world"

        ctx.await_orchestrator = _await_orchestrator

        broker = ContextBroker.__new__(ContextBroker)
        broker.ctx = ctx
        broker._active_section_budgets = None

        payload = await ContextBroker.expand_world(
            broker,
            entities=["city"],
            aspects=["time"],
            depth="full",
            scene_scope=scope,
        )

        assert payload == {"state": {"time": "noon"}}
        assert orchestrator.calls[0]["entities"] == ["city"]
        assert orchestrator.calls[0]["aspects"] == ["time"]
        assert orchestrator.calls[0]["depth"] == "full"
        assert orchestrator.calls[0]["scope"] is scope

    asyncio.run(_run())


def test_expand_world_handles_timeout():
    async def _run():
        async def _slow_call():
            await asyncio.sleep(0.05)
            return {"state": {}}

        orchestrator = _StubWorldOrchestrator(_slow_call)

        ctx = SimpleNamespace(
            world_orchestrator=orchestrator,
            world_expand_timeout=0.01,
            is_orchestrator_ready=lambda name: True,
        )

        async def _await_orchestrator(name):
            return True

        ctx.await_orchestrator = _await_orchestrator

        broker = ContextBroker.__new__(ContextBroker)
        broker.ctx = ctx
        broker._active_section_budgets = None

        result = await ContextBroker.expand_world(broker)
        assert result == {}

    asyncio.run(_run())
