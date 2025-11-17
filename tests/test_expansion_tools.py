import asyncio
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock

# Stub nyx.tasks packages before importing Nyx modules to avoid heavy side-effects
stub_tasks_pkg = ModuleType("nyx.tasks")
stub_tasks_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("nyx.tasks", stub_tasks_pkg)
background_stub = ModuleType("nyx.tasks.background")
background_stub.place_enrichment = ModuleType("nyx.tasks.background.place_enrichment")
sys.modules.setdefault("nyx.tasks.background", background_stub)
sys.modules.setdefault("nyx.tasks.heavy", ModuleType("nyx.tasks.heavy"))
sys.modules.setdefault("nyx.tasks.light", ModuleType("nyx.tasks.light"))

from nyx.nyx_agent.assembly import ExpansionTools


def test_expand_world_state_logs_and_returns_payload():
    async def _run():
        broker = SimpleNamespace()
        broker.expand_world = AsyncMock(return_value={"state": {"time": "noon"}})
        governance = SimpleNamespace(log_context_expansion=AsyncMock())
        broker.ctx = SimpleNamespace(governance=governance)

        tools = ExpansionTools(broker)

        result = await tools.expand_world_state(
            entities=["region"],
            aspects=["time", "tension"],
            depth="full",
            scene_scope=None,
        )

        broker.expand_world.assert_awaited_once()
        assert result["world"] == {"state": {"time": "noon"}}
        assert result["requested_entities"] == ["region"]
        assert result["requested_aspects"] == ["time", "tension"]
        assert tools._expansion_history[-1]["type"] == "world_state"
        governance.log_context_expansion.assert_awaited_once()
        call_kwargs = governance.log_context_expansion.await_args.kwargs
        assert call_kwargs["expansion_type"] == "world_state"
        assert call_kwargs["metadata"]["aspects"] == ["time", "tension"]

    asyncio.run(_run())


def test_expand_world_state_handles_missing_governance():
    async def _run():
        broker = SimpleNamespace()
        broker.expand_world = AsyncMock(return_value={"summary": {}})
        broker.ctx = SimpleNamespace(governance=None)

        tools = ExpansionTools(broker)

        result = await tools.expand_world_state()
        assert result["world"] == {"summary": {}}

    asyncio.run(_run())
