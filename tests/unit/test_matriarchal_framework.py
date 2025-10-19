import asyncio
import importlib.util
import os
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment shim
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Stub heavy dependencies to keep the test lightweight
# ---------------------------------------------------------------------------

class _StubAgent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def clone(self, **kwargs):
        return self


class _StubRunner:
    @staticmethod
    async def run(*args, **kwargs):  # pragma: no cover - stubbed runner
        class _Result:
            final_output = ""

            def final_output_as(self, _type):
                return _type()

        return _Result()


def _identity_decorator(func=None, **_kwargs):
    def decorator(inner):
        return inner

    if callable(func):
        return decorator(func)
    return decorator


def _handoff(*args, **kwargs):  # pragma: no cover - stubbed handoff
    return {"args": args, "kwargs": kwargs}


class _RunResultStreaming:  # pragma: no cover - sentinel placeholder
    pass


class _GuardrailFunctionOutput:  # pragma: no cover - sentinel placeholder
    pass


class _InputGuardrail:  # pragma: no cover - sentinel placeholder
    pass


class _OutputGuardrail:  # pragma: no cover - sentinel placeholder
    pass


agents_module = types.ModuleType("agents")
agents_module.__path__ = []
agents_module.Agent = _StubAgent
agents_module.Runner = _StubRunner
agents_module.function_tool = _identity_decorator
agents_module.trace = _identity_decorator
agents_module.handoff = _handoff
agents_module.RunResultStreaming = _RunResultStreaming
agents_module.GuardrailFunctionOutput = _GuardrailFunctionOutput
agents_module.InputGuardrail = _InputGuardrail
agents_module.OutputGuardrail = _OutputGuardrail
agents_module.RunConfig = None  # placeholder, set after class definition

agents_run_module = types.ModuleType("agents.run")


class _RunConfig:  # pragma: no cover - stub RunConfig
    def __init__(self, **kwargs):
        self.config = kwargs


agents_run_module.RunConfig = _RunConfig
agents_module.RunConfig = _RunConfig

agents_run_context_module = types.ModuleType("agents.run_context")


class _RunContextWrapper:
    def __init__(self, context=None):
        self.context = context or {}


agents_run_context_module.RunContextWrapper = _RunContextWrapper
agents_module.RunContextWrapper = _RunContextWrapper


embedding_vector_module = types.ModuleType("embedding.vector_store")


async def _generate_embedding(_text):  # pragma: no cover - deterministic stub
    return [0.0]


async def _compute_similarity(_a, _b):  # pragma: no cover - deterministic stub
    return 1.0


embedding_vector_module.generate_embedding = _generate_embedding
embedding_vector_module.compute_similarity = _compute_similarity


@asynccontextmanager
async def _db_connection_context():  # pragma: no cover - stubbed DB context
    class _Conn:
        async def fetchval(self, *args, **kwargs):
            return None

        async def execute(self, *args, **kwargs):
            return None

    yield _Conn()


db_connection_module = types.ModuleType("db.connection")
db_connection_module.get_db_connection_context = _db_connection_context


class _Cache:
    def get(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None

    async def invalidate_pattern(self, *args, **kwargs):
        return None

    def invalidate(self, *args, **kwargs):
        return None


lore_cache_module = types.ModuleType("lore.core.cache")
lore_cache_module.GLOBAL_LORE_CACHE = _Cache()


nyx_directive_handler_module = types.ModuleType("nyx.directive_handler")


class _DirectiveHandler:  # pragma: no cover - minimal directive handler
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


nyx_directive_handler_module.DirectiveHandler = _DirectiveHandler

nyx_governance_helpers_module = types.ModuleType("nyx.governance_helpers")
nyx_governance_helpers_module.with_governance = _identity_decorator
nyx_governance_helpers_module.with_governance_permission = _identity_decorator
nyx_governance_helpers_module.with_action_reporting = _identity_decorator

nyx_core_memory_module = types.ModuleType("nyx.core.memory.vector_store")
nyx_core_memory_module.add = lambda *args, **kwargs: "stubbed"
nyx_core_memory_module.query = lambda *args, **kwargs: []

STUB_MODULES = {
    "agents": agents_module,
    "agents.run": agents_run_module,
    "agents.run_context": agents_run_context_module,
    "agents.tool": types.ModuleType("agents.tool"),
    "embedding.vector_store": embedding_vector_module,
    "db.connection": db_connection_module,
    "lore.core.cache": lore_cache_module,
    "nyx.directive_handler": nyx_directive_handler_module,
    "nyx.governance_helpers": nyx_governance_helpers_module,
    "nyx.core.memory.vector_store": nyx_core_memory_module,
}

for name, module in STUB_MODULES.items():
    sys.modules[name] = module

sys.modules["agents.tool"].FunctionTool = type("FunctionTool", (), {})

# Inject lightweight lore packages to avoid executing heavy __init__ modules
lore_package = types.ModuleType("lore")
lore_package.__path__ = [str(ROOT / "lore")]
sys.modules["lore"] = lore_package

lore_managers_package = types.ModuleType("lore.managers")
lore_managers_package.__path__ = [str(ROOT / "lore/managers")]
sys.modules["lore.managers"] = lore_managers_package

base_manager_spec = importlib.util.spec_from_file_location(
    "lore.managers.base_manager",
    ROOT / "lore/managers/base_manager.py",
)
base_manager_module = importlib.util.module_from_spec(base_manager_spec)
sys.modules["lore.managers.base_manager"] = base_manager_module
base_manager_spec.loader.exec_module(base_manager_module)

lore_frameworks_package = types.ModuleType("lore.frameworks")
lore_frameworks_package.__path__ = [str(ROOT / "lore/frameworks")]
sys.modules["lore.frameworks"] = lore_frameworks_package

matriarchal_spec = importlib.util.spec_from_file_location(
    "lore.frameworks.matriarchal",
    ROOT / "lore/frameworks/matriarchal.py",
)
matriarchal_module = importlib.util.module_from_spec(matriarchal_spec)
sys.modules["lore.frameworks.matriarchal"] = matriarchal_module
matriarchal_spec.loader.exec_module(matriarchal_module)

MatriarchalPowerStructureFramework = matriarchal_module.MatriarchalPowerStructureFramework
BaseLoreManager = base_manager_module.BaseLoreManager

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from nyx.nyx_governance import AgentType


class _GovernorStub:
    def __init__(self):
        self.register_calls = []
        self.directive_calls = []

    async def register_agent(self, agent_type, agent_instance, agent_id):
        self.register_calls.append((agent_type, agent_id, agent_instance))
        return {"success": True}

    async def issue_directive(
        self,
        agent_type,
        agent_id,
        directive_type,
        directive_data,
        priority,
        duration_minutes,
    ):
        self.directive_calls.append(
            {
                "agent_type": agent_type,
                "agent_id": agent_id,
                "directive_type": directive_type,
                "directive_data": directive_data,
                "priority": priority,
                "duration_minutes": duration_minutes,
            }
        )
        return {"success": True}


def test_matriarchal_framework_registers_with_governance_successfully():
    async def _run():
        framework = MatriarchalPowerStructureFramework.__new__(MatriarchalPowerStructureFramework)
        BaseLoreManager.__init__(framework, user_id=1, conversation_id=42)
        governor = _GovernorStub()
        framework.set_governor(governor)

        result = await framework.register_with_governance()

        assert result is True
        assert governor.register_calls
        registered_type, registered_id, _ = governor.register_calls[0]
        assert registered_type == AgentType.NARRATIVE_CRAFTER
        assert registered_id == "matriarchal_power_framework"

    asyncio.run(_run())
