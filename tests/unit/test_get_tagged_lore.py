"""Regression coverage for the lore orchestrator's tagged lookup."""

from __future__ import annotations

from typing import Any, Dict, List

import importlib.util
import os
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment shim
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class _SentenceTransformerStub:
    def __init__(self, *args, **kwargs):  # pragma: no cover - lightweight stub
        self._dim = 384

    def encode(self, *args, **kwargs):  # pragma: no cover - lightweight stub
        return []

    def get_sentence_embedding_dimension(self) -> int:  # pragma: no cover - lightweight stub
        return self._dim

    def to(self, *args, **kwargs):  # pragma: no cover - compatibility shim
        return self


def _passthrough_decorator(*_args, **_kwargs):  # pragma: no cover - decorator shim
    def _decorator(func):
        return func

    return _decorator


_STUB_MODULES: Dict[str, ModuleType] = {
    "sentence_transformers": ModuleType("sentence_transformers"),
    "sentence_transformers.models": ModuleType("sentence_transformers.models"),
    "faiss": ModuleType("faiss"),
    "faiss_cpu": ModuleType("faiss_cpu"),
    "torch": ModuleType("torch"),
    "tensorflow": ModuleType("tensorflow"),
    "tensorflow_hub": ModuleType("tensorflow_hub"),
    "huggingface_hub": ModuleType("huggingface_hub"),
    "nyx": ModuleType("nyx"),
    "nyx.integrate": ModuleType("nyx.integrate"),
    "nyx.nyx_governance": ModuleType("nyx.nyx_governance"),
    "nyx.governance_helpers": ModuleType("nyx.governance_helpers"),
    "nyx.scene_keys": ModuleType("nyx.scene_keys"),
    "nyx.constants": ModuleType("nyx.constants"),
    "nyx.directive_handler": ModuleType("nyx.directive_handler"),
    "nyx.core.memory.vector_store": ModuleType("nyx.core.memory.vector_store"),
    "lore.core.cache": ModuleType("lore.core.cache"),
    "lore.core.canon": ModuleType("lore.core.canon"),
    "lore.managers.education": ModuleType("lore.managers.education"),
    "lore.managers.geopolitical": ModuleType("lore.managers.geopolitical"),
    "lore.managers.local_lore": ModuleType("lore.managers.local_lore"),
    "lore.managers.politics": ModuleType("lore.managers.politics"),
    "lore.managers.religion": ModuleType("lore.managers.religion"),
    "logic.conflict_system.conflict_synthesizer": ModuleType("logic.conflict_system.conflict_synthesizer"),
    "logic.conflict_system.dynamic_conflict_template": ModuleType("logic.conflict_system.dynamic_conflict_template"),
    "logic.conflict_system.background_processor": ModuleType("logic.conflict_system.background_processor"),
}


_STUB_MODULES["sentence_transformers"].SentenceTransformer = _SentenceTransformerStub
_STUB_MODULES["sentence_transformers"].models = _STUB_MODULES["sentence_transformers.models"]
_STUB_MODULES["sentence_transformers.models"].Transformer = lambda *args, **kwargs: object()
_STUB_MODULES["sentence_transformers.models"].Pooling = lambda *args, **kwargs: object()


_STUB_MODULES["nyx.integrate"].get_central_governance = lambda *args, **kwargs: None


class _AgentTypeStub:
    NARRATIVE_CRAFTER = "narrative_crafter"
    NPC = "npc"
    LORE = "lore"

    @classmethod
    def __getattr__(cls, name):  # pragma: no cover - fallback
        return name.lower()


class _DirectiveTypeStub:
    LORE = "lore"

    @classmethod
    def __getattr__(cls, name):  # pragma: no cover - fallback
        return name.lower()


class _DirectivePriorityStub:
    NORMAL = "normal"

    @classmethod
    def __getattr__(cls, name):  # pragma: no cover - fallback
        return name.lower()


_STUB_MODULES["nyx.nyx_governance"].AgentType = _AgentTypeStub
_STUB_MODULES["nyx.nyx_governance"].DirectiveType = _DirectiveTypeStub
_STUB_MODULES["nyx.nyx_governance"].DirectivePriority = _DirectivePriorityStub

_STUB_MODULES["nyx.constants"].AgentType = _AgentTypeStub
_STUB_MODULES["nyx.constants"].DirectiveType = _DirectiveTypeStub
_STUB_MODULES["nyx.constants"].DirectivePriority = _DirectivePriorityStub

_STUB_MODULES["nyx.directive_handler"].DirectiveHandler = type("DirectiveHandler", (), {})

_STUB_MODULES["nyx.governance_helpers"].with_governance = _passthrough_decorator
_STUB_MODULES["nyx.governance_helpers"].with_governance_permission = _passthrough_decorator
_STUB_MODULES["nyx.scene_keys"].generate_scene_cache_key = lambda *args, **kwargs: "stub-key"

_STUB_MODULES["nyx"].integrate = _STUB_MODULES["nyx.integrate"]
_STUB_MODULES["nyx"].nyx_governance = _STUB_MODULES["nyx.nyx_governance"]
_STUB_MODULES["nyx"].governance_helpers = _STUB_MODULES["nyx.governance_helpers"]
_STUB_MODULES["nyx"].scene_keys = _STUB_MODULES["nyx.scene_keys"]
_STUB_MODULES["nyx"].constants = _STUB_MODULES["nyx.constants"]
_STUB_MODULES["nyx"].directive_handler = _STUB_MODULES["nyx.directive_handler"]
_STUB_MODULES["nyx"].__path__ = []  # pragma: no cover - package shim


async def _vector_add(*args, **kwargs):  # pragma: no cover - async shim
    return "stub-id"


async def _vector_query(*args, **kwargs):  # pragma: no cover - async shim
    return []


_STUB_MODULES["nyx.core.memory.vector_store"].add = _vector_add
_STUB_MODULES["nyx.core.memory.vector_store"].query = _vector_query


def _canon_noop(*_args, **_kwargs):  # pragma: no cover - canon shim
    return None


for _func in [
    "initialize_canon_memory_integration",
    "log_canonical_event",
    "find_or_create_npc",
    "find_or_create_nation",
    "find_or_create_location",
    "find_or_create_faction",
    "find_or_create_historical_event",
    "find_or_create_urban_myth",
    "find_or_create_landmark",
    "find_or_create_event",
    "find_or_create_quest",
    "sync_entity_to_memory",
    "ensure_embedding_columns",
]:
    setattr(_STUB_MODULES["lore.core.canon"], _func, _canon_noop)


def _manager_getattr(_name):  # pragma: no cover - manager shim
    return SimpleNamespace()


for _mgr in [
    "lore.managers.education",
    "lore.managers.geopolitical",
    "lore.managers.local_lore",
    "lore.managers.politics",
    "lore.managers.religion",
]:
    _STUB_MODULES[_mgr].__getattr__ = lambda name, _getter=_manager_getattr: _getter(name)


_STUB_MODULES["lore.core.cache"].GLOBAL_LORE_CACHE = {}

_STUB_MODULES["logic.conflict_system.conflict_synthesizer"].get_synthesizer = lambda *args, **kwargs: None
_STUB_MODULES["logic.conflict_system.background_processor"].get_conflict_scheduler = lambda *args, **kwargs: None


for _name, _module in _STUB_MODULES.items():
    sys.modules.setdefault(_name, _module)


module_path = ROOT / "lore" / "lore_orchestrator.py"
spec = importlib.util.spec_from_file_location("lore.lore_orchestrator", module_path)
assert spec and spec.loader  # pragma: no cover - defensive
_lore_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _lore_module
spec.loader.exec_module(_lore_module)
LoreOrchestrator = _lore_module.LoreOrchestrator
OrchestratorConfig = _lore_module.OrchestratorConfig


class FakeConnection:
    """Minimal asyncpg-like connection for testing."""

    def __init__(self, table_exists: Dict[str, bool], data: Dict[str, List[Dict[str, Any]]]):
        self._table_exists = table_exists
        self._data = data
        self.fetch_queries: List[str] = []
        self.fetchval_queries: List[Any] = []

    async def fetch(self, query: str, *params: Any):  # pragma: no cover - signature parity
        self.fetch_queries.append(query)
        lowered = query.lower()
        if "from nations" in lowered:
            return self._data.get("nations", [])
        if "from religions" in lowered:
            return self._data.get("religions", [])
        if "from events" in lowered:
            return self._data.get("events", [])
        raise AssertionError(f"Unexpected fetch query: {query}")

    async def fetchval(self, query: str, *params: Any):  # pragma: no cover - signature parity
        self.fetchval_queries.append((query, params))
        lowered = query.lower()
        if "information_schema.tables" in lowered:
            table_name = params[0]
            return self._table_exists.get(table_name, False)
        raise AssertionError(f"Unexpected fetchval query: {query}")


class FakeConnectionContext:
    """Async context manager returning the fake connection."""

    def __init__(self, connection: FakeConnection):
        self._connection = connection

    async def __aenter__(self):  # pragma: no cover - trivial
        return self._connection

    async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():  # pragma: no cover - test backend shim
    return "asyncio"


async def test_get_tagged_lore_queries_existing_columns(monkeypatch):
    """Ensure tagged lookups use the real schema and skip missing tables."""

    orchestrator = LoreOrchestrator(
        user_id=1,
        conversation_id=1,
        config=OrchestratorConfig(enable_cache=False),
    )

    fake_data = {
        "nations": [
            {
                "id": 42,
                "name": "Aurelia",
                "description": "A matriarchal nation of valor",
                "cultural_traits": ["valor", "honor"],
                "major_resources": ["iron"],
                "major_cities": ["Aurel"],
                "neighboring_nations": ["Borea"],
                "notable_features": "Ancient forest canopy",
            }
        ],
        "events": [
            {
                "id": 5,
                "event_name": "Battle of Dawn",
                "description": "A pivotal battle tied to valor",
                "location": "Aurelia",
                "year": 512,
                "month": 3,
                "day": 21,
                "time_of_day": "Morning",
            }
        ],
    }

    fake_connection = FakeConnection(
        table_exists={"nations": True, "religions": False, "events": True},
        data=fake_data,
    )

    def fake_get_db_connection_context(*args: Any, **kwargs: Any):  # pragma: no cover - passthrough
        return FakeConnectionContext(fake_connection)

    monkeypatch.setattr(_lore_module, "get_db_connection_context", fake_get_db_connection_context)

    results = await orchestrator.get_tagged_lore(["Valor"])

    assert "Valor" in results
    valor_items = results["Valor"]
    assert any(item["type"] == "nation" and item["id"] == 42 for item in valor_items)

    event_item = next(item for item in valor_items if item["type"] == "event")
    assert event_item["id"] == 5
    assert event_item["details"]["location"] == "Aurelia"
    assert event_item["details"]["date"] == {"year": 512, "month": 3, "day": 21, "time_of_day": "Morning"}

    lowered_queries = [query.lower() for query in fake_connection.fetch_queries]
    assert all("nation_id" not in query for query in lowered_queries)
    assert all("event_id" not in query for query in lowered_queries)
    assert not any("from religions" in query for query in lowered_queries)

    assert any(params[0] == "religions" for _, params in fake_connection.fetchval_queries)
