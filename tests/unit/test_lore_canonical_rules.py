from __future__ import annotations

import os
from pathlib import Path
import sys
import typing
from types import ModuleType, SimpleNamespace

# Stub heavy optional dependencies before importing the orchestrator module.
_stub_modules = {
    "sentence_transformers": ModuleType("sentence_transformers"),
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
    "lore.core.cache": ModuleType("lore.core.cache"),
    "lore.core.canon": ModuleType("lore.core.canon"),
    "lore.managers.education": ModuleType("lore.managers.education"),
    "lore.managers.geopolitical": ModuleType("lore.managers.geopolitical"),
    "lore.managers.local_lore": ModuleType("lore.managers.local_lore"),
    "lore.managers.politics": ModuleType("lore.managers.politics"),
    "lore.managers.religion": ModuleType("lore.managers.religion"),
}


class _SentenceTransformerStub:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):  # pragma: no cover - compatibility stub
        return []


_stub_modules["sentence_transformers"].SentenceTransformer = _SentenceTransformerStub

def _passthrough_decorator(*_args, **_kwargs):
    def _decorator(func):
        return func

    return _decorator


_stub_modules["nyx.integrate"].get_central_governance = lambda *args, **kwargs: None
class _AgentTypeStub:
    NARRATIVE_CRAFTER = "narrative_crafter"
    NPC = "npc"
    LORE = "lore"

    @classmethod
    def __getattr__(cls, name):
        return name.lower()


class _DirectiveTypeStub:
    LORE = "lore"
    @classmethod
    def __getattr__(cls, name):
        return name.lower()


class _DirectivePriorityStub:
    NORMAL = "normal"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def __getattr__(cls, name):
        return name.lower()


_stub_modules["nyx.nyx_governance"].AgentType = _AgentTypeStub
_stub_modules["nyx.nyx_governance"].DirectiveType = _DirectiveTypeStub
_stub_modules["nyx.nyx_governance"].DirectivePriority = _DirectivePriorityStub
_stub_modules["nyx.nyx_governance"].NyxUnifiedGovernor = type(
    "NyxUnifiedGovernor",
    (),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "ensure_initialized": lambda self: None,
    },
)
_stub_modules["nyx.constants"].AgentType = _AgentTypeStub
_stub_modules["nyx.constants"].DirectiveType = _DirectiveTypeStub
_stub_modules["nyx.constants"].DirectivePriority = _DirectivePriorityStub
_stub_modules["nyx.directive_handler"].DirectiveHandler = type("DirectiveHandler", (), {})
_stub_modules["nyx.governance_helpers"].with_governance = _passthrough_decorator
_stub_modules["nyx.governance_helpers"].with_governance_permission = _passthrough_decorator
_stub_modules["nyx.governance_helpers"].with_action_reporting = _passthrough_decorator
_stub_modules["nyx.scene_keys"].generate_scene_cache_key = lambda *args, **kwargs: "stub-key"

_stub_modules["nyx"].integrate = _stub_modules["nyx.integrate"]
_stub_modules["nyx"].nyx_governance = _stub_modules["nyx.nyx_governance"]
_stub_modules["nyx"].governance_helpers = _stub_modules["nyx.governance_helpers"]
_stub_modules["nyx"].scene_keys = _stub_modules["nyx.scene_keys"]
_stub_modules["nyx"].constants = _stub_modules["nyx.constants"]
_stub_modules["nyx"].directive_handler = _stub_modules["nyx.directive_handler"]
_stub_modules["nyx"].__path__ = []

_stub_modules["lore.core.cache"].GLOBAL_LORE_CACHE = {}

def _canon_noop(*_args, **_kwargs):
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
    setattr(_stub_modules["lore.core.canon"], _func, _canon_noop)

def _manager_getattr(_name):
    return SimpleNamespace()


for _mgr in [
    "lore.managers.education",
    "lore.managers.geopolitical",
    "lore.managers.local_lore",
    "lore.managers.politics",
    "lore.managers.religion",
]:
    _stub_modules[_mgr].__getattr__ = lambda name, _getter=_manager_getattr: _getter(name)

for _name, _module in _stub_modules.items():
    sys.modules.setdefault(_name, _module)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

sys.path.append(str(Path(__file__).resolve().parents[2]))

from contextlib import asynccontextmanager

import importlib.util

import pytest

pytestmark = pytest.mark.anyio

_module_path = Path(__file__).resolve().parents[2] / "lore" / "lore_orchestrator.py"
_spec = importlib.util.spec_from_file_location("lore.lore_orchestrator", _module_path)
assert _spec and _spec.loader
_lore_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _lore_module
_spec.loader.exec_module(_lore_module)
LoreOrchestrator = _lore_module.LoreOrchestrator


class _ValidatorStub:
    async def get_global_rules(self):
        return [
            "Keep timeline coherent",
            "Protect relationship continuity",
            "Uphold established governance",
        ]


async def _noop_canon_module(self) -> object:  # pragma: no cover - patched helper
    return object()


async def _validator_factory(self) -> _ValidatorStub:
    return _ValidatorStub()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.unit
async def test_canonical_rules_include_location_canon(monkeypatch):
    orchestrator = LoreOrchestrator(user_id=1, conversation_id=1)

    monkeypatch.setattr(LoreOrchestrator, "_get_canon_module", _noop_canon_module)
    monkeypatch.setattr(LoreOrchestrator, "_get_canon_validation", _validator_factory)

    location_row = {
        "location_name": "Eldertree Market",
        "notable_features": ["Floating stalls", "Bioluminescent canopy"],
        "hidden_aspects": [{"name": "Secret catacombs", "description": "Beneath the market"}],
        "access_restrictions": ["Permit required after dusk"],
        "local_customs": [
            {"title": "Morning salute", "details": "Merchants ring bells at sunrise"}
        ],
    }

    class _Conn:
        async def fetchrow(self, query, *args, **kwargs):
            if "FROM Locations" in query:
                return location_row
            return None

        async def fetch(self, query, *args, **kwargs):
            return []

    @asynccontextmanager
    async def _ctx():
        yield _Conn()

    monkeypatch.setattr(_lore_module, "get_db_connection_context", _ctx)

    scope = SimpleNamespace(location_id=42, nation_ids=set())

    rules = await orchestrator._get_canonical_rules_for_scope(scope)

    assert any("Floating stalls" in rule for rule in rules)
    assert any("Secret catacombs" in rule for rule in rules)
    assert any("Morning salute" in rule for rule in rules)


@pytest.mark.unit
async def test_canonical_rules_support_slug_lookup(monkeypatch):
    orchestrator = LoreOrchestrator(user_id=5, conversation_id=6)

    monkeypatch.setattr(LoreOrchestrator, "_get_canon_module", _noop_canon_module)
    monkeypatch.setattr(LoreOrchestrator, "_get_canon_validation", _validator_factory)

    orchestrator._table_columns_cache["locations"] = {
        "location_name",
        "notable_features",
        "hidden_aspects",
        "access_restrictions",
        "local_customs",
    }

    observed: dict[str, typing.Any] = {}

    class _Conn:
        async def fetchrow(self, query, *args, **kwargs):
            if "FROM Locations" in query:
                observed["query"] = " ".join(query.split())
                observed["args"] = args
                return {
                    "location_name": "Twilight Market",
                    "notable_features": ["Starlit bazaar"],
                    "hidden_aspects": [],
                    "access_restrictions": [],
                    "local_customs": [],
                }
            return None

        async def fetch(self, query, *args, **kwargs):
            return []

    @asynccontextmanager
    async def _ctx():
        yield _Conn()

    monkeypatch.setattr(_lore_module, "get_db_connection_context", _ctx)

    scope = SimpleNamespace(location_id=None, location_name="twilight-market", nation_ids=set())

    rules = await orchestrator._get_canonical_rules_for_scope(scope)

    assert observed["args"] == (5, 6, "twilight-market")
    assert "location_name = $3" in observed["query"]
    assert any("Starlit bazaar" in rule for rule in rules)


@pytest.mark.unit
async def test_canonical_rules_include_nation_canon_and_fallback(monkeypatch):
    orchestrator = LoreOrchestrator(user_id=2, conversation_id=3)

    monkeypatch.setattr(LoreOrchestrator, "_get_canon_module", _noop_canon_module)
    monkeypatch.setattr(LoreOrchestrator, "_get_canon_validation", _validator_factory)

    nation_rows = [
        {
            "id": 7,
            "name": "Auroria",
            "cultural_traits": ["Honor duels", {"name": "Artistry", "details": "Lightwoven"}],
            "major_resources": ["Sunstone"],
            "major_cities": [{"name": "Auris", "description": "Capital of light"}],
            "neighboring_nations": [{"name": "Thornfell"}],
            "notable_features": "Sky bridges connect the floating isles.",
        },
        {
            "id": 9,
            "name": "Noctis",
            "cultural_traits": [],
            "major_resources": [],
            "major_cities": [],
            "neighboring_nations": [],
            "notable_features": None,
        },
    ]

    class _Conn:
        async def fetchrow(self, query, *args, **kwargs):
            return None

        async def fetch(self, query, *args, **kwargs):
            if "FROM Nations" in query:
                return nation_rows
            return []

    @asynccontextmanager
    async def _ctx():
        yield _Conn()

    monkeypatch.setattr(_lore_module, "get_db_connection_context", _ctx)

    scope = SimpleNamespace(location_id=None, nation_ids={7, 9})

    rules = await orchestrator._get_canonical_rules_for_scope(scope)

    assert any("Auroria cultural trait" in rule for rule in rules)
    assert any("Auroria resource" in rule for rule in rules)
    assert any("Maintain known canon for Noctis" in rule for rule in rules)
    # Ensure rules are trimmed and stringified even for nested JSON
    assert all(isinstance(rule, str) and len(rule) <= 160 for rule in rules)
