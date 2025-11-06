import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import importlib.util
import sys
from types import ModuleType
from typing import Any, Dict, List

import pytest

_canonical_spec = importlib.util.spec_from_file_location(
    "tests.unit._lore_canonical_rules_helper",
    Path(__file__).resolve().parent / "test_lore_canonical_rules.py",
)
assert _canonical_spec and _canonical_spec.loader
_canonical_module = importlib.util.module_from_spec(_canonical_spec)

_nyx_gateway = ModuleType("nyx.gateway")
_nyx_gateway.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("nyx.gateway", _nyx_gateway)
_nyx_gateway_module = ModuleType("nyx.gateway.llm_gateway")
_nyx_gateway_module.LLMRequest = object()
sys.modules.setdefault("nyx.gateway.llm_gateway", _nyx_gateway_module)
setattr(_nyx_gateway, "llm_gateway", _nyx_gateway_module)
_nyx_location = ModuleType("nyx.location")
_nyx_location.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("nyx.location", _nyx_location)
_nyx_location_hierarchy = ModuleType("nyx.location.hierarchy")
_nyx_location_hierarchy.get_or_create_location = lambda *args, **kwargs: None
sys.modules.setdefault("nyx.location.hierarchy", _nyx_location_hierarchy)
_nyx_location_types = ModuleType("nyx.location.types")
_nyx_location_types.Candidate = type("Candidate", (), {})
_nyx_location_types.Place = type("Place", (), {})
sys.modules.setdefault("nyx.location.types", _nyx_location_types)
_lore_cache_module = sys.modules.setdefault("lore.core.cache", ModuleType("lore.core.cache"))
_lore_cache_module.LoreCache = type("LoreCache", (), {})
_lore_cache_module.GLOBAL_LORE_CACHE = {}

_canonical_spec.loader.exec_module(_canonical_module)

LoreOrchestrator = _canonical_module.LoreOrchestrator
_lore_module = _canonical_module._lore_module
OrchestratorConfig = _lore_module.OrchestratorConfig

pytestmark = pytest.mark.anyio


class _DummyScope:
    location_id = "Central-Plaza"
    npc_ids: set[int] = set()
    lore_tags: set[str] = set()
    topics: set[str] = set()
    conflict_ids: set[int] = set()
    nation_ids: List[int] = []
    link_hints: Dict[str, Any] = {}

    def to_key(self) -> str:
        return "dummy-scope"


class _DummyConnection:
    def __init__(self, captured: List[tuple[str, Any]]) -> None:
        self._captured = captured

    async def fetchrow(self, query: str, *params: Any):
        param = params[0] if params else None
        self._captured.append((query.strip(), param))
        lowered = query.lower()
        if "location_name_lc" in lowered:
            return {
                "location_id": 42,
                "location_name": "Central Plaza",
                "notable_features": ["A bustling market square"],
                "hidden_aspects": [],
                "access_restrictions": [],
                "local_customs": ["Festival night"],
            }
        if "from locations" in lowered:
            return {
                "id": 42,
                "location_name": "Central Plaza",
                "description": "The heart of the city.",
                "notable_features": ["A bustling market square"],
                "hidden_aspects": [],
                "access_restrictions": [],
                "local_customs": ["Festival night"],
            }
        return None

    async def fetch(self, query: str, *params: Any):
        self._captured.append((query.strip(), params))
        return []


@asynccontextmanager
async def _dummy_db_context(captured: List[tuple[str, Any]]):
    yield _DummyConnection(captured)


class _DummyValidator:
    async def get_global_rules(self) -> List[str]:
        await asyncio.sleep(0)
        return ["Maintain timeline consistency"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_scene_brief_handles_slug_location(monkeypatch):
    captured_queries: List[tuple[str, Any]] = []

    orchestrator = LoreOrchestrator(1, 1, OrchestratorConfig(enable_governance=False))

    async def fake_get_table_columns(self, conn, table_name: str):
        if table_name.lower() == "locations":
            return {
                "id",
                "location_id",
                "location_name",
                "description",
                "notable_features",
                "hidden_aspects",
                "access_restrictions",
                "local_customs",
            }
        return set()

    async def fake_get_canon_module(self):
        return object()

    async def fake_get_canon_validation(self):
        return _DummyValidator()

    monkeypatch.setattr(
        _lore_module,
        "get_db_connection_context",
        lambda: _dummy_db_context(captured_queries),
    )
    monkeypatch.setattr(LoreOrchestrator, "_get_table_columns", fake_get_table_columns, raising=False)
    monkeypatch.setattr(LoreOrchestrator, "_get_canon_module", fake_get_canon_module, raising=False)
    monkeypatch.setattr(LoreOrchestrator, "_get_canon_validation", fake_get_canon_validation, raising=False)

    scope = _DummyScope()
    brief = await orchestrator.get_scene_brief(scope)

    canonical_rules = brief["signals"].get("canonical_rules", [])
    assert canonical_rules, "Expected canonical rules to be returned for slug location"
    assert any("Location feature" in rule for rule in canonical_rules)

    slug_queries = [param for _, param in captured_queries if param == scope.location_id]
    assert slug_queries, "Expected slug location name to be used in DB lookup"

    lowercase_checks = [
        query for query, _ in captured_queries if "location_name_lc" in query.lower()
    ]
    assert lowercase_checks, "Expected case-insensitive location lookup for slug reuse"
