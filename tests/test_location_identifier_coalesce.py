from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio
import asyncpg
import pytest
import sys
import types
import os

# Ensure the repository root is importable when tests are executed in isolation.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    parent_name, _, attr = name.rpartition(".")
    if parent_name:
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        setattr(parent, attr, module)
    return module


# Stub heavy embedding dependencies to avoid network calls during tests.
stub_sentence_transformers = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, *_: object, **__: object) -> None:
        self._dim = 4

    def encode(self, texts, **__: object):  # pragma: no cover - deterministic stub
        return [[0.0] * self._dim for _ in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
stub_sentence_transformers.models = types.SimpleNamespace(
    Transformer=lambda *_, **__: types.SimpleNamespace(get_word_embedding_dimension=lambda: 4),
    Pooling=lambda *_, **__: object(),
)

sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

faiss_module = types.ModuleType("faiss")


class _DummyIndexFlatIP:
    def __init__(self, *_: object, **__: object) -> None:
        pass


class _DummyIndexIDMap:
    def __init__(self, inner: object):
        self.inner = inner

    def add_with_ids(self, *_: object, **__: object) -> None:
        return None

    def remove_ids(self, *_: object, **__: object) -> None:
        return None


faiss_module.IndexFlatIP = _DummyIndexFlatIP
faiss_module.IndexIDMap = _DummyIndexIDMap
sys.modules.setdefault("faiss", faiss_module)
sys.modules.setdefault("faiss.contrib", types.ModuleType("faiss.contrib"))

# Stub the heavy Nyx vector store to avoid loading FAISS/SentenceTransformers implementations.
_stub_module("nyx.core.memory.vector_store")
_vector_store_stub = sys.modules["nyx.core.memory.vector_store"]


async def _async_noop(*_: object, **__: object) -> None:
    return None


_vector_store_stub.add = _async_noop  # type: ignore[attr-defined]
_vector_store_stub.update = _async_noop  # type: ignore[attr-defined]
_vector_store_stub.query = _async_noop  # type: ignore[attr-defined]
_vector_store_stub.delete = _async_noop  # type: ignore[attr-defined]

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from lore.data_access import LocationDataAccess
import lore.data_access as data_access_module
from lore.managers import base_manager as base_manager_module
from lore.managers.local_lore import LocalLoreManager
import lore.managers.local_lore as local_lore_module
from lore.managers.religion import HolySiteParams, ReligionManager
import lore.managers.religion as religion_module


class LocationOnlySchemaConnection:
    """Connection stub that emulates a schema without a Locations.id column."""

    def __init__(self):
        self.executed_queries = []

    def _normalize(self, query: str) -> str:
        return " ".join(query.split()).upper()

    def _check_locations_id(self, query: str) -> None:
        normalized = self._normalize(query)
        if "LOCATIONS" in normalized and " ID" in normalized and "COALESCE" not in normalized:
            raise asyncpg.exceptions.UndefinedColumnError('column "id" does not exist')

    async def fetchrow(self, query: str, *args):
        self._check_locations_id(query)
        normalized = self._normalize(query)
        if "SELECT * FROM LOCATIONS" in normalized:
            location_id = args[0]
            return {
                "location_id": location_id,
                "location_name": "Harbor of Veils",
                "description": "A misty harbor for clandestine meetings.",
            }
        if "SELECT COALESCE(ID, LOCATION_ID) AS ID, LOCATION_NAME" in normalized:
            location_id = args[0]
            return {
                "id": location_id,
                "location_name": "Harbor of Veils",
                "location_type": "port",
                "description": "A misty harbor for clandestine meetings.",
            }
        if "FROM PANTHEONS" in normalized:
            return {"name": "Moon Court", "description": "A luminous order of priestesses."}
        return None

    async def fetchval(self, query: str, *args):
        self._check_locations_id(query)
        normalized = self._normalize(query)
        if "SELECT LOCATION_NAME FROM LOCATIONS" in normalized:
            return "Harbor of Veils"
        return None

    async def fetch(self, query: str, *args):
        self._check_locations_id(query)
        normalized = self._normalize(query)
        if "FROM LOCALHISTORIES" in normalized:
            location_id = args[0]
            return [
                {
                    "id": 201,
                    "location_id": location_id,
                    "event_name": "Harbor Founding",
                    "description": "A matriarch established safe passage through the mists.",
                    "date_description": "A generation ago",
                    "significance": 7,
                    "impact_type": "cultural",
                    "notable_figures": [],
                    "current_relevance": "High",
                    "commemoration": None,
                    "connected_myths": [],
                    "related_landmarks": [],
                    "narrative_category": "historical",
                }
            ]
        if "FROM LANDMARKS" in normalized and "COALESCE" not in normalized:
            # Handle both the lore fetch and id list queries.
            if "SELECT * FROM LANDMARKS" in normalized:
                location_id = args[0]
                return [
                    {
                        "id": 301,
                        "name": "Moonlit Spire",
                        "location_id": location_id,
                        "landmark_type": "structure",
                        "description": "A spiraling tower honoring the moon matriarchs.",
                        "historical_significance": None,
                        "current_use": None,
                        "controlled_by": None,
                        "legends": [],
                        "connected_histories": [],
                        "architectural_style": None,
                        "symbolic_meaning": None,
                        "matriarchal_significance": "moderate",
                    }
                ]
            return []
        if "FROM URBANMYTHS" in normalized:
            if "SELECT *" in normalized:
                return [
                    {
                        "id": 401,
                        "name": "Whispers in the Veil",
                        "description": "Sailors hear matriarchal spirits guiding their path.",
                        "origin_location": "Harbor of Veils",
                        "origin_event": None,
                        "believability": 6,
                        "spread_rate": 5,
                        "regions_known": [],
                        "narrative_style": "folklore",
                        "themes": [],
                        "variations": [],
                        "matriarchal_elements": [],
                    }
                ]
            return [
                {"id": 401}
            ]
        if "FROM NARRATIVECONNECTIONS" in normalized:
            return []
        if "FROM DEITIES" in normalized:
            return [
                {"id": 11, "name": "Selene", "gender": "female", "domain": "moon", "rank": 9},
            ]
        if "FROM LOCATIONS LIMIT 10" in normalized:
            return [
                {"id": 1, "location_name": "Harbor of Veils", "description": "A misty harbor."},
                {"id": 2, "location_name": "Velvet Sanctum", "description": "Hidden sanctuary."},
            ]
        if "FROM REGIONALRELIGIOUSPRACTICE" in normalized:
            return []
        return []

    async def execute(self, query: str, *args):
        self._check_locations_id(query)
        return "OK"


def fake_db_context_factory():
    @asynccontextmanager
    async def _fake_context():
        yield LocationOnlySchemaConnection()

    return _fake_context


def test_location_data_access_coalesces_identifiers(monkeypatch):
    async def _run():
        monkeypatch.setattr(data_access_module, "get_db_connection_context", fake_db_context_factory())

        data_access = LocationDataAccess(user_id=7, conversation_id=13)
        data_access.initialized = True

        result = await data_access.get_location_details(location_id=99)

        assert result["id"] == 99
        assert result["location_id"] == 99

    asyncio.run(_run())


def test_local_lore_manager_handles_location_id_schema(monkeypatch):
    async def _run():
        fake_context = fake_db_context_factory()
        monkeypatch.setattr(local_lore_module, "get_db_connection_context", fake_context)
        monkeypatch.setattr(base_manager_module, "get_db_connection_context", fake_context)

        manager = LocalLoreManager(user_id=5, conversation_id=8)
        manager.get_cache = lambda *_: None  # type: ignore[assignment]
        manager.set_cache = lambda *_, **__: None  # type: ignore[assignment]
        location_lore = await manager._get_location_lore_impl({}, location_id=55)

        assert location_lore.location["id"] == 55
        assert location_lore.location["location_id"] == 55
        assert location_lore.histories, "Expected mock history data to be returned"
        assert location_lore.landmarks, "Expected mock landmark data to be returned"

    asyncio.run(_run())


def test_religion_manager_fetches_locations_without_id_column(monkeypatch):
    async def _run():
        fake_context = fake_db_context_factory()
        monkeypatch.setattr(base_manager_module, "get_db_connection_context", fake_context)
        monkeypatch.setattr(religion_module, "GeopoliticalSystemManager", lambda *_, **__: SimpleNamespace())

        fake_sites = [
            HolySiteParams(
                name="Lunar Bastion",
                site_type="temple",
                description="A shining temple dedicated to moonlit rites.",
                clergy_type="High Priestess",
                location_id=1,
            )
        ]

        class _FakeResult:
            def final_output_as(self, _model):
                return fake_sites

        monkeypatch.setattr(religion_module.Runner, "run", AsyncMock(return_value=_FakeResult()))

        manager = ReligionManager(user_id=2, conversation_id=3)

        async def _fake_add_holy_site(ctx, site):
            return 501

        manager.add_holy_site = _fake_add_holy_site  # type: ignore[assignment]
        ctx = SimpleNamespace(context={"user_id": 2, "conversation_id": 3})

        created = await manager._generate_holy_sites(ctx, pantheon_id=1)

        assert created
        assert created[0]["id"] == 501
        assert created[0]["location_id"] == 1

    asyncio.run(_run())
