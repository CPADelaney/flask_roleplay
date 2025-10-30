from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional

import sys
import os
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

class _PreImportHFEmbeddings:
    def __init__(self, *_, **__):
        self._vector = [0.0] * 1536

    def embed_query(self, text: str) -> List[float]:  # pragma: no cover - deterministic stub
        return list(self._vector)

hf_module = ModuleType("langchain_community.embeddings")
hf_module.HuggingFaceEmbeddings = _PreImportHFEmbeddings
sys.modules["langchain_community.embeddings"] = hf_module


class _FakeSentenceTransformer:
    def __init__(self, *_, **__):  # pragma: no cover - deterministic stub
        pass

    def encode(self, sentences, *_, **__):
        return [[0.0] * 1536 for _ in sentences]


sentence_module = ModuleType("sentence_transformers")
sentence_module.SentenceTransformer = _FakeSentenceTransformer
sys.modules["sentence_transformers"] = sentence_module

import pytest

from memory.memory_service import MemoryEmbeddingService


class _StubVectorDatabase:
    """Minimal in-memory vector store used for unit testing."""

    def __init__(self, *_, **__):
        self.dimensions: Dict[str, int] = {}
        self.records: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def initialize(self) -> None:  # pragma: no cover - interface placeholder
        return None

    async def close(self) -> None:  # pragma: no cover - interface placeholder
        self.records.clear()
        self.dimensions.clear()

    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        existing = self.dimensions.get(collection_name)
        if existing is not None and existing != dimension:
            # Reset mismatched collections to simulate rebuild
            self.records[collection_name] = {}
        self.dimensions[collection_name] = dimension
        self.records.setdefault(collection_name, {})
        return True

    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> bool:
        expected_dim = self.dimensions[collection_name]
        bucket = self.records.setdefault(collection_name, {})

        for doc_id, vector, meta in zip(ids, vectors, metadata):
            if len(vector) != expected_dim:
                raise ValueError("Vector dimension mismatch in stub store")
            bucket[doc_id] = {"vector": vector, "metadata": meta}
        return True

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        bucket = self.records.get(collection_name, {})
        expected_dim = self.dimensions.get(collection_name)
        if expected_dim is None:
            return []

        def matches_filter(meta: Dict[str, Any]) -> bool:
            if not filter_dict:
                return True
            for key, value in filter_dict.items():
                if meta.get(key) != value:
                    return False
            return True

        results: List[Dict[str, Any]] = []
        for doc_id, payload in bucket.items():
            if not matches_filter(payload["metadata"]):
                continue
            vector = payload["vector"]
            score = sum(a * b for a, b in zip(query_vector, vector))
            results.append({
                "id": doc_id,
                "score": score,
                "metadata": payload["metadata"],
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str],
    ) -> List[Dict[str, Any]]:
        bucket = self.records.get(collection_name, {})
        output: List[Dict[str, Any]] = []
        for doc_id in ids:
            payload = bucket.get(doc_id)
            if payload:
                output.append({"id": doc_id, "metadata": payload["metadata"]})
        return output


@pytest.fixture(autouse=True)
def _patch_vector_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyHFEmbeddings:
        def __init__(self, *_, **__):
            self._vector = [0.0] * 1536

        def embed_query(self, text: str) -> List[float]:  # pragma: no cover - deterministic stub
            return list(self._vector)

    monkeypatch.setattr("memory.memory_service.ChromaVectorDatabase", _StubVectorDatabase)
    monkeypatch.setattr("memory.memory_service.FAISSVectorDatabase", _StubVectorDatabase)
    monkeypatch.setattr("memory.memory_service.create_vector_database", lambda _: _StubVectorDatabase())
    monkeypatch.setattr("memory.memory_service.HuggingFaceEmbeddings", _DummyHFEmbeddings)
    monkeypatch.setattr("memory.memory_service.hosted_vector_store_enabled", lambda *_, **__: True)
    monkeypatch.setattr(
        "memory.memory_service.get_hosted_vector_store_ids",
        lambda *_: ["agents-memory"],
    )
    monkeypatch.setattr("memory.memory_service.search_hosted_vector_store", lambda *_, **__: [])
    monkeypatch.setattr("memory.memory_service.upsert_hosted_vector_documents", lambda *_, **__: None)


@pytest.fixture
def memory_config() -> Dict[str, Any]:
    return {
        "vector_store": {
            "persist_base_dir": "./vector_stores",
            "use_legacy_vector_store": False,
            "hosted_vector_store_ids": ["agents-memory"],
        },
        "embedding": {
            "type": "openai",
            "openai_model": "text-embedding-3-small",
        },
    }


@pytest.fixture
def legacy_memory_config(memory_config: Dict[str, Any]) -> Dict[str, Any]:
    config = {
        "vector_store": {
            **memory_config["vector_store"],
            "use_legacy_vector_store": True,
            "hosted_vector_store_ids": [],
            "dimension": 8,
        },
        "embedding": {
            **memory_config["embedding"],
            "embedding_dim": 8,
        },
    }
    return config


@pytest.fixture
def legacy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memory.memory_service.hosted_vector_store_enabled", lambda *_, **__: False)
    monkeypatch.setattr("memory.memory_service.get_hosted_vector_store_ids", lambda *_: [])


@pytest.fixture(autouse=True)
def _patch_openai_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_ask(
        text: str,
        *,
        mode: str = "retrieval",
        metadata: Optional[Dict[str, Any]] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        limit: Optional[int] = None,
        legacy_fallback=None,
    ) -> Dict[str, Any]:
        base = float(len(text) % 31)
        size = dimensions or 8
        embedding = [base + i for i in range(size)]
        return {"embedding": embedding, "provider": "test"}

    monkeypatch.setattr("memory.memory_service.rag_ask", _fake_ask)


def test_generate_embedding_uses_agents_dimension(memory_config: Dict[str, Any]) -> None:
    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=1,
            conversation_id=42,
            vector_store_type="chroma",
            embedding_model="openai",
            config=memory_config,
        )

        await service.initialize()
        embedding = await service.generate_embedding("hello world")

        assert len(embedding) == 8
        assert service._get_target_dimension() == 8

    asyncio.run(_run())


def test_add_and_search_memory_returns_inserted_record(
    legacy_backend: None, legacy_memory_config: Dict[str, Any]
) -> None:
    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=7,
            conversation_id=99,
            vector_store_type="faiss",
            embedding_model="openai",
            config=legacy_memory_config,
        )

        await service.initialize()
        memory_id = await service.add_memory("A remembered moment", {"memory_id": "memory-1"})

        results = await service.search_memories("A remembered moment", top_k=1)
        assert results
        assert results[0]["id"] == memory_id

    asyncio.run(_run())


def test_service_initializes_when_agents_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
    memory_config: Dict[str, Any],
) -> None:
    from rag import vector_store as rag_vector_store

    monkeypatch.setattr("memory.memory_service.hosted_vector_store_enabled", lambda *_, **__: False)
    monkeypatch.setattr(
        "memory.memory_service.get_hosted_vector_store_ids",
        lambda *_: ["agents-memory"],
    )
    monkeypatch.setattr(rag_vector_store, "agents_setup", None, raising=False)
    monkeypatch.setattr(
        rag_vector_store,
        "_AGENTS_IMPORT_ERROR",
        ImportError("Agents SDK missing"),
        raising=False,
    )

    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=14,
            conversation_id=28,
            vector_store_type="chroma",
            embedding_model="openai",
            config=memory_config,
        )

        assert service._legacy_vector_store_enabled is True
        assert service._use_hosted_vector_store is False

        await service.initialize()
        assert isinstance(service.vector_db, _StubVectorDatabase)

    asyncio.run(_run())


def test_legacy_vector_store_guard(monkeypatch: pytest.MonkeyPatch, memory_config: Dict[str, Any]) -> None:
    monkeypatch.setattr("memory.memory_service.hosted_vector_store_enabled", lambda *_, **__: False)
    monkeypatch.setattr("memory.memory_service.get_hosted_vector_store_ids", lambda *_: [])
    monkeypatch.delenv("ENABLE_LEGACY_VECTOR_STORE", raising=False)
    monkeypatch.delenv("ALLOW_LEGACY_EMBEDDINGS", raising=False)

    legacy_config = {
        **memory_config,
        "vector_store": {
            **memory_config["vector_store"],
            "use_legacy_vector_store": False,
            "hosted_vector_store_ids": [],
        },
    }

    with pytest.raises(RuntimeError, match="Legacy vector store backend disabled"):
        MemoryEmbeddingService(
            user_id=7,
            conversation_id=13,
            vector_store_type="chroma",
            embedding_model="openai",
            config=legacy_config,
        )


def test_add_memory_rejects_invalid_embedding(
    legacy_backend: None, legacy_memory_config: Dict[str, Any]
) -> None:
    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=3,
            conversation_id=5,
            vector_store_type="chroma",
            embedding_model="openai",
            config=legacy_memory_config,
        )

        await service.initialize()

        with pytest.raises(ValueError):
            await service.add_memory(
                "Custom embedding mismatch",
                {"memory_id": "manual"},
                embedding=[0.0, 1.0],
            )

    asyncio.run(_run())


def test_legacy_vector_store_enabled_via_env(monkeypatch: pytest.MonkeyPatch, memory_config: Dict[str, Any]) -> None:
    monkeypatch.setattr("memory.memory_service.hosted_vector_store_enabled", lambda *_, **__: False)
    monkeypatch.setattr("memory.memory_service.get_hosted_vector_store_ids", lambda *_: [])
    monkeypatch.setenv("ENABLE_LEGACY_VECTOR_STORE", "1")
    monkeypatch.delenv("ALLOW_LEGACY_EMBEDDINGS", raising=False)

    service = MemoryEmbeddingService(
        user_id=11,
        conversation_id=21,
        vector_store_type="chroma",
        embedding_model="openai",
        config=memory_config,
    )

    assert service._legacy_vector_store_enabled is True


def test_add_memory_accepts_pgvector_like_embedding(
    monkeypatch: pytest.MonkeyPatch,
    legacy_backend: None,
    legacy_memory_config: Dict[str, Any],
) -> None:
    class _PgVector:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=8,
            conversation_id=12,
            vector_store_type="faiss",
            embedding_model="openai",
            config=legacy_memory_config,
        )

        await service.initialize()

        async def _fail_generate(text: str) -> List[float]:  # pragma: no cover - defensive
            raise AssertionError("generate_embedding should not be called")

        monkeypatch.setattr(service, "generate_embedding", _fail_generate)

        vector = [float(i) for i in range(service._get_target_dimension())]
        pg_vector = _PgVector(vector)

        memory_id = await service.add_memory(
            "Precomputed vector",
            {"memory_id": "npc-1"},
            entity_type="npc",
            embedding=pg_vector,
        )

        stored = service.vector_db.records["npc_embeddings"][memory_id]
        assert stored["vector"] == vector

    asyncio.run(_run())


def test_add_memory_regenerates_on_unusable_embedding(
    monkeypatch: pytest.MonkeyPatch,
    legacy_backend: None,
    legacy_memory_config: Dict[str, Any],
) -> None:
    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=2,
            conversation_id=3,
            vector_store_type="chroma",
            embedding_model="openai",
            config=legacy_memory_config,
        )

        await service.initialize()

        calls: List[str] = []

        async def _generate(text: str) -> List[float]:
            calls.append(text)
            return [1.0 for _ in range(service._get_target_dimension())]

        monkeypatch.setattr(service, "generate_embedding", _generate)

        memory_id = await service.add_memory(
            "Regenerate vector",
            {},
            embedding=object(),
        )

        assert calls == ["Regenerate vector"]
        stored = service.vector_db.records["memory_embeddings"][memory_id]
        assert stored["vector"][0] == pytest.approx(1.0)

    asyncio.run(_run())
