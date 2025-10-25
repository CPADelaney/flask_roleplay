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


@pytest.fixture
def memory_config() -> Dict[str, Any]:
    return {
        "vector_store": {
            "persist_base_dir": "./vector_stores",
            "dimension": 1536,
        },
        "embedding": {
            "type": "openai",
            "openai_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        },
    }


@pytest.fixture(autouse=True)
def _patch_openai_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_embedding(text: str, model: str = "text-embedding-3-small", dimensions: Optional[int] = None) -> List[float]:
        base = float(len(text) % 31)
        return [base + i for i in range(1536)]

    monkeypatch.setattr("memory.memory_service.get_text_embedding", _fake_embedding)


def test_generate_embedding_matches_configured_dimension(memory_config: Dict[str, Any]) -> None:
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

        assert len(embedding) == 1536
        assert service._get_target_dimension() == 1536

    asyncio.run(_run())


def test_add_and_search_memory_returns_inserted_record(memory_config: Dict[str, Any]) -> None:
    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=7,
            conversation_id=99,
            vector_store_type="faiss",
            embedding_model="openai",
            config=memory_config,
        )

        await service.initialize()
        memory_id = await service.add_memory("A remembered moment", {"memory_id": "memory-1"})

        results = await service.search_memories("A remembered moment", top_k=1)
        assert results
        assert results[0]["id"] == memory_id

    asyncio.run(_run())


def test_add_memory_rejects_invalid_embedding(memory_config: Dict[str, Any]) -> None:
    async def _run() -> None:
        service = MemoryEmbeddingService(
            user_id=3,
            conversation_id=5,
            vector_store_type="chroma",
            embedding_model="openai",
            config=memory_config,
        )

        await service.initialize()

        with pytest.raises(ValueError):
            await service.add_memory(
                "Custom embedding mismatch",
                {"memory_id": "manual"},
                embedding=[0.0, 1.0],
            )

    asyncio.run(_run())
