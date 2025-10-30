import importlib
import pathlib
import sys
from unittest.mock import AsyncMock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import embedding.vector_store as legacy_vector_store


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_generate_embedding_routes_through_rag(monkeypatch):
    stub = AsyncMock(return_value={"embedding": [0.1, 0.2, 0.3]})
    monkeypatch.setattr(legacy_vector_store, "rag_ask", stub)

    embedding = await legacy_vector_store.generate_embedding("hello world")
    assert embedding[:3] == [0.1, 0.2, 0.3]
    assert len(embedding) == legacy_vector_store.EMBEDDING_DIMENSIONS
    stub.assert_awaited_once()


@pytest.mark.anyio
async def test_similarity_helpers_continue_to_work(monkeypatch):
    module = importlib.reload(legacy_vector_store)
    vector = [0.2, 0.3, 0.4]
    similarity = await module.compute_similarity(vector, vector)
    assert similarity == pytest.approx(1.0, rel=1e-3)

    results = await module.find_most_similar(vector, {"memory": vector})
    assert results[0]["id"] == "memory"
    assert results[0]["similarity"] == pytest.approx(1.0, rel=1e-3)
