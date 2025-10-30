import importlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import embedding.vector_store as legacy_vector_store


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_legacy_embedding_helpers_raise_without_flag(monkeypatch):
    monkeypatch.delenv("ENABLE_LEGACY_EMBEDDINGS", raising=False)

    with pytest.raises(RuntimeError):
        await legacy_vector_store.generate_embedding("hello world")

    with pytest.raises(RuntimeError):
        await legacy_vector_store.compute_similarity([0.1], [0.2])

    with pytest.raises(RuntimeError):
        await legacy_vector_store.find_most_similar([0.1], {"a": [0.2]})


@pytest.mark.anyio
async def test_legacy_embedding_helpers_respect_feature_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_LEGACY_EMBEDDINGS", "1")

    module = importlib.reload(legacy_vector_store)

    embedding = await module.generate_embedding("hello world")
    assert isinstance(embedding, list)
    assert len(embedding) == module.EMBEDDING_DIMENSIONS

    similarity = await module.compute_similarity(embedding, embedding)
    assert similarity == pytest.approx(1.0, rel=1e-3)

    results = await module.find_most_similar(embedding, {"memory": embedding})
    assert results[0]["id"] == "memory"
    assert results[0]["similarity"] == pytest.approx(1.0, rel=1e-3)
