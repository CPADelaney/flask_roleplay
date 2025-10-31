import os

os.environ.setdefault("NYX_MEMORY_EMBEDDING_MODEL", "offline-test-model")

import numpy as np
import pytest

from nyx.core.memory import vector_store


@pytest.mark.asyncio
async def test_vector_store_add_and_query(tmp_path):
    # Use temporary directory for persistence
    path = os.path.join(tmp_path, "vs")
    vector_store.load(path)
    uid = await vector_store.add("2+2=4", {"source": "unit"})
    hits = await vector_store.query("What is 2+2?")
    assert hits[0]["text"] == "2+2=4"
    assert hits[0]["meta"]["uid"] == uid

    expected_dim = vector_store.get_vector_dimension()
    assert vector_store._index.d == expected_dim  # type: ignore[attr-defined]

    # The FAISS index stores normalised float32 vectors with the configured width
    stored_vec = vector_store._index.reconstruct(vector_store._uid_to_faiss[uid])  # type: ignore[attr-defined]
    assert isinstance(stored_vec, np.ndarray)
    assert stored_vec.shape == (expected_dim,)

    await vector_store.save(path)
    vector_store.load(path)
    hits = await vector_store.query("What is 2+2?")
    assert hits[0]["text"] == "2+2=4"
    assert hits[0]["meta"]["uid"] == uid
