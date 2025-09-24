import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from memory.chroma_vector_store import ChromaVectorDatabase


@pytest.mark.asyncio
async def test_chroma_query_with_multi_field_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    store = ChromaVectorDatabase(persist_directory=str(tmp_path))
    await store.initialize()

    ids = ["m1", "m2", "m3"]
    vectors = [
        [0.1, 0.2, 0.3],
        [0.9, 0.1, 0.1],
        [0.05, 0.2, 0.4],
    ]
    metadata = [
        {"user_id": 1, "conversation_id": 42, "label": "match"},
        {"user_id": 1, "conversation_id": 43, "label": "other-conversation"},
        {"user_id": 2, "conversation_id": 42, "label": "other-user"},
    ]

    inserted = await store.insert_vectors("memories", ids, vectors, metadata)
    assert inserted

    results = await store.search_vectors(
        "memories",
        query_vector=[0.1, 0.2, 0.3],
        top_k=3,
        filter_dict={"user_id": 1, "conversation_id": 42},
    )

    await store.close()

    assert results
    assert all(result["metadata"]["user_id"] == 1 for result in results)
    assert all(result["metadata"]["conversation_id"] == 42 for result in results)
    assert results[0]["id"] == "m1"
