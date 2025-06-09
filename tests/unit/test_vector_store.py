import os

from nyx.core.memory import vector_store
import pytest


@pytest.mark.asyncio
async def test_vector_store_add_and_query(tmp_path):
    # Use temporary directory for persistence
    path = os.path.join(tmp_path, "vs")
    vector_store.load(path)
    uid = await vector_store.add("2+2=4", {"source": "unit"})
    hits = await vector_store.query("What is 2+2?")
    assert hits[0]["text"] == "2+2=4"
    assert hits[0]["meta"]["uid"] == uid
    await vector_store.save(path)
    vector_store.load(path)
    hits = await vector_store.query("What is 2+2?")
    assert hits[0]["text"] == "2+2=4"
    assert hits[0]["meta"]["uid"] == uid
