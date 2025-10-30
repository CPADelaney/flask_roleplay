from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ask import ask as rag_ask
from scripts import load_vector_store

pytestmark = pytest.mark.integration


@pytest.mark.requires_openai
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required")
def test_loader_script_upserts_documents(monkeypatch, tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()

    fixture_path = fixture_dir / "memories.json"
    payload = [
        {
            "id": "memory-1",
            "text": "First memory",
            "metadata": {"existing": "value"},
        },
        {
            "text": "Second memory",
            "metadata": {"memory_id": "memory-2"},
        },
    ]
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    captured: dict[str, object] = {}

    async def fake_upsert(documents, *, vector_store_id, metadata):
        captured["documents"] = [dict(doc) for doc in documents]
        captured["vector_store_id"] = vector_store_id
        captured["metadata"] = dict(metadata)
        return [doc.get("id", f"generated-{index}") for index, doc in enumerate(documents)]

    monkeypatch.setattr(load_vector_store, "upsert_hosted_vector_documents", fake_upsert)
    monkeypatch.setattr(
        load_vector_store,
        "hosted_vector_store_enabled",
        lambda configured_ids=None, *, config=None: True,
    )
    monkeypatch.setattr(
        load_vector_store,
        "legacy_vector_store_enabled",
        lambda config=None: False,
    )
    monkeypatch.setattr(
        load_vector_store,
        "get_hosted_vector_store_ids",
        lambda config=None: ["vs-default-from-config"],
    )

    exit_code = load_vector_store.main(
        [
            "--dir",
            str(fixture_dir),
            "--vector-store-id",
            "vs-explicit",
            "--collection",
            "memory_embeddings",
            "--metadata",
            "component=fixture",
            "--metadata",
            "import_batch=integration",
        ]
    )

    assert exit_code == 0
    assert captured["vector_store_id"] == "vs-explicit"
    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["component"] == "scripts.load_vector_store"
    assert metadata["document_path"] == str(fixture_dir)

    uploaded = captured["documents"]
    assert isinstance(uploaded, list)
    assert len(uploaded) == 2

    first_doc = uploaded[0]
    assert first_doc["id"] == "memory-1"
    assert first_doc["metadata"]["existing"] == "value"
    assert first_doc["metadata"]["collection"] == "memory_embeddings"
    assert first_doc["metadata"]["component"] == "fixture"

    second_doc = uploaded[1]
    assert second_doc["metadata"]["memory_id"] == "memory-2"
    assert second_doc["metadata"]["import_batch"] == "integration"


@pytest.mark.requires_openai
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required")
def test_agents_retrieval_integration(monkeypatch):
    calls: list[dict[str, object]] = []

    async def fake_agents_ask(*, prompt, mode, metadata, model, dimensions, limit):
        calls.append(
            {
                "prompt": prompt,
                "mode": mode,
                "metadata": dict(metadata),
                "model": model,
                "dimensions": dimensions,
                "limit": limit,
            }
        )
        return {
            "documents": [
                {
                    "text": "Retrieved snippet",
                    "metadata": {"memory_id": "memory-xyz", "collection": "memory_embeddings"},
                }
            ],
            "metadata": {"tool_usage": {"file_search": 1}},
        }

    fake_agents_module = types.SimpleNamespace(ask=fake_agents_ask)
    monkeypatch.setitem(sys.modules, "agents", fake_agents_module)

    result = asyncio.run(
        rag_ask(
            "Find the latest memory",
            backend="agents",
            metadata={"request_id": "req-123"},
            limit=3,
        )
    )

    assert result["provider"] == "agents"
    assert result["documents"][0]["text"] == "Retrieved snippet"
    assert result["metadata"]["request_id"] == "req-123"

    assert calls, "Agents backend was not invoked"
    call = calls[0]
    assert call["mode"] == "retrieval"
    assert call["metadata"]["request_id"] == "req-123"
    assert call["limit"] == 3
