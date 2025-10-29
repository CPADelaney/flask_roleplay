"""Tests for lore context retrieval relying on stored embeddings."""

from __future__ import annotations

from unittest.mock import AsyncMock

import os
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment shim
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import importlib.util
import types


if "lore" not in sys.modules:
    lore_pkg = types.ModuleType("lore")
    lore_pkg.__path__ = []  # pragma: no cover - package shim
    sys.modules["lore"] = lore_pkg

module_path = ROOT / "lore" / "data_access.py"
spec = importlib.util.spec_from_file_location("lore.data_access", module_path)
assert spec and spec.loader  # pragma: no cover - defensive
_data_access = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _data_access
spec.loader.exec_module(_data_access)
LoreKnowledgeAccess = _data_access.LoreKnowledgeAccess


import asyncio


def test_generate_available_lore_uses_stored_embeddings(monkeypatch):
    async def _run_test():
        access = LoreKnowledgeAccess()
        access.initialized = True

        knowledge_items = [
            {
                "lore_type": "Factions",
                "lore_id": 1,
                "knowledge_level": "known",
                "is_secret": False,
            },
            {
                "lore_type": "WorldLore",
                "lore_id": 2,
                "knowledge_level": "rumor",
                "is_secret": True,
            },
        ]

        monkeypatch.setattr(
            access,
            "get_entity_knowledge",
            AsyncMock(return_value=knowledge_items),
        )

        async def _get_lore_by_id(lore_type: str, lore_id: int):
            if lore_id == 1:
                return {
                    "id": lore_id,
                    "lore_type": lore_type,
                    "name": "Lore One",
                    "embedding": [0.1, 0.2, 0.3],
                }

            if lore_id == 2:
                return {
                    "id": lore_id,
                    "lore_type": lore_type,
                    "name": "Lore Two",
                    "description": "missing vector",
                    "embedding": None,
                }

            return {}

        monkeypatch.setattr(
            access,
            "get_lore_by_id",
            AsyncMock(side_effect=_get_lore_by_id),
        )

        mock_generate_embedding = AsyncMock(return_value=[0.9, 0.1, 0.0])
        monkeypatch.setattr(
            _data_access,
            "generate_embedding",
            mock_generate_embedding,
        )

        mock_similarity = AsyncMock(return_value=0.75)
        monkeypatch.setattr(
            _data_access,
            "compute_similarity",
            mock_similarity,
        )

        results = await access.generate_available_lore_for_context(
            "query text",
            "npc",
            42,
            limit=5,
        )

        mock_generate_embedding.assert_awaited_once_with("query text")
        assert mock_similarity.await_count == 1

        # Only the lore item with an embedding should be returned.
        assert len(results) == 1
        result = results[0]
        assert result["knowledge_level"] == "known"
        assert result["is_secret"] is False
        assert result["relevance"] == pytest.approx(0.75)
        assert "embedding" not in result

    asyncio.run(_run_test())
