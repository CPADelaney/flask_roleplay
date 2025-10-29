import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.reconsolidation import ReconsolidationManager


class _VectorSearchConnection:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = rows or []
        self.last_query: Optional[str] = None
        self.last_args: Optional[tuple[Any, ...]] = None

    async def fetch(self, query: str, *args: Any):
        self.last_query = query
        self.last_args = args
        return self.rows

    async def fetchrow(self, query: str, *args: Any):
        # No refetch required when the embedding is provided.
        return None


class _KeywordFallbackConnection:
    def __init__(self) -> None:
        self.vector_calls: List[str] = []
        self.keyword_calls: List[str] = []

    async def fetch(self, query: str, *args: Any):
        if "ILIKE" in query:
            self.keyword_calls.append(query)
            return [
                {
                    "id": 2,
                    "memory_text": "Alpha remembers distant adventures",
                    "significance": 1,
                    "emotional_intensity": 10,
                }
            ]
        self.vector_calls.append(query)
        return []

    async def fetchrow(self, query: str, *args: Any):
        # Simulate a record without an embedding forcing the fallback path.
        return {"embedding": None}


def test_find_similar_memories_uses_stored_embedding() -> None:
    manager = ReconsolidationManager(user_id=1, conversation_id=99)
    connection = _VectorSearchConnection(
        rows=[
            {
                "id": 42,
                "memory_text": "A heroic tale",
                "significance": 1,
                "emotional_intensity": 5,
                "similarity": 0.1,
            },
            {
                "id": 43,
                "memory_text": "A distant memory",
                "significance": 1,
                "emotional_intensity": 5,
                "similarity": 0.35,
            },
        ]
    )

    result = asyncio.run(
        manager._find_similar_memories(
            memory_id=1,
            entity_type="npc",
            entity_id=7,
            memory_text="A heroic tale of bravery",
            memory_embedding=[0.1, 0.2, 0.3],
            conn=connection,
        )
    )

    assert "embedding <=> $1::vector" in (connection.last_query or "")
    assert connection.last_args is not None
    assert connection.last_args[0] == [0.1, 0.2, 0.3]
    assert len(result) == 1
    assert result[0]["id"] == 42


def test_find_similar_memories_falls_back_to_keywords() -> None:
    manager = ReconsolidationManager(user_id=3, conversation_id=3)
    connection = _KeywordFallbackConnection()

    result = asyncio.run(
        manager._find_similar_memories(
            memory_id=2,
            entity_type="npc",
            entity_id=4,
            memory_text="Alpha beta gamma delta epsilon",
            memory_embedding=None,
            conn=connection,
        )
    )

    assert connection.vector_calls == []
    assert len(connection.keyword_calls) == 1
    assert result and result[0]["id"] == 2
