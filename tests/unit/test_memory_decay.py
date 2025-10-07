from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock
import asyncio
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import memory.core as core


class _StubEmbedding:
    async def get_embedding(self, text: str):  # pragma: no cover - simple stub
        return [0.0] * core.EMBEDDING_DIMENSION

    async def get_embeddings(self, texts):  # pragma: no cover - simple stub
        return [[0.0] * core.EMBEDDING_DIMENSION for _ in texts]


def test_apply_memory_decay_handles_decimal_days_old(monkeypatch):
    monkeypatch.setattr(core, "FallbackEmbedding", lambda: _StubEmbedding())
    monkeypatch.setattr(core.MemoryTelemetry, "record", AsyncMock())

    manager = core.UnifiedMemoryManager(entity_type="npc", entity_id=1, user_id=1, conversation_id=1)

    conn = AsyncMock()
    conn.fetch.return_value = [
        {
            "id": 42,
            "significance": Decimal("3"),
            "times_recalled": 0,
            "days_old": Decimal("120"),
        }
    ]

    affected = asyncio.run(manager.apply_memory_decay(conn=conn, decay_rate=1.0))

    assert affected == 1
    assert conn.execute.await_count == 1
    update_call = conn.execute.await_args_list[0]
    assert update_call.args[1] == 2
    assert update_call.args[2] == 42
