import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from memory.core import MemoryCache, MemoryType, UnifiedMemoryManager


def test_add_memory_embeds_before_db_connection(monkeypatch):
    async def run_test():
        class DummyEmbeddingProvider:
            async def get_embedding(self, text: str):
                return [0.0]

            async def get_embeddings(self, texts):
                return [[0.0] for _ in texts]

        manager = UnifiedMemoryManager.__new__(UnifiedMemoryManager)
        manager.entity_type = "npc"
        manager.entity_id = 1
        manager.user_id = 2
        manager.conversation_id = 3
        manager.cache = MemoryCache()
        manager.embedding_provider = DummyEmbeddingProvider()

        connection_active = False

        fetchval_mock = AsyncMock(return_value=123)

        class FakeTransaction:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeConn:
            def __init__(self):
                self.fetchval = fetchval_mock

            def transaction(self):
                return FakeTransaction()

        @asynccontextmanager
        async def fake_connection_context():
            nonlocal connection_active
            connection_active = True
            try:
                yield FakeConn()
            finally:
                connection_active = False

        monkeypatch.setattr(
            "db.connection.get_db_connection_context",
            fake_connection_context
        )

        async def fake_embedding(text: str):
            assert connection_active is False, "Embedding should occur before DB connection acquisition"
            return [42.0]

        manager.embedding_provider.get_embedding = AsyncMock(side_effect=fake_embedding)

        memory_id = await manager.add_memory("test memory", memory_type=MemoryType.OBSERVATION)

        assert memory_id == 123
        manager.embedding_provider.get_embedding.assert_awaited_once()

        # Ensure the computed embedding was sent to the insert query
        assert fetchval_mock.await_args_list, "No database calls were made"
        insert_args = fetchval_mock.await_args_list[0].args
        # args[0] is the SQL string; args[10] corresponds to the embedding payload
        assert insert_args[10] == [42.0]
        # args[9] should be the serialized tags array
        assert json.loads(insert_args[9]) == []

        assert connection_active is False

    asyncio.run(run_test())
