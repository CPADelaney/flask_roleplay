import sys
import time as pytime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import core as memory_core
from memory.core import EMBEDDING_DIMENSION, FallbackEmbedding


class DummyProvider:
    def __init__(self):
        self.call_count = 0

    async def get_embedding(self, text: str):
        self.call_count += 1
        # Return a deterministic vector that encodes the call count for visibility
        return [float(self.call_count)] + [0.0] * (EMBEDDING_DIMENSION - 1)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_fallback_embedding_reuses_cached_vector():
    provider = DummyProvider()
    fallback = FallbackEmbedding(providers=[provider], cache_ttl=60, cache_maxsize=16)

    first = await fallback.get_embedding("hello world")
    second = await fallback.get_embedding("hello world")

    assert provider.call_count == 1
    assert second == first


@pytest.mark.anyio
async def test_fallback_embedding_cache_expires(monkeypatch):
    provider = DummyProvider()
    fallback = FallbackEmbedding(providers=[provider], cache_ttl=1, cache_maxsize=16)

    base_time = pytime.time()
    monkeypatch.setattr(memory_core.time, "time", lambda: base_time)
    await fallback.get_embedding("hello world")
    assert provider.call_count == 1

    monkeypatch.setattr(memory_core.time, "time", lambda: base_time + fallback._cache_ttl + 1)
    await fallback.get_embedding("hello world")
    assert provider.call_count == 2
