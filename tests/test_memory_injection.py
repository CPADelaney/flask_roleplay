import pytest
import types
import sys
import importlib

@pytest.mark.asyncio
async def test_prepare_context_injects_memories():
    async def fake_fetch(user_msg, k=5):
        return [{"text": "Paris is the capital of France", "meta": {"uid": "m1"}, "score": 0.9}]

    class FakeMM:
        fetch_relevant = staticmethod(fake_fetch)

    sys.modules['nyx.core.memory.memory_manager'] = types.SimpleNamespace(MemoryManager=FakeMM)
    orchestrator = importlib.reload(importlib.import_module('nyx.core.orchestrator'))

    ctx = "System prompt"
    user_msg = "What is the capital of France?"
    new_ctx = await orchestrator.prepare_context(ctx, user_msg)

    assert new_ctx.startswith("KNOWLEDGE:")
    assert "Paris is the capital of France" in new_ctx
    assert "<!--MEM:m1,0.90-->" in new_ctx.split("\n")[2]

    del sys.modules['nyx.core.memory.memory_manager']
