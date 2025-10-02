import asyncio
import importlib
import asyncio
import os
import sys
import types
import typing
from contextlib import asynccontextmanager

import pytest
import typing_extensions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

typing.TypedDict = typing_extensions.TypedDict

dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: None
dummy_models.Pooling = lambda *args, **kwargs: None


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):
        return 3


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

feasibility = importlib.import_module("nyx.nyx_agent.feasibility")


def test_load_context_uses_setting_kind_fallback(monkeypatch):
    run_calls = {"count": 0}

    async def fake_run(cls, *args, **kwargs):
        run_calls["count"] += 1
        return types.SimpleNamespace(final_output="{}")

    monkeypatch.setattr(feasibility.Runner, "run", classmethod(fake_run))

    class DummyConnection:
        async def fetch(self, query, *args):
            if "FROM CurrentRoleplay" in query and "key = ANY" in query:
                return [{"key": "SettingKind", "value": "science_fiction"}]
            if "FROM GameRules" in query:
                return []
            if "FROM PlayerInventory" in query:
                return []
            if "FROM NPCStats" in query:
                return []
            if "FROM messages" in query:
                return []
            return []

        async def fetchval(self, *args, **kwargs):
            return None

        async def fetchrow(self, *args, **kwargs):
            return None

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    monkeypatch.setattr(feasibility, "get_db_connection_context", fake_db_context)

    class DummyNyxContext:
        def __init__(self, user_id: int, conversation_id: int):
            self.user_id = user_id
            self.conversation_id = conversation_id

    async def _run():
        ctx = DummyNyxContext(user_id=1, conversation_id=2)
        context = await feasibility._load_comprehensive_context(ctx)

        assert context["type"] == "sci_fi_futuristic"
        assert context["kind"] == "science_fiction"
        assert context["capabilities"].get("technology") == "futuristic"
        assert context["technology_level"] == "futuristic"
        assert context["setting_era"] == "far_future"
        assert context["magic_system"] == "none"

    asyncio.run(_run())
    assert run_calls["count"] == 0
