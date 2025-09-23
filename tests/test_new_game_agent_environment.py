import asyncio
import os
import pathlib
import sys
from contextlib import asynccontextmanager

import pytest
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def encode(self, texts, **kwargs):
        return [[0.0] * self._dim for _ in texts]

    def get_sentence_embedding_dimension(self):
        return self._dim


class DummyTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def get_word_embedding_dimension(self):
        return self._dim


class DummyPooling:
    def __init__(self, dim, pooling_mode="mean"):
        self.dim = dim
        self.pooling_mode = pooling_mode


dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: DummyTransformer(*args, **kwargs)
dummy_models.Pooling = lambda *args, **kwargs: DummyPooling(*args, **kwargs)

dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules["sentence_transformers"] = dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import new_game_agent


def test_generate_environment_uses_fallback_payload(monkeypatch):
    async def _run():
        agent = new_game_agent.NewGameAgent()

        run_calls = {"count": 0}

        async def failing_run(cls, *args, **kwargs):
            run_calls["count"] += 1
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(new_game_agent.Runner, "run", classmethod(failing_run))

        async def fake_create_calendar(self, ctx, params):
            return {"days": ["Sol"], "months": ["Primus"], "seasons": ["Bloom"]}

        async def fake_require_day_names(self, user_id, conversation_id, timeout=15.0):
            return ["Sol"]

        monkeypatch.setattr(new_game_agent.NewGameAgent, "create_calendar", fake_create_calendar)
        monkeypatch.setattr(new_game_agent.NewGameAgent, "_require_day_names", fake_require_day_names)

        class DummyConnection:
            def __init__(self):
                self.execute_calls = []

            async def execute(self, query, *args):
                self.execute_calls.append((query, args))
                return None

            async def fetchrow(self, query, *args):
                return None

            async def fetchval(self, query, *args):
                return None

            def transaction(self):
                @asynccontextmanager
                async def _txn():
                    yield self

                return _txn()

        @asynccontextmanager
        async def fake_db_context():
            yield DummyConnection()

        monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

        async def fake_create_game_setting(cctx, conn, setting_name, **kwargs):
            return None

        async def fake_find_or_create_event(
            cctx, conn, name, description, start_time, end_time, location, year, month, day, time_of_day
        ):
            return None

        async def fake_find_or_create_location(
            cctx, conn, location_name, description, location_type, notable_features, open_hours
        ):
            return None

        async def fake_find_or_create_quest(cctx, conn, quest_name, progress_detail, status):
            return None

        async def fake_update_current_roleplay(cctx, conn, key, value):
            return None

        async def fake_synthesize_setting_rules(desc, name):
            return {
                "capabilities": {},
                "setting_kind": "test_kind",
                "_reality_context": "normal",
                "hard_rules": [],
                "soft_rules": [],
            }

        monkeypatch.setattr(new_game_agent.canon, "create_game_setting", fake_create_game_setting)
        monkeypatch.setattr(new_game_agent.canon, "find_or_create_event", fake_find_or_create_event)
        monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", fake_find_or_create_location)
        monkeypatch.setattr(new_game_agent.canon, "find_or_create_quest", fake_find_or_create_quest)
        monkeypatch.setattr(new_game_agent.canon, "update_current_roleplay", fake_update_current_roleplay)
        monkeypatch.setattr(new_game_agent, "synthesize_setting_rules", fake_synthesize_setting_rules)

        ctx = new_game_agent.RunContextWrapper(context={"user_id": 1, "conversation_id": 99})

        params = new_game_agent.GenerateEnvironmentParams(mega_name="Test Mega", mega_desc="Test description")

        result = await new_game_agent.NewGameAgent.generate_environment.__wrapped__(agent, ctx, params)

        assert isinstance(result, new_game_agent.EnvironmentData)
        assert result.setting_name == new_game_agent.FALLBACK_ENVIRONMENT_PAYLOAD["setting_name"]
        assert len(result.locations) == len(new_game_agent.FALLBACK_ENVIRONMENT_PAYLOAD["locations"])
        assert result.quest_data.quest_name == new_game_agent.FALLBACK_ENVIRONMENT_PAYLOAD["quest_data"]["quest_name"]
        assert run_calls["count"] == 3

    asyncio.run(_run())
