import json
import os
import sys
import types
from contextlib import asynccontextmanager

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self._dim = 384

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return np.zeros((len(texts), self._dim), dtype=np.float32)

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

sys.modules.setdefault("sentence_transformers", dummy_sentence_transformers)
sys.modules.setdefault("sentence_transformers.models", dummy_models)

import new_game_agent
from lore.core.context import CanonicalContext


class StubConnection:
    def __init__(self, story_payload, expected_conversation_id, expected_user_id):
        self.story_payload = story_payload
        self.expected_conversation_id = expected_conversation_id
        self.expected_user_id = expected_user_id
        self.fetchrow_calls = []
        self.execute_calls = []

    async def fetchrow(self, query, *args):
        normalized_query = " ".join(query.split())
        self.fetchrow_calls.append((normalized_query, args))

        if "SELECT story_data FROM PresetStories" in normalized_query:
            return {"story_data": json.dumps(self.story_payload)}

        if "SELECT id FROM conversations" in normalized_query:
            if args == (self.expected_conversation_id, self.expected_user_id):
                return {"id": self.expected_conversation_id}
            return None

        raise AssertionError(f"Unexpected fetchrow: {normalized_query} {args}")

    async def execute(self, query, *args):
        normalized_query = " ".join(query.split())
        self.execute_calls.append((normalized_query, args))
        return None


@pytest.mark.asyncio
async def test_process_preset_game_reuses_existing_conversation(monkeypatch):
    agent = new_game_agent.NewGameAgent()

    user_id = 77
    existing_conversation_id = 4242
    preset_story_id = "test_story"
    conversation_data = {"conversation_id": existing_conversation_id, "preset_story_id": preset_story_id}

    story_payload = {
        "name": "Test Preset",
        "synopsis": "A quiet town hides many secrets.",
        "theme": "mystery",
        "locations": [],
        "npcs": [],
    }

    stub_conn = StubConnection(story_payload, existing_conversation_id, user_id)

    @asynccontextmanager
    async def fake_db_context():
        yield stub_conn

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    calls = {
        "environment": [],
        "calendar": [],
        "locations": [],
        "npcs": [],
        "opening": [],
        "stats": [],
    }

    async def fake_insert_default_player_stats_chase(uid, cid):
        calls["stats"].append((uid, cid))

    async def fake_setup_environment(self, ctx_wrap, preset_data):
        calls["environment"].append((ctx_wrap.context["conversation_id"], preset_data["name"]))

    async def fake_setup_calendar(self, ctx_wrap):
        calls["calendar"].append(ctx_wrap.context["conversation_id"])

    async def fake_create_locations(self, ctx_wrap, preset_data):
        calls["locations"].append(ctx_wrap.context["conversation_id"])
        return [101]

    async def fake_create_npcs(self, ctx_wrap, preset_data):
        calls["npcs"].append(ctx_wrap.context["conversation_id"])
        return [201]

    async def fake_create_opening(self, ctx_wrap, preset_data):
        calls["opening"].append(ctx_wrap.context["conversation_id"])
        return "Opening narrative"

    async def fake_rules(env_desc, setting_name):
        return {
            "capabilities": {"travel": True},
            "setting_kind": "test_kind",
            "_reality_context": "normal",
            "hard_rules": [],
            "soft_rules": [],
        }

    async def fake_canon_update(ctx_wrap, conn, key, value):
        # Simulate canonical update without touching the database
        return None

    monkeypatch.setattr(new_game_agent, "insert_default_player_stats_chase", fake_insert_default_player_stats_chase)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_setup_preset_environment", fake_setup_environment)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_setup_standard_calendar", fake_setup_calendar)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_preset_locations", fake_create_locations)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_preset_npcs", fake_create_npcs)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_preset_opening", fake_create_opening)
    monkeypatch.setattr(new_game_agent, "synthesize_setting_rules", fake_rules)
    monkeypatch.setattr(new_game_agent.canon, "update_current_roleplay", fake_canon_update)

    ctx = CanonicalContext(user_id, existing_conversation_id)

    result = await agent.process_preset_game_direct(ctx, conversation_data, preset_story_id)

    assert result.conversation_id == existing_conversation_id
    assert conversation_data["conversation_id"] == existing_conversation_id
    assert result.opening_narrative == "Opening narrative"

    # Ensure the existing conversation was prepared for reuse
    delete_tables = {"Events", "PlannedEvents", "PlayerInventory", "Quests", "NPCStats", "Locations", "SocialLinks", "CurrentRoleplay"}
    delete_queries = {
        query
        for query, args in stub_conn.execute_calls
        if query.startswith("DELETE FROM") and args == (user_id, existing_conversation_id)
    }
    assert delete_tables == {q.split()[2] for q in delete_queries}

    # Status should be set back to processing and eventually ready
    assert any("SET status='processing'" in query and args == (existing_conversation_id, user_id, "Test Preset")
               for query, args in stub_conn.execute_calls)
    assert any("SET status='ready'" in query and args[0] == existing_conversation_id for query, args in stub_conn.execute_calls)

    # All seeding helpers should have targeted the original conversation
    assert calls["stats"] == [(user_id, existing_conversation_id)]
    assert calls["environment"] == [(existing_conversation_id, "Test Preset")]
    assert calls["calendar"] == [existing_conversation_id]
    assert calls["locations"] == [existing_conversation_id]
    assert calls["npcs"] == [existing_conversation_id]
    assert calls["opening"] == [existing_conversation_id]

    # The preset flow should not have inserted a new conversation row
    assert not any("INSERT INTO conversations" in query for query, _ in stub_conn.fetchrow_calls)
