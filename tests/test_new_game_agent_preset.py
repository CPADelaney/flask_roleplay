import asyncio
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
import npcs.preset_npc_handler as preset_module
from lore.core.context import CanonicalContext


class StubConnection:
    NPC_STATS_COLUMNS = [
        "user_id",
        "conversation_id",
        "npc_name",
        "sex",
        "age",
        "physical_description",
        "role",
        "introduced",
        "current_location",
        "dominance",
        "cruelty",
        "closeness",
        "trust",
        "respect",
        "affection",
        "intensity",
        "personality_traits",
        "likes",
        "dislikes",
        "hobbies",
        "relationships",
        "affiliations",
        "schedule",
        "archetypes",
        "archetype_summary",
        "archetype_extras_summary",
        "memory",
        "special_mechanics",
    ]

    def __init__(
        self,
        story_payload,
        expected_conversation_id,
        expected_user_id,
        *,
        return_raw_story_data=False,
    ):
        self.story_payload = story_payload
        self.expected_conversation_id = expected_conversation_id
        self.expected_user_id = expected_user_id
        self.return_raw_story_data = return_raw_story_data
        self.fetchrow_calls = []
        self.fetch_calls = []
        self.execute_calls = []
        self.npc_stat_inserts = []
        self.npc_memory_inserts = []
        self.unified_memory_inserts = []
        self.social_link_inserts = []
        self._next_npc_id = 1000
        self.start_location_name = "Preset Hub"

    async def fetchrow(self, query, *args):
        normalized_query = " ".join(query.split())
        self.fetchrow_calls.append((normalized_query, args))

        if "SELECT story_data FROM PresetStories" in normalized_query:
            if self.return_raw_story_data:
                return {"story_data": self.story_payload}
            return {"story_data": json.dumps(self.story_payload)}

        if "SELECT id FROM conversations" in normalized_query:
            if args == (self.expected_conversation_id, self.expected_user_id):
                return {"id": self.expected_conversation_id}
            return None

        if normalized_query.startswith("INSERT INTO NPCStats"):
            record = dict(zip(self.NPC_STATS_COLUMNS, args))
            record["npc_id"] = self._next_npc_id
            self._next_npc_id += 1
            self.npc_stat_inserts.append(record)
            return {"npc_id": record["npc_id"]}

        raise AssertionError(f"Unexpected fetchrow: {normalized_query} {args}")

    async def fetch(self, query, *args):
        normalized_query = " ".join(query.split())
        self.fetch_calls.append((normalized_query, args))

        if "SELECT location_name FROM Locations" in normalized_query:
            if args == (self.expected_user_id, self.expected_conversation_id):
                return [{"location_name": self.start_location_name}]
            return []

        raise AssertionError(f"Unexpected fetch: {normalized_query} {args}")

    async def execute(self, query, *args):
        normalized_query = " ".join(query.split())
        self.execute_calls.append((normalized_query, args))

        if normalized_query.startswith("INSERT INTO NPCMemories"):
            self.npc_memory_inserts.append(args)
        elif normalized_query.startswith("INSERT INTO unified_memories"):
            self.unified_memory_inserts.append(args)
        elif normalized_query.startswith("INSERT INTO SocialLinks"):
            self.social_link_inserts.append(args)

        return None


def _install_common_preset_patches(monkeypatch, calls):
    async def fake_insert_default_player_stats_chase(uid, cid):
        calls["stats"].append((uid, cid))

    async def fake_setup_environment(self, ctx_wrap, preset_data):
        calls["environment"].append((ctx_wrap.context["conversation_id"], preset_data["name"]))

    async def fake_setup_calendar(self, ctx_wrap):
        calls["calendar"].append(ctx_wrap.context["conversation_id"])

    async def fake_create_locations(self, ctx_wrap, preset_data):
        calls["locations"].append(ctx_wrap.context["conversation_id"])
        return [101]

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

    async def fake_load_calendar_names(uid, cid):
        calls.setdefault("calendar_names", []).append((uid, cid))
        return {"months": ["Dawn"], "days": ["Sol"]}

    async def fake_set_current_time(uid, cid, year, month_idx, day_num, phase):
        calls.setdefault("set_current_time", []).append((uid, cid, year, month_idx, day_num, phase))

    async def fake_canon_update(ctx_wrap, conn, key, value):
        # Simulate canonical update without touching the database
        calls.setdefault("canon_updates", []).append((key, value))
        return None

    async def fail_preset_handler(*args, **kwargs):
        raise AssertionError("PresetNPCHandler.create_detailed_npc should not be called in fast path")

    def fake_randint(a, b):
        return a

    def fake_choice(seq):
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        return seq[0]

    monkeypatch.setattr(new_game_agent, "insert_default_player_stats_chase", fake_insert_default_player_stats_chase)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_setup_preset_environment", fake_setup_environment)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_setup_standard_calendar", fake_setup_calendar)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_preset_locations", fake_create_locations)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_preset_opening", fake_create_opening)
    monkeypatch.setattr(new_game_agent, "synthesize_setting_rules", fake_rules)
    monkeypatch.setattr(new_game_agent, "load_calendar_names", fake_load_calendar_names)
    monkeypatch.setattr(new_game_agent, "set_current_time", fake_set_current_time)
    monkeypatch.setattr(new_game_agent.canon, "update_current_roleplay", fake_canon_update)
    monkeypatch.setattr(
        preset_module.PresetNPCHandler,
        "create_detailed_npc",
        staticmethod(fail_preset_handler),
    )
    monkeypatch.setattr(new_game_agent.random, "randint", fake_randint)
    monkeypatch.setattr(new_game_agent.random, "choice", fake_choice)


def test_process_preset_game_reuses_existing_conversation(monkeypatch):
    agent = new_game_agent.NewGameAgent()

    user_id = 77
    existing_conversation_id = 4242
    preset_story_id = "test_story"
    conversation_data = {"conversation_id": existing_conversation_id, "preset_story_id": preset_story_id}

    story_payload = {
        "id": preset_story_id,
        "name": "Test Preset",
        "synopsis": "A quiet town hides many secrets.",
        "theme": "mystery",
        "locations": [],
        "npcs": [],
        "required_locations": [],
        "required_npcs": [
            {
                "id": "npc_alpha",
                "name": "Alex Storm",
                "traits": ["observant"],
                "stats": {"dominance": 55, "trust": 20, "affection": 15},
                "personality": {
                    "likes": ["coffee"],
                    "dislikes": ["lies"],
                    "hobbies": ["painting"],
                },
                "memories": [
                    {
                        "memory_text": "Alex met the player at the station.",
                        "tags": ["meeting"],
                        "significance": 4,
                    }
                ],
                "relationships": [
                    {"type": "ally", "strength": 70}
                ],
                "special_mechanics": {"ritual": "sunrise briefing"},
            }
        ],
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
        "opening": [],
        "stats": [],
        "canon_updates": [],
        "set_current_time": [],
        "calendar_names": [],
    }

    _install_common_preset_patches(monkeypatch, calls)

    ctx = CanonicalContext(user_id, existing_conversation_id)

    result = asyncio.run(agent.process_preset_game_direct(ctx, conversation_data, preset_story_id))

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
    assert calls["opening"] == [existing_conversation_id]
    assert calls["calendar_names"] == [(user_id, existing_conversation_id)]

    assert stub_conn.npc_stat_inserts
    npc_record = stub_conn.npc_stat_inserts[0]
    assert npc_record["user_id"] == user_id
    assert npc_record["conversation_id"] == existing_conversation_id

    relationships = json.loads(npc_record["relationships"])
    assert relationships[0]["relationship_label"] == "ally"

    assert len(stub_conn.npc_memory_inserts) == 1
    assert len(stub_conn.unified_memory_inserts) == 1
    assert len(stub_conn.social_link_inserts) == 1

    # The preset flow should not have inserted a new conversation row
    assert not any("INSERT INTO conversations" in query for query, _ in stub_conn.fetchrow_calls)

    expected_time = f"Year 1 Dawn Sol {new_game_agent.TIME_PHASES[0]}"
    assert any(
        key == "CurrentLocation" and value == stub_conn.start_location_name
        for key, value in calls["canon_updates"]
    )
    assert any(
        key == "CurrentTime" and value == expected_time
        for key, value in calls["canon_updates"]
    )
    assert calls["set_current_time"] == [
        (user_id, existing_conversation_id, 1, 1, 1, new_game_agent.TIME_PHASES[0])
    ]


def test_process_preset_game_handles_dict_story_data(monkeypatch):
    agent = new_game_agent.NewGameAgent()

    user_id = 88
    existing_conversation_id = 5150
    preset_story_id = "dict_story"
    conversation_data = {"conversation_id": existing_conversation_id, "preset_story_id": preset_story_id}

    story_payload = {
        "id": preset_story_id,
        "name": "Dict Preset",
        "synopsis": "An adventure stored as a dict.",
        "theme": "exploration",
        "locations": [],
        "npcs": [],
        "required_locations": [],
        "required_npcs": [
            {
                "id": "npc_beta",
                "name": "Blair Rowan",
                "traits": ["curious"],
                "stats": {"dominance": 30, "trust": 25},
                "relationships": [
                    {"type": "mentor", "strength": 60}
                ],
            }
        ],
    }

    stub_conn = StubConnection(
        story_payload,
        existing_conversation_id,
        user_id,
        return_raw_story_data=True,
    )

    @asynccontextmanager
    async def fake_db_context():
        yield stub_conn

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    calls = {
        "environment": [],
        "calendar": [],
        "locations": [],
        "opening": [],
        "stats": [],
        "canon_updates": [],
        "set_current_time": [],
        "calendar_names": [],
    }

    _install_common_preset_patches(monkeypatch, calls)

    ctx = CanonicalContext(user_id, existing_conversation_id)

    result = asyncio.run(agent.process_preset_game_direct(ctx, conversation_data, preset_story_id))

    assert result.conversation_id == existing_conversation_id
    assert result.scenario_name == "Dict Preset"
    assert result.status == "ready"
    assert any(
        "SET status='ready'" in query and args == (existing_conversation_id, user_id, "Dict Preset")
        for query, args in stub_conn.execute_calls
    )

    assert stub_conn.npc_stat_inserts

class SetupCheckStubConnection:
    def __init__(
        self,
        npc_count,
        location_count,
        roleplay_values,
        pool_status,
        lore_status,
        preset_story_id,
        preset_story_data,
    ):
        self.npc_count = npc_count
        self.location_count = location_count
        self.roleplay_values = roleplay_values
        self.pool_status = pool_status
        self.lore_status = lore_status
        self.preset_story_id = preset_story_id
        self.preset_story_data = preset_story_data
        self.fetchval_calls = []
        self.fetchrow_calls = []

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        self.fetchval_calls.append((normalized, args))

        if "SELECT COUNT(*) FROM NPCStats" in normalized:
            return self.npc_count

        if "SELECT COUNT(*) FROM Locations" in normalized:
            return self.location_count

        if "SELECT story_id FROM PresetStoryProgress" in normalized:
            return self.preset_story_id

        if "SELECT value FROM CurrentRoleplay" in normalized:
            key = args[2] if len(args) > 2 else None
            if key == 'LoreGenerationStatus':
                return self.lore_status
            if key == 'NPCPoolStatus':
                return self.pool_status
            if key is not None:
                return self.roleplay_values.get(key)
            return None

        return None

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())
        self.fetchrow_calls.append((normalized, args))

        if "SELECT value FROM CurrentRoleplay" in normalized and "NPCPoolStatus" in normalized:
            if self.pool_status is None:
                return None
            return {"value": self.pool_status}

        if "SELECT story_data FROM PresetStories" in normalized:
            return {"story_data": self.preset_story_data}

        return None

    async def execute(self, query, *args):
        return None

    async def fetch(self, query, *args):
        return []


def test_setup_check_allows_queued_lore_and_npcs(monkeypatch):
    agent = new_game_agent.NewGameAgent()

    pool_status = json.dumps({"status": "queued", "target": 5})
    roleplay_values = {
        'CurrentSetting': 'Queen of Thorns',
        'EnvironmentDesc': 'A hidden network of power in the Bay Area.',
        'ChaseSchedule': json.dumps({'status': 'pending'}),
        'LoreSummary': 'Lore generation pending (background task queued)',
        'NPCPoolStatus': pool_status,
    }

    stub_conn = SetupCheckStubConnection(
        npc_count=1,
        location_count=2,
        roleplay_values=roleplay_values,
        pool_status=pool_status,
        lore_status=json.dumps({"status": "queued"}),
        preset_story_id="queen_of_thorns",
        preset_story_data={'required_locations': [{'name': 'Velvet Sanctum'}, {'name': 'Thorn Garden'}]},
    )

    @asynccontextmanager
    async def fake_db_context():
        yield stub_conn

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    complete, missing, pending = asyncio.run(agent._is_setup_complete(user_id=42, conversation_id=77))

    assert complete is True
    assert missing == []
    assert any('NPC pool' in entry for entry in pending)
    assert any('Lore' in entry for entry in pending)



def test_queen_of_thorns_preset_uses_fast_path(monkeypatch):
    agent = new_game_agent.NewGameAgent()

    user_id = 55
    conversation_id = 9901
    conversation_data = {"conversation_id": conversation_id, "preset_story_id": "queen_of_thorns"}

    from story_templates.preset_story_loader import PresetStoryLoader
    from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY

    story_payload = PresetStoryLoader._serialize_story(QUEEN_OF_THORNS_STORY)

    stub_conn = StubConnection(story_payload, conversation_id, user_id)

    @asynccontextmanager
    async def fake_db_context():
        yield stub_conn

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    calls = {
        "environment": [],
        "calendar": [],
        "locations": [],
        "opening": [],
        "stats": [],
    }

    _install_common_preset_patches(monkeypatch, calls)

    async def fail_generate(*args, **kwargs):
        raise AssertionError("generate_memories should not be called for preset NPC fast path")

    monkeypatch.setattr(new_game_agent.NPCCreationHandler, "generate_memories", fail_generate)

    ctx = CanonicalContext(user_id, conversation_id)
    result = asyncio.run(agent.process_preset_game_direct(ctx, conversation_data, "queen_of_thorns"))

    assert result.status == "ready"
    assert conversation_data["conversation_id"] == conversation_id

    ready_updates = [
        (query, args)
        for query, args in stub_conn.execute_calls
        if "UPDATE conversations" in query and "SET status='ready'" in query
    ]
    assert ready_updates

    assert len(stub_conn.npc_stat_inserts) == len(story_payload["required_npcs"])
    assert len(stub_conn.social_link_inserts) >= len(story_payload["required_npcs"])
