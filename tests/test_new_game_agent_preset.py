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
    def __init__(self, story_payload, expected_conversation_id, expected_user_id, *, return_raw_story_data=False):
        self.story_payload = story_payload
        self.expected_conversation_id = expected_conversation_id
        self.expected_user_id = expected_user_id
        self.fetchrow_calls = []
        self.execute_calls = []
        self.return_raw_story_data = return_raw_story_data

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

        raise AssertionError(f"Unexpected fetchrow: {normalized_query} {args}")

    async def execute(self, query, *args):
        normalized_query = " ".join(query.split())
        self.execute_calls.append((normalized_query, args))
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

    _install_common_preset_patches(monkeypatch, calls)

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


@pytest.mark.asyncio
async def test_process_preset_game_handles_dict_story_data(monkeypatch):
    agent = new_game_agent.NewGameAgent()

    user_id = 88
    existing_conversation_id = 5150
    preset_story_id = "dict_story"
    conversation_data = {"conversation_id": existing_conversation_id, "preset_story_id": preset_story_id}

    story_payload = {
        "name": "Dict Preset",
        "synopsis": "An adventure stored as a dict.",
        "theme": "exploration",
        "locations": [],
        "npcs": [],
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
        "npcs": [],
        "opening": [],
        "stats": [],
    }

    _install_common_preset_patches(monkeypatch, calls)

    ctx = CanonicalContext(user_id, existing_conversation_id)

    result = await agent.process_preset_game_direct(ctx, conversation_data, preset_story_id)

    assert result.conversation_id == existing_conversation_id
    assert result.scenario_name == "Dict Preset"
    assert result.status == "ready"
    assert any(
        "SET status='ready'" in query and args == (existing_conversation_id, user_id, "Dict Preset")
        for query, args in stub_conn.execute_calls
    )

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


@pytest.mark.asyncio
async def test_setup_check_allows_queued_lore_and_npcs(monkeypatch):
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

    complete, missing, pending = await agent._is_setup_complete(user_id=42, conversation_id=77)

    assert complete is True
    assert missing == []
    assert any('NPC pool' in entry for entry in pending)
    assert any('Lore' in entry for entry in pending)


class QueenPipelineStubConnection:
    def __init__(self, story_payload, user_id, conversation_id):
        self.story_payload = story_payload
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.fetchrow_calls = []
        self.fetchval_calls = []
        self.execute_calls = []
        self.fetch_calls = []

    async def fetchrow(self, query, *args):
        normalized = " ".join(query.split())
        self.fetchrow_calls.append((normalized, args))

        if "SELECT story_data FROM PresetStories" in normalized:
            return {"story_data": json.dumps(self.story_payload)}

        if "SELECT id FROM conversations" in normalized:
            if args == (self.conversation_id, self.user_id):
                return {"id": self.conversation_id}
            return None

        if "SELECT npc_id FROM NPCStats" in normalized:
            return None

        if "SELECT age, birthdate" in normalized:
            return None

        if "SELECT relationships FROM NPCStats" in normalized:
            return {"relationships": json.dumps([])}

        if "SELECT special_mechanics FROM NPCStats" in normalized:
            return {"special_mechanics": None}

        return None

    async def fetchval(self, query, *args):
        normalized = " ".join(query.split())
        self.fetchval_calls.append((normalized, args))

        if "SELECT COUNT(*) FROM NPCMemories" in normalized:
            return 0

        if "SELECT value FROM CurrentRoleplay" in normalized:
            return None

        return None

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        self.execute_calls.append((normalized, args))
        return None

    async def fetch(self, query, *args):
        normalized = " ".join(query.split())
        self.fetch_calls.append((normalized, args))
        return []


@pytest.mark.asyncio
async def test_queen_of_thorns_preset_skips_memory_generation(monkeypatch):
    agent = new_game_agent.NewGameAgent()

    user_id = 55
    conversation_id = 9901
    conversation_data = {"conversation_id": conversation_id, "preset_story_id": "queen_of_thorns"}

    from story_templates.preset_story_loader import PresetStoryLoader
    from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
    import lore.core.lore_system as lore_system_module
    import memory.wrapper as memory_wrapper_module
    import memory.schemas as memory_schemas_module
    import npcs.preset_npc_handler as preset_module
    import db.connection as db_connection_module
    from lore.core import canon as canon_module
    from logic import dynamic_relationships as dynamic_relationships_module

    story_payload = PresetStoryLoader._serialize_story(QUEEN_OF_THORNS_STORY)

    stub_conn = QueenPipelineStubConnection(story_payload, user_id, conversation_id)

    @asynccontextmanager
    async def fake_db_context():
        yield stub_conn

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(db_connection_module, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(preset_module, "get_db_connection_context", fake_db_context)

    async def fake_insert_stats(uid, cid):
        return None

    async def fake_rules(env_desc, setting_name):
        return {
            "capabilities": {},
            "setting_kind": "preset",
            "_reality_context": "normal",
            "hard_rules": [],
            "soft_rules": [],
        }

    async def fake_opening(self, ctx_wrap, preset_data):
        return "Opening narrative"

    async def fake_locations(self, ctx_wrap, preset_data):
        return [101]

    monkeypatch.setattr(new_game_agent, "insert_default_player_stats_chase", fake_insert_stats)
    monkeypatch.setattr(new_game_agent, "synthesize_setting_rules", fake_rules)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_preset_opening", fake_opening)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_preset_locations", fake_locations)

    npc_counter = iter(range(3000, 3010))
    location_counter = iter(range(4000, 4010))

    async def fake_find_or_create_npc(ctx_obj, conn, npc_name: str, **kwargs):
        return next(npc_counter)

    async def fake_find_or_create_location(ctx_wrap, conn, name: str, **kwargs):
        return next(location_counter)

    async def fake_update_current_roleplay(ctx_wrap, conn, key, value):
        return None

    async def fake_update_entity(canon_ctx, conn, entity_name, entity_id, updates, reason):
        return None

    monkeypatch.setattr(canon_module, "find_or_create_npc", fake_find_or_create_npc)
    monkeypatch.setattr(canon_module, "find_or_create_location", fake_find_or_create_location)
    monkeypatch.setattr(canon_module, "update_current_roleplay", fake_update_current_roleplay)
    monkeypatch.setattr(canon_module, "update_entity_canonically", fake_update_entity)

    class DummyLoreSystem:
        async def propose_and_enact_change(self, **kwargs):
            return {"status": "ok"}

    async def fake_lore_instance(uid, cid):
        return DummyLoreSystem()

    monkeypatch.setattr(lore_system_module.LoreSystem, "get_instance", fake_lore_instance)

    memory_instances = {}

    class DummyMemorySystem:
        def __init__(self, uid, cid):
            self.user_id = uid
            self.conversation_id = cid
            self.memories = []
            self.beliefs = []
            self.emotions = {}

        async def remember(self, entity_type, entity_id, memory_text, importance="medium", emotional=True, tags=None):
            self.memories.append(memory_text)
            return {"memory_id": len(self.memories)}

        async def update_npc_emotion(self, npc_id, emotion, intensity=0.5):
            self.emotions[npc_id] = {"emotion": emotion, "intensity": intensity}
            return {"status": "ok"}

        async def create_belief(self, entity_type, entity_id, belief_text, confidence=0.7):
            self.beliefs.append(belief_text)
            return {"belief_id": len(self.beliefs)}

    async def fake_memory_instance(uid, cid):
        key = (uid, cid)
        if key not in memory_instances:
            memory_instances[key] = DummyMemorySystem(uid, cid)
        return memory_instances[key]

    monkeypatch.setattr(memory_wrapper_module.MemorySystem, "get_instance", fake_memory_instance)

    class DummyMemorySchemaManager:
        def __init__(self, uid, cid):
            self.user_id = uid
            self.conversation_id = cid
            self.created = []

        async def create_schema(self, **kwargs):
            self.created.append(kwargs.get("schema_name"))

    monkeypatch.setattr(memory_schemas_module, "MemorySchemaManager", DummyMemorySchemaManager)

    class DummyDimensions:
        def __init__(self):
            self.trust = 0
            self.respect = 0
            self.affection = 0
            self.intimacy = 0
            self.fascination = 0
            self.influence = 0
            self.volatility = 0
            self.unresolved_conflict = 0

        def clamp(self):
            for field in (
                "trust",
                "respect",
                "affection",
                "intimacy",
                "fascination",
                "influence",
                "volatility",
                "unresolved_conflict",
            ):
                value = getattr(self, field)
                if value > 100:
                    setattr(self, field, 100)
                elif value < -100:
                    setattr(self, field, -100)

    class DummyRelationshipState:
        def __init__(self):
            self.dimensions = DummyDimensions()

    class DummyRelationshipManager:
        def __init__(self, uid, cid):
            self.user_id = uid
            self.conversation_id = cid
            self.states = []

        async def get_relationship_state(self, **kwargs):
            state = DummyRelationshipState()
            self.states.append(state)
            return state

        async def _queue_update(self, state):
            return None

        async def _flush_updates(self):
            return None

    monkeypatch.setattr(dynamic_relationships_module, "OptimizedRelationshipManager", DummyRelationshipManager)
    monkeypatch.setattr(preset_module, "OptimizedRelationshipManager", DummyRelationshipManager)

    async def fake_update_context(*args, **kwargs):
        return None

    monkeypatch.setattr(preset_module, "update_relationship_context_tool", fake_update_context)

    async def fake_special_mechanics(*args, **kwargs):
        return None

    monkeypatch.setattr(
        preset_module.PresetNPCHandler,
        "_initialize_special_mechanics",
        staticmethod(fake_special_mechanics),
    )

    calls = {"generate": 0}

    async def fake_generate_memories(self, ctx_wrap, npc_name):
        calls["generate"] += 1
        return []

    monkeypatch.setattr(new_game_agent.NPCCreationHandler, "generate_memories", fake_generate_memories)

    ctx = CanonicalContext(user_id, conversation_id)
    result = await agent.process_preset_game_direct(ctx, conversation_data, "queen_of_thorns")

    assert result.status == "ready"
    assert conversation_data["conversation_id"] == conversation_id
    assert calls["generate"] == 0

    ready_updates = [
        (query, args)
        for query, args in stub_conn.execute_calls
        if "UPDATE conversations" in query and "SET status='ready'" in query
    ]
    assert ready_updates

    memory_key = (user_id, conversation_id)
    assert memory_key in memory_instances
    assert memory_instances[memory_key].memories  # preset memories seeded
