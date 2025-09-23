import asyncio
import copy
import importlib
import json
import os
import pathlib
import sys
from contextlib import asynccontextmanager, contextmanager

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

nyx_novelty_engine = types.ModuleType("nyx.core.novelty_engine")


class DummyNoveltyEngine:
    async def initialize(self, *args, **kwargs):
        return None


class DummyNoveltyIdea:
    pass


nyx_novelty_engine.NoveltyEngine = DummyNoveltyEngine
nyx_novelty_engine.NoveltyIdea = DummyNoveltyIdea
sys.modules["nyx.core.novelty_engine"] = nyx_novelty_engine

nyx_recognition_memory = types.ModuleType("nyx.core.recognition_memory")


class DummyRecognitionMemorySystem:
    async def initialize(self, *args, **kwargs):
        return None


class DummyRecognitionResult:
    pass


nyx_recognition_memory.RecognitionMemorySystem = DummyRecognitionMemorySystem
nyx_recognition_memory.RecognitionResult = DummyRecognitionResult
sys.modules["nyx.core.recognition_memory"] = nyx_recognition_memory

nyx_brain_base = types.ModuleType("nyx.core.brain.base")


class DummyNyxBrain:
    @classmethod
    async def get_instance(cls, *args, **kwargs):
        return cls()


nyx_brain_base.NyxBrain = DummyNyxBrain

nyx_brain_checkpoint = types.ModuleType("nyx.core.brain.checkpointing_agent")


class DummyCheckpointingPlannerAgent:
    async def plan(self, *args, **kwargs):
        return None


nyx_brain_checkpoint.CheckpointingPlannerAgent = DummyCheckpointingPlannerAgent

nyx_module = importlib.import_module("nyx")
nyx_core_module = importlib.import_module("nyx.core")

nyx_brain_module = types.ModuleType("nyx.core.brain")
nyx_brain_module.base = nyx_brain_base
nyx_brain_module.checkpointing_agent = nyx_brain_checkpoint

nyx_core_module.brain = nyx_brain_module

sys.modules["nyx.core.brain"] = nyx_brain_module
sys.modules["nyx.core.brain.base"] = nyx_brain_base
sys.modules["nyx.core.brain.checkpointing_agent"] = nyx_brain_checkpoint

nyx_sdk_module = types.ModuleType("nyx.nyx_agent_sdk")


class DummyNyxSDKConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyNyxAgentSDK:
    def __init__(self, config):
        self.config = config

    async def initialize_agent(self):
        return None


nyx_sdk_module.NyxSDKConfig = DummyNyxSDKConfig
nyx_sdk_module.NyxAgentSDK = DummyNyxAgentSDK
sys.modules["nyx.nyx_agent_sdk"] = nyx_sdk_module

world_director_stub = types.ModuleType("story_agent.world_director_agent")


class DummyWorldDirector:
    def __init__(self, *_args, **_kwargs):
        pass

    async def initialize(self):
        return None

    def start_background_processing(self):
        return None


world_director_stub.CompleteWorldDirector = DummyWorldDirector
sys.modules["story_agent.world_director_agent"] = world_director_stub

import new_game_agent
import tasks


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
        ctx.user_id = 1
        ctx.conversation_id = 99

        params = new_game_agent.GenerateEnvironmentParams(mega_name="Test Mega", mega_desc="Test description")

        result = await new_game_agent.NewGameAgent.generate_environment.__wrapped__(agent, ctx, params)

        assert isinstance(result, new_game_agent.EnvironmentData)
        assert result.setting_name == new_game_agent.FALLBACK_ENVIRONMENT_PAYLOAD["setting_name"]
        assert len(result.locations) == len(new_game_agent.FALLBACK_ENVIRONMENT_PAYLOAD["locations"])
        assert result.quest_data.quest_name == new_game_agent.FALLBACK_ENVIRONMENT_PAYLOAD["quest_data"]["quest_name"]
        assert run_calls["count"] == 3

    asyncio.run(_run())


def test_process_new_game_task_runs_environment_without_attribute_error(monkeypatch):
    user_id = 7
    conversation_id = 321
    location_calls: list[str] = []

    @contextmanager
    def noop_trace(**kwargs):
        yield

    monkeypatch.setattr(tasks, "trace", noop_trace)

    def immediate_run(coro):
        return asyncio.run(coro)

    monkeypatch.setattr(tasks, "run_async_in_worker_loop", immediate_run)

    class DummyConnection:
        async def execute(self, query, *args):
            return None

        async def fetchrow(self, query, *args):
            if "RETURNING id" in query or "SELECT id FROM conversations" in query:
                return {"id": conversation_id}
            return None

        async def fetchval(self, query, *args):
            if "SELECT COUNT(*) FROM messages" in query:
                return 0
            if "SELECT 1 FROM messages" in query:
                return 0
            return None

        async def fetch(self, query, *args):
            return []

        def transaction(self):
            @asynccontextmanager
            async def _txn():
                yield self

            return _txn()

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    monkeypatch.setattr(tasks, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    env_payload = {
        "setting_name": "TestTown",
        "environment_desc": "A detailed description",
        "environment_history": "A storied past",
        "events": [
            {
                "name": "Morning Briefing",
                "description": "Start of day",
                "start_time": "08:00",
                "end_time": "09:00",
                "location": "Central Plaza",
                "year": 1,
                "month": 1,
                "day": 1,
                "time_of_day": "Morning",
            }
        ],
        "locations": [
            {
                "location_name": f"Location {idx}",
                "description": "A place",
                "type": "public",
                "features": ["feature"],
                "open_hours_json": json.dumps({"Mon-Sun": "Always"}),
            }
            for idx in range(10)
        ],
        "scenario_name": "Scenario Alpha",
        "quest_data": {"quest_name": "Quest", "quest_description": "Do things"},
    }

    async def fake_runner_run(cls, prompt, *args, context=None, **kwargs):
        return types.SimpleNamespace(final_output=copy.deepcopy(env_payload))

    monkeypatch.setattr(new_game_agent.Runner, "run", classmethod(fake_runner_run))

    async def fake_create_calendar(self, ctx, params):
        return {"days": ["Sol"], "months": ["Primus"], "seasons": ["Bloom"]}

    async def fake_require_day_names(self, user_id_value, conversation_id_value, timeout=15.0):
        assert user_id_value == user_id
        assert conversation_id_value == conversation_id
        return ["Sol"]

    async def fake_create_chase_schedule(self, ctx, params):
        return json.dumps({"schedule": []})

    async def fake_apply_stat_modifiers(self, user_id_value, conversation_id_value, modifiers):
        assert user_id_value == user_id
        assert conversation_id_value == conversation_id

    async def fake_queue_npc_pool(self, user_id_value, conversation_id_value, target_count=5):
        assert user_id_value == user_id
        assert conversation_id_value == conversation_id

    async def fake_player_schedule(self, ctx, environment_desc):
        return new_game_agent.NPCScheduleData(
            npc_ids=[], chase_schedule_json=json.dumps({"schedule": []})
        )

    async def fake_opening(self, ctx, params):
        return "Opening narrative"

    async def fake_finalize(self, ctx, params):
        return new_game_agent.FinalizeResult(
            status="ready",
            welcome_image_url=None,
            lore_summary="Generated lore",
            initial_conflict="Conflict",
            currency_system="Credits",
        )

    async def fake_is_setup_complete(self, user_id_value, conversation_id_value):
        assert user_id_value == user_id
        assert conversation_id_value == conversation_id
        return True, [], []

    async def fake_insert_default_stats(user_id_value, conversation_id_value):
        assert user_id_value == user_id
        assert conversation_id_value == conversation_id

    class DummyDirectiveHandler:
        async def process_directives(self, force_check=False):
            return None

        def start_background_processing(self):
            return None

    async def fake_initialize_directive_handler(self, user_id_value, conversation_id_value):
        assert user_id_value == user_id
        assert conversation_id_value == conversation_id
        self.directive_handler = DummyDirectiveHandler()

    class DummyGovernance:
        async def register_agent(self, *args, **kwargs):
            return None

        async def check_action_permission(self, *args, **kwargs):
            return {"approved": True}

        async def process_agent_action_report(self, *args, **kwargs):
            return {"success": True}

    async def fake_get_central_governance(user_id_value, conversation_id_value):
        assert user_id_value == user_id
        return DummyGovernance()

    nyx_integrate_module = types.ModuleType("nyx.integrate")
    nyx_integrate_module.get_central_governance = fake_get_central_governance
    sys.modules["nyx.integrate"] = nyx_integrate_module

    async def fake_generate_mega_setting_logic():
        return {
            "selected_settings": ["Component"],
            "enhanced_features": [],
            "stat_modifiers": {},
            "mega_name": "Mega",
            "mega_description": "Mega description",
        }

    async def fake_create_game_setting(cctx, conn, setting_name, **kwargs):
        assert getattr(cctx, "user_id", None) == user_id
        assert getattr(cctx, "conversation_id", None) == conversation_id

    async def fake_find_or_create_event(
        cctx,
        conn,
        name,
        description,
        start_time,
        end_time,
        location,
        year,
        month,
        day,
        time_of_day,
    ):
        assert getattr(cctx, "user_id", None) == user_id
        assert getattr(cctx, "conversation_id", None) == conversation_id

    async def fake_find_or_create_location(
        cctx,
        conn,
        location_name,
        description,
        location_type,
        notable_features,
        open_hours,
    ):
        assert getattr(cctx, "user_id", None) == user_id
        assert getattr(cctx, "conversation_id", None) == conversation_id
        location_calls.append(location_name)

    async def fake_find_or_create_quest(cctx, conn, quest_name, progress_detail, status):
        assert getattr(cctx, "user_id", None) == user_id
        assert getattr(cctx, "conversation_id", None) == conversation_id

    async def fake_update_current_roleplay(cctx, conn, key, value):
        assert getattr(cctx, "user_id", None) == user_id
        assert getattr(cctx, "conversation_id", None) == conversation_id

    original_generate_environment = (
        new_game_agent.NewGameAgent.generate_environment.__wrapped__
    )

    async def passthrough_generate_environment(self, ctx, params):
        return await original_generate_environment(self, ctx, params)

    monkeypatch.setattr(new_game_agent.NewGameAgent, "create_calendar", fake_create_calendar)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_require_day_names", fake_require_day_names)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "create_chase_schedule", fake_create_chase_schedule)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_apply_setting_stat_modifiers", fake_apply_stat_modifiers)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_queue_npc_pool_fill", fake_queue_npc_pool)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_player_schedule_data", fake_player_schedule)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "create_opening_narrative", fake_opening)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "finalize_game_setup", fake_finalize)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_is_setup_complete", fake_is_setup_complete)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "initialize_directive_handler", fake_initialize_directive_handler)
    monkeypatch.setattr(new_game_agent, "insert_default_player_stats_chase", fake_insert_default_stats)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "generate_environment", passthrough_generate_environment)

    import routes.settings_routes as settings_routes

    monkeypatch.setattr(settings_routes, "generate_mega_setting_logic", fake_generate_mega_setting_logic)

    monkeypatch.setattr(new_game_agent.canon, "create_game_setting", fake_create_game_setting)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_event", fake_find_or_create_event)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_location", fake_find_or_create_location)
    monkeypatch.setattr(new_game_agent.canon, "find_or_create_quest", fake_find_or_create_quest)
    monkeypatch.setattr(new_game_agent.canon, "update_current_roleplay", fake_update_current_roleplay)

    result = tasks.process_new_game_task(user_id, {})

    assert result["status"] == "ready"
    assert location_calls
