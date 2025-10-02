import asyncio
import contextlib
import json
import os
import pathlib
import sys
import types
from contextlib import asynccontextmanager
from datetime import datetime

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")


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

sys.modules.setdefault("sentence_transformers", dummy_sentence_transformers)
sys.modules.setdefault("sentence_transformers.models", dummy_models)

import new_game_agent
from lore.core.context import CanonicalContext
from nyx import integrate as nyx_integrate
import routes.settings_routes as settings_routes


@pytest.mark.asyncio
async def test_process_new_game_does_not_block_on_background(monkeypatch):
    created_conversation_id = 4242

    class DummyConnection:
        async def fetchrow(self, query, *args):
            if "INSERT INTO conversations" in query:
                return {"id": created_conversation_id}
            return {"id": created_conversation_id}

        async def execute(self, query, *args):
            return None

        async def fetchval(self, query, *args):
            return 0

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    async def fake_insert_default_player_stats(user_id, conversation_id):
        assert conversation_id == created_conversation_id

    monkeypatch.setattr(
        new_game_agent,
        "insert_default_player_stats_chase",
        fake_insert_default_player_stats,
    )

    async def fake_generate_mega_setting_logic():
        return {
            "selected_settings": ["A quiet testing town"],
            "enhanced_features": [],
            "stat_modifiers": {"kindness": 1},
            "mega_name": "Test Mega",
            "mega_description": "Test description",
        }

    monkeypatch.setattr(
        settings_routes,
        "generate_mega_setting_logic",
        fake_generate_mega_setting_logic,
    )

    async def fake_apply_modifiers(self, user_id, conversation_id, modifiers):
        assert conversation_id == created_conversation_id

    monkeypatch.setattr(
        new_game_agent.NewGameAgent,
        "_apply_setting_stat_modifiers",
        fake_apply_modifiers,
    )

    async def fake_generate_environment(self, ctx_wrap, params):
        return new_game_agent.EnvironmentData(
            setting_name="Testville",
            environment_desc="Test desc",
            environment_history="Test history",
            scenario_name="Testing Scenario",
            events=[],
            locations=[],
            quest_data=new_game_agent.QuestData(),
        )

    monkeypatch.setattr(
        new_game_agent.NewGameAgent,
        "generate_environment",
        fake_generate_environment,
    )

    async def fake_create_player_schedule(self, ctx_wrap, desc):
        return new_game_agent.NPCScheduleData(npc_ids=[], chase_schedule_json="{}")

    monkeypatch.setattr(
        new_game_agent.NewGameAgent,
        "_create_player_schedule_data",
        fake_create_player_schedule,
    )

    async def fake_queue_npcs(self, user_id, conversation_id, target_count=5):
        assert conversation_id == created_conversation_id

    monkeypatch.setattr(
        new_game_agent.NewGameAgent,
        "_queue_npc_pool_fill",
        fake_queue_npcs,
    )

    async def fake_create_opening(self, ctx_wrap, params):
        return "Opening narrative"

    monkeypatch.setattr(
        new_game_agent.NewGameAgent,
        "create_opening_narrative",
        fake_create_opening,
    )

    async def fake_finalize(self, ctx_wrap, params):
        return new_game_agent.FinalizeResult(
            status="ok",
            welcome_image_url=None,
            lore_summary="Lore",
            initial_conflict="",
            currency_system="",
        )

    monkeypatch.setattr(
        new_game_agent.NewGameAgent,
        "finalize_game_setup",
        fake_finalize,
    )

    async def fake_is_setup_complete(self, user_id, conversation_id):
        return True, [], []

    monkeypatch.setattr(
        new_game_agent.NewGameAgent,
        "_is_setup_complete",
        fake_is_setup_complete,
    )

    class DummyWorldDirector:
        def __init__(self, user_id, conversation_id):
            self.user_id = user_id
            self.conversation_id = conversation_id

        async def initialize(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "story_agent.world_director_agent",
        types.SimpleNamespace(CompleteWorldDirector=DummyWorldDirector),
    )

    class FakeGovernance:
        def __init__(self):
            self.registered = []

        async def register_agent(self, agent_type, agent_instance, agent_id):
            self.registered.append((agent_type, agent_id))
            return {"success": True}

        async def get_agent_directives(self, agent_type, agent_id):
            return []

        async def check_action_permission(self, *args, **kwargs):
            return {"approved": True}

        async def process_agent_action_report(self, *args, **kwargs):
            return {"success": True}

    fake_governance = FakeGovernance()

    async def fake_get_central_governance(user_id, conversation_id):
        return fake_governance

    monkeypatch.setattr(
        nyx_integrate,
        "get_central_governance",
        fake_get_central_governance,
    )

    directive_handler_instance = {}

    class StubDirectiveHandler:
        def __init__(self, user_id, conversation_id, agent_type, agent_id, governance=None):
            directive_handler_instance["instance"] = self
            self.user_id = user_id
            self.conversation_id = conversation_id
            self.agent_type = agent_type
            self.agent_id = agent_id
            self.governance = governance
            self.handlers = {}
            self.start_calls = 0
            self.background_task = None

        def register_handler(self, directive_type, handler):
            self.handlers[directive_type] = handler

        async def process_directives(self, force_check=False):
            self.force_check = force_check
            return {"processed": 0}

        def start_background_processing(self, interval=60.0):
            self.start_calls += 1
            self.interval = interval
            self.background_task = asyncio.create_task(asyncio.sleep(3600))
            return self.background_task

        async def stop_background_processing(self):
            if self.background_task and not self.background_task.done():
                self.background_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.background_task
            self.background_task = None

    monkeypatch.setattr(new_game_agent, "DirectiveHandler", StubDirectiveHandler)

    agent = new_game_agent.NewGameAgent()
    ctx = CanonicalContext(user_id=7, conversation_id=0)

    result = await asyncio.wait_for(
        agent.process_new_game(ctx, {}),
        timeout=1.0,
    )

    assert isinstance(result, new_game_agent.ProcessNewGameResult)
    assert result.conversation_id == created_conversation_id

    handler = directive_handler_instance["instance"]
    assert handler.start_calls == 1
    assert agent._directive_task is handler.background_task
    assert not agent._directive_task.done()

    await agent.shutdown()

    # Shutdown should cancel the background task
    assert handler.background_task is None or handler.background_task.cancelled()
    assert agent._directive_task is None


def test_process_new_game_task_uses_environment_name(monkeypatch):
    import importlib
    import sys

    user_id = 11
    conversation_id = 57
    captured_names: list[str] = []

    fake_brain_base = types.ModuleType("nyx.core.brain.base")
    fake_brain_base.NyxBrain = object
    fake_checkpoint = types.ModuleType("nyx.core.brain.checkpointing_agent")
    fake_checkpoint.CheckpointingPlannerAgent = object
    fake_nyx_agent_sdk = types.ModuleType("nyx.nyx_agent_sdk")

    class _StubSDK:
        async def initialize_agent(self):
            return None

    fake_nyx_agent_sdk.NyxAgentSDK = _StubSDK
    fake_nyx_agent_sdk.NyxSDKConfig = object

    fake_nyx = types.ModuleType("nyx")
    fake_nyx_core = types.ModuleType("nyx.core")
    fake_nyx_core_brain = types.ModuleType("nyx.core.brain")
    fake_nyx.core = fake_nyx_core
    fake_nyx_core.brain = fake_nyx_core_brain
    fake_nyx_core_brain.base = fake_brain_base
    fake_nyx_core_brain.checkpointing_agent = fake_checkpoint

    monkeypatch.setitem(sys.modules, "nyx", fake_nyx)
    monkeypatch.setitem(sys.modules, "nyx.core", fake_nyx_core)
    monkeypatch.setitem(sys.modules, "nyx.core.brain", fake_nyx_core_brain)
    monkeypatch.setitem(sys.modules, "nyx.core.brain.base", fake_brain_base)
    monkeypatch.setitem(sys.modules, "nyx.core.brain.checkpointing_agent", fake_checkpoint)
    fake_nyx.nyx_agent_sdk = fake_nyx_agent_sdk

    sys.modules.pop("tasks", None)
    monkeypatch.setitem(sys.modules, "nyx.nyx_agent_sdk", fake_nyx_agent_sdk)

    tasks = importlib.import_module("tasks")

    @contextlib.contextmanager
    def noop_trace(**kwargs):
        yield

    monkeypatch.setattr(tasks, "trace", noop_trace)

    def immediate_run(coro):
        return asyncio.run(coro)

    monkeypatch.setattr(tasks, "run_async_in_worker_loop", immediate_run)

    class DummyConnection:
        async def execute(self, query, *args):
            if "UPDATE conversations" in query:
                captured_names.append(args[2])
            return None

        async def fetchval(self, query, *args):
            if "SELECT conversation_name" in query:
                return "Existing Name"
            if "SELECT 1 FROM messages" in query:
                return 0
            return None

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    monkeypatch.setattr(tasks, "get_db_connection_context", fake_db_context)

    class DummyAgent:
        async def process_new_game(self, ctx, payload):
            return types.SimpleNamespace(
                message="ok",
                scenario_name="Scenario",
                environment_name="Forest of Tests",
                environment_desc="",
                lore_summary="",
                conversation_id=conversation_id,
                welcome_image_url=None,
                status="ready",
                opening_narrative="Intro",
            )

        async def process_preset_game_direct(self, ctx, payload, preset_story_id):
            return await self.process_new_game(ctx, payload)

    monkeypatch.setattr(tasks, "NewGameAgent", lambda: DummyAgent())

    result = tasks.process_new_game_task(user_id, {"conversation_id": conversation_id})

    assert result["status"] == "ready"
    assert captured_names == ["Forest of Tests"]


def test_process_new_game_task_handles_image_timeout(monkeypatch):
    import importlib
    import time

    user_id = 21
    conversation_id = 404

    # Ensure Nyx dependencies resolve during task import
    fake_brain_base = types.ModuleType("nyx.core.brain.base")
    fake_brain_base.NyxBrain = object
    fake_checkpoint = types.ModuleType("nyx.core.brain.checkpointing_agent")
    fake_checkpoint.CheckpointingPlannerAgent = object
    fake_nyx_agent_sdk = types.ModuleType("nyx.nyx_agent_sdk")

    class _StubSDK:
        async def initialize_agent(self):
            return None

    fake_nyx_agent_sdk.NyxAgentSDK = _StubSDK
    fake_nyx_agent_sdk.NyxSDKConfig = object

    fake_nyx = types.ModuleType("nyx")
    fake_nyx_core = types.ModuleType("nyx.core")
    fake_nyx_core_brain = types.ModuleType("nyx.core.brain")
    fake_nyx.core = fake_nyx_core
    fake_nyx_core.brain = fake_nyx_core_brain
    fake_nyx_core_brain.base = fake_brain_base
    fake_nyx_core_brain.checkpointing_agent = fake_checkpoint

    fake_integrate = types.ModuleType("nyx.integrate")

    class FakeGovernance:
        async def register_agent(self, *args, **kwargs):
            return {"success": True}

        async def get_agent_directives(self, *args, **kwargs):
            return []

        async def check_action_permission(self, *args, **kwargs):
            return {"approved": True}

        async def process_agent_action_report(self, *args, **kwargs):
            return {"success": True}

    async def fake_get_central_governance(*_args, **_kwargs):
        return FakeGovernance()

    fake_integrate.get_central_governance = fake_get_central_governance

    monkeypatch.setitem(sys.modules, "nyx", fake_nyx)
    monkeypatch.setitem(sys.modules, "nyx.core", fake_nyx_core)
    monkeypatch.setitem(sys.modules, "nyx.core.brain", fake_nyx_core_brain)
    monkeypatch.setitem(sys.modules, "nyx.core.brain.base", fake_brain_base)
    monkeypatch.setitem(sys.modules, "nyx.core.brain.checkpointing_agent", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "nyx.integrate", fake_integrate)
    fake_nyx.integrate = fake_integrate
    fake_nyx.nyx_agent_sdk = fake_nyx_agent_sdk
    monkeypatch.setitem(sys.modules, "nyx.nyx_agent_sdk", fake_nyx_agent_sdk)

    sys.modules.pop("tasks", None)
    tasks = importlib.import_module("tasks")

    @contextlib.contextmanager
    def noop_trace(**kwargs):
        yield

    monkeypatch.setattr(tasks, "trace", noop_trace)
    monkeypatch.setattr(tasks, "run_async_in_worker_loop", asyncio.run)

    status_updates: list[tuple[str, tuple]] = []

    class DummyConnection:
        async def execute(self, query, *args):
            if "UPDATE conversations" in query:
                status_updates.append((query, args))
            return None

        async def fetchval(self, query, *args):
            if "SELECT conversation_name" in query:
                return "Existing Name"
            if "SELECT 1 FROM messages" in query:
                return 0
            return None

        async def fetchrow(self, query, *args):
            return None

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    monkeypatch.setattr(tasks, "get_db_connection_context", fake_db_context)
    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    # Patch new game agent helpers to avoid heavy dependencies
    async def fake_queue_lore(self, user_id_arg, conv_id_arg):
        assert conv_id_arg == conversation_id
        return "Generated lore"

    async def fake_queue_conflict(self, user_id_arg, conv_id_arg):
        assert conv_id_arg == conversation_id
        return "Conflict"

    async def fake_init_player_context(self, canon_ctx, user_id_arg, conv_id_arg):
        assert conv_id_arg == conversation_id

    monkeypatch.setattr(new_game_agent.NewGameAgent, "_queue_lore_generation", fake_queue_lore)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_queue_conflict_generation", fake_queue_conflict)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_initialize_player_context", fake_init_player_context)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "_image_gen_available", lambda self: True)

    from logic import currency_generator as currency_module

    class FakeCurrencyGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def get_currency_system(self):
            return {
                "currency_name": "Test Coin",
                "currency_plural": "Test Coins",
            }

    monkeypatch.setattr(currency_module, "CurrencyGenerator", FakeCurrencyGenerator)

    from lore.core import canon as canon_module

    async def fake_find_currency(*args, **kwargs):
        return None

    async def fake_update_roleplay(*args, **kwargs):
        return None

    monkeypatch.setattr(canon_module, "find_or_create_currency_system", fake_find_currency)
    monkeypatch.setattr(canon_module, "update_current_roleplay", fake_update_roleplay)

    import routes.ai_image_generator as image_module

    async def fake_process_scene_data(gpt_response, user_id_arg, conv_id_arg):
        return {
            "npcs": [
                {"id": 1, "name": "Nyx", "visual_seed": "seed"},
            ]
        }

    def fake_generate_prompt(scene_data):
        return {"image_prompt": "A dramatic scene", "negative_prompt": "blurry"}

    def fake_cached_images(prompt):
        return []

    def fake_save_image(url, prompt, variation_id):
        return f"cached_{variation_id}.png"

    async def fake_update_attrs(user_id_arg, conv_id_arg, npc_id, prompt_data, image_path=None):
        return {}, {}

    async def fake_track_evolution(*args, **kwargs):
        return None

    timeouts_seen: list[float | None] = []

    def fake_generate_ai_image(*args, **kwargs):
        timeouts_seen.append(kwargs.get("timeout"))
        time.sleep(0.01)
        return None

    monkeypatch.setattr(image_module, "process_gpt_scene_data", fake_process_scene_data)
    monkeypatch.setattr(image_module, "generate_image_prompt", fake_generate_prompt)
    monkeypatch.setattr(image_module, "get_cached_images", fake_cached_images)
    monkeypatch.setattr(image_module, "save_image_to_cache", fake_save_image)
    monkeypatch.setattr(image_module, "update_npc_visual_attributes", fake_update_attrs)
    monkeypatch.setattr(image_module, "track_visual_evolution", fake_track_evolution)
    monkeypatch.setattr(image_module, "generate_ai_image", fake_generate_ai_image)

    # Ensure new_game_agent uses the patched image generator
    monkeypatch.setattr(new_game_agent, "generate_roleplay_image_from_gpt", image_module.generate_roleplay_image_from_gpt)

    async def fake_process_new_game(self, ctx, payload):
        payload.setdefault("conversation_id", conversation_id)
        finalize_params = new_game_agent.FinalizeGameSetupParams(
            opening_narrative="Welcome to the arena."
        )
        ctx_wrap = new_game_agent._build_run_context_wrapper(ctx.user_id, conversation_id)
        finalize_result = await self.finalize_game_setup(ctx_wrap, finalize_params)
        return new_game_agent.ProcessNewGameResult(
            message="ok",
            scenario_name="Scenario",
            environment_name="Test Environment",
            environment_desc="",
            lore_summary=finalize_result.lore_summary,
            conversation_id=conversation_id,
            welcome_image_url=finalize_result.welcome_image_url,
            status="ready",
            opening_narrative=finalize_params.opening_narrative,
        )

    async def tracking_generate_roleplay_image(*args, **kwargs):
        return await image_module.generate_roleplay_image_from_gpt(*args, **kwargs)

    monkeypatch.setattr(new_game_agent, "generate_roleplay_image_from_gpt", tracking_generate_roleplay_image)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "process_new_game", fake_process_new_game)
    monkeypatch.setattr(new_game_agent.NewGameAgent, "process_preset_game_direct", fake_process_new_game)

    result = tasks.process_new_game_task(user_id, {"conversation_id": conversation_id})

    assert result["status"] == "ready"
    assert result["welcome_image_url"] is None
    assert timeouts_seen == [5.0]
    assert any("status='ready'" in query for query, _ in status_updates)


def test_initialize_player_context_aligns_with_schedule(monkeypatch):
    user_id = 99
    conversation_id = 123

    fixed_now = datetime(2024, 3, 18, 9, 30)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now
            return fixed_now.replace(tzinfo=tz)

    monkeypatch.setattr(new_game_agent, "datetime", FixedDateTime)

    schedule = {
        "Monday": {
            "Morning": "Chase prepares for the day at Town Square",
            "Afternoon": "Chase attends to responsibilities at Observation Park",
            "Evening": "Chase spends time on personal activities at Observation Park",
            "Night": "Chase returns home and rests",
        }
    }

    location_rows = [
        {"location_name": "Town Square"},
        {"location_name": "Observation Park"},
    ]

    class DummyConnection:
        async def fetch(self, query, *args):
            if "FROM Locations" in query:
                return location_rows
            return []

        async def fetchrow(self, query, *args):
            if "ChaseSchedule" in query:
                return {"value": json.dumps(schedule)}
            return None

        async def execute(self, query, *args):
            return None

    @asynccontextmanager
    async def fake_db_context():
        yield DummyConnection()

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    async def fake_load_calendar_names(unused_user_id, unused_conversation_id):
        return {"months": ["Month"], "days": ["Monday"], "seasons": []}

    monkeypatch.setattr(new_game_agent, "load_calendar_names", fake_load_calendar_names)

    recorded_time_args = {}

    async def fake_set_current_time(*args):
        recorded_time_args["args"] = args

    monkeypatch.setattr(new_game_agent, "set_current_time", fake_set_current_time)

    updates = []

    async def fake_update_current_roleplay(ctx, conn, key, value):
        updates.append((key, value))

    monkeypatch.setattr(new_game_agent.canon, "update_current_roleplay", fake_update_current_roleplay)

    async def runner():
        agent = new_game_agent.NewGameAgent()
        ctx_wrap = new_game_agent._build_run_context_wrapper(user_id, conversation_id)

        await agent._initialize_player_context(ctx_wrap, user_id, conversation_id)

    asyncio.run(runner())

    assert ("CurrentLocation", "Town Square") in updates
    expected_time = "Year 2024 Month Monday Morning"
    assert ("CurrentTime", expected_time) in updates
    assert recorded_time_args["args"] == (user_id, conversation_id, 2024, 1, 18, "Morning")

