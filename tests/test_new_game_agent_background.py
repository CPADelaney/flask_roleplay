import asyncio
import contextlib
import os
import pathlib
import sys
import types
from contextlib import asynccontextmanager

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
