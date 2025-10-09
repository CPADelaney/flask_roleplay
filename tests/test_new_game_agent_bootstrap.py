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


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_process_new_game_bootstrap_seeds_governance_state(monkeypatch):
    created_conversation_id = 1337
    final_location = "Final Plaza"
    final_time = "Year 1 MonthOne Monday Morning"

    state = {
        "current_roleplay": {},
        "set_time": None,
        "updates": [],
        "provisional_snapshot": None,
    }
    warnings: list[str] = []

    async def failing_log(*args, **kwargs):
        raise RuntimeError("canonical logging offline")

    monkeypatch.setattr(new_game_agent.canon, "log_canonical_event", failing_log)

    class DummyConnection:
        def __init__(self):
            self._state = state

        async def fetchrow(self, query, *args):
            if "INSERT INTO conversations" in query:
                return {"id": created_conversation_id}
            if "SELECT id FROM conversations" in query:
                return {"id": created_conversation_id}
            if "SELECT value FROM CurrentRoleplay" in query:
                key = query.split("key='")[-1].split("'")[0]
                value = self._state["current_roleplay"].get(key)
                return {"value": value} if value is not None else None
            return None

        async def execute(self, query, *args):
            if "INSERT INTO CurrentRoleplay" in query and len(args) >= 4:
                key = args[2]
                value = args[3]
                self._state["current_roleplay"][key] = value
                self._state["updates"].append((key, value))
            if query.strip().upper().startswith("DELETE FROM CURRENTROLEPLAY"):
                self._state["current_roleplay"].clear()
            return None

        async def fetch(self, query, *args):
            if "FROM Locations" in query:
                return [{"location_name": "Bootstrap Square"}]
            return []

        async def fetchval(self, query, *args):
            if "SELECT COUNT(*) FROM messages" in query:
                return 0
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

    async def fake_set_current_time(user_id, conversation_id, year, month_idx, day_num, phase):
        state["set_time"] = {
            "year": year,
            "month": month_idx,
            "day": day_num,
            "phase": phase,
        }

    monkeypatch.setattr(new_game_agent, "set_current_time", fake_set_current_time)

    async def fake_load_calendar_names(user_id, conversation_id):
        return {
            "months": ["MonthOne"],
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        }

    monkeypatch.setattr(new_game_agent, "load_calendar_names", fake_load_calendar_names)

    async def fake_generate_mega_setting_logic():
        return {
            "selected_settings": ["Test Locale"],
            "enhanced_features": [],
            "stat_modifiers": {},
            "mega_name": "Bootstrap Mega",
            "mega_description": "Bootstrap description",
        }

    monkeypatch.setattr(settings_routes, "generate_mega_setting_logic", fake_generate_mega_setting_logic)

    async def fake_apply_modifiers(self, user_id, conversation_id, modifiers):
        return None

    monkeypatch.setattr(new_game_agent.NewGameAgent, "_apply_setting_stat_modifiers", fake_apply_modifiers)

    async def fake_generate_environment(self, ctx_wrap, params):
        return new_game_agent.EnvironmentData(
            setting_name="Bootstrap Town",
            environment_desc="Bootstrap environment",
            environment_history="Bootstrap history",
            scenario_name="Bootstrap Scenario",
            events=[],
            locations=[],
            quest_data=new_game_agent.QuestData(),
        )

    monkeypatch.setattr(new_game_agent.NewGameAgent, "generate_environment", fake_generate_environment)

    async def fake_create_player_schedule(self, ctx_wrap, desc):
        return new_game_agent.NPCScheduleData(npc_ids=[], chase_schedule_json="{}")

    monkeypatch.setattr(new_game_agent.NewGameAgent, "_create_player_schedule_data", fake_create_player_schedule)

    class DummyNPCreator:
        async def spawn_multiple_npcs(self, ctx, count=5):
            return []

    monkeypatch.setattr(new_game_agent, "NPCCreationHandler", lambda: DummyNPCreator())
    monkeypatch.setattr(
        sys.modules.setdefault("npcs.new_npc_creation", types.SimpleNamespace()),
        "NPCCreationHandler",
        lambda: DummyNPCreator(),
    )

    async def fake_create_opening(self, ctx_wrap, params):
        assert state["current_roleplay"].get("CurrentLocation")
        assert state["current_roleplay"].get("CurrentTime")
        return "Bootstrap opening"

    monkeypatch.setattr(new_game_agent.NewGameAgent, "create_opening_narrative", fake_create_opening)

    async def fake_finalize(self, ctx_wrap, params):
        async with new_game_agent.get_db_connection_context() as conn:
            await new_game_agent.canon.update_current_roleplay(
                ctx_wrap, conn, "CurrentLocation", final_location
            )
        async with new_game_agent.get_db_connection_context() as conn:
            await new_game_agent.canon.update_current_roleplay(
                ctx_wrap, conn, "CurrentTime", final_time
            )
        return new_game_agent.FinalizeResult(
            status="ok",
            welcome_image_url=None,
            lore_summary="Bootstrap lore",
            initial_conflict="",
            currency_system="",
        )

    monkeypatch.setattr(new_game_agent.NewGameAgent, "finalize_game_setup", fake_finalize)

    class DummyDirectiveHandler:
        async def process_directives(self, force_check=False):
            return {"processed": force_check}

        def start_background_processing(self):
            return None

    async def fake_initialize_directive_handler(self, user_id, conversation_id):
        self.directive_handler = DummyDirectiveHandler()

    monkeypatch.setattr(new_game_agent.NewGameAgent, "initialize_directive_handler", fake_initialize_directive_handler)

    class FakeGovernance:
        def __init__(self):
            self.calls: list[bool] = []
            self.last_snapshot: dict[str, str] | None = None

        async def register_agent(self, agent_type, agent_instance, agent_id):
            if "CurrentLocation" not in state["current_roleplay"]:
                warnings.append("missing location before register")
            if "CurrentTime" not in state["current_roleplay"]:
                warnings.append("missing time before register")
            return {"success": True}

        async def get_agent_directives(self, *args, **kwargs):
            return []

        async def check_action_permission(self, *args, **kwargs):
            return {"approved": True}

        async def process_agent_action_report(self, *args, **kwargs):
            return {"success": True}

        async def initialize_game_state(self, force: bool = False):
            snapshot = {
                "CurrentLocation": state["current_roleplay"].get("CurrentLocation"),
                "CurrentTime": state["current_roleplay"].get("CurrentTime"),
            }
            if not snapshot["CurrentLocation"] or not snapshot["CurrentTime"]:
                warnings.append("missing snapshot during initialize")
            self.calls.append(force)
            self.last_snapshot = snapshot
            return snapshot

    fake_governance = FakeGovernance()
    governance_call_count = {"count": 0}

    async def fake_get_central_governance(user_id, conversation_id):
        governance_call_count["count"] += 1
        if governance_call_count["count"] > 1:
            state["provisional_snapshot"] = dict(state["current_roleplay"])
            if "CurrentLocation" not in state["current_roleplay"]:
                warnings.append("missing location before governance bootstrap")
            if "CurrentTime" not in state["current_roleplay"]:
                warnings.append("missing time before governance bootstrap")
        return fake_governance

    monkeypatch.setattr(nyx_integrate, "get_central_governance", fake_get_central_governance)

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

    agent = new_game_agent.NewGameAgent()
    ctx = CanonicalContext(user_id=99, conversation_id=0)

    result = await agent.process_new_game(ctx, {})

    assert result.conversation_id == created_conversation_id
    assert warnings == []
    assert state["provisional_snapshot"] is not None
    assert fake_governance.calls == [True]
    assert fake_governance.last_snapshot == {
        "CurrentLocation": final_location,
        "CurrentTime": final_time,
    }
    assert state["current_roleplay"]["CurrentLocation"] == final_location
    assert state["current_roleplay"]["CurrentTime"] == final_time


@pytest.mark.anyio("asyncio")
async def test_initialize_player_context_persists_on_canon_log_failure(monkeypatch):
    user_id = 7
    conversation_id = 21
    ctx = CanonicalContext(user_id=user_id, conversation_id=conversation_id)

    state = {
        "current_roleplay": {},
        "set_time_calls": [],
    }

    async def failing_log(*args, **kwargs):
        raise RuntimeError("memory system offline")

    monkeypatch.setattr(new_game_agent.canon, "log_canonical_event", failing_log)

    class DummyConnection:
        def __init__(self):
            self._state = state

        async def fetch(self, query, *args):
            if "FROM Locations" in query:
                return [{"location_name": "Starter Plaza"}]
            return []

        async def fetchrow(self, query, *args):
            if "ChaseSchedule" in query:
                return None
            return None

        async def execute(self, query, *args):
            if "INSERT INTO CurrentRoleplay" in query and len(args) >= 4:
                key = args[2]
                value = args[3]
                self._state["current_roleplay"][key] = value
            return None

        async def fetchval(self, query, *args):
            return None

    connection = DummyConnection()

    @asynccontextmanager
    async def fake_db_context():
        yield connection

    monkeypatch.setattr(new_game_agent, "get_db_connection_context", fake_db_context)

    async def fake_set_current_time(user, convo, year, month_idx, day_num, phase):
        state["set_time_calls"].append((year, month_idx, day_num, phase))

    monkeypatch.setattr(new_game_agent, "set_current_time", fake_set_current_time)

    async def fake_load_calendar_names(user, convo):
        return {"months": ["MonthOne"], "days": ["Monday"]}

    monkeypatch.setattr(new_game_agent, "load_calendar_names", fake_load_calendar_names)

    agent = new_game_agent.NewGameAgent()

    await agent._initialize_player_context(ctx, user_id, conversation_id)

    assert state["current_roleplay"].get("CurrentLocation") == "Starter Plaza"
    assert "CurrentTime" in state["current_roleplay"]
    assert state["current_roleplay"]["CurrentTime"].startswith("Year")
    assert state["set_time_calls"]

