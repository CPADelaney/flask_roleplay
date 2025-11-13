"""Ensure async route handlers await request.get_json."""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Any, Dict
import sys

import pytest
from quart import Quart

sys.path.append(str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

stub_sentence_transformers = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover - simple shim
        return None

    def encode(self, *_: Any, **__: Any) -> list[Any]:  # pragma: no cover - unused helper
        return []


stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer

sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("faiss.contrib", types.ModuleType("faiss.contrib"))

from routes.player_input import player_input_bp


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    parent_name, _, attr = name.rpartition(".")
    if parent_name:
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        setattr(parent, attr, module)
    return module


async def _async_noop(*_: Any, **__: Any) -> Any:
    return {}


def _sync_noop(*_: Any, **__: Any) -> Any:
    return {}


# Stub heavy logic modules to avoid cascading imports during tests
logic_module = _stub_module("logic")

activity_module = _stub_module("logic.activity_analyzer")


class _StubActivityAnalyzer:
    async def analyze_activity(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {}


activity_module.ActivityAnalyzer = _StubActivityAnalyzer

npc_progression_module = _stub_module("logic.npc_narrative_progression")
npc_progression_module.check_for_npc_revelation = _async_noop
npc_progression_module.get_npc_narrative_stage = _async_noop
npc_progression_module.progress_npc_narrative_stage = _async_noop
npc_progression_module.NPC_NARRATIVE_STAGES = []

narrative_events_module = _stub_module("logic.narrative_events")
narrative_events_module.get_relationship_overview = _async_noop
narrative_events_module.check_for_personal_revelations = _async_noop
narrative_events_module.check_for_narrative_moments = _async_noop
narrative_events_module.add_dream_sequence = _async_noop
narrative_events_module.add_moment_of_clarity = _async_noop
narrative_events_module.initialize_player_stats = _async_noop
narrative_events_module.analyze_narrative_tone = _async_noop

conflict_pkg = _stub_module("logic.conflict_system")
conflict_module = _stub_module("logic.conflict_system.conflict_synthesizer")


class _StubSystemEvent:  # pragma: no cover - sentinel placeholder
    pass


class _StubEventType:  # pragma: no cover - sentinel placeholder
    pass


class _StubSubsystemType:  # pragma: no cover - sentinel placeholder
    pass


async def _stub_get_synthesizer(*_: Any, **__: Any) -> Any:
    class _Synth:
        async def get_system_state(self) -> Dict[str, Any]:
            return {}

        async def get_conflict_state(self, *_: Any, **__: Any) -> Dict[str, Any]:
            return {}

    return _Synth()


conflict_module.get_synthesizer = _stub_get_synthesizer
conflict_module.SystemEvent = _StubSystemEvent
conflict_module.EventType = _StubEventType
conflict_module.SubsystemType = _StubSubsystemType

db_helpers_module = _stub_module("utils.db_helpers")


def _decorator_passthrough(func):
    return func


db_helpers_module.db_transaction = _decorator_passthrough
db_helpers_module.with_transaction = _decorator_passthrough
db_helpers_module.handle_database_operation = _decorator_passthrough
db_helpers_module.fetch_row_async = _async_noop
db_helpers_module.fetch_all_async = _async_noop
db_helpers_module.execute_async = _async_noop

performance_module = _stub_module("utils.performance")


class _StubPerformanceTracker:
    def __init__(self, *_: Any, **__: Any) -> None:
        return None

    def start_phase(self, *_: Any, **__: Any) -> None:
        return None

    def end_phase(self) -> None:
        return None


def _timed_function(func=None, *, name: str | None = None):  # pragma: no cover - passthrough decorator
    if func is None:
        return lambda f: f
    return func


class _StubStats:
    def record_request(self, *_: Any, **__: Any) -> None:
        return None


performance_module.PerformanceTracker = _StubPerformanceTracker
performance_module.timed_function = _timed_function
performance_module.STATS = _StubStats()

caching_module = _stub_module("utils.caching")


class _StubCacheManager:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, *_: Any, **__: Any) -> Any:
        return None

    def set(self, *_: Any, **__: Any) -> None:
        return None

    def remove(self, *_: Any, **__: Any) -> None:
        return None

    def remove_pattern(self, *_: Any, **__: Any) -> None:  # pragma: no cover - unused here
        return None


caching_module.NPC_CACHE = _StubCacheManager()
caching_module.LOCATION_CACHE = _StubCacheManager()
caching_module.AGGREGATOR_CACHE = _StubCacheManager()
caching_module.TIME_CACHE = _StubCacheManager()
caching_module.COMPUTATION_CACHE = _StubCacheManager()


class _StubCacheDecorator:
    def cached(self, *_: Any, **__: Any):
        def decorator(func):
            return func

        return decorator


caching_module.cache = _StubCacheDecorator()

conversation_history_module = _stub_module("utils.conversation_history")
conversation_history_module.fetch_recent_turns = _async_noop

conversation_store_module = _stub_module("nyx.conversation.store")


class _StubConversationStore:
    def __init__(self, *_: Any, **__: Any) -> None:
        return None

    async def fetch_recent_turns(self, *_: Any, **__: Any) -> list[Dict[str, Any]]:
        return []

    async def append_turn(self, *_: Any, **__: Any) -> None:
        return None


conversation_store_module.ConversationStore = _StubConversationStore

universal_updater_module = _stub_module("logic.universal_updater_agent")
universal_updater_module.apply_universal_updates = _async_noop

aggregator_module = _stub_module("logic.aggregator_sdk")
aggregator_module.get_aggregated_roleplay_context = _async_noop

time_cycle_module = _stub_module("logic.time_cycle")
time_cycle_module.get_current_time = _async_noop
time_cycle_module.should_advance_time = _async_noop
time_cycle_module.nightly_maintenance = _async_noop

inventory_module = _stub_module("logic.inventory_system_sdk")


class _StubInventoryContext:
    async def __aenter__(self) -> "_StubInventoryContext":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


inventory_module.InventoryContext = _StubInventoryContext

chatgpt_module = _stub_module("logic.chatgpt_integration")
chatgpt_module.get_chatgpt_response = _async_noop
chatgpt_module.get_openai_client = _sync_noop
chatgpt_module.build_message_history = _sync_noop

resource_module = _stub_module("logic.resource_management")


class _ResourceManagerPlaceholder:
    pass


resource_module.ResourceManager = _ResourceManagerPlaceholder

settings_routes_module = _stub_module("routes.settings_routes")
settings_routes_module.generate_mega_setting_logic = _async_noop

gpt_image_module = _stub_module("logic.gpt_image_decision")
gpt_image_module.should_generate_image_for_response = _async_noop

ai_image_module = _stub_module("routes.ai_image_generator")
ai_image_module.generate_roleplay_image_from_gpt = _async_noop

lore_core_module = _stub_module("lore.core")
lore_system_module = _stub_module("lore.core.lore_system")


class _StubLoreSystem:
    pass


lore_system_module.LoreSystem = _StubLoreSystem

integrated_npc_module = _stub_module("logic.fully_integrated_npc_system")


class _StubIntegratedNPCSystem:
    def __init__(self, *_: Any, **__: Any) -> None:
        return None


integrated_npc_module.IntegratedNPCSystem = _StubIntegratedNPCSystem

npcs_module = _stub_module("npcs")
npcs_creation_module = _stub_module("npcs.new_npc_creation")
npcs_creation_module.NPCCreationHandler = type("NPCCreationHandler", (), {})
npcs_creation_module.RunContextWrapper = type("RunContextWrapper", (), {})

stats_module = _stub_module("logic.stats_logic")
stats_module.get_player_current_tier = _async_noop
stats_module.check_for_combination_triggers = _async_noop
stats_module.apply_stat_change = _async_noop
stats_module.apply_activity_effects = _async_noop

social_links_module = _stub_module("logic.social_links")
social_links_module.get_relationship_dynamic_level = _async_noop
social_links_module.update_relationship_dynamic = _async_noop
social_links_module.check_for_relationship_crossroads = _async_noop
social_links_module.check_for_relationship_ritual = _async_noop
social_links_module.get_relationship_summary = _async_noop
social_links_module.apply_crossroads_choice = _async_noop

addiction_module = _stub_module("logic.addiction_system_sdk")
addiction_module.check_addiction_levels = _async_noop
addiction_module.update_addiction_level = _async_noop
addiction_module.process_addiction_effects = _async_noop
addiction_module.get_addiction_status = _async_noop
addiction_module.get_addiction_label = _async_noop

nyx_module = _stub_module("nyx")
nyx_agent_sdk_module = _stub_module("nyx.nyx_agent_sdk")
nyx_agent_sdk_module.process_user_input = _async_noop

lore_module = _stub_module("lore")
lore_generator_module = _stub_module("lore.lore_generator")
lore_generator_module.DynamicLoreGenerator = type("DynamicLoreGenerator", (), {})

from routes.story_routes import story_bp


class _StubCursor:
    async def execute(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - behaviourless stub
        return None

    async def fetchall(self) -> list[Any]:
        return []

    async def __aenter__(self) -> "_StubCursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _StubConnection:
    async def commit(self) -> None:
        return None

    def cursor(self) -> _StubCursor:
        return _StubCursor()

    async def __aenter__(self) -> "_StubConnection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _StubDBContext:
    async def __aenter__(self) -> _StubConnection:
        return _StubConnection()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _StubCache:
    def get(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - behaviourless stub
        return None

    def remove(self, *args: Any, **kwargs: Any) -> None:
        return None

    def remove_pattern(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - unused here
        return None


class _StubResourceManager:
    def __init__(self, *_: Any, **__: Any) -> None:
        return None

    async def modify_money(self, amount: int, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"resource": "money", "amount": amount}

    async def modify_supplies(self, amount: int, *_: Any, **__: Any) -> Dict[str, Any]:  # pragma: no cover - unused fallback
        return {"resource": "supplies", "amount": amount}

    async def modify_influence(self, amount: int, *_: Any, **__: Any) -> Dict[str, Any]:  # pragma: no cover
        return {"resource": "influence", "amount": amount}

    async def modify_hunger(self, amount: int, *_: Any, **__: Any) -> Dict[str, Any]:  # pragma: no cover
        return {"resource": "hunger", "amount": amount}

    async def modify_energy(self, amount: int, *_: Any, **__: Any) -> Dict[str, Any]:  # pragma: no cover
        return {"resource": "energy", "amount": amount}


@pytest.mark.asyncio
async def test_player_input_routes_parse_json(monkeypatch: pytest.MonkeyPatch) -> None:
    app = Quart(__name__)
    app.secret_key = "test-key"
    app.register_blueprint(player_input_bp)

    monkeypatch.setattr("routes.player_input.get_db_connection_context", lambda: _StubDBContext())

    test_client = app.test_client()

    async with test_client.session_transaction() as session:
        session["user_id"] = "user-123"

    start_response = await test_client.post(
        "/start_chat",
        json={"user_input": "Hello", "conversation_id": 1, "universal_update": {}},
    )
    assert start_response.status_code == 200
    start_payload = await start_response.get_json()
    assert start_payload["status"] == "success"

    player_response = await test_client.post("/player_input", json={"text": "Testing"})
    assert player_response.status_code == 200
    player_payload = await player_response.get_json()
    assert player_payload["original_text"] == "Testing"


@pytest.mark.asyncio
async def test_story_routes_parse_json(monkeypatch: pytest.MonkeyPatch) -> None:
    app = Quart(__name__)
    app.secret_key = "test-key"
    app.register_blueprint(story_bp)

    monkeypatch.setattr("routes.story_routes.ResourceManager", _StubResourceManager)
    monkeypatch.setattr("routes.story_routes.NPC_CACHE", _StubCache())

    test_client = app.test_client()

    async with test_client.session_transaction() as session:
        session["user_id"] = "user-123"

    response = await test_client.post(
        "/player/resources/modify",
        json={"conversation_id": 1, "resource_type": "money", "amount": 7},
    )

    assert response.status_code == 200
    payload = await response.get_json()
    assert payload == {"resource": "money", "amount": 7}
