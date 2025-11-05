import enum
import json
import types
from pathlib import Path
from contextlib import asynccontextmanager
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def stub_world_director_dependencies():
    import sys
    import types as types_module

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    modules: dict[str, types_module.ModuleType] = {}

    def create_module(name: str) -> types_module.ModuleType:
        if name in modules:
            return modules[name]
        module = types_module.ModuleType(name)
        modules[name] = module
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent_module = create_module(parent_name)
            setattr(parent_module, attr_name, module)
        return module

    async def async_noop(*args, **kwargs):  # pragma: no cover - helper
        return None

    def noop(*args, **kwargs):  # pragma: no cover - helper
        return None

    # agents module
    agents_mod = create_module("agents")

    class _RunContextWrapper:
        def __init__(self, context):
            self.context = context

    def _function_tool_stub(*d_args, **d_kwargs):
        if d_args and callable(d_args[0]) and not d_kwargs:
            return d_args[0]

        def _inner(fn):
            return fn

        return _inner

    agents_mod.Agent = type("Agent", (), {})
    agents_mod.function_tool = _function_tool_stub
    agents_mod.Runner = type("Runner", (), {})
    agents_mod.trace = _function_tool_stub
    agents_mod.ModelSettings = type("ModelSettings", (), {})
    agents_mod.RunContextWrapper = _RunContextWrapper

    # db connection stub
    db_conn_mod = create_module("db.connection")

    @asynccontextmanager
    async def dummy_db_connection():
        yield None

    db_conn_mod.get_db_connection_context = dummy_db_connection

    # logic submodules with minimal attributes
    logic_universal = create_module("logic.universal_updater_agent")
    logic_universal.UniversalUpdaterAgent = type("UniversalUpdaterAgent", (), {})
    logic_universal.process_universal_update = async_noop

    logic_memory = create_module("logic.memory_logic")
    logic_memory.MemoryManager = type("MemoryManager", (), {})
    logic_memory.EnhancedMemory = type("EnhancedMemory", (), {})
    logic_memory.MemoryType = type("MemoryType", (), {})
    logic_memory.MemorySignificance = type("MemorySignificance", (), {})
    logic_memory.get_shared_memory = async_noop
    logic_memory.propagate_shared_memories = async_noop
    logic_memory.fetch_formatted_locations = async_noop

    memory_masks = create_module("memory.masks")
    memory_masks.ProgressiveRevealManager = type("ProgressiveRevealManager", (), {})
    memory_masks.RevealType = type("RevealType", (), {})
    memory_masks.RevealSeverity = type("RevealSeverity", (), {})
    memory_masks.NPCMask = type("NPCMask", (), {})

    logic_stats = create_module("logic.stats_logic")
    logic_stats.get_player_visible_stats = async_noop
    logic_stats.get_player_hidden_stats = async_noop
    logic_stats.get_all_player_stats = async_noop
    logic_stats.apply_stat_changes = async_noop
    logic_stats.check_for_combination_triggers = async_noop
    logic_stats.apply_activity_effects = async_noop
    logic_stats.STAT_THRESHOLDS = {}
    logic_stats.STAT_COMBINATIONS = {}
    logic_stats.ACTIVITY_EFFECTS = {}
    logic_stats.detect_deception = async_noop
    logic_stats.calculate_social_insight = async_noop
    logic_stats.update_hunger_from_time = async_noop
    logic_stats.consume_food = async_noop
    logic_stats.apply_damage = async_noop
    logic_stats.heal_player = async_noop

    logic_rules = create_module("logic.rule_enforcement")
    logic_rules.enforce_all_rules_on_player = async_noop
    logic_rules.evaluate_condition = noop
    logic_rules.parse_condition = noop
    logic_rules.apply_effect = async_noop
    logic_rules.get_player_stats = async_noop
    logic_rules.get_npc_stats = async_noop

    logic_inventory = create_module("logic.inventory_system_sdk")
    logic_inventory.get_inventory = async_noop
    logic_inventory.add_item = async_noop
    logic_inventory.remove_item = async_noop
    logic_inventory.InventoryContext = type("InventoryContext", (), {})
    logic_inventory.register_with_governance = async_noop

    logic_time = create_module("logic.time_cycle")
    logic_time.get_current_time_model = async_noop
    logic_time.advance_time_with_events = async_noop
    logic_time.get_current_vitals = async_noop
    logic_time.process_activity_vitals = async_noop
    logic_time.ActivityManager = type("ActivityManager", (), {})
    logic_time.ActivityType = type("ActivityType", (), {})

    logic_calendar = create_module("logic.calendar")
    logic_calendar.load_calendar_names = async_noop
    logic_calendar.update_calendar_names = async_noop
    logic_calendar.add_calendar_event = async_noop

    logic_relationships = create_module("logic.dynamic_relationships")
    logic_relationships.OptimizedRelationshipManager = type("OptimizedRelationshipManager", (), {})
    logic_relationships.drain_relationship_events_tool = async_noop
    logic_relationships.get_relationship_summary_tool = async_noop
    logic_relationships.process_relationship_interaction_tool = async_noop
    logic_relationships.poll_relationship_events_tool = async_noop
    logic_relationships.RelationshipState = type("RelationshipState", (), {})
    logic_relationships.RelationshipDimensions = type("RelationshipDimensions", (), {})
    logic_relationships.RelationshipArchetypes = type("RelationshipArchetypes", (), {})
    logic_relationships.RelationshipPatternDetector = type("RelationshipPatternDetector", (), {})

    logic_narrative = create_module("logic.narrative_events")
    logic_narrative.check_for_personal_revelations = async_noop
    logic_narrative.check_for_narrative_moments = async_noop
    logic_narrative.add_dream_sequence = async_noop
    logic_narrative.add_moment_of_clarity = async_noop
    logic_narrative.get_relationship_overview = async_noop
    logic_narrative.generate_inner_monologue = async_noop

    logic_npc = create_module("logic.npc_narrative_progression")
    logic_npc.get_npc_narrative_stage = async_noop
    logic_npc.check_for_npc_revelation = async_noop
    logic_npc.progress_npc_narrative_stage = async_noop
    logic_npc.NPC_NARRATIVE_STAGES = {}
    logic_npc.NPCNarrativeStage = type("NPCNarrativeStage", (), {})

    logic_addiction = create_module("logic.addiction_system_sdk")
    logic_addiction.AddictionContext = type("AddictionContext", (), {})
    logic_addiction.addiction_system_agent = async_noop
    logic_addiction.process_addiction_update = async_noop
    logic_addiction.check_addiction_status = async_noop
    logic_addiction.get_addiction_status = async_noop

    logic_currency = create_module("logic.currency_generator")
    logic_currency.CurrencyGenerator = type("CurrencyGenerator", (), {})

    logic_event_system = create_module("logic.event_system")
    logic_event_system.EventSystem = type("EventSystem", (), {})

    # context modules
    context_service = create_module("context.context_service")
    context_service.get_context_service = async_noop

    context_memory = create_module("context.memory_manager")
    context_memory.get_memory_manager = async_noop
    context_memory.MemoryAddRequest = type("MemoryAddRequest", (), {})

    context_models = create_module("context.models")
    context_models.MemoryMetadata = type("MemoryMetadata", (), {})

    context_vector = create_module("context.vector_service")
    context_vector.get_vector_service = async_noop

    context_perf = create_module("context.context_performance")
    context_perf.PerformanceMonitor = type("PerformanceMonitor", (), {})

    def _track_performance_stub(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    context_perf.track_performance = _track_performance_stub

    # nyx modules
    nyx_governance = create_module("nyx.nyx_governance")
    nyx_governance.NyxUnifiedGovernor = type("NyxUnifiedGovernor", (), {})
    nyx_governance.AgentType = type("AgentType", (), {})
    nyx_governance.DirectiveType = type("DirectiveType", (), {})
    nyx_governance.DirectivePriority = type("DirectivePriority", (), {})

    nyx_governance_ids = create_module("nyx.governance.ids")
    nyx_governance_ids.format_agent_id = lambda *args, **kwargs: "agent"

    nyx_gateway = create_module("nyx.gateway")

    class _LLMRequest:  # pragma: no cover - helper
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _LLMOperation(enum.Enum):  # pragma: no cover - helper
        ORCHESTRATION = "orchestration"

    async def _execute(_request, **_kwargs):  # pragma: no cover - helper
        return types.SimpleNamespace(raw=None, text="")

    nyx_gateway.llm_gateway = types.SimpleNamespace(
        LLMRequest=_LLMRequest,
        LLMOperation=_LLMOperation,
        execute=_execute,
    )

    nyx_integrate = create_module("nyx.integrate")
    nyx_integrate.get_central_governance = async_noop

    nyx_directive = create_module("nyx.directive_handler")
    nyx_directive.DirectiveHandler = type("DirectiveHandler", (), {})

    logic_conflict_pkg = create_module("logic.conflict_system")
    logic_conflict = create_module("logic.conflict_system.conflict_synthesizer")
    logic_conflict.get_synthesizer = async_noop
    logic_conflict.ConflictSynthesizer = type("ConflictSynthesizer", (), {})
    setattr(logic_conflict_pkg, "conflict_synthesizer", logic_conflict)

    # lore modules
    lore_core_pkg = create_module("lore.core")
    lore_canon = create_module("lore.core.canon")

    def ensure_canonical_context(data):
        return data

    lore_canon.find_or_create_npc = async_noop
    lore_canon.find_or_create_location = async_noop
    lore_canon.find_or_create_event = async_noop
    lore_canon.find_or_create_faction = async_noop
    lore_canon.find_or_create_historical_event = async_noop
    lore_canon.find_or_create_notable_figure = async_noop
    lore_canon.log_canonical_event = async_noop
    lore_canon.ensure_canonical_context = ensure_canonical_context
    lore_canon.update_entity_with_governance = async_noop
    lore_canon.get_entity_by_id = async_noop
    lore_canon.find_entity_by_name = async_noop
    lore_canon.create_message = async_noop
    lore_canon.update_current_roleplay = async_noop

    lore_context = create_module("lore.core.context")
    lore_context.CanonicalContext = type("CanonicalContext", (), {})

    # chatgpt integration stub
    logic_chatgpt = create_module("logic.chatgpt_integration")

    class OpenAIClientManager:  # pragma: no cover - helper
        pass

    async def generate_embedding(text: str):
        return [float(len(text))]

    def cosine_similarity(left, right):
        return 0.95 if left == right else 0.5

    async def generate_text_completion(*args, **kwargs):
        return "analysis"

    async def generate_reflection(*args, **kwargs):
        return {}

    async def analyze_preferences(*args, **kwargs):
        return {}

    async def create_semantic_abstraction(*args, **kwargs):
        return {}

    logic_chatgpt.OpenAIClientManager = OpenAIClientManager
    logic_chatgpt.get_chatgpt_response = generate_text_completion
    logic_chatgpt.generate_text_completion = generate_text_completion
    logic_chatgpt.generate_embedding = generate_embedding
    logic_chatgpt.get_text_embedding = generate_embedding
    logic_chatgpt.generate_reflection = generate_reflection
    logic_chatgpt.analyze_preferences = analyze_preferences
    logic_chatgpt.create_semantic_abstraction = create_semantic_abstraction
    logic_chatgpt.cosine_similarity = cosine_similarity

    # rag backend stub
    rag_module = create_module("rag")
    rag_backend_module = create_module("rag.backend")

    class _BackendPreference(enum.Enum):  # pragma: no cover - helper
        AUTO = "auto"
        AGENTS = "agents"
        LEGACY = "legacy"

    async def fake_rag_ask(prompt, **kwargs):
        metadata = kwargs.get("metadata") or {}
        if kwargs.get("mode") == "embedding":
            return {"embedding": [float(len(prompt))], "provider": "test", "metadata": metadata}

        legacy = kwargs.get("legacy_fallback")
        documents = []
        if callable(legacy):
            documents = await legacy()
        return {"documents": documents, "provider": "legacy", "metadata": metadata}

    rag_backend_module.BackendPreference = _BackendPreference
    rag_backend_module.ask = fake_rag_ask
    rag_backend_module.get_configured_backend = lambda: _BackendPreference.AUTO

    rag_module.ask = fake_rag_ask
    rag_module.BackendPreference = _BackendPreference
    rag_module.get_configured_backend = rag_backend_module.get_configured_backend

    with mock.patch.dict(sys.modules, modules):
        yield


@pytest.mark.asyncio
async def test_check_all_emergent_patterns_normalizes_keyvalues():
    import importlib
    from story_agent.world_simulation_models import CompleteWorldState, KeyValue

    world_director_agent = importlib.import_module("story_agent.world_director_agent")

    memory_values = [
        "Shared memory insight",
        "Shared memory insight",
        "An unrelated observation",
        "Another log entry",
        "Different topic",
        "Closing remark",
    ]

    recent_memories = [
        KeyValue(key=str(idx), value=json.dumps({"text": text}))
        for idx, text in enumerate(memory_values)
    ]

    active_npcs = [
        KeyValue(
            key="0",
            value=json.dumps(
                {
                    "npc_name": "Astra",
                    "relationship": {
                        "patterns": ["protective guidance", "structured discipline"],
                        "archetype": "mentor",
                    },
                }
            ),
        )
    ]

    addiction_status = [
        KeyValue(key="has_addictions", value=True),
        KeyValue(
            key="addictions",
            value=json.dumps(
                {
                    "sweets": {
                        "level": 4,
                        "recent_increases": 3,
                    }
                }
            ),
        ),
    ]

    stat_combinations = [
        KeyValue(
            key="0",
            value=json.dumps(
                {
                    "name": "Focus & Poise",
                    "behaviors": ["calm execution"],
                }
            ),
        )
    ]

    triggered_effects = [
        KeyValue(
            key="0",
            value=json.dumps({"rule": {"rule_name": "RespectCurfew"}}),
        ),
        KeyValue(
            key="1",
            value=json.dumps({"rule_name": "RespectCurfew"}),
        ),
        KeyValue(
            key="2",
            value=json.dumps({"rule": {"rule_name": "SelfCare"}}),
        ),
    ]

    world_state = CompleteWorldState(
        recent_memories=recent_memories,
        active_npcs=active_npcs,
        addiction_status=addiction_status,
        active_stat_combinations=stat_combinations,
        triggered_effects=triggered_effects,
    )

    director_context = world_director_agent.CompleteWorldDirectorContext(
        user_id=1, conversation_id=1
    )
    director_context.current_world_state = world_state

    ctx_wrapper = types.SimpleNamespace(context=director_context)

    result = await world_director_agent.check_all_emergent_patterns(ctx_wrapper)

    assert any(pattern.similarity > 0.8 for pattern in result.memory_patterns)
    assert result.relationship_patterns
    assert result.relationship_patterns[0].npc == "Astra"
    assert result.relationship_patterns[0].patterns == [
        "protective guidance",
        "structured discipline",
    ]
    assert result.addiction_patterns and result.addiction_patterns[0].type == "sweets"
    assert result.stat_patterns and result.stat_patterns[0].combination == "Focus & Poise"
    assert any(p.rule == "RespectCurfew" and p.frequency == 2 for p in result.rule_patterns)
    assert result.narrative_analysis == "analysis"


def test_warm_initialize_restores_currency_generator(monkeypatch):
    import asyncio
    import importlib

    world_director_agent = importlib.import_module("story_agent.world_director_agent")

    creation_calls: list[tuple[int, int]] = []

    class DummyCurrencyGenerator:
        def __init__(self, user_id: int, conversation_id: int):
            creation_calls.append((user_id, conversation_id))

        async def get_currency_system(self) -> dict[str, str]:
            return {"name": "tokens"}

    monkeypatch.setattr(world_director_agent, "CurrencyGenerator", DummyCurrencyGenerator)
    monkeypatch.setattr(world_director_agent, "create_complete_world_director", lambda: object())

    async def exercise():
        director = world_director_agent.CompleteWorldDirector(user_id=7, conversation_id=9)
        director.context = world_director_agent.CompleteWorldDirectorContext(
            user_id=7,
            conversation_id=9,
        )
        director.context.currency_generator = None

        await director.initialize(warmed=True)

        assert isinstance(director.context.currency_generator, DummyCurrencyGenerator)
        assert creation_calls == [(7, 9)]

        director.context.currency_generator = None
        currency_system = await director.context._safe_get_currency_system()

        assert currency_system == {"name": "tokens"}
        assert creation_calls == [(7, 9), (7, 9)]
        assert isinstance(director.context.currency_generator, DummyCurrencyGenerator)

    asyncio.run(exercise())
