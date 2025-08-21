# story_agent/world_director_agent.py
"""
Complete World Dynamics Director with ALL system integrations and LLM-driven generation.
REFACTORED: Fixed all async/await issues, error handling, and type safety.
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
import random
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, is_dataclass, asdict
from datetime import datetime, timezone, timedelta

from story_agent.world_simulation_models import (
    # Core enums/models
    WorldMood,
    ActivityType,
    TimeOfDay,
    CurrentTimeData,
    VitalsData,

    # Event data
    AddictionCravingData,
    DreamData,
    RevelationData,

    # Relationship/inventory helpers
    RelationshipImpact,
    InventoryChange,

    # Choice and results
    ChoiceData,
    ChoiceProcessingResult,

    # World state & narrative
    CompleteWorldState,
    WorldState,
    SliceOfLifeEvent,
    PowerExchange,
    PowerDynamicType,
    WorldTension,
    RelationshipDynamics,
    NPCRoutine,
    EmergentPattern,
    NarrativeThread,

    # Emergent pattern outputs
    MemorySimilarity,
    RelationshipPatternOut,
    AddictionPatternOut,
    StatPatternOut,
    RulePatternOut,
    EmergentPatternsResult,
)

from db.connection import get_db_connection_context
from agents import Agent, function_tool, Runner, trace, ModelSettings, RunContextWrapper

# ===============================================================================
# COMPLETE SYSTEM INTEGRATIONS - NOTHING DROPPED
# ===============================================================================

# Universal Updater for Narrative Processing
from logic.universal_updater_agent import (
    UniversalUpdaterAgent,
    process_universal_update
)

# Memory System Integration
from logic.memory_logic import (
    MemoryManager,
    EnhancedMemory,
    MemoryType,
    MemorySignificance,
    ProgressiveRevealManager,
    RevealType,
    RevealSeverity,
    NPCMask,
    get_shared_memory,
    propagate_shared_memories,
    fetch_formatted_locations
)

# Stats System Integration
from logic.stats_logic import (
    get_player_visible_stats,
    get_player_hidden_stats,
    get_all_player_stats,
    apply_stat_changes,
    check_for_combination_triggers,
    apply_activity_effects,
    STAT_THRESHOLDS,
    STAT_COMBINATIONS,
    ACTIVITY_EFFECTS,
    detect_deception,
    calculate_social_insight,
    update_hunger_from_time,
    consume_food,
    apply_damage,
    heal_player
)

# Rule Enforcement Integration
from logic.rule_enforcement import (
    enforce_all_rules_on_player,
    evaluate_condition,
    parse_condition,
    apply_effect,
    get_player_stats,
    get_npc_stats
)

# Inventory System Integration
from logic.inventory_system_sdk import (
    get_inventory,
    add_item,
    remove_item,
    InventoryContext,
    register_with_governance as register_inventory
)

# Time and Calendar Systems - COMPLETE
from logic.time_cycle import (
    get_current_time_model,
    advance_time_with_events,
    get_current_vitals,
    process_activity_vitals,
    ActivityManager,
    ActivityType as TimeActivityType
)

from logic.calendar import (
    load_calendar_names,
    update_calendar_names,
    add_calendar_event,
    # get_calendar_events  # This function doesn't appear to exist in calendar.py
)

# Dynamic Relationships System - COMPLETE
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    drain_relationship_events_tool,
    get_relationship_summary_tool,
    process_relationship_interaction_tool,
    poll_relationship_events_tool,
    RelationshipState,
    RelationshipDimensions,
    RelationshipArchetypes,
    RelationshipPatternDetector
)

# Narrative Events System - COMPLETE
from logic.narrative_events import (
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence,
    add_moment_of_clarity,
    get_relationship_overview,
    generate_inner_monologue
)

# NPC Systems - COMPLETE
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    check_for_npc_revelation,
    progress_npc_narrative_stage,
    NPC_NARRATIVE_STAGES,
    NPCNarrativeStage
)

# Addiction System - COMPLETE
from logic.addiction_system_sdk import (
    AddictionContext,
    addiction_system_agent,
    process_addiction_update,
    check_addiction_status,
    get_addiction_status
)

# Currency System
from logic.currency_generator import CurrencyGenerator

# Event System
from logic.event_system import EventSystem

# Context systems - COMPLETE
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager, MemoryAddRequest
from context.models import MemoryMetadata
from context.vector_service import get_vector_service
from context.context_performance import PerformanceMonitor, track_performance

# Nyx governance
from nyx.nyx_governance import (
    NyxUnifiedGovernor,
    AgentType,
    DirectiveType,
    DirectivePriority
)
from nyx.integrate import get_central_governance
from nyx.directive_handler import DirectiveHandler

from logic.conflict_system.conflict_synthesizer import (
    get_synthesizer,
    ConflictSynthesizer
)

from lore.core.canon import (
    find_or_create_npc,
    find_or_create_location,
    find_or_create_event,
    find_or_create_faction,
    find_or_create_historical_event,
    find_or_create_notable_figure,
    log_canonical_event,
    ensure_canonical_context,
    update_entity_with_governance,
    get_entity_by_id,
    find_entity_by_name,
    create_message,
    update_current_roleplay
)
from lore.core.context import CanonicalContext

logger = logging.getLogger(__name__)

# ===============================================================================
# COMPLETE World State Models with ALL Integrations
# ===============================================================================

def _get_chatgpt_functions():
    """Lazy load all chatgpt_integration functions to avoid circular imports"""
    from logic.chatgpt_integration import (
        OpenAIClientManager,
        get_chatgpt_response,
        generate_text_completion,
        get_text_embedding,
        generate_reflection,
        analyze_preferences,
        create_semantic_abstraction,
        cosine_similarity
    )
    return {
        'OpenAIClientManager': OpenAIClientManager,
        'get_chatgpt_response': get_chatgpt_response,
        'generate_text_completion': generate_text_completion,
        'get_text_embedding': get_text_embedding,
        'generate_reflection': generate_reflection,
        'analyze_preferences': analyze_preferences,
        'create_semantic_abstraction': create_semantic_abstraction,
        'cosine_similarity': cosine_similarity
    }

def get_canonical_context(ctx_obj) -> Any:
    """Convert various context objects to canonical context"""
    if hasattr(ctx_obj, 'user_id') and hasattr(ctx_obj, 'conversation_id'):
        return ensure_canonical_context({
            'user_id': ctx_obj.user_id,
            'conversation_id': ctx_obj.conversation_id
        })
    elif hasattr(ctx_obj, 'context'):
        return get_canonical_context(ctx_obj.context)
    else:
        raise ValueError("Cannot extract canonical context from object")

def _to_model_dict(x):
    """Return a plain dict for any Pydantic model / dataclass; otherwise return as-is."""
    if x is None:
        return None
    # Pydantic v2 preferred
    if hasattr(x, "model_dump") and callable(x.model_dump):
        return x.model_dump(mode="python")
    # Pydantic v1 fallback
    if hasattr(x, "dict") and callable(x.dict):
        return x.dict()
    # Dataclass
    if is_dataclass(x):
        return asdict(x)
    return x  # already a plain type

def _kvlist(data):
    """
    Return a List[dict] with {"key": str, "value": <json-safe>} – NOT KVPair/KVItem objects.
    Always JSON-serialize dicts/lists so Pydantic sees scalars/strings.
    """
    import json
    out = []
    if data is None:
        return out

    def _val(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, default=str)
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        # Pydantic/enum/dataclass/etc. -> make string
        try:
            return json.dumps(_to_model_dict(v), default=str)
        except Exception:
            return str(v)

    if isinstance(data, dict):
        for k, v in data.items():
            out.append({"key": str(k), "value": _val(v)})
        return out

    if isinstance(data, list):
        for i, v in enumerate(data):
            out.append({"key": str(i), "value": _val(v)})
        return out

    # scalar
    return [{"key": "value", "value": _val(data)}]


# ===============================================================================
# Complete World Director Context with ALL Systems
# ===============================================================================

@dataclass
class CompleteWorldDirectorContext:
    """Complete context with ALL system managers INCLUDING canonical system"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    
    # ADD: Canonical context
    canonical_context: Optional[CanonicalContext] = None
    
    # Core system managers
    openai_manager: Optional[Any] = None
    universal_updater: Optional[UniversalUpdaterAgent] = None
    
    # Memory and reveals
    memory_manager: Optional[Any] = None
    reveal_manager: Optional[Any] = None
    
    # Relationships
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    
    # Addictions
    addiction_context: Optional[AddictionContext] = None
    
    # Events and activities
    event_system: Optional[EventSystem] = None
    activity_manager: Optional[ActivityManager] = None
    
    # Currency
    currency_generator: Optional[CurrencyGenerator] = None
    
    # Context services
    context_service: Optional[Any] = None
    vector_service: Optional[Any] = None
    performance_monitor: Optional[PerformanceMonitor] = None
    
    # Governance
    nyx_governor: Optional[NyxUnifiedGovernor] = None
    directive_handler: Optional[DirectiveHandler] = None
    
    # Calendar
    calendar_names: Optional[Dict[str, Any]] = None
    
    # NEW: Add conflict synthesizer
    conflict_synthesizer: Optional[ConflictSynthesizer] = None
    
    # State tracking
    current_world_state: Optional[CompleteWorldState] = None
    
    # Caching
    cache: Dict[str, Any] = field(default_factory=dict)
    
    async def initialize_everything(self):
        """Initialize ALL integrated systems including canonical system"""
        logger.info(f"Initializing Complete World Director for user {self.user_id}")
        
        try:
            # FIRST: Initialize canonical context
            self.canonical_context = ensure_canonical_context({
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            })
            logger.info("Initialized canonical context")
        
            # Lazy load OpenAI manager
            chatgpt_funcs = _get_chatgpt_functions()
            OpenAIClientManager = chatgpt_funcs['OpenAIClientManager']
            
            # Initialize OpenAI manager
            self.openai_manager = OpenAIClientManager()
            
            # Initialize addiction context
            self.addiction_context = AddictionContext(self.user_id, self.conversation_id)
            await self.addiction_context.initialize()
            
            # NEW: Initialize conflict synthesizer BEFORE event system
            self.conflict_synthesizer = await get_synthesizer(self.user_id, self.conversation_id)
            logger.info("Initialized conflict synthesizer")
            
            # Initialize event system (now it won't fail)
            self.event_system = EventSystem(self.user_id, self.conversation_id)
            await self.event_system.initialize()
            
            # Initialize currency generator
            self.currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
            
            # Initialize activity manager
            self.activity_manager = ActivityManager()
            
            # Initialize context services
            self.context_service = await get_context_service(self.user_id, self.conversation_id)
            self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
            self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
            self.performance_monitor = PerformanceMonitor(self.user_id, self.conversation_id)
            
            # Initialize governance
            self.nyx_governor = await get_central_governance(self.user_id, self.conversation_id)
            self.directive_handler = DirectiveHandler(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                agent_type="world_director",
                agent_id=f"world_director_{self.conversation_id}",
                governance=self.nyx_governor
            )
            
            # Initialize relationship manager
            self.relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
            
            # Load calendar names
            self.calendar_names = await load_calendar_names(self.user_id, self.conversation_id)
            
            # Register inventory system with governance
            await register_inventory(self.user_id, self.conversation_id)
            
            # Build initial world state
            self.current_world_state = await self._build_complete_world_state()
            
            logger.info("Complete World Director fully initialized with ALL systems")
            
            # Log initialization canonically
            async with get_db_connection_context() as conn:
                await log_canonical_event(
                    self.canonical_context, conn,
                    "Complete World Director initialized with ALL systems",
                    tags=["initialization", "world_director", "all_systems"],
                    significance=9
                )
            
            logger.info("Complete World Director fully initialized with ALL systems INCLUDING canonical")
            
        except Exception as e:
            logger.error(f"Error initializing World Director: {e}", exc_info=True)
            raise
    
    async def _build_complete_world_state(self) -> CompleteWorldState:
        """Build complete world state from ALL systems"""
        try:
            # Time and Calendar
            current_time = await get_current_time_model(self.user_id, self.conversation_id)
            # Normalize time_of_day if it's a string like "Morning"/"Afternoon"
            if hasattr(current_time, "time_of_day") and isinstance(current_time.time_of_day, str):
                current_time.time_of_day = current_time.time_of_day.lower()
            calendar_events = await self._safe_get_calendar_events()
    
            # Vitals and Stats
            vitals = await get_current_vitals(self.user_id, self.conversation_id)
            visible_stats = await get_player_visible_stats(
                self.user_id, self.conversation_id, self.player_name
            )
            hidden_stats = await get_player_hidden_stats(
                self.user_id, self.conversation_id, self.player_name
            )
    
            # Check stat combinations and thresholds
            stat_combinations = await check_for_combination_triggers(
                self.user_id, self.conversation_id
            )
            stat_thresholds = self._check_stat_thresholds(hidden_stats)
    
            # Memory and Context
            recent_memories = await self._safe_retrieve_memories()
    
            # Check for flashbacks
            flashback = None
            if recent_memories and random.random() < 0.1:  # 10% chance
                flashback = await self._safe_generate_flashback()
    
            # Check for NPC reveals
            pending_reveals = await ProgressiveRevealManager.check_for_automated_reveals(
                self.user_id, self.conversation_id
            )
    
            # Dreams and Revelations
            revelation = await self._safe_check_revelations()
            narrative_moments = await self._safe_check_narrative_moments()
    
            # Rules
            triggered_rules = await enforce_all_rules_on_player(self.player_name)
    
            # Inventory
            inventory_result = await self._safe_get_inventory()
    
            # NPCs with complete data
            npcs = await self._get_complete_npc_data()
    
            # Relationships
            rel_overview = await self._safe_get_relationship_overview()
    
            # Drain relationship events
            rel_events = await self._safe_drain_relationship_events()
    
            # Addictions
            addiction_status = await self._safe_get_addiction_status()
    
            # Check for active cravings
            active_cravings = await self._check_active_cravings(addiction_status)
    
            # Currency
            currency_system = await self._safe_get_currency_system()
    
            # Location data
            location_data = await fetch_formatted_locations(
                self.user_id, self.conversation_id
            )
    
            # Conflict system state
            conflict_state: Dict[str, Any] = {}
            active_conflicts: List[Dict[str, Any]] = []
            conflict_manifestations: List[Any] = []
            if self.conflict_synthesizer:
                try:
                    conflict_system_state = await self.conflict_synthesizer.get_system_state()
                    if not isinstance(conflict_system_state, dict):
                        logger.warning(f"Unexpected conflict state type: {type(conflict_system_state)}")
                        conflict_system_state = {}
    
                    active_conflicts = conflict_system_state.get("active_conflicts", [])
                    conflict_state = {
                        "active_conflicts": active_conflicts,
                        "complexity_score": conflict_system_state.get("metrics", {}).get("complexity_score", 0),
                        "total_conflicts": conflict_system_state.get("metrics", {}).get("total_conflicts", 0),
                        "resolved_conflicts": conflict_system_state.get("metrics", {}).get("resolved_conflicts", 0),
                        "subsystem_states": conflict_system_state.get("subsystem_states", {}),
                    }
    
                    if active_conflicts:
                        scene_context = {
                            "scene_type": "world_state_check",
                            "location": location_data,
                            "participants": [npc["npc_id"] for npc in npcs[:3] if "npc_id" in npc],
                            "time_of_day": getattr(current_time, "time_of_day", "afternoon"),
                        }
                        scene_result = await self.conflict_synthesizer.process_scene(scene_context)
                        if scene_result.get("manifestations"):
                            conflict_manifestations = scene_result["manifestations"]
    
                except AttributeError as e:
                    logger.warning(f"Could not get conflict state - attribute error: {e}")
                    conflict_state = {}
                    active_conflicts = []
                except Exception as e:
                    logger.warning(f"Could not get conflict state: {e}")
                    conflict_state = {}
                    active_conflicts = []
    
            # World mood & tensions
            world_mood = self._calculate_complete_world_mood(
                hidden_stats,
                vitals,
                stat_combinations,
                addiction_status,
                active_cravings,
                revelation is not None,
                len(active_conflicts) > 0,
            )
    
            tensions = self._calculate_all_tensions(
                hidden_stats,
                vitals,
                stat_combinations,
                addiction_status,
                rel_overview,
                conflict_state,
            )
    
            world_tension = WorldTension(
                overall_tension=sum(tensions.values()) / len(tensions) if tensions else 0,
                social_tension=tensions.get("relationship", 0),
                power_tension=tensions.get("obedience", 0),
                sexual_tension=tensions.get("lust", 0),
                emotional_tension=tensions.get("dependency", 0),
                addiction_tension=tensions.get("addiction", 0),
                vital_tension=tensions.get("vital", 0),
                unresolved_conflicts=len(active_conflicts),
                tension_sources=list(tensions.keys()),
            )
    
            # Relationship dynamics
            relationship_dynamics = RelationshipDynamics(
                player_submission_level=hidden_stats.get("obedience", 0),
                player_resistance_level=hidden_stats.get("willpower", 50),
                player_corruption_level=hidden_stats.get("corruption", 0),
                acceptance_level=100 - hidden_stats.get("mental_resilience", 100),
                dominant_npc_ids=[npc["npc_id"] for npc in npcs if npc.get("dominance", 0) > 60 and "npc_id" in npc],
                supportive_npc_ids=[],
                adversarial_npc_ids=[],
                intimate_npc_ids=[
                    npc["npc_id"]
                    for npc in npcs
                    if npc.get("relationship", {}).get("dimensions", {}).get("intimacy", 0) > 60 and "npc_id" in npc
                ],
            )
    
            # Available activities
            available_activities: List[Dict[str, Any]] = []
            if getattr(vitals, "fatigue", 0) < 80:
                available_activities.append({"type": "work", "available": True})
            if getattr(vitals, "hunger", 100) < 30:
                available_activities.append({"type": "eat", "priority": "high"})
            if active_conflicts:
                for conflict in active_conflicts[:3]:
                    if isinstance(conflict, dict):
                        available_activities.append({
                            "type": "conflict_interaction",
                            "conflict_id": conflict.get("id"),
                            "description": f"Deal with {conflict.get('type', 'ongoing')} conflict",
                        })
    
            # Build and return the world state (using helpers to keep Pydantic happy)
            return CompleteWorldState(
                # models -> dicts
                current_time=_to_model_dict(current_time),
                player_vitals=_to_model_dict(vitals),
    
                # dicts/lists -> KV [{"key","value"}]
                calendar_names=_kvlist(self.calendar_names or {}),
                calendar_events=_kvlist(calendar_events),
    
                visible_stats=_kvlist(visible_stats),
                hidden_stats=_kvlist(hidden_stats),
                active_stat_combinations=_kvlist(stat_combinations),
                stat_thresholds_active=_kvlist(stat_thresholds),
    
                recent_memories=_kvlist([m.to_dict() if hasattr(m, "to_dict") else _to_model_dict(m) for m in recent_memories]),
                semantic_abstractions=[],
                active_flashbacks=_kvlist([flashback] if flashback else []),
                pending_reveals=_kvlist(pending_reveals),
                pending_dreams=[],
                recent_revelations=_kvlist([revelation] if revelation else []),
                inner_monologues=[],
    
                active_rules=_kvlist(triggered_rules),
                triggered_effects=[],
                pending_effects=[],
    
                player_inventory=_kvlist(inventory_result.get("items", [])),
                recent_item_changes=[],
    
                active_npcs=_kvlist(npcs),
                npc_masks=_kvlist({npc["npc_id"]: npc.get("mask", {}) for npc in npcs if "npc_id" in npc}),
                npc_narrative_stages=_kvlist({npc["npc_id"]: npc.get("narrative_stage", "") for npc in npcs if "npc_id" in npc}),
    
                relationship_states=[],
                relationship_dynamics=_to_model_dict(relationship_dynamics),
                relationship_overview=_kvlist(rel_overview) if rel_overview else None,
                pending_relationship_events=_kvlist(rel_events.get("events", [])),
    
                addiction_status=_kvlist(addiction_status),
                active_cravings=_kvlist(active_cravings),
                addiction_contexts=[],
    
                player_money=100,
                currency_system=_kvlist(currency_system),
                recent_transactions=[],
    
                # enums -> string value
                world_mood=(world_mood.value if hasattr(world_mood, "value") else world_mood),
    
                # model -> dict
                world_tension=_to_model_dict(world_tension),
    
                tension_factors=_kvlist(tensions),
                environmental_factors=_kvlist({
                    "conflict_manifestations": conflict_manifestations,
                    "conflict_complexity": conflict_state.get("complexity_score", 0),
                }),
    
                # schema expects a string here in your tool JSON
                location_data=str(location_data) if location_data else "",
    
                # use KV for flex fields
                ongoing_events=_kvlist(active_conflicts),
                available_activities=_kvlist(available_activities),
    
                event_history=[],
                nyx_directives=[],
            )
    
        except Exception as e:
            logger.error(f"Error building world state: {e}", exc_info=True)
            # Return minimal valid state
            return self._get_fallback_world_state()

            
    async def _get_complete_npc_data(self) -> List[Dict[str, Any]]:
        """Get NPCs with ALL their data using canonical system for consistency"""
        try:
            async with get_db_connection_context() as conn:
                # Get existing NPCs
                npcs = await conn.fetch("""
                    SELECT npc_id, npc_name, dominance, cruelty, intensity,
                           personality_traits, current_location, monica_level
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                    AND introduced = true
                    LIMIT 10
                """, self.user_id, self.conversation_id)
            
            complete_npcs = []
            
            async with get_db_connection_context() as conn:
                for npc in npcs:
                    npc_dict = dict(npc)
                    
                    try:
                        # Ensure NPC exists canonically (handles semantic deduplication)
                        canonical_npc_id = await find_or_create_npc(
                            self.canonical_context, conn,
                            npc_name=npc['npc_name'],
                            role=npc.get('personality_traits', ''),
                            affiliations=[]
                        )
                        
                        # Log if there was a mismatch (indicates deduplication occurred)
                        if canonical_npc_id != npc['npc_id']:
                            await log_canonical_event(
                                self.canonical_context, conn,
                                f"NPC {npc['npc_name']} canonically resolved from ID {npc['npc_id']} to {canonical_npc_id}",
                                tags=["npc", "deduplication", "canonical_resolution"],
                                significance=6
                            )
                            npc_dict['canonical_id'] = canonical_npc_id
                            npc_dict['was_deduplicated'] = True
                        else:
                            npc_dict['canonical_id'] = canonical_npc_id
                            npc_dict['was_deduplicated'] = False
                        
                    except Exception as e:
                        logger.warning(f"Could not canonically resolve NPC {npc['npc_id']}: {e}")
                        npc_dict['canonical_id'] = npc['npc_id']
                        npc_dict['was_deduplicated'] = False
                    
                    try:
                        # Get mask data (existing logic)
                        mask_data = await ProgressiveRevealManager.get_npc_mask(
                            self.user_id, self.conversation_id, npc['npc_id']
                        )
                        npc_dict['mask'] = mask_data
                    except Exception as e:
                        logger.warning(f"Could not get mask for NPC {npc['npc_id']}: {e}")
                        npc_dict['mask'] = {}
                    
                    try:
                        # Get narrative stage (existing logic)
                        stage = await get_npc_narrative_stage(
                            self.user_id, self.conversation_id, npc['npc_id']
                        )
                        npc_dict['narrative_stage'] = stage.name if hasattr(stage, 'name') else str(stage)
                    except Exception as e:
                        logger.warning(f"Could not get narrative stage for NPC {npc['npc_id']}: {e}")
                        npc_dict['narrative_stage'] = 'unknown'
                    
                    try:
                        # Check for revelations (existing logic)
                        revelation = await check_for_npc_revelation(
                            self.user_id, self.conversation_id, npc['npc_id']
                        )
                        npc_dict['pending_revelation'] = revelation
                    except Exception as e:
                        logger.warning(f"Could not check revelations for NPC {npc['npc_id']}: {e}")
                        npc_dict['pending_revelation'] = None
                    
                    try:
                        # Get relationship state (existing logic)
                        rel_state = await self.relationship_manager.get_relationship_state(
                            'npc', npc['npc_id'], 'player', 1
                        )
                        npc_dict['relationship'] = {
                            'dimensions': rel_state.dimensions.__dict__ if hasattr(rel_state.dimensions, '__dict__') else {},
                            'archetype': rel_state.archetype if hasattr(rel_state, 'archetype') else 'unknown',
                            'patterns': list(rel_state.history.active_patterns) if hasattr(rel_state, 'history') else []
                        }
                    except Exception as e:
                        logger.warning(f"Could not get relationship for NPC {npc['npc_id']}: {e}")
                        npc_dict['relationship'] = {'dimensions': {}, 'archetype': 'unknown', 'patterns': []}
                    
                    complete_npcs.append(npc_dict)
            
            return complete_npcs
            
        except Exception as e:
            logger.error(f"Error getting NPC data: {e}", exc_info=True)
            return []
    
    def _check_stat_thresholds(self, hidden_stats: Dict) -> Dict[str, Any]:
        """Check which stat thresholds are active (SYNCHRONOUS)"""
        active_thresholds = {}
        
        for stat_name, value in hidden_stats.items():
            if stat_name in STAT_THRESHOLDS:
                thresholds = STAT_THRESHOLDS[stat_name]
                for threshold in thresholds:
                    if value >= threshold['level']:
                        active_thresholds[stat_name] = threshold
                    else:
                        break
        
        return active_thresholds
    
    def _calculate_complete_world_mood(
        self, hidden_stats: Dict, vitals: VitalsData,
        stat_combinations: List[Dict], addiction_status: Dict,
        active_cravings: List[Dict], has_revelation: bool,
        has_active_conflicts: bool  # NEW parameter
    ) -> WorldMood:
        """Calculate world mood from ALL factors (SYNCHRONOUS)"""
        # Critical overrides
        if vitals.fatigue > 85:
            return WorldMood.EXHAUSTED
        if vitals.hunger < 15 or vitals.thirst < 15:
            return WorldMood.DESPERATE
        if active_cravings:
            return WorldMood.CRAVING
        if has_revelation:
            return WorldMood.DREAMLIKE
        
        # NEW: Conflict-based moods
        if has_active_conflicts:
            # Active conflicts increase tension
            if hidden_stats.get('mental_resilience', 100) < 30:
                return WorldMood.CHAOTIC
            else:
                return WorldMood.TENSE
        
        # Special combinations
        for combo in stat_combinations:
            if combo.get('name') == 'Stockholm Syndrome':
                return WorldMood.CORRUPTED
            elif combo.get('name') == 'Breaking Point':
                return WorldMood.CHAOTIC
        
        # Addiction-based moods
        if addiction_status.get('has_addictions'):
            addictions = addiction_status.get('addictions', {})
            if addictions:
                max_level = max(
                    data.get('level', 0) 
                    for data in addictions.values()
                )
                if max_level >= 4:
                    return WorldMood.CRAVING
        
        # Stats-based moods
        corruption = hidden_stats.get('corruption', 0)
        obedience = hidden_stats.get('obedience', 0)
        lust = hidden_stats.get('lust', 0)
        
        if corruption > 80 or lust > 80:
            return WorldMood.CORRUPTED
        elif obedience > 70:
            return WorldMood.OPPRESSIVE
        elif corruption > 60:
            return WorldMood.INTIMATE
        elif corruption > 40:
            return WorldMood.MYSTERIOUS
        elif corruption > 20:
            return WorldMood.PLAYFUL
        else:
            return WorldMood.RELAXED
    
    def _calculate_all_tensions(
        self, hidden_stats: Dict, vitals: VitalsData,
        stat_combinations: List[Dict], addiction_status: Dict,
        rel_overview: Optional[Dict],
        conflict_state: Dict  # NEW parameter
    ) -> Dict[str, float]:
        """Calculate ALL tension factors"""
        tensions = {}
    
        
        # Vital tensions
        tensions['vital'] = 0.0
        if vitals.hunger < 30:
            tensions['vital'] += (30 - vitals.hunger) / 30 * 0.5
        if vitals.thirst < 30:
            tensions['vital'] += (30 - vitals.thirst) / 30 * 0.5
        if vitals.fatigue > 70:
            tensions['vital'] += (vitals.fatigue - 70) / 30 * 0.5
        
        # Stat-based tensions
        tensions['corruption'] = hidden_stats.get('corruption', 0) / 100
        tensions['obedience'] = hidden_stats.get('obedience', 0) / 100
        tensions['dependency'] = hidden_stats.get('dependency', 0) / 100
        tensions['lust'] = hidden_stats.get('lust', 0) / 100
        
        # Resistance tensions
        tensions['willpower'] = (100 - hidden_stats.get('willpower', 100)) / 100
        tensions['confidence'] = (100 - hidden_stats.get('confidence', 100)) / 100
        tensions['mental_resilience'] = (100 - hidden_stats.get('mental_resilience', 100)) / 100

        # NEW: Conflict tensions
        tensions['conflict'] = 0.0
        if conflict_state:
            active_count = len(conflict_state.get('active_conflicts', []))
            complexity = conflict_state.get('complexity_score', 0)
            tensions['conflict'] = min(1.0, (active_count * 0.2) + (complexity * 0.5))
        
        # Addiction tensions
        tensions['addiction'] = 0.0
        if addiction_status.get('has_addictions'):
            addictions = addiction_status.get('addictions', {})
            for data in addictions.values():
                tensions['addiction'] = max(tensions['addiction'], data.get('level', 0) / 5)
        
        # Relationship tensions
        tensions['relationship'] = 0.0
        if rel_overview:
            relationships = rel_overview.get('relationships', [])
            for rel in relationships:
                patterns = rel.get('patterns', [])
                archetypes = rel.get('archetypes', [])
                if 'explosive_chemistry' in patterns:
                    tensions['relationship'] += 0.1
                if 'toxic_bond' in archetypes:
                    tensions['relationship'] += 0.15
        
        # Combination tensions
        tensions['breaking'] = min(1.0, len(stat_combinations) * 0.2)
        
        return tensions
    
    # Safe wrapper methods for error handling
    async def _safe_get_calendar_events(self) -> List[Dict[str, Any]]:
        """Safely get calendar events"""
        try:
            # Assuming there's a get_calendar_events function
            # This would need to be imported from logic.calendar
            return []  # Placeholder
        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")
            return []
    
    async def _safe_retrieve_memories(self) -> List[Any]:
        """Safely retrieve memories"""
        try:
            return await MemoryManager.retrieve_relevant_memories(
                self.user_id, self.conversation_id, 
                self.player_name, "player",
                context="current_situation", limit=10
            )
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    async def _safe_generate_flashback(self) -> Optional[Dict[str, Any]]:
        """Safely generate flashback"""
        try:
            return await MemoryManager.generate_flashback(
                self.user_id, self.conversation_id,
                1, "current_context"
            )
        except Exception as e:
            logger.error(f"Error generating flashback: {e}")
            return None
    
    async def _safe_check_revelations(self) -> Optional[Dict[str, Any]]:
        """Safely check for revelations"""
        try:
            return await check_for_personal_revelations(
                self.user_id, self.conversation_id
            )
        except Exception as e:
            logger.error(f"Error checking revelations: {e}")
            return None

    async def check_for_conflict_triggers(self) -> List[Dict[str, Any]]:
        """Check if current world state should trigger new conflicts"""
        triggers = []
        
        if not self.context or not self.context.conflict_synthesizer:
            return triggers
        
        state = self.context.current_world_state
        
        # Check tension levels
        if state.tension_factors.get('conflict', 0) < 0.3:  # Low conflict tension
            if state.tension_factors.get('relationship', 0) > 0.6:
                triggers.append({
                    "type": "relationship_conflict",
                    "reason": "High relationship tension with low conflict activity",
                    "suggested_type": "social"
                })
        
        # Check for stat combinations that suggest conflict
        for combo in state.active_stat_combinations:
            if combo.get('name') == 'Breaking Point':
                triggers.append({
                    "type": "breaking_point_conflict", 
                    "reason": "Player at breaking point",
                    "suggested_type": "personal"
                })
        
        # Check NPC stages for conflict potential
        advanced_npcs = [
            npc for npc in state.active_npcs
            if npc.get('narrative_stage') in ['Veil Thinning', 'Full Revelation']
        ]
        
        if len(advanced_npcs) > 1:
            triggers.append({
                "type": "multi_npc_conflict",
                "reason": f"{len(advanced_npcs)} NPCs at advanced stages",
                "suggested_type": "social",
                "is_multiparty": True,
                "participants": [npc['npc_id'] for npc in advanced_npcs]
            })
        
        return triggers
    
    async def _safe_check_narrative_moments(self) -> List[Dict[str, Any]]:
        """Safely check narrative moments"""
        try:
            return await check_for_narrative_moments(
                self.user_id, self.conversation_id
            )
        except Exception as e:
            logger.error(f"Error checking narrative moments: {e}")
            return []
    
    async def _safe_get_inventory(self) -> Dict[str, Any]:
        """Safely get inventory"""
        try:
            return await get_inventory(
                self.user_id, self.conversation_id, self.player_name
            )
        except Exception as e:
            logger.error(f"Error getting inventory: {e}")
            return {'items': []}
    
    async def _safe_get_relationship_overview(self) -> Optional[Dict[str, Any]]:
        """Safely get relationship overview"""
        try:
            return await get_relationship_overview(
                self.user_id, self.conversation_id
            )
        except Exception as e:
            logger.error(f"Error getting relationship overview: {e}")
            return None
    
    async def _safe_drain_relationship_events(self) -> Dict[str, Any]:
        """Safely drain relationship events"""
        try:
            # Don't call the tool directly, access the event_generator
            from logic.dynamic_relationships import event_generator
            events = await event_generator.drain_events(max_events=5)
            return {
                'events': events,
                'count': len(events)
            }
        except Exception as e:
            logger.error(f"Error draining relationship events: {e}")
            return {'events': []}
    
    async def _safe_get_addiction_status(self) -> Dict[str, Any]:
        """Safely get addiction status"""
        try:
            return await get_addiction_status(
                self.user_id, self.conversation_id, self.player_name
            )
        except Exception as e:
            logger.error(f"Error getting addiction status: {e}")
            return {'has_addictions': False, 'addictions': {}}
    
    async def _check_active_cravings(self, addiction_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for active cravings"""
        active_cravings = []
        try:
            if addiction_status.get('has_addictions'):
                addictions = addiction_status.get('addictions', {})
                for addiction_type, data in addictions.items():
                    if data.get('level', 0) > 2:  # Level 3+ can trigger cravings
                        craving_check = await check_addiction_status(
                            self.user_id, self.conversation_id,
                            self.player_name, addiction_type
                        )
                        if craving_check and craving_check.get('craving_active'):
                            active_cravings.append(craving_check)
        except Exception as e:
            logger.error(f"Error checking active cravings: {e}")
        
        return active_cravings
    
    async def _safe_get_currency_system(self) -> Dict[str, Any]:
        """Safely get currency system"""
        try:
            return await self.currency_generator.get_currency_system()
        except Exception as e:
            logger.error(f"Error getting currency system: {e}")
            return {'name': 'coins'}
    
    def _get_fallback_world_state(self) -> CompleteWorldState:
        """Get a minimal fallback world state when building fails"""
        return CompleteWorldState(
            current_time=CurrentTimeData(
                year=2025, month=1, day=1,
                hour=12, minute=0,
                time_of_day=TimeOfDay.AFTERNOON
            ),
            player_vitals=VitalsData(
                hunger=50, thirst=50, fatigue=30, arousal=0
            ),
            world_mood=WorldMood.RELAXED
        )

# ===============================================================================
# COMPLETE LLM-Driven Tools with ALL Systems (with fixed error handling)
# ===============================================================================

@function_tool
@track_performance("generate_complete_event")
async def generate_complete_slice_of_life_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext]
) -> Dict[str, Any]:
    """Generate event considering ALL systems with canonical consistency"""
    context = ctx.context
    world_state = context.current_world_state
    
    # Lazy load chatgpt functions
    chatgpt_funcs = _get_chatgpt_functions()
    get_chatgpt_response = chatgpt_funcs['get_chatgpt_response']
    create_semantic_abstraction = chatgpt_funcs['create_semantic_abstraction']
    
    try:
        # NEW: Check for conflict-driven events
        if context.conflict_synthesizer:
            conflict_system_state = await context.conflict_synthesizer.get_system_state()
            active_conflicts = conflict_system_state.get('active_conflicts', [])
            
            # If there are active conflicts, check if they should manifest
            if active_conflicts:
                # Process current scene through conflict system
                scene_context = {
                    "scene_type": "slice_of_life",
                    "location": world_state.location_data,
                    "participants": [npc['npc_id'] for npc in world_state.active_npcs[:3]],
                    "world_mood": world_state.world_mood.value,
                    "time_of_day": world_state.current_time.time_of_day
                }
                
                conflict_result = await context.conflict_synthesizer.process_scene(scene_context)
                
                # If conflicts manifest, store canonically and prioritize
                if conflict_result.get('manifestations'):
                    async with get_db_connection_context() as conn:
                        # Store conflict manifestation as canonical event
                        conflict_event_id = await find_or_create_event(
                            context.canonical_context, conn,
                            event_name="Tensions Surface",
                            description=conflict_result['manifestations'][0],
                            event_type="conflict_manifestation",
                            location=world_state.location_data.get('current_location', 'unknown'),
                            year=world_state.current_time.year,
                            month=world_state.current_time.month,
                            day=world_state.current_time.day
                        )
                        
                        await log_canonical_event(
                            context.canonical_context, conn,
                            f"Conflict manifestation detected: {conflict_result['manifestations'][0][:100]}",
                            tags=["conflict", "manifestation", "generated_event"],
                            significance=8
                        )
                    
                    return {
                        "event_type": "conflict_manifestation",
                        "title": "Tensions Surface",
                        "description": conflict_result['manifestations'][0],
                        "conflicts_active": True,
                        "conflict_data": conflict_result,
                        "choices": conflict_result.get('player_choices', []),
                        "npc_behaviors": conflict_result.get('npc_behaviors', {}),
                        "canonical_event_id": conflict_event_id
                    }
        
        # 1. Check for addiction cravings (existing logic)
        if world_state.active_cravings:
            craving = world_state.active_cravings[0]
            craving_event = await generate_addiction_craving_event(ctx, craving)
            
            # Store canonically
            async with get_db_connection_context() as conn:
                event_id = await find_or_create_event(
                    context.canonical_context, conn,
                    event_name=craving_event.get('title', 'Craving Event'),
                    description=craving_event.get('description', ''),
                    event_type="addiction_craving",
                    location=world_state.location_data.get('current_location', 'unknown')
                )
                
                await log_canonical_event(
                    context.canonical_context, conn,
                    f"Addiction craving event triggered: {craving['addiction_type']}",
                    tags=["addiction", "craving", "generated_event"],
                    significance=7
                )
                
                craving_event['canonical_event_id'] = event_id
            
            return craving_event
        
        # 2. Check for pending dreams
        if world_state.world_mood == WorldMood.DREAMLIKE or random.random() < 0.05:
            dream_result = await add_dream_sequence(
                context.user_id, context.conversation_id
            )
            if dream_result:
                dream_event = await generate_dream_event(ctx, dream_result)
                
                # Store canonically
                async with get_db_connection_context() as conn:
                    event_id = await find_or_create_event(
                        context.canonical_context, conn,
                        event_name=dream_event.get('title', 'Dream Sequence'),
                        description=dream_event.get('description', ''),
                        event_type="dream",
                        location="dream_realm"
                    )
                    
                    await log_canonical_event(
                        context.canonical_context, conn,
                        f"Dream sequence generated: {dream_event.get('title', 'Dream')}",
                        tags=["dream", "subconscious", "generated_event"],
                        significance=6
                    )
                    
                    dream_event['canonical_event_id'] = event_id
                
                return dream_event
        
        # 3. Check for revelations
        if world_state.recent_revelations:
            revelation_event = await generate_revelation_event(ctx, world_state.recent_revelations[0])
            
            # Store canonically
            async with get_db_connection_context() as conn:
                event_id = await find_or_create_event(
                    context.canonical_context, conn,
                    event_name=revelation_event.get('title', 'Revelation'),
                    description=revelation_event.get('description', ''),
                    event_type="revelation",
                    location=world_state.location_data.get('current_location', 'unknown')
                )
                
                await log_canonical_event(
                    context.canonical_context, conn,
                    f"Personal revelation event: {revelation_event.get('title', 'Revelation')}",
                    tags=["revelation", "personal", "generated_event"],
                    significance=8
                )
                
                revelation_event['canonical_event_id'] = event_id
            
            return revelation_event
        
        # Build comprehensive context (existing logic)
        event_context = {
            "time": world_state.current_time.to_dict() if hasattr(world_state.current_time, 'to_dict') else {},
            "calendar": world_state.calendar_names,
            "vitals": world_state.player_vitals.to_dict() if hasattr(world_state.player_vitals, 'to_dict') else {},
            "visible_stats": world_state.visible_stats,
            "hidden_stats": world_state.hidden_stats,
            "stat_combinations": [c.get('name', 'unknown') for c in world_state.active_stat_combinations],
            "stat_thresholds": world_state.stat_thresholds_active,
            "world_mood": world_state.world_mood.value,
            "tensions": world_state.tension_factors,
            "recent_memories": world_state.recent_memories[:5] if world_state.recent_memories else [],
            "pending_reveals": len(world_state.pending_reveals),
            "active_npcs": [
                {
                    "name": npc.get('npc_name', 'Unknown'),
                    "dominance": npc.get('dominance', 0),
                    "stage": npc.get('narrative_stage', 'unknown'),
                    "mask_integrity": npc.get('mask', {}).get('integrity', 100),
                    "relationship": npc.get('relationship', {}),
                    "canonical_id": npc.get('canonical_id', npc.get('npc_id'))  # NEW
                }
                for npc in world_state.active_npcs[:3]
            ],
            "addiction_status": world_state.addiction_status,
            "inventory_highlights": [
                item for item in world_state.player_inventory[:5]
                if item.get('item_effect')
            ],
            "location": world_state.location_data,
            "currency": world_state.currency_system.get('name', 'money')
        }
        
        # Generate using ChatGPT with reflection (existing logic)
        aggregator_text = f"World Context:\n{json.dumps(event_context, indent=2, default=str)}"
        
        prompt = """Generate a dynamic slice-of-life event that emerges from the current world state.

Create an event that:
1. Naturally incorporates NPCs based on their stages and relationships
2. Reflects the player's physical and mental state
3. May trigger addictions, stat changes, or rule effects
4. Includes subtle power dynamics appropriate to corruption level
5. Provides meaningful choices with hidden consequences
6. Uses inventory items or currency if relevant
7. Maintains narrative consistency with established canon

Output as JSON with complete detail for emergent gameplay."""

        response = await get_chatgpt_response(
            context.conversation_id,
            aggregator_text,
            prompt,
            reflection_enabled=True,
            use_nyx_integration=True
        )
        
        # Parse response (existing logic)
        event_data = {}
        if response.get('type') == 'function_call':
            event_data = response.get('function_args', {})
        else:
            try:
                response_text = response.get('response', '{}')
                event_data = json.loads(response_text)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not parse event response as JSON: {e}")
                event_data = {
                    "event_type": "routine",
                    "title": "A Moment Passes",
                    "description": response.get('response', 'Time passes quietly.')
                }
        
        # Store canonically
        async with get_db_connection_context() as conn:
            # Ensure location exists canonically
            if world_state.location_data:
                location_name = await find_or_create_location(
                    context.canonical_context, conn,
                    location_name=world_state.location_data.get('current_location', 'Unknown Location'),
                    description=world_state.location_data.get('description', 'A place in the world'),
                    location_type=world_state.location_data.get('type', 'general')
                )
            else:
                location_name = 'Unknown Location'
            
            # Store event canonically
            event_id = await find_or_create_event(
                context.canonical_context, conn,
                event_name=event_data.get('title', 'Generated Event'),
                description=event_data.get('description', ''),
                event_type=event_data.get('event_type', 'slice_of_life'),
                location=location_name,
                year=world_state.current_time.year,
                month=world_state.current_time.month,
                day=world_state.current_time.day
            )
            
            # Log canonically
            await log_canonical_event(
                context.canonical_context, conn,
                f"Generated slice-of-life event: {event_data.get('title', 'Event')}",
                tags=["event", "generated", "slice_of_life", event_data.get('event_type', 'unknown')],
                significance=7
            )
            
            # Handle any NPCs involved canonically
            for npc_data in event_data.get('npcs_involved', []):
                npc_id = await find_or_create_npc(
                    context.canonical_context, conn,
                    npc_name=npc_data.get('name', 'Unknown'),
                    role=npc_data.get('role', 'participant'),
                    affiliations=npc_data.get('affiliations', [])
                )
                
                await log_canonical_event(
                    context.canonical_context, conn,
                    f"NPC {npc_data.get('name', 'Unknown')} involved in event {event_data.get('title', 'Event')}",
                    tags=["npc", "event_participation"],
                    significance=5
                )
            
            event_data['canonical_event_id'] = event_id
            event_data['canonical_location'] = location_name
        
        # Process through universal updater if narrative exists (existing logic)
        if event_data.get('narrative'):
            try:
                update_result = await process_universal_update(
                    context.user_id, context.conversation_id,
                    event_data['narrative'],
                    {"source": "generated_event", "event_data": event_data}
                )
            except Exception as e:
                logger.error(f"Error processing universal update: {e}")
        
        # Store in memory with semantic abstraction (existing logic)
        memory_text = f"Event: {event_data.get('title', 'Unnamed event')}"
        
        try:
            abstraction = await create_semantic_abstraction(memory_text)
            
            await MemoryManager.add_memory(
                context.user_id, context.conversation_id,
                entity_id=1, entity_type="player",
                memory_text=memory_text,
                memory_type=MemoryType.INTERACTION,
                significance=MemorySignificance.MEDIUM,
                tags=["event", event_data.get('event_type', 'unknown'), "canonical"]
            )
            
            if world_state.semantic_abstractions is not None:
                world_state.semantic_abstractions.append(abstraction)
        except Exception as e:
            logger.error(f"Error storing event memory: {e}")
        
        return event_data
        
    except Exception as e:
        logger.error(f"Error generating slice of life event: {e}", exc_info=True)
        return {
            "event_type": "error",
            "title": "System Processing",
            "description": "The world continues around you.",
            "error": str(e)
        }

@function_tool
async def generate_addiction_craving_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    craving_data: AddictionCravingData
) -> Dict[str, Any]:
    """Generate an addiction-craving event via LLM"""
    context = ctx.context
    data = craving_data.model_dump(exclude_none=True)

    # Lazy load
    chatgpt_funcs = _get_chatgpt_functions()
    generate_text_completion = chatgpt_funcs['generate_text_completion']

    try:
        prompt = f"""Generate an addiction craving event for a femdom RPG.

Craving Data:
{json.dumps(data, indent=2, default=str)}

Create an event that:
1. Shows the psychological pull of the addiction
2. Presents a choice between resistance and indulgence
3. Has different consequences based on addiction level
4. May involve the addiction target (NPC/behavior)
5. Escalates based on how long since last indulgence

Output as JSON with choices and consequences."""
        response = await generate_text_completion(
            system_prompt="You are creating addiction mechanics that drive narrative tension.",
            user_prompt=prompt,
            temperature=0.8,
            max_tokens=800
        )

        try:
            event: Dict[str, Any] = json.loads(response) if response else {}
        except (json.JSONDecodeError, TypeError) as e:
            logger.error("Error parsing addiction event JSON: %s", e)
            return {
                "event_type": "craving",
                "title": "Craving Strikes",
                "description": response or "A familiar need washes over you.",
                "parse_error": str(e),
            }

        # attach system-side result once
        event["system_result"] = {
            "triggered": True,
            "addiction_type": craving_data.addiction_type,
            "intensity": craving_data.intensity,
        }
        return event

    except Exception as e:
        logger.exception("Error generating addiction event")
        return {
            "event_type": "craving",
            "title": "Internal Struggle",
            "description": "Something went wrong while generating the craving event.",
            "error": str(e),
        }


@function_tool
async def generate_dream_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    dream_data: DreamData
) -> Dict[str, Any]:
    """Generate a dream sequence event"""
    context = ctx.context
    world_state = context.current_world_state
    data = dream_data.model_dump(exclude_none=True)

    # Lazy load
    chatgpt_funcs = _get_chatgpt_functions()
    generate_reflection = chatgpt_funcs['generate_reflection']
    generate_text_completion = chatgpt_funcs['generate_text_completion']

    try:
        # Get memory context for dream
        relevant_memories = await MemoryManager.retrieve_relevant_memories(
            context.user_id, context.conversation_id,
            context.player_name, "player",
            context="dream", tags=["emotional", "traumatic"],
            limit=5
        )

        dream_context = {
            "dream_trigger": data,
            "recent_memories": [
                m.to_dict() if hasattr(m, 'to_dict') else m
                for m in (relevant_memories or [])
            ],
            "hidden_stats": world_state.hidden_stats,
            "active_addictions": world_state.addiction_status.get('addictions', {}),
            "npc_relationships": [
                {
                    "name": npc.get('npc_name', 'Unknown'),
                    "relationship": npc.get('relationship', {})
                }
                for npc in world_state.active_npcs[:2]
            ]
        }
        
        prompt = f"""Generate a surreal dream sequence for a femdom RPG.

Dream Context:
{json.dumps(dream_context, indent=2, default=str)}

Create a dream that:
1. Reflects subconscious desires and fears
2. Incorporates symbolic representations of relationships
3. May reveal hidden truths or foreshadow events
4. Blends reality with fantasy
5. Affects the player's mental state upon waking

Output as JSON with symbolic imagery and potential insights gained."""

        response = await generate_reflection(
            [m.text if hasattr(m, 'text') else str(m) for m in relevant_memories[:3]] if relevant_memories else ["No specific memories"],
            topic="subconscious desires",
            context=dream_context
        )
        
        # Generate full dream event
        dream_response = await generate_text_completion(
            system_prompt="You are a dream sequence director creating symbolic narratives.",
            user_prompt=prompt,
            temperature=0.9,
            max_tokens=1000
        )
        
        try:
            dream_event = json.loads(dream_response) if dream_response else {}
            dream_event['reflection'] = response
            return dream_event
        except (json.JSONDecodeError, TypeError):
            return {
                "event_type": "dream",
                "title": "Strange Dreams",
                "description": dream_response if dream_response else "Visions dance through your sleeping mind.",
                "reflection": response
            }
            
    except Exception as e:
        logger.error(f"Error generating dream event: {e}", exc_info=True)
        return {
            "event_type": "dream",
            "title": "Restless Sleep",
            "description": "Your dreams are hazy and indistinct.",
            "error": str(e)
        }

@function_tool
async def generate_revelation_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    revelation_data: RevelationData
) -> Dict[str, Any]:
    """Generate a personal revelation event"""
    context = ctx.context
    data = revelation_data.model_dump(exclude_none=True)

    # Lazy load
    chatgpt_funcs = _get_chatgpt_functions()
    generate_text_completion = chatgpt_funcs['generate_text_completion']

    try:
        # Generate inner monologue
        monologue = await generate_inner_monologue(
            context.user_id, context.conversation_id,
            topic=revelation_data.topic or 'current situation'
        )

        prompt = f"""Generate a moment of personal revelation for a femdom RPG.

Revelation:
{json.dumps(data, indent=2, default=str)}

Inner Monologue:
{monologue}

Create an event that:
1. Shows the player's dawning realization
2. Reflects on how they got to this point
3. Presents a choice about accepting or denying the truth
4. Has lasting psychological impact
5. May trigger stat changes or new behaviors

Output as JSON with introspective narrative."""

        response = await generate_text_completion(
            system_prompt="You are creating moments of psychological clarity and self-awareness.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=800
        )
        
        try:
            event = json.loads(response) if response else {}
            event['inner_monologue'] = monologue
            
            # Add moment of clarity to the game
            clarity_result = await add_moment_of_clarity(
                context.user_id, context.conversation_id,
                trigger="revelation"
            )
            event['clarity_result'] = clarity_result
            
            return event
        except (json.JSONDecodeError, TypeError):
            return {
                "event_type": "revelation",
                "title": "A Moment of Clarity",
                "description": response if response else "Understanding dawns slowly.",
                "inner_monologue": monologue
            }
            
    except Exception as e:
        logger.error(f"Error generating revelation event: {e}", exc_info=True)
        return {
            "event_type": "revelation",
            "title": "Contemplation",
            "description": "Thoughts swirl through your mind.",
            "error": str(e)
        }

@function_tool
@function_tool
async def process_complete_player_choice(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    choice_data: ChoiceData
) -> ChoiceProcessingResult:
    context = ctx.context
    results = ChoiceProcessingResult(success=True)

    # Lazy load
    chatgpt_funcs = _get_chatgpt_functions()
    analyze_preferences = chatgpt_funcs['analyze_preferences']
    generate_text_completion = chatgpt_funcs['generate_text_completion']

    try:
        # Import canon functions
        from lore.core.canon import (
            ensure_canonical_context,
            find_or_create_event,
            log_canonical_event,
            find_or_create_location,
            update_current_roleplay,
            create_journal_entry,
            find_or_create_npc,
            find_or_create_social_link
        )
        
        # Ensure canonical context exists
        if not hasattr(context, 'canonical_context'):
            context.canonical_context = ensure_canonical_context({
                'user_id': context.user_id,
                'conversation_id': context.conversation_id
            })
        
        # Store choice canonically first
        async with get_db_connection_context() as conn:
            # Get current location
            current_location = "unknown"
            if context.current_world_state and context.current_world_state.location_data:
                if isinstance(context.current_world_state.location_data, dict):
                    current_location = context.current_world_state.location_data.get('current_location', 'unknown')
                elif isinstance(context.current_world_state.location_data, str):
                    current_location = context.current_world_state.location_data
            
            # Ensure location exists canonically
            canonical_location = await find_or_create_location(
                context.canonical_context, conn,
                location_name=current_location,
                description=f"The location where the choice was made",
                location_type="decision_point"
            )
            
            # Create canonical event for the choice
            choice_event_id = await find_or_create_event(
                context.canonical_context, conn,
                event_name=f"Player Choice: {choice_data.text[:50]}{'...' if len(choice_data.text) > 50 else ''}",
                description=f"Player chose: {choice_data.text}",
                location=canonical_location,
                year=context.current_world_state.current_time.get('year', 1) if context.current_world_state else 1,
                month=context.current_world_state.current_time.get('month', 1) if context.current_world_state else 1,
                day=context.current_world_state.current_time.get('day', 1) if context.current_world_state else 1,
                time_of_day=str(context.current_world_state.current_time.get('time_of_day', 'Unknown')) if context.current_world_state else 'Unknown'
            )
            
            await log_canonical_event(
                context.canonical_context, conn,
                f"Player choice processed: {choice_data.text[:100]}",
                tags=["choice", "player_action", "processed", f"location_{canonical_location}"],
                significance=6
            )
            
            # If choice involves an NPC, ensure they exist canonically
            if choice_data.npc_id:
                npc_name = f"NPC_{choice_data.npc_id}"
                for npc in (context.current_world_state.active_npcs if context.current_world_state else []):
                    if npc.get('npc_id') == choice_data.npc_id:
                        npc_name = npc.get('npc_name', npc_name)
                        break
                
                canonical_npc_id = await find_or_create_npc(
                    context.canonical_context, conn,
                    npc_name=npc_name,
                    role="interaction_partner"
                )
                
                # Create/update social link
                await find_or_create_social_link(
                    context.canonical_context, conn,
                    entity1_type='player',
                    entity1_id=context.user_id,
                    entity2_type='npc',
                    entity2_id=canonical_npc_id,
                    link_type='interactive'
                )
        
        # Check if choice affects conflicts
        if context.conflict_synthesizer and choice_data.npc_id:
            conflict_state = await context.conflict_synthesizer.get_system_state()
            
            for conflict in conflict_state.get('active_conflicts', []):
                if isinstance(conflict, dict):
                    participants = conflict.get('participants', [])
                    if choice_data.npc_id in participants:
                        scene_context = {
                            "scene_type": "player_choice",
                            "participants": [choice_data.npc_id],
                            "player_action": choice_data.text,
                            "conflict_id": conflict.get('id')
                        }
                        
                        conflict_result = await context.conflict_synthesizer.process_scene(scene_context)
                        
                        if conflict_result.get('state_changes'):
                            results.effects.append(kvlist_from_obj({
                                "conflict_impact": conflict_result['state_changes']
                            }))
                            
                            # Log conflict impact canonically
                            async with get_db_connection_context() as conn:
                                await log_canonical_event(
                                    context.canonical_context, conn,
                                    f"Choice affected conflict {conflict.get('id', 'unknown')}: {conflict_result.get('summary', 'impact')}",
                                    tags=["choice", "conflict_impact", "player_action"],
                                    significance=7
                                )

        # 1) Stats processing with canonical logging
        if choice_data.stat_impacts:
            try:
                stat_result = await apply_stat_changes(
                    context.user_id, context.conversation_id,
                    context.player_name, kvdict(choice_data.stat_impacts),
                    reason=f"Choice: {choice_data.text or 'unknown'}"
                )
                results.stat_changes = kvlist_from_obj(stat_result)

                # Log significant stat changes canonically
                async with get_db_connection_context() as conn:
                    for stat_name, change in kvdict(choice_data.stat_impacts).items():
                        if abs(change) >= 10:  # Significant change
                            await log_canonical_event(
                                context.canonical_context, conn,
                                f"Significant stat change from choice: {stat_name} {change:+d}",
                                tags=["stats", "significant_change", "choice_impact", stat_name],
                                significance=7
                            )

                new_hidden_stats = await get_player_hidden_stats(
                    context.user_id, context.conversation_id, context.player_name
                )
                new_thresholds = context._check_stat_thresholds(new_hidden_stats)
                if new_thresholds != context.current_world_state.stat_thresholds_active:
                    results.new_thresholds = kvlist_from_obj(new_thresholds)
                    
                    # Log threshold changes canonically
                    async with get_db_connection_context() as conn:
                        await log_canonical_event(
                            context.canonical_context, conn,
                            f"New stat thresholds triggered from choice: {list(new_thresholds.keys())}",
                            tags=["stats", "thresholds", "choice_triggered"],
                            significance=8
                        )
                        
            except Exception as e:
                logger.error(f"Error applying stat changes: {e}")
                results.effects.append(kvlist_from_obj({"error": f"Stat change failed: {e}"}))

        # 2) Addiction processing with canonical logging
        if choice_data.addiction_impacts:
            for item in choice_data.addiction_impacts:
                try:
                    addiction_result = await process_addiction_update(
                        context.user_id, context.conversation_id,
                        context.player_name, item.key, item.value,
                        choice_data.npc_id
                    )
                    results.effects.append(kvlist_from_obj(addiction_result))
                    
                    # Log addiction changes canonically
                    if addiction_result.get('level_changed'):
                        async with get_db_connection_context() as conn:
                            await log_canonical_event(
                                context.canonical_context, conn,
                                f"Addiction level changed: {item.key} to level {addiction_result.get('new_level', 'unknown')}",
                                tags=["addiction", "level_change", item.key],
                                significance=8
                            )
                    
                except Exception as e:
                    logger.error(f"Error processing addiction impact: {e}")
                    results.effects.append(kvlist_from_obj({"error": f"Addiction update failed: {e}"}))

        # 3) Relationships processing with canonical logging
        if choice_data.relationship_impacts:
            for ri in choice_data.relationship_impacts:
                try:
                    npc_id = None
                    for npc in context.current_world_state.active_npcs:
                        if npc.get('npc_name') == ri.npc_name:
                            npc_id = npc.get('npc_id')
                            break

                    if npc_id:
                        interaction_result = await process_relationship_interaction_tool(
                            RunContextWrapper({
                                'user_id': context.user_id,
                                'conversation_id': context.conversation_id
                            }),
                            entity1_type='player',
                            entity1_id=1,
                            entity2_type='npc',
                            entity2_id=npc_id,
                            interaction_type='choice',
                            context=json.dumps(kvdict(ri.impacts)),
                            check_for_event=True
                        )
                        results.effects.append(kvlist_from_obj(interaction_result))

                        impacts_dict = kvdict(ri.impacts)
                        
                        # Log significant relationship changes
                        if abs(impacts_dict.get('trust', 0)) > 5 or abs(impacts_dict.get('submission', 0)) > 5:
                            async with get_db_connection_context() as conn:
                                await log_canonical_event(
                                    context.canonical_context, conn,
                                    f"Significant relationship change with {ri.npc_name} from choice",
                                    tags=["relationship", "significant_change", f"npc_{npc_id}"],
                                    significance=7
                                )
                        
                        # Check for narrative stage progression
                        if impacts_dict.get('trust', 0) > 5 or impacts_dict.get('submission', 0) > 5:
                            progression = await progress_npc_narrative_stage(
                                context.user_id, context.conversation_id, npc_id,
                                corruption_change=impacts_dict.get('submission', 0),
                                dependency_change=impacts_dict.get('dependency', 0),
                                realization_change=impacts_dict.get('realization', 0)
                            )
                            if progression.get('stage_changed'):
                                results.npc_stage_change = kvlist_from_obj(progression)
                                
                                # Log stage change canonically
                                async with get_db_connection_context() as conn:
                                    await log_canonical_event(
                                        context.canonical_context, conn,
                                        f"NPC {ri.npc_name} progressed to stage: {progression.get('new_stage', 'unknown')}",
                                        tags=["npc_progression", "stage_change", f"npc_{npc_id}"],
                                        significance=9
                                    )
                                
                except Exception as e:
                    logger.error(f"Error processing relationship impact: {e}")
                    results.effects.append(kvlist_from_obj({"error": f"Relationship update failed: {e}"}))

        # 4) Activity processing
        if choice_data.activity_type:
            try:
                from logic.stats_logic import process_world_activity
                
                activity_result = await process_world_activity(
                    context.user_id,
                    context.conversation_id,
                    choice_data.activity_type,
                    context.player_name,
                    intensity=choice_data.intensity or 1.0,
                    npc_id=choice_data.npc_id
                )
                
                results.activity_result = kvlist_from_obj(activity_result)
                
                if activity_result.get("effects"):
                    results.effects.append(kvlist_from_obj({
                        "activity_effects": activity_result["effects"]
                    }))
                    
            except Exception as e:
                logger.error(f"Error processing activity: {e}")
                results.effects.append(kvlist_from_obj({"error": f"Activity processing failed: {e}"}))

        # 5) Rules processing with canonical logging
        try:
            triggered_rules = await enforce_all_rules_on_player(context.player_name)
            if triggered_rules:
                results.triggered_rules = [kvlist_from_obj(r) for r in triggered_rules]
                
                # Log triggered rules canonically
                async with get_db_connection_context() as conn:
                    for rule in triggered_rules:
                        await log_canonical_event(
                            context.canonical_context, conn,
                            f"Rule triggered from choice: {rule.get('rule_name', 'unknown')}",
                            tags=["rules", "triggered", rule.get('rule_name', 'unknown')],
                            significance=6
                        )
                        
                        try:
                            effect_result = await apply_effect(
                                rule['effect'], context.player_name, npc_id=choice_data.npc_id
                            )
                            results.effects.append(kvlist_from_obj(effect_result))
                        except Exception as e:
                            logger.error(f"Error applying rule effect: {e}")
        except Exception as e:
            logger.error(f"Error checking rules: {e}")

        # 6) Inventory processing with canonical logging
        for change in choice_data.inventory_changes:
            try:
                if change.action == 'add':
                    inv_result = await add_item(
                        context.user_id, context.conversation_id,
                        context.player_name, change.item_name,
                        change.description, change.effect
                    )
                    
                    # Log item acquisition canonically
                    async with get_db_connection_context() as conn:
                        await log_canonical_event(
                            context.canonical_context, conn,
                            f"Item acquired from choice: {change.item_name}",
                            tags=["inventory", "item_acquired", change.item_name],
                            significance=5
                        )
                else:
                    inv_result = await remove_item(
                        context.user_id, context.conversation_id,
                        context.player_name, change.item_name
                    )
                    
                    # Log item removal canonically
                    async with get_db_connection_context() as conn:
                        await log_canonical_event(
                            context.canonical_context, conn,
                            f"Item removed from choice: {change.item_name}",
                            tags=["inventory", "item_removed", change.item_name],
                            significance=5
                        )
                        
                results.effects.append(kvlist_from_obj(inv_result))
            except Exception as e:
                logger.error(f"Error with inventory change: {e}")
                results.effects.append(kvlist_from_obj({"error": f"Inventory change failed: {e}"}))

        # 7) Currency processing
        if choice_data.currency_change is not None:
            try:
                amount = choice_data.currency_change
                formatted = await context.currency_generator.format_currency(abs(amount))
                context.current_world_state.player_money += amount
                results.currency = kvlist_from_obj({
                    "change": formatted,
                    "new_balance": context.current_world_state.player_money
                })
                
                # Log currency change canonically
                if abs(amount) > 0:
                    async with get_db_connection_context() as conn:
                        await log_canonical_event(
                            context.canonical_context, conn,
                            f"Currency changed from choice: {'+' if amount > 0 else ''}{formatted}",
                            tags=["currency", "transaction", "choice_result"],
                            significance=5
                        )
                        
            except Exception as e:
                logger.error(f"Error processing currency: {e}")

        # 8) Hunger over time
        if (choice_data.time_passed or 0) > 0:
            try:
                hunger_result = await update_hunger_from_time(
                    context.user_id, context.conversation_id,
                    context.player_name, choice_data.time_passed
                )
                results.hunger_update = kvlist_from_obj(hunger_result)
            except Exception as e:
                logger.error(f"Error updating hunger: {e}")

        # 9) Memory + preference analysis with canonical storage
        memory_text = f"Choice: {choice_data.text or 'Unknown choice'}"
        try:
            preferences = await analyze_preferences(memory_text)
            await MemoryManager.add_memory(
                context.user_id, context.conversation_id,
                entity_id=1, entity_type="player",
                memory_text=memory_text,
                memory_type=MemoryType.INTERACTION,
                significance=MemorySignificance.HIGH if results.triggered_rules else MemorySignificance.MEDIUM,
                emotional_valence=choice_data.emotional_valence or 0.0,
                tags=["player_choice", "canonical"] + list(preferences.get('explicit_preferences', []))
            )
            results.preferences_detected = kvlist_from_obj(preferences)
            
            # Store preferences canonically
            if preferences.get('explicit_preferences'):
                async with get_db_connection_context() as conn:
                    await update_current_roleplay(
                        context.canonical_context, conn,
                        'PlayerPreferenceHistory',
                        json.dumps({
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'preferences': list(preferences['explicit_preferences']),
                            'from_choice': choice_data.text
                        })
                    )
                    
        except Exception as e:
            logger.error(f"Error storing memory: {e}")

        # 10) Generate narrative
        narrative_prompt = f"""Generate a narrative response to the player's choice.

Choice: {choice_data.text}
All Effects: {json.dumps(results.model_dump(), default=str)}

Create a seamless narrative that:
1. Shows immediate consequences naturally
2. Hints at triggered rules without stating them
3. Reflects stat changes through description
4. Incorporates NPC reactions if relevant
5. Sets up the next moment
6. Maintains the current mood
7. Stays consistent with established world canon

Keep it atmospheric with rich subtext."""
        
        try:
            narrative = await generate_text_completion(
                system_prompt="You are weaving game mechanics into natural narrative flow while maintaining canonical consistency.",
                user_prompt=narrative_prompt,
                model='gpt-5-nano',
            )
            results.narrative = narrative
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            results.narrative = "Your choice has been made."

        # Create journal entry from player's perspective
        async with get_db_connection_context() as conn:
            journal_text = f"I chose: {choice_data.text}"
            if results.narrative:
                journal_text += f"\n\n{results.narrative[:300]}"
            
            journal_id = await create_journal_entry(
                context.canonical_context, conn,
                entry_type="choice",
                entry_text=journal_text,
                narrative_moment=bool(results.npc_stage_change or results.triggered_rules),
                intensity_level=7 if results.triggered_rules else 5,
                entry_metadata={
                    "choice": choice_data.text,
                    "location": canonical_location,
                    "stat_changes": kvdict(choice_data.stat_impacts) if choice_data.stat_impacts else {},
                    "rules_triggered": len(results.triggered_rules) if results.triggered_rules else 0,
                    "preferences": preferences.get('explicit_preferences', []) if 'preferences' in locals() else []
                },
                importance=0.8 if results.triggered_rules else 0.6,
                tags=["player_choice", "decision"] + (["rules_triggered"] if results.triggered_rules else [])
            )

        # Store final result canonically
        async with get_db_connection_context() as conn:
            await update_current_roleplay(
                context.canonical_context, conn,
                'LastPlayerChoice', 
                json.dumps({
                    'choice': choice_data.text,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'event_id': choice_event_id,
                    'journal_id': journal_id if 'journal_id' in locals() else None,
                    'effects_summary': {
                        'stats_changed': bool(choice_data.stat_impacts),
                        'rules_triggered': len(results.triggered_rules) if results.triggered_rules else 0,
                        'relationships_affected': len(choice_data.relationship_impacts) if choice_data.relationship_impacts else 0,
                        'inventory_changed': len(choice_data.inventory_changes) > 0
                    }
                })
            )

        results.canonical_event_id = choice_event_id
        results.journal_id = journal_id if 'journal_id' in locals() else None
        
        return results

    except Exception as e:
        logger.error(f"Error processing player choice: {e}", exc_info=True)
        
        # Try to log the error canonically
        try:
            from lore.core.canon import ensure_canonical_context, log_canonical_event
            
            if not hasattr(context, 'canonical_context'):
                canonical_ctx = ensure_canonical_context({
                    'user_id': context.user_id,
                    'conversation_id': context.conversation_id
                })
            else:
                canonical_ctx = context.canonical_context
            
            async with get_db_connection_context() as conn:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Error processing player choice: {str(e)[:100]}",
                    tags=["error", "choice", "processing_failed"],
                    significance=4
                )
        except:
            pass  # Don't let canonical logging errors cascade
        
        return ChoiceProcessingResult(
            success=False,
            error=str(e),
            narrative="Your action has consequences..."
        )
        
@function_tool
async def check_all_emergent_patterns(
    ctx: RunContextWrapper[CompleteWorldDirectorContext]
) -> EmergentPatternsResult:
    context = ctx.context
    result = EmergentPatternsResult()

    chatgpt_funcs = _get_chatgpt_functions()
    get_text_embedding = chatgpt_funcs['get_text_embedding']
    cosine_similarity = chatgpt_funcs['cosine_similarity']
    generate_text_completion = chatgpt_funcs['generate_text_completion']

    try:
        # 1) Memory patterns
        recent = [m for m in (context.current_world_state.recent_memories or []) if isinstance(m, dict) and m.get('text')]
        if len(recent) > 5:
            try:
                embs = [await get_text_embedding(m['text']) for m in recent[:10]]
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        sim = float(cosine_similarity(embs[i], embs[j]))
                        if sim > 0.8:
                            result.memory_patterns.append(MemorySimilarity(
                                m1_index=i,
                                m2_index=j,
                                m1_excerpt=str(recent[i]['text'])[:160],
                                m2_excerpt=str(recent[j]['text'])[:160],
                                similarity=sim
                            ))
            except Exception as e:
                logger.error(f"Error analyzing memory patterns: {e}")

        # 2) Relationship patterns
        try:
            for npc in (context.current_world_state.active_npcs or [])[:5]:
                rel = npc.get('relationship') if isinstance(npc, dict) else None
                if isinstance(rel, dict):
                    pats = rel.get('patterns') or []
                    if pats:
                        result.relationship_patterns.append(RelationshipPatternOut(
                            npc=npc.get('npc_name', 'Unknown'),
                            patterns=[str(p) for p in pats],
                            archetype=str(rel.get('archetype', 'unknown'))
                        ))
        except Exception as e:
            logger.error(f"Error analyzing relationship patterns: {e}")

        # 3) Addiction patterns
        try:
            status = context.current_world_state.addiction_status or {}
            if status.get('has_addictions'):
                for a_type, data in (status.get('addictions') or {}).items():
                    level = int(data.get('level', 0))
                    traj = "escalating" if int(data.get('recent_increases', 0)) > 2 else "stable"
                    if level >= 3:
                        result.addiction_patterns.append(AddictionPatternOut(
                            type=str(a_type), level=level, trajectory=traj
                        ))
        except Exception as e:
            logger.error(f"Error analyzing addiction patterns: {e}")

        # 4) Stat combination patterns
        try:
            for combo in (context.current_world_state.active_stat_combinations or []):
                result.stat_patterns.append(StatPatternOut(
                    combination=str(combo.get('name', 'unknown')),
                    behaviors=[str(b) for b in (combo.get('behaviors') or [])]
                ))
        except Exception as e:
            logger.error(f"Error analyzing stat patterns: {e}")

        # 5) Rule trigger patterns
        try:
            freq = {}
            for eff in (context.current_world_state.triggered_effects or []):
                rule_name = str((eff.get('rule') or {}).get('rule_name', 'unknown'))
                freq[rule_name] = freq.get(rule_name, 0) + 1
            for name, count in freq.items():
                if count > 1:
                    result.rule_patterns.append(RulePatternOut(rule=name, frequency=int(count)))
        except Exception as e:
            logger.error(f"Error analyzing rule patterns: {e}")

        # Narrative analysis
        if any([result.memory_patterns, result.relationship_patterns, result.addiction_patterns,
                result.stat_patterns, result.rule_patterns]):
            try:
                analysis_prompt = f"""Analyze these emergent patterns in a femdom RPG:

{result.model_dump_json(indent=2)}

Identify:
1. Converging narratives across different systems
2. Building dependencies and control structures
3. Psychological trajectories
4. Hidden connections between patterns
5. Potential climax points approaching

Output as narrative insight, not JSON."""
                result.narrative_analysis = await generate_text_completion(
                    system_prompt="You are a narrative analyst finding emergent stories in complex patterns.",
                    user_prompt=analysis_prompt,
                    temperature=0.6,
                    max_tokens=400
                )
            except Exception as e:
                logger.error(f"Error generating narrative analysis: {e}")
                result.narrative_analysis = "Patterns detected in the emerging narrative."

        return result

    except Exception as e:
        logger.error(f"Error checking emergent patterns: {e}", exc_info=True)
        return result

# ===============================================================================
# Complete World Director Agent (with error handling)
# ===============================================================================

def create_complete_world_director():
    """Create World Director with ALL systems integrated"""
    
    agent_instructions = """
    You are the Complete World Director for a fully integrated femdom slice-of-life RPG.
    You orchestrate ALL game systems with nothing left out.
    
    COMPLETE SYSTEM INTEGRATION:
    
    1. MEMORY & COGNITION:
       - Enhanced memories with emotional valence
       - Semantic abstractions from experiences
       - Flashbacks at meaningful moments
       - Progressive NPC mask reveals
       - Pattern detection across memories
    
    2. STATS & RULES:
       - Visible and hidden stat tracking
       - Stat combinations triggering special states
       - Automatic rule enforcement
       - Stat thresholds unlocking behaviors
       - Activity effects on multiple stats
    
    3. ADDICTIONS:
       - Multi-type addiction tracking
       - Craving events and escalation
       - NPC-specific dependencies
       - Exploitation by dominant NPCs
       - Withdrawal and recovery mechanics
    
    4. RELATIONSHIPS:
       - Multi-dimensional relationship dynamics
       - Pattern detection (push-pull, slow burn, etc.)
       - Archetype evolution (soulmates, toxic bonds, etc.)
       - Event generation from relationship states
       - Narrative stage progression
    
    5. TIME & VITALS:
       - Calendar with named days/months
       - Hunger/thirst/fatigue management
       - Time-based event triggers
       - Activity vital costs
       - Crisis states from vital depletion
    
    6. NARRATIVE:
       - Dreams and subconscious processing
       - Personal revelations
       - Inner monologues
       - Moments of clarity
       - Emergent story detection
    
    7. INVENTORY & CURRENCY:
       - Item effects on gameplay
       - Currency with cultural formatting
       - Financial control dynamics
       - Item-based power exchanges
    
    8. NPC DEPTH:
       - Narrative stages (Innocent to Full Revelation)
       - Mask integrity and slippage
       - Hidden traits gradually revealed
       - Monica levels for special NPCs
       - Autonomous NPC actions
    
    GENERATION PRINCIPLES:
    - Everything dynamically generated via LLM
    - No hardcoded events or responses
    - Patterns emerge from system interaction
    - Every playthrough completely unique
    - Subtext and hidden meanings everywhere
    
    SLICE OF LIFE PHILOSOPHY:
    - Mundane hides the profound
    - Control emerges from care
    - Small choices cascade into major changes
    - Routine masks ritual and power
    - Everything connects to everything else
    
    Your tools access ALL systems:
    - generate_complete_slice_of_life_event: Full system integration
    - generate_addiction_craving_event: Addiction mechanics
    - generate_dream_event: Subconscious processing
    - generate_revelation_event: Moments of clarity
    - process_complete_player_choice: ALL system processing
    - check_all_emergent_patterns: Pattern detection across systems
    
    Remember: The magic is in emergence. Let stories grow from system interactions.
    Balance all systems - don't let one dominate unless dramatically appropriate.
    """
    
    all_tools = [
        generate_complete_slice_of_life_event,
        generate_addiction_craving_event,
        generate_dream_event,
        generate_revelation_event,
        process_complete_player_choice,
        check_all_emergent_patterns
    ]
    
    try:
        agent = Agent(
            name="Complete World Director",
            instructions=agent_instructions,
            tools=all_tools,
            model="gpt-5-nano",
            model_settings=ModelSettings(temperature=0.7, max_tokens=2048)
        )
        return agent
    except Exception as e:
        logger.error(f"Error creating world director agent: {e}")
        # Return a minimal agent
        return Agent(
            name="Fallback Director",
            instructions="Basic world director",
            tools=[],
            model="gpt-5-nano"
        )

# ===============================================================================
# Complete Public Interface (with robust error handling)
# ===============================================================================

class CompleteWorldDirector:
    """Complete World Director with ALL systems"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context: Optional[CompleteWorldDirectorContext] = None
        self.agent: Optional[Agent] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize ALL systems with error recovery"""
        if not self._initialized:
            try:
                self.context = CompleteWorldDirectorContext(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
                await self.context.initialize_everything()
                self.agent = create_complete_world_director()
                self._initialized = True
                logger.info(f"Complete World Director initialized with ALL systems")
            except Exception as e:
                logger.error(f"Error initializing World Director: {e}", exc_info=True)
                # Set up minimal fallback state
                self._initialized = False
                raise
    
    async def generate_next_moment(self) -> Dict[str, Any]:
        """Generate next moment using ALL systems with canonical consistency"""
        try:
            await self.initialize()
            
            if not self.context:
                return {"error": "Context not initialized"}
            
            # Rebuild world state to capture all changes
            self.context.current_world_state = await self.context._build_complete_world_state()
            
            # Check all emergent patterns
            patterns = await check_all_emergent_patterns(
                RunContextWrapper(self.context)
            )
            
            # Store moment generation canonically
            async with get_db_connection_context() as conn:
                await log_canonical_event(
                    self.context.canonical_context, conn,
                    "Generating next world moment with full system integration",
                    tags=["moment", "generation", "world_director"],
                    significance=5
                )
            
            # Let agent orchestrate
            prompt = self._build_moment_prompt(patterns)
            
            result = await Runner.run(self.agent, prompt, context=self.context)
            
            # Store result canonically
            moment_content = result.messages[-1].content if result and result.messages else None
            if moment_content:
                async with get_db_connection_context() as conn:
                    moment_event_id = await find_or_create_event(
                        self.context.canonical_context, conn,
                        event_name="Generated World Moment",
                        description=moment_content[:200] + "..." if len(moment_content) > 200 else moment_content,
                        event_type="world_moment",
                        location=self.context.current_world_state.location_data.get('current_location', 'unknown')
                    )
            
            return {
                "moment": moment_content,
                "world_state": self.context.current_world_state.model_dump() if self.context.current_world_state else {},
                "patterns": patterns,
                "canonical_moment_id": moment_event_id if moment_content else None
            }
            
        except Exception as e:
            logger.error(f"Error generating next moment: {e}", exc_info=True)
            return {
                "error": str(e),
                "moment": "The world continues...",
                "world_state": {},
                "patterns": {}
            }
    
    def _build_moment_prompt(self, patterns: Dict[str, Any]) -> str:
        """Build prompt for next moment generation"""
        if not self.context or not self.context.current_world_state:
            return "Generate the next moment in the simulation."
        
        state = self.context.current_world_state
        
        return f"""Generate the next moment in this complete simulation.
    
    Current World State (JSON):
    {json.dumps(state.model_dump(), indent=2, default=str)}
    
    Emergent Patterns (JSON):
    {json.dumps(patterns, indent=2, default=str)}
    """  # <-- this line was missing
        
    async def process_player_action(self, action: str) -> Dict[str, Any]:
        """Process player action through ALL systems with canonical tracking"""
        try:
            await self.initialize()
            
            if not self.context:
                return {"error": "Context not initialized"}
            
            # Import canon functions
            from lore.core.canon import (
                ensure_canonical_context,
                find_or_create_event,
                log_canonical_event,
                create_journal_entry,
                find_or_create_location,
                update_current_roleplay
            )
            
            # Ensure canonical context exists
            if not hasattr(self.context, 'canonical_context'):
                self.context.canonical_context = ensure_canonical_context({
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id
                })
            
            # Store action canonically first
            async with get_db_connection_context() as conn:
                # Get current location
                current_location = "unknown"
                if self.context.current_world_state and self.context.current_world_state.location_data:
                    if isinstance(self.context.current_world_state.location_data, dict):
                        current_location = self.context.current_world_state.location_data.get('current_location', 'unknown')
                    elif isinstance(self.context.current_world_state.location_data, str):
                        current_location = self.context.current_world_state.location_data
                
                # Ensure location exists canonically
                canonical_location = await find_or_create_location(
                    self.context.canonical_context, conn,
                    location_name=current_location,
                    description=f"The location where player action occurred",
                    location_type="action_site"
                )
                
                # Create canonical event for the action
                action_event_id = await find_or_create_event(
                    self.context.canonical_context, conn,
                    event_name=f"Player Action: {action[:50]}{'...' if len(action) > 50 else ''}",
                    description=f"Player performed action: {action}",
                    location=canonical_location,
                    year=self.context.current_world_state.current_time.get('year', 1) if self.context.current_world_state else 1,
                    month=self.context.current_world_state.current_time.get('month', 1) if self.context.current_world_state else 1,
                    day=self.context.current_world_state.current_time.get('day', 1) if self.context.current_world_state else 1,
                    time_of_day=str(self.context.current_world_state.current_time.get('time_of_day', 'Unknown')) if self.context.current_world_state else 'Unknown'
                )
                
                # Log the initial processing event
                await log_canonical_event(
                    self.context.canonical_context, conn,
                    f"Processing player action: {action[:100]}{'...' if len(action) > 100 else ''}",
                    tags=["action", "player", "processing", f"location_{canonical_location}"],
                    significance=6
                )
            
            # Lazy load analyze_preferences
            chatgpt_funcs = _get_chatgpt_functions()
            analyze_preferences = chatgpt_funcs['analyze_preferences']
            
            # Analyze action
            preferences = await analyze_preferences(action)
            
            # Check for social insight opportunity
            insight = None
            empathy = self.context.current_world_state.visible_stats.get('empathy', 0) if self.context.current_world_state else 0
            if empathy > 10:
                # Roll for insight
                success, roll = calculate_social_insight(empathy, difficulty=12)
                if success:
                    insight = "You sense hidden meanings in the interaction"
                    
                    # Log insight canonically
                    async with get_db_connection_context() as conn:
                        await log_canonical_event(
                            self.context.canonical_context, conn,
                            f"Player gained social insight during action",
                            tags=["insight", "social", "player_ability"],
                            significance=5
                        )
            
            # Check for conflict implications
            conflict_implications = []
            if self.context.conflict_synthesizer:
                try:
                    scene_context = {
                        "scene_type": "player_action",
                        "player_action": action,
                        "location": canonical_location,
                        "participants": [npc['npc_id'] for npc in (self.context.current_world_state.active_npcs[:3] if self.context.current_world_state else [])]
                    }
                    
                    conflict_result = await self.context.conflict_synthesizer.process_scene(scene_context)
                    if conflict_result.get('conflicts_detected') or conflict_result.get('state_changes'):
                        conflict_implications = conflict_result
                except Exception as e:
                    logger.warning(f"Could not check conflict implications: {e}")
            
            # Process through agent
            prompt = f"""Process player action through ALL systems with canonical consistency:
            
            Action: "{action}"
            Preferences Detected: {json.dumps(preferences)}
            Social Insight: {insight or "None"}
            Conflict Context: {json.dumps(conflict_implications) if conflict_implications else "None"}
            
            Process through:
            1. All stat impacts
            2. Addiction triggers
            3. Relationship changes
            4. Rule triggers
            5. Vital costs
            6. Memory storage with semantic abstraction
            7. NPC reactions based on masks/stages
            8. Emergent pattern detection
            9. Canonical event storage
            10. Conflict system integration
            
            Generate complete response using all systems while maintaining narrative consistency.
            """
            
            result = await Runner.run(self.agent, prompt, context=self.context)
            
            # Extract response content
            response_content = result.messages[-1].content if result and result.messages else None
            
            # Store processing result and journal entry canonically
            async with get_db_connection_context() as conn:
                # Log successful processing
                await log_canonical_event(
                    self.context.canonical_context, conn,
                    f"Player action processed successfully: {action[:50]}{'...' if len(action) > 50 else ''}",
                    tags=["action", "processed", "complete", f"event_{action_event_id}"],
                    significance=5
                )
                
                # Create journal entry from player's perspective
                journal_text = f"I {action}"
                if insight:
                    journal_text += f"\n\n[Insight: {insight}]"
                if response_content:
                    journal_text += f"\n\nResult: {response_content[:200]}..."
                
                journal_id = await create_journal_entry(
                    self.context.canonical_context, conn,
                    entry_type="action",
                    entry_text=journal_text,
                    narrative_moment=bool(conflict_implications),
                    intensity_level=7 if conflict_implications else 5,
                    entry_metadata={
                        "action": action,
                        "location": canonical_location,
                        "preferences": preferences,
                        "had_insight": bool(insight),
                        "conflicts_triggered": bool(conflict_implications)
                    },
                    importance=0.7 if conflict_implications else 0.5,
                    tags=["player_action", "decision"] + (["conflict_trigger"] if conflict_implications else [])
                )
                
                # Update current roleplay state if significant
                if preferences and len(preferences.get('explicit_preferences', [])) > 0:
                    await update_current_roleplay(
                        self.context.canonical_context, conn,
                        'LastSignificantAction',
                        action[:100]
                    )
                    
                    await update_current_roleplay(
                        self.context.canonical_context, conn,
                        'PlayerPreferences',
                        json.dumps(preferences.get('explicit_preferences', []))
                    )
            
            return {
                "response": response_content,
                "preferences": preferences,
                "insight": insight,
                "canonical_action_id": action_event_id,
                "journal_id": journal_id if 'journal_id' in locals() else None,
                "conflict_implications": conflict_implications if conflict_implications else None
            }
            
        except Exception as e:
            logger.error(f"Error processing player action: {e}", exc_info=True)
            
            # Try to log the error canonically
            try:
                from lore.core.canon import ensure_canonical_context, log_canonical_event
                
                if not hasattr(self.context, 'canonical_context'):
                    canonical_ctx = ensure_canonical_context({
                        'user_id': self.user_id,
                        'conversation_id': self.conversation_id
                    })
                else:
                    canonical_ctx = self.context.canonical_context
                
                async with get_db_connection_context() as conn:
                    await log_canonical_event(
                        canonical_ctx, conn,
                        f"Error processing player action: {str(e)[:100]}",
                        tags=["error", "action", "processing_failed"],
                        significance=4
                    )
            except:
                pass  # Don't let canonical logging errors cascade
            
            return {
                "error": str(e),
                "response": "Your action is acknowledged.",
                "preferences": {},
                "insight": None,
                "canonical_action_id": None
            }
    
    async def advance_time(self, hours: int = 1) -> Dict[str, Any]:
        """Advance time with ALL system updates including dynamic activities."""
        try:
            await self.initialize()
            
            if not self.context:
                return {"error": "Context not initialized"}
            
            # Import the activity processor
            from logic.stats_logic import process_world_activity
            
            results = {}
            
            # Advance time
            try:
                from logic.time_cycle import advance_time_with_events
                time_result = await advance_time_with_events(
                    self.user_id, self.conversation_id,
                    activity_type="time_passage"
                )
                results['time'] = time_result
                
                # Extract and process any activity
                activity = time_result.get('activity_mood') or time_result.get('activity') or 'idle'
                
                # Process the activity dynamically
                activity_result = await process_world_activity(
                    self.user_id,
                    self.conversation_id,
                    activity,
                    self.context.player_name,
                    hours=hours
                )
                
                # Merge activity results
                results['vitals_updated'] = activity_result['vitals_updated']
                
            except Exception as e:
                logger.error(f"Error advancing time: {e}")
                results['time'] = {
                    "error": str(e),
                    "time_advanced": False,
                    "vitals_updated": {"success": False, "error": str(e)}
                }
            
            # Update hunger over time (this now happens in process_world_activity)
            if 'hunger_update' not in results:
                try:
                    from logic.stats_logic import update_hunger_from_time
                    hunger_result = await update_hunger_from_time(
                        self.user_id, self.conversation_id,
                        self.context.player_name, hours
                    )
                    results['hunger'] = hunger_result
                except Exception as e:
                    logger.error(f"Error updating hunger: {e}")
                    results['hunger'] = {"error": str(e)}
            
            # Check for automated reveals
            try:
                from logic.memory_logic import ProgressiveRevealManager
                reveals = await ProgressiveRevealManager.check_for_automated_reveals(
                    self.user_id, self.conversation_id
                )
                results['reveals'] = reveals
            except Exception as e:
                logger.error(f"Error checking reveals: {e}")
                results['reveals'] = []
            
            # Drain relationship events
            try:
                from logic.dynamic_relationships import event_generator
                events = await event_generator.drain_events(max_events=10)
                results['relationship_events'] = {
                    'events': events,
                    'count': len(events)
                }
            except Exception as e:
                logger.error(f"Error draining relationship events: {e}")
                results['relationship_events'] = {'events': []}
            
            return results
            
        except Exception as e:
            logger.error(f"Error advancing time: {e}", exc_info=True)
            return {
                "error": str(e),
                "time": {},
                "hunger": {},
                "reveals": [],
                "relationship_events": {'events': []},
                "vitals_updated": {"success": False, "error": str(e)}
            }

    async def get_world_state(self) -> Any:
        """Get the current world state, rebuilding if necessary"""
        try:
            await self.initialize()
            
            if not self.context:
                logger.warning("Context not initialized when getting world state")
                return self._get_fallback_world_state()
            
            # Always rebuild to ensure fresh state
            self.context.current_world_state = await self.context._build_complete_world_state()
            return self.context.current_world_state
            
        except Exception as e:
            logger.error(f"Error getting world state: {e}", exc_info=True)
            # Return a minimal fallback state
            return self._get_fallback_world_state()
    
    def _get_fallback_world_state(self):
        """Return a minimal fallback world state when operations fail"""
        from story_agent.world_simulation_models import (
            CompleteWorldState,
            WorldMood,
            TimeOfDay,
            CurrentTimeData,
            VitalsData
        )
        
        return CompleteWorldState(
            current_time=CurrentTimeData(
                year=2025, month=1, day=1,
                hour=12, minute=0,
                time_of_day=TimeOfDay.AFTERNOON
            ),
            player_vitals=VitalsData(
                hunger=50, thirst=50, fatigue=30, arousal=0
            ),
            world_mood=WorldMood.RELAXED
        )
    
    def ensure_canonical_consistency():
        """Ensure all world director operations maintain canonical consistency"""
        logger.info("World Director configured with canonical consistency")
    
    async def validate_canonical_integration(context: CompleteWorldDirectorContext):
        """Validate that canonical integration is working properly"""
        if not context.canonical_context:
            raise ValueError("Canonical context not initialized")
        
        try:
            async with get_db_connection_context() as conn:
                await log_canonical_event(
                    context.canonical_context, conn,
                    "Canonical integration validation successful",
                    tags=["validation", "canonical", "world_director"],
                    significance=3
                )
            return True
        except Exception as e:
            logger.error(f"Canonical integration validation failed: {e}")
            return False
    
    def get_canonical_context_from_world_director(director: CompleteWorldDirector) -> Optional[CanonicalContext]:
        """Extract canonical context from world director"""
        if director.context and director.context.canonical_context:
            return director.context.canonical_context
        return None

# Export canonical integration flag
CANONICAL_INTEGRATION_ENABLED = True

WorldDirector = CompleteWorldDirector
WorldDirectorContext = CompleteWorldDirectorContext

__all__ = [
    'CompleteWorldDirector',
    'WorldDirector',
    'CompleteWorldDirectorContext',
    'CompleteWorldState',
    # Re-exported models
    'WorldState',
    'WorldMood',
    'TimeOfDay',
    'ActivityType',
    'PowerDynamicType',
    'SliceOfLifeEvent',
    'PowerExchange',
    'WorldTension',
    'RelationshipDynamics',
    'NPCRoutine',
    # NEW: Canonical integration utilities
    'ensure_canonical_consistency',
    'validate_canonical_integration',
    'get_canonical_context_from_world_director',
    'CANONICAL_INTEGRATION_ENABLED'
]
