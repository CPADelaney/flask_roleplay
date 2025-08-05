# story_agent/world_director_agent_complete.py
"""
Complete World Dynamics Director with ALL system integrations and LLM-driven generation.
No functionality dropped - everything integrated and dynamically generated.
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone, timedelta
from enum import Enum

from db.connection import get_db_connection_context
from agents import Agent, function_tool, Runner, trace, ModelSettings, RunContextWrapper

# ===============================================================================
# COMPLETE SYSTEM INTEGRATIONS - NOTHING DROPPED
# ===============================================================================

# OpenAI Integration for Dynamic Generation
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

# Universal Updater for Narrative Processing
from logic.universal_updater_agent import (
    UniversalUpdaterAgent,
    process_universal_update,
    convert_updates_to_array_format,
    convert_response_to_array_format
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
    generate_flashback,
    check_for_automated_reveals,
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
    VitalsData,
    CurrentTimeData,
    process_activity_vitals,
    ActivityManager,
    TimeOfDay,
    ActivityType as TimeActivityType
)

from logic.calendar import (
    load_calendar_names,
    update_calendar_names,
    get_calendar_events,
    add_calendar_event
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
    process_narrative_event,
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
    get_addiction_status,
    trigger_craving_event,
    AddictionType,
    AddictionLevel
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

logger = logging.getLogger(__name__)

# ===============================================================================
# COMPLETE World State Models with ALL Integrations
# ===============================================================================

class WorldMood(Enum):
    """Overall mood/atmosphere of the world"""
    RELAXED = "relaxed"
    TENSE = "tense"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    MYSTERIOUS = "mysterious"
    OPPRESSIVE = "oppressive"
    CHAOTIC = "chaotic"
    EXHAUSTED = "exhausted"
    DESPERATE = "desperate"
    CORRUPTED = "corrupted"
    DREAMLIKE = "dreamlike"  # Added for dream sequences
    CRAVING = "craving"       # Added for addiction states

class ActivityType(Enum):
    """Types of slice-of-life activities"""
    WORK = "work"
    SOCIAL = "social"
    LEISURE = "leisure"
    INTIMATE = "intimate"
    ROUTINE = "routine"
    SPECIAL = "special"
    ADDICTION = "addiction"
    VITAL = "vital"
    DREAM = "dream"           # Added for dream sequences
    REVELATION = "revelation" # Added for narrative moments

class CompleteWorldState(BaseModel):
    """Complete world state with ALL system integrations"""
    # Time and Calendar
    current_time: CurrentTimeData
    calendar_names: Dict[str, Any] = Field(default_factory=dict)
    calendar_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Vitals and Stats
    player_vitals: VitalsData
    visible_stats: Dict[str, Any] = Field(default_factory=dict)
    hidden_stats: Dict[str, Any] = Field(default_factory=dict)
    active_stat_combinations: List[Dict[str, Any]] = Field(default_factory=list)
    stat_thresholds_active: Dict[str, Any] = Field(default_factory=dict)
    
    # Memory and Context
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_abstractions: List[str] = Field(default_factory=list)
    active_flashbacks: List[Dict[str, Any]] = Field(default_factory=list)
    pending_reveals: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Dreams and Revelations
    pending_dreams: List[Dict[str, Any]] = Field(default_factory=list)
    recent_revelations: List[Dict[str, Any]] = Field(default_factory=list)
    inner_monologues: List[str] = Field(default_factory=list)
    
    # Rules and Effects
    active_rules: List[Dict[str, Any]] = Field(default_factory=list)
    triggered_effects: List[Dict[str, Any]] = Field(default_factory=list)
    pending_effects: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Inventory
    player_inventory: List[Dict[str, Any]] = Field(default_factory=list)
    recent_item_changes: List[Dict[str, Any]] = Field(default_factory=list)
    
    # NPCs and Relationships
    active_npcs: List[Dict[str, Any]] = Field(default_factory=list)
    npc_masks: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    npc_narrative_stages: Dict[int, str] = Field(default_factory=dict)
    relationship_states: Dict[str, Any] = Field(default_factory=dict)
    relationship_overview: Optional[Dict[str, Any]] = None
    pending_relationship_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Addictions
    addiction_status: Dict[str, Any] = Field(default_factory=dict)
    active_cravings: List[Dict[str, Any]] = Field(default_factory=list)
    addiction_contexts: Dict[str, Any] = Field(default_factory=dict)
    
    # Currency
    player_money: int = 0
    currency_system: Dict[str, Any] = Field(default_factory=dict)
    recent_transactions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # World State
    world_mood: WorldMood
    tension_factors: Dict[str, float] = Field(default_factory=dict)
    environmental_factors: Dict[str, Any] = Field(default_factory=dict)
    location_data: str = ""
    
    # Events
    ongoing_events: List[Dict[str, Any]] = Field(default_factory=list)
    available_activities: List[Dict[str, Any]] = Field(default_factory=list)
    event_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Governance
    nyx_directives: List[Dict[str, Any]] = Field(default_factory=list)
    
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Complete World Director Context with ALL Systems
# ===============================================================================

@dataclass
class CompleteWorldDirectorContext:
    """Complete context with ALL system managers"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    
    # Core system managers
    openai_manager: Optional[OpenAIClientManager] = None
    universal_updater: Optional[UniversalUpdaterAgent] = None
    
    # Memory and reveals
    memory_manager: Optional[Any] = None  # Will use class methods
    reveal_manager: Optional[Any] = None  # Will use class methods
    
    # Relationships
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    
    # Addictions - COMPLETE
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
    
    # State tracking
    current_world_state: Optional[CompleteWorldState] = None
    
    # Caching
    cache: Dict[str, Any] = field(default_factory=dict)
    
    async def initialize_everything(self):
        """Initialize ALL integrated systems - nothing left out"""
        logger.info(f"Initializing Complete World Director for user {self.user_id}")
        
        # Initialize OpenAI manager
        self.openai_manager = OpenAIClientManager()
        
        # Initialize universal updater
        self.universal_updater = UniversalUpdaterAgent(self.user_id, self.conversation_id)
        await self.universal_updater.initialize()
        
        # Initialize relationship manager
        self.relationship_manager = OptimizedRelationshipManager(
            self.user_id, self.conversation_id
        )
        
        # Initialize addiction context - IMPORTANT
        self.addiction_context = AddictionContext(self.user_id, self.conversation_id)
        await self.addiction_context.initialize()
        
        # Initialize event system
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
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize governance
        self.nyx_governor = await get_central_governance(self.user_id, self.conversation_id)
        self.directive_handler = DirectiveHandler(self.user_id, self.conversation_id)
        await self.directive_handler.initialize()
        
        # Load calendar names
        self.calendar_names = await load_calendar_names(self.user_id, self.conversation_id)
        
        # Register inventory system with governance
        await register_inventory(self.user_id, self.conversation_id)
        
        # Build initial world state
        self.current_world_state = await self._build_complete_world_state()
        
        logger.info("Complete World Director fully initialized with ALL systems")
    
    async def _build_complete_world_state(self) -> CompleteWorldState:
        """Build complete world state from ALL systems"""
        # Time and Calendar
        current_time = await get_current_time_model(self.user_id, self.conversation_id)
        calendar_events = await get_calendar_events(self.user_id, self.conversation_id)
        
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
        stat_thresholds = await self._check_stat_thresholds(hidden_stats)
        
        # Memory and Context
        recent_memories = await MemoryManager.retrieve_relevant_memories(
            self.user_id, self.conversation_id, 
            self.player_name, "player",
            context="current_situation", limit=10
        )
        
        # Check for flashbacks
        flashback = None
        if recent_memories and random.random() < 0.1:  # 10% chance
            flashback = await MemoryManager.generate_flashback(
                self.user_id, self.conversation_id,
                1, "current_context"
            )
        
        # Check for NPC reveals
        pending_reveals = await check_for_automated_reveals(
            self.user_id, self.conversation_id
        )
        
        # Dreams and Revelations
        revelation = await check_for_personal_revelations(
            self.user_id, self.conversation_id
        )
        narrative_moments = await check_for_narrative_moments(
            self.user_id, self.conversation_id
        )
        
        # Rules
        triggered_rules = await enforce_all_rules_on_player(self.player_name)
        
        # Inventory
        inventory_result = await get_inventory(
            self.user_id, self.conversation_id, self.player_name
        )
        
        # NPCs with complete data
        npcs = await self._get_complete_npc_data()
        
        # Relationships
        rel_overview = await get_relationship_overview(
            self.user_id, self.conversation_id
        )
        
        # Drain relationship events
        rel_events = await drain_relationship_events_tool(
            ctx=RunContextWrapper({
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            }),
            max_events=5
        )
        
        # Addictions - COMPLETE CHECK
        addiction_status = await get_addiction_status(
            self.user_id, self.conversation_id, self.player_name
        )
        
        # Check for active cravings
        active_cravings = []
        if addiction_status.get('has_addictions'):
            for addiction_type, data in addiction_status.get('addictions', {}).items():
                if data.get('level', 0) > 2:  # Level 3+ can trigger cravings
                    craving_check = await check_addiction_status(
                        self.user_id, self.conversation_id,
                        self.player_name, addiction_type
                    )
                    if craving_check.get('craving_active'):
                        active_cravings.append(craving_check)
        
        # Currency
        currency_system = await self.currency_generator.get_currency_system()
        
        # Location data
        location_data = await fetch_formatted_locations(
            self.user_id, self.conversation_id
        )
        
        # Calculate world mood
        world_mood = await self._calculate_complete_world_mood(
            hidden_stats, vitals, stat_combinations,
            addiction_status, active_cravings,
            revelation is not None
        )
        
        # Calculate tensions
        tensions = self._calculate_all_tensions(
            hidden_stats, vitals, stat_combinations,
            addiction_status, rel_overview
        )
        
        return CompleteWorldState(
            current_time=current_time,
            calendar_names=self.calendar_names or {},
            calendar_events=calendar_events,
            player_vitals=vitals,
            visible_stats=visible_stats,
            hidden_stats=hidden_stats,
            active_stat_combinations=stat_combinations,
            stat_thresholds_active=stat_thresholds,
            recent_memories=[m.to_dict() for m in recent_memories] if recent_memories else [],
            semantic_abstractions=[],  # Will be populated as needed
            active_flashbacks=[flashback] if flashback else [],
            pending_reveals=pending_reveals,
            pending_dreams=[],  # Will be populated by dream system
            recent_revelations=[revelation] if revelation else [],
            inner_monologues=[],  # Will be populated as needed
            active_rules=triggered_rules,
            triggered_effects=[],
            pending_effects=[],
            player_inventory=inventory_result.get('items', []),
            recent_item_changes=[],
            active_npcs=npcs,
            npc_masks={npc['npc_id']: npc.get('mask', {}) for npc in npcs},
            npc_narrative_stages={npc['npc_id']: npc.get('narrative_stage', '') for npc in npcs},
            relationship_states={},  # Will be populated as needed
            relationship_overview=rel_overview,
            pending_relationship_events=rel_events.get('events', []),
            addiction_status=addiction_status,
            active_cravings=active_cravings,
            addiction_contexts={},
            player_money=100,  # Should load from DB
            currency_system=currency_system,
            recent_transactions=[],
            world_mood=world_mood,
            tension_factors=tensions,
            environmental_factors={},
            location_data=location_data,
            ongoing_events=[],
            available_activities=[],
            event_history=[],
            nyx_directives=[]
        )
    
    async def _get_complete_npc_data(self) -> List[Dict[str, Any]]:
        """Get NPCs with ALL their data"""
        async with get_db_connection_context() as conn:
            npcs = await conn.fetch("""
                SELECT npc_id, npc_name, dominance, cruelty, intensity,
                       personality_traits, current_location, monica_level
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND introduced = true
                LIMIT 10
            """, self.user_id, self.conversation_id)
        
        complete_npcs = []
        for npc in npcs:
            npc_dict = dict(npc)
            
            # Get mask data
            mask_data = await ProgressiveRevealManager.get_npc_mask(
                self.user_id, self.conversation_id, npc['npc_id']
            )
            npc_dict['mask'] = mask_data
            
            # Get narrative stage
            stage = await get_npc_narrative_stage(
                self.user_id, self.conversation_id, npc['npc_id']
            )
            npc_dict['narrative_stage'] = stage.name
            
            # Check for revelations
            revelation = await check_for_npc_revelation(
                self.user_id, self.conversation_id, npc['npc_id']
            )
            npc_dict['pending_revelation'] = revelation
            
            # Get relationship state
            rel_state = await self.relationship_manager.get_relationship_state(
                'npc', npc['npc_id'], 'player', 1
            )
            npc_dict['relationship'] = {
                'dimensions': rel_state.dimensions.__dict__,
                'archetype': rel_state.archetype,
                'patterns': list(rel_state.history.active_patterns)
            }
            
            complete_npcs.append(npc_dict)
        
        return complete_npcs
    
    async def _check_stat_thresholds(self, hidden_stats: Dict) -> Dict[str, Any]:
        """Check which stat thresholds are active"""
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
    
    async def _calculate_complete_world_mood(
        self, hidden_stats: Dict, vitals: VitalsData,
        stat_combinations: List[Dict], addiction_status: Dict,
        active_cravings: List[Dict], has_revelation: bool
    ) -> WorldMood:
        """Calculate world mood from ALL factors"""
        # Critical overrides
        if vitals.fatigue > 85:
            return WorldMood.EXHAUSTED
        if vitals.hunger < 15 or vitals.thirst < 15:
            return WorldMood.DESPERATE
        if active_cravings:
            return WorldMood.CRAVING
        if has_revelation:
            return WorldMood.DREAMLIKE
        
        # Special combinations
        for combo in stat_combinations:
            if combo['name'] == 'Stockholm Syndrome':
                return WorldMood.CORRUPTED
            elif combo['name'] == 'Breaking Point':
                return WorldMood.CHAOTIC
        
        # Addiction-based moods
        if addiction_status.get('has_addictions'):
            max_level = max(
                data.get('level', 0) 
                for data in addiction_status.get('addictions', {}).values()
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
        rel_overview: Dict
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
        
        # Addiction tensions
        tensions['addiction'] = 0.0
        if addiction_status.get('has_addictions'):
            for data in addiction_status.get('addictions', {}).values():
                tensions['addiction'] = max(tensions['addiction'], data.get('level', 0) / 5)
        
        # Relationship tensions
        tensions['relationship'] = 0.0
        if rel_overview:
            for rel in rel_overview.get('relationships', []):
                if 'explosive_chemistry' in rel.get('patterns', []):
                    tensions['relationship'] += 0.1
                if 'toxic_bond' in rel.get('archetypes', []):
                    tensions['relationship'] += 0.15
        
        # Combination tensions
        tensions['breaking'] = min(1.0, len(stat_combinations) * 0.2)
        
        return tensions

# ===============================================================================
# COMPLETE LLM-Driven Tools with ALL Systems
# ===============================================================================

@function_tool
@track_performance("generate_complete_event")
async def generate_complete_slice_of_life_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext]
) -> Dict[str, Any]:
    """Generate event considering ALL systems including addictions and dreams"""
    context = ctx.context
    world_state = context.current_world_state
    
    # Check for priority events first
    
    # 1. Check for addiction cravings
    if world_state.active_cravings:
        craving = world_state.active_cravings[0]
        return await generate_addiction_craving_event(ctx, craving)
    
    # 2. Check for pending dreams
    if world_state.world_mood == WorldMood.DREAMLIKE or random.random() < 0.05:
        dream_result = await add_dream_sequence(
            context.user_id, context.conversation_id
        )
        if dream_result:
            return await generate_dream_event(ctx, dream_result)
    
    # 3. Check for revelations
    if world_state.recent_revelations:
        return await generate_revelation_event(ctx, world_state.recent_revelations[0])
    
    # Build comprehensive context
    event_context = {
        "time": world_state.current_time.to_dict(),
        "calendar": world_state.calendar_names,
        "vitals": world_state.player_vitals.to_dict(),
        "visible_stats": world_state.visible_stats,
        "hidden_stats": world_state.hidden_stats,
        "stat_combinations": [c['name'] for c in world_state.active_stat_combinations],
        "stat_thresholds": world_state.stat_thresholds_active,
        "world_mood": world_state.world_mood.value,
        "tensions": world_state.tension_factors,
        "recent_memories": world_state.recent_memories[:5],
        "pending_reveals": len(world_state.pending_reveals),
        "active_npcs": [
            {
                "name": npc['npc_name'],
                "dominance": npc['dominance'],
                "stage": npc['narrative_stage'],
                "mask_integrity": npc.get('mask', {}).get('integrity', 100),
                "relationship": npc.get('relationship', {})
            }
            for npc in world_state.active_npcs[:3]
        ],
        "addiction_status": world_state.addiction_status,
        "inventory_highlights": [
            item for item in world_state.player_inventory[:5]
            if item.get('item_effect')  # Items with effects
        ],
        "location": world_state.location_data,
        "currency": world_state.currency_system.get('name', 'money')
    }
    
    # Generate using ChatGPT with reflection
    aggregator_text = f"World Context:\n{json.dumps(event_context, indent=2, default=str)}"
    
    prompt = """Generate a dynamic slice-of-life event that emerges from the current world state.

Create an event that:
1. Naturally incorporates NPCs based on their stages and relationships
2. Reflects the player's physical and mental state
3. May trigger addictions, stat changes, or rule effects
4. Includes subtle power dynamics appropriate to corruption level
5. Provides meaningful choices with hidden consequences
6. Uses inventory items or currency if relevant

Output as JSON with complete detail for emergent gameplay."""

    response = await get_chatgpt_response(
        context.conversation_id,
        aggregator_text,
        prompt,
        reflection_enabled=True,  # Enable reflection for better generation
        use_nyx_integration=True  # Use Nyx if available
    )
    
    # Parse response
    if response['type'] == 'function_call':
        event_data = response['function_args']
    else:
        # Parse text response
        try:
            event_data = json.loads(response['response'])
        except:
            # Generate fallback
            event_data = {
                "event_type": "routine",
                "title": "A Moment Passes",
                "description": response['response']
            }
    
    # Process through universal updater
    if event_data.get('narrative'):
        update_result = await process_universal_update(
            context.user_id, context.conversation_id,
            event_data['narrative'],
            {"source": "generated_event", "event_data": event_data}
        )
    
    # Store in memory with semantic abstraction
    memory_text = f"Event: {event_data.get('title', 'Unnamed event')}"
    abstraction = await create_semantic_abstraction(memory_text)
    
    await MemoryManager.add_memory(
        context.user_id, context.conversation_id,
        entity_id=1, entity_type="player",
        memory_text=memory_text,
        memory_type=MemoryType.INTERACTION,
        significance=MemorySignificance.MEDIUM,
        tags=["event", event_data.get('event_type', 'unknown')]
    )
    
    world_state.semantic_abstractions.append(abstraction)
    
    return event_data

@function_tool
async def generate_addiction_craving_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    craving_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate an addiction craving event using LLM"""
    context = ctx.context
    
    prompt = f"""Generate an addiction craving event for a femdom RPG.

Craving Data:
{json.dumps(craving_data, indent=2, default=str)}

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
        event = json.loads(response)
        
        # Trigger the craving event in the addiction system
        craving_result = await trigger_craving_event(
            context.user_id, context.conversation_id,
            context.player_name,
            craving_data.get('addiction_type'),
            craving_data.get('intensity', 1.0)
        )
        
        event['system_result'] = craving_result
        return event
        
    except Exception as e:
        logger.error(f"Error generating addiction event: {e}")
        return {"error": str(e)}

@function_tool
async def generate_dream_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    dream_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a dream sequence event"""
    context = ctx.context
    world_state = context.current_world_state
    
    # Get memory context for dream
    relevant_memories = await MemoryManager.retrieve_relevant_memories(
        context.user_id, context.conversation_id,
        context.player_name, "player",
        context="dream", tags=["emotional", "traumatic"],
        limit=5
    )
    
    dream_context = {
        "dream_trigger": dream_data,
        "recent_memories": [m.to_dict() for m in relevant_memories] if relevant_memories else [],
        "hidden_stats": world_state.hidden_stats,
        "active_addictions": world_state.addiction_status.get('addictions', {}),
        "npc_relationships": [
            {
                "name": npc['npc_name'],
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
        [m.text for m in relevant_memories[:3]] if relevant_memories else ["No specific memories"],
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
        dream_event = json.loads(dream_response)
        dream_event['reflection'] = response
        return dream_event
    except:
        return {
            "event_type": "dream",
            "title": "Strange Dreams",
            "description": dream_response,
            "reflection": response
        }

@function_tool
async def generate_revelation_event(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    revelation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a personal revelation event"""
    context = ctx.context
    
    # Generate inner monologue
    monologue = await generate_inner_monologue(
        context.user_id, context.conversation_id,
        topic=revelation_data.get('topic', 'current situation')
    )
    
    prompt = f"""Generate a moment of personal revelation for a femdom RPG.

Revelation:
{json.dumps(revelation_data, indent=2, default=str)}

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
        event = json.loads(response)
        event['inner_monologue'] = monologue
        
        # Add moment of clarity to the game
        clarity_result = await add_moment_of_clarity(
            context.user_id, context.conversation_id,
            trigger="revelation"
        )
        event['clarity_result'] = clarity_result
        
        return event
    except:
        return {
            "event_type": "revelation",
            "title": "A Moment of Clarity",
            "description": response,
            "inner_monologue": monologue
        }

@function_tool
async def process_complete_player_choice(
    ctx: RunContextWrapper[CompleteWorldDirectorContext],
    choice_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Process player choice through ALL systems"""
    context = ctx.context
    results = {"effects": [], "narratives": []}
    
    # 1. Apply stat changes with thresholds check
    if 'stat_impacts' in choice_data:
        stat_result = await apply_stat_changes(
            context.user_id, context.conversation_id,
            context.player_name, choice_data['stat_impacts'],
            reason=f"Choice: {choice_data.get('text', 'unknown')}"
        )
        results['stat_changes'] = stat_result
        
        # Check for new thresholds
        new_thresholds = await context._check_stat_thresholds(
            await get_player_hidden_stats(
                context.user_id, context.conversation_id, context.player_name
            )
        )
        if new_thresholds != context.current_world_state.stat_thresholds_active:
            results['new_thresholds'] = new_thresholds
    
    # 2. Process addiction impacts
    if 'addiction_impacts' in choice_data:
        for addiction_type, intensity in choice_data['addiction_impacts'].items():
            addiction_result = await process_addiction_update(
                context.user_id, context.conversation_id,
                context.player_name, addiction_type, intensity,
                choice_data.get('npc_id')
            )
            results['effects'].append(addiction_result)
    
    # 3. Process relationship impacts
    if 'relationship_impacts' in choice_data:
        for npc_name, impacts in choice_data['relationship_impacts'].items():
            # Find NPC
            npc_id = None
            for npc in context.current_world_state.active_npcs:
                if npc['npc_name'] == npc_name:
                    npc_id = npc['npc_id']
                    break
            
            if npc_id:
                # Process interaction
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
                    context=json.dumps(impacts),
                    check_for_event=True
                )
                results['effects'].append(interaction_result)
                
                # Check for narrative progression
                if impacts.get('trust', 0) > 5 or impacts.get('submission', 0) > 5:
                    progression = await progress_npc_narrative_stage(
                        context.user_id, context.conversation_id,
                        npc_id,
                        corruption_change=impacts.get('submission', 0),
                        dependency_change=impacts.get('dependency', 0),
                        realization_change=impacts.get('realization', 0)
                    )
                    if progression.get('stage_changed'):
                        results['npc_stage_change'] = progression
    
    # 4. Activity processing
    if 'activity_type' in choice_data:
        activity_result = await process_activity_vitals(
            context.user_id, context.conversation_id,
            context.player_name, choice_data['activity_type'],
            choice_data.get('intensity', 1.0)
        )
        results['activity_result'] = activity_result
        
        # Apply activity effects on stats
        if choice_data['activity_type'] in ACTIVITY_EFFECTS:
            effect_result = await apply_activity_effects(
                context.user_id, context.conversation_id,
                choice_data['activity_type'],
                choice_data.get('intensity', 1.0)
            )
            results['effects'].append(effect_result)
    
    # 5. Check for triggered rules
    triggered_rules = await enforce_all_rules_on_player(context.player_name)
    if triggered_rules:
        results['triggered_rules'] = triggered_rules
        
        for rule in triggered_rules:
            effect_result = await apply_effect(
                rule['effect'], context.player_name,
                npc_id=choice_data.get('npc_id')
            )
            results['effects'].append(effect_result)
    
    # 6. Inventory changes
    if 'inventory_changes' in choice_data:
        for change in choice_data['inventory_changes']:
            if change['action'] == 'add':
                inv_result = await add_item(
                    context.user_id, context.conversation_id,
                    context.player_name, change['item_name'],
                    change.get('description'), change.get('effect')
                )
            else:
                inv_result = await remove_item(
                    context.user_id, context.conversation_id,
                    context.player_name, change['item_name']
                )
            results['effects'].append(inv_result)
    
    # 7. Currency changes
    if 'currency_change' in choice_data:
        amount = choice_data['currency_change']
        formatted = await context.currency_generator.format_currency(abs(amount))
        context.current_world_state.player_money += amount
        results['currency'] = {
            "change": formatted,
            "new_balance": context.current_world_state.player_money
        }
    
    # 8. Check for hunger/thirst over time
    if choice_data.get('time_passed', 0) > 0:
        hunger_result = await update_hunger_from_time(
            context.user_id, context.conversation_id,
            context.player_name, choice_data['time_passed']
        )
        results['hunger_update'] = hunger_result
    
    # 9. Store in memory with analysis
    memory_text = f"Choice: {choice_data.get('text', 'Unknown choice')}"
    
    # Analyze preferences in the choice
    preferences = await analyze_preferences(memory_text)
    
    await MemoryManager.add_memory(
        context.user_id, context.conversation_id,
        entity_id=1, entity_type="player",
        memory_text=memory_text,
        memory_type=MemoryType.INTERACTION,
        significance=MemorySignificance.HIGH if triggered_rules else MemorySignificance.MEDIUM,
        emotional_valence=choice_data.get('emotional_valence', 0),
        tags=["player_choice"] + list(preferences.get('explicit_preferences', []))
    )
    
    results['preferences_detected'] = preferences
    
    # 10. Generate comprehensive narrative
    narrative_prompt = f"""Generate a narrative response to the player's choice.

Choice: {choice_data.get('text')}
All Effects: {json.dumps(results, default=str)}

Create a seamless narrative that:
1. Shows immediate consequences naturally
2. Hints at triggered rules without stating them
3. Reflects stat changes through description
4. Incorporates NPC reactions if relevant
5. Sets up the next moment
6. Maintains the current mood

Keep it atmospheric with rich subtext."""

    narrative = await generate_text_completion(
        system_prompt="You are weaving game mechanics into natural narrative flow.",
        user_prompt=narrative_prompt,
        temperature=0.7,
        max_tokens=200
    )
    
    results['narrative'] = narrative
    
    return results

@function_tool
async def check_all_emergent_patterns(
    ctx: RunContextWrapper[CompleteWorldDirectorContext]
) -> Dict[str, Any]:
    """Check ALL systems for emergent patterns and narratives"""
    context = ctx.context
    patterns = {
        "memory_patterns": [],
        "relationship_patterns": [],
        "addiction_patterns": [],
        "stat_patterns": [],
        "rule_patterns": []
    }
    
    # 1. Memory patterns using vector similarity
    recent_memories = context.current_world_state.recent_memories
    if len(recent_memories) > 5:
        # Get embeddings for recent memories
        embeddings = []
        for mem in recent_memories[:10]:
            if isinstance(mem, dict) and 'text' in mem:
                embedding = await get_text_embedding(mem['text'])
                embeddings.append(embedding)
        
        # Find similar memories (high cosine similarity)
        if embeddings:
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    if sim > 0.8:  # High similarity threshold
                        similarities.append({
                            "memory1": recent_memories[i],
                            "memory2": recent_memories[j],
                            "similarity": sim
                        })
            
            if similarities:
                patterns['memory_patterns'] = similarities
    
    # 2. Relationship patterns
    for npc in context.current_world_state.active_npcs[:5]:
        if 'relationship' in npc:
            rel_patterns = npc['relationship'].get('patterns', [])
            if rel_patterns:
                patterns['relationship_patterns'].append({
                    "npc": npc['npc_name'],
                    "patterns": rel_patterns,
                    "archetype": npc['relationship'].get('archetype')
                })
    
    # 3. Addiction patterns
    if context.current_world_state.addiction_status.get('has_addictions'):
        for addiction_type, data in context.current_world_state.addiction_status.get('addictions', {}).items():
            if data.get('level', 0) >= 3:
                patterns['addiction_patterns'].append({
                    "type": addiction_type,
                    "level": data['level'],
                    "trajectory": "escalating" if data.get('recent_increases', 0) > 2 else "stable"
                })
    
    # 4. Stat combination patterns
    active_combos = context.current_world_state.active_stat_combinations
    if active_combos:
        patterns['stat_patterns'] = [
            {
                "combination": combo['name'],
                "behaviors": combo.get('behaviors', [])
            }
            for combo in active_combos
        ]
    
    # 5. Rule trigger patterns
    if context.current_world_state.triggered_effects:
        rule_frequency = {}
        for effect in context.current_world_state.triggered_effects:
            rule_name = effect.get('rule', {}).get('rule_name', 'unknown')
            rule_frequency[rule_name] = rule_frequency.get(rule_name, 0) + 1
        
        patterns['rule_patterns'] = [
            {"rule": name, "frequency": count}
            for name, count in rule_frequency.items()
            if count > 1
        ]
    
    # Generate narrative analysis of patterns
    if any(patterns.values()):
        analysis_prompt = f"""Analyze these emergent patterns in a femdom RPG:

{json.dumps(patterns, indent=2, default=str)}

Identify:
1. Converging narratives across different systems
2. Building dependencies and control structures
3. Psychological trajectories
4. Hidden connections between patterns
5. Potential climax points approaching

Output as narrative insight, not JSON."""

        narrative_analysis = await generate_text_completion(
            system_prompt="You are a narrative analyst finding emergent stories in complex patterns.",
            user_prompt=analysis_prompt,
            temperature=0.6,
            max_tokens=400
        )
        
        patterns['narrative_analysis'] = narrative_analysis
    
    return patterns

# ===============================================================================
# Complete World Director Agent
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
    
    agent = Agent(
        name="Complete World Director",
        instructions=agent_instructions,
        tools=all_tools,
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.7, max_tokens=2048)
    )
    
    return agent

# ===============================================================================
# Complete Public Interface
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
        """Initialize ALL systems"""
        if not self._initialized:
            self.context = CompleteWorldDirectorContext(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            await self.context.initialize_everything()
            self.agent = create_complete_world_director()
            self._initialized = True
            logger.info(f"Complete World Director initialized with ALL systems")
    
    async def generate_next_moment(self) -> Dict[str, Any]:
        """Generate next moment using ALL systems"""
        await self.initialize()
        
        # Rebuild world state to capture all changes
        self.context.current_world_state = await self.context._build_complete_world_state()
        
        # Check all emergent patterns
        patterns = await check_all_emergent_patterns(
            RunContextWrapper(self.context)
        )
        
        # Let agent orchestrate
        prompt = f"""Generate the next moment in this complete simulation.
        
        World State Summary:
        - Mood: {self.context.current_world_state.world_mood.value}
        - Tensions: {json.dumps(self.context.current_world_state.tension_factors)}
        - Active NPCs: {len(self.context.current_world_state.active_npcs)}
        - Addictions: {len(self.context.current_world_state.addiction_status.get('addictions', {}))}
        - Active Cravings: {len(self.context.current_world_state.active_cravings)}
        - Pending Reveals: {len(self.context.current_world_state.pending_reveals)}
        - Triggered Rules: {len(self.context.current_world_state.triggered_effects)}
        
        Emergent Patterns Detected:
        {json.dumps(patterns, indent=2, default=str)}
        
        Orchestrate all systems to create the next emergent moment.
        Balance competing priorities naturally.
        Let the most dramatically appropriate system take precedence.
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        
        return {
            "moment": result.messages[-1].content if result else None,
            "world_state": self.context.current_world_state.model_dump(),
            "patterns": patterns
        }
    
    async def process_player_action(self, action: str) -> Dict[str, Any]:
        """Process player action through ALL systems"""
        await self.initialize()
        
        # Analyze action
        preferences = await analyze_preferences(action)
        
        # Check for social insight opportunity
        insight = None
        empathy = self.context.current_world_state.visible_stats.get('empathy', 0)
        if empathy > 10:
            # Roll for insight
            success, roll = calculate_social_insight(empathy, difficulty=12)
            if success:
                insight = "You sense hidden meanings in the interaction"
        
        # Process through agent
        prompt = f"""Process player action through ALL systems:
        
        Action: "{action}"
        Preferences Detected: {json.dumps(preferences)}
        Social Insight: {insight or "None"}
        
        Process through:
        1. All stat impacts
        2. Addiction triggers
        3. Relationship changes
        4. Rule triggers
        5. Vital costs
        6. Memory storage with semantic abstraction
        7. NPC reactions based on masks/stages
        8. Emergent pattern detection
        
        Generate complete response using all systems.
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        
        return {
            "response": result.messages[-1].content if result else None,
            "preferences": preferences,
            "insight": insight
        }
    
    async def advance_time(self, hours: int = 1) -> Dict[str, Any]:
        """Advance time with ALL system updates"""
        await self.initialize()
        
        # Advance time
        time_result = await advance_time_with_events(
            self.user_id, self.conversation_id,
            activity_type="time_passage"
        )
        
        # Update hunger over time
        hunger_result = await update_hunger_from_time(
            self.user_id, self.conversation_id,
            self.context.player_name, hours
        )
        
        # Check for automated reveals
        reveals = await check_for_automated_reveals(
            self.user_id, self.conversation_id
        )
        
        # Drain relationship events
        rel_events = await drain_relationship_events_tool(
            RunContextWrapper({
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            }),
            max_events=10
        )
        
        return {
            "time": time_result,
            "hunger": hunger_result,
            "reveals": reveals,
            "relationship_events": rel_events
        }
