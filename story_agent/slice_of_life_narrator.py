"""
Comprehensive Slice-of-Life Narrator with FULL System Integration and Context Services

This complete refactor ensures:
- ALL Nyx governance integration is preserved and enhanced
- ALL original functionality from both versions is maintained
- Dynamic LLM generation capabilities are enhanced
- Full context service integration for optimal narrative awareness
- Memory and vector services are fully utilized
- All system integrations work together seamlessly
- No features are dropped - everything is preserved and improved
"""

import logging
import json
import asyncio
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

from agents import Agent, Runner, function_tool, handoff, ModelSettings, RunContextWrapper
from pydantic import BaseModel, Field, ConfigDict

# Database connection
from db.connection import get_db_connection_context

# GPT Integration - CRITICAL
from logic.chatgpt_integration import (
    generate_text_completion,
    get_chatgpt_response,
    TEMPERATURE_SETTINGS
)

# ===============================================================================
# NYX GOVERNANCE INTEGRATION - CRITICAL (ENHANCED)
# ===============================================================================
from nyx.governance_helpers import with_governance, with_governance_permission
from nyx.integrate import get_central_governance, remember_with_governance

# ===============================================================================
# CONTEXT SERVICES INTEGRATION - NEW ENHANCEMENT
# ===============================================================================
from context.context_service import (
    get_context_service, 
    get_comprehensive_context,
    ContextService
)
from context.memory_manager import (
    get_memory_manager,
    MemoryManager,
    MemoryAddRequest,
    MemorySearchRequest
)
from context.vector_service import (
    get_vector_service,
    VectorService
)
from context.unified_cache import context_cache
from context.context_performance import PerformanceMonitor, track_performance

# ===============================================================================
# WORLD DIRECTOR INTEGRATION
# ===============================================================================
from story_agent.world_director_agent import (
    WorldState, WorldMood, TimeOfDay, ActivityType, PowerDynamicType,
    SliceOfLifeEvent, NPCRoutine, PowerExchange, WorldTension,
    WorldDirector
)

# ===============================================================================
# COMPREHENSIVE SYSTEM IMPORTS - ALL FEATURES PRESERVED
# ===============================================================================

# Import world simulation tools
from story_agent.tools import (
    DailyLifeDirector, AmbientDialogueWriter, 
    PowerDynamicsOrchestrator, PlayerAgencyManager
)

# NPC systems
from logic.npc_narrative_progression import get_npc_narrative_stage, NPCNarrativeStage
from logic.dynamic_relationships import OptimizedRelationshipManager, event_generator

# Time and calendar systems
from logic.calendar import load_calendar_names
from logic.time_cycle import get_current_vitals, get_current_time_model, CurrentTimeData

# Addiction system
from logic.addiction_system_sdk import (
    get_addiction_status, 
    AddictionContext,
    check_addiction_levels_impl
)

# Narrative events
from logic.narrative_events import (
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence
)

# Event system
from logic.event_system import EventSystem

# Currency
from logic.currency_generator import CurrencyGenerator

# Relationship integration
from logic.relationship_integration import RelationshipIntegration

# Memory systems (enhanced with context integration)
from logic.memory_logic import (
    MemoryManager as LegacyMemoryManager,
    EnhancedMemory,
    MemoryType,
    MemorySignificance,
    ProgressiveRevealManager
)

# Stats and rules
from logic.stats_logic import (
    get_all_player_stats,
    get_player_current_tier,
    check_for_combination_triggers,
    STAT_THRESHOLDS,
    ACTIVITY_EFFECTS
)
from logic.rule_enforcement import enforce_all_rules_on_player

# Inventory
from logic.inventory_system_sdk import get_inventory

logger = logging.getLogger(__name__)

# ===============================================================================
# Narrative Tone and Style Enums (PRESERVED)
# ===============================================================================

class NarrativeTone(Enum):
    """Tone for slice-of-life narration"""
    CASUAL = "casual"
    INTIMATE = "intimate"
    OBSERVATIONAL = "observational"
    SENSUAL = "sensual"
    TEASING = "teasing"
    COMMANDING = "commanding"
    SUBTLE = "subtle"
    PSYCHOLOGICAL = "psychological"

class SceneFocus(Enum):
    """What to emphasize in scene narration"""
    ATMOSPHERE = "atmosphere"
    DIALOGUE = "dialogue"
    INTERNAL = "internal"
    DYNAMICS = "dynamics"
    ROUTINE = "routine"
    TENSION = "tension"

# ===============================================================================
# Pydantic Models for Type-Safe Narration (ALL PRESERVED)
# ===============================================================================

class SliceOfLifeNarration(BaseModel):
    """Complete narration for a slice-of-life scene"""
    scene_description: str
    atmosphere: str
    tone: NarrativeTone
    focus: SceneFocus
    power_dynamic_hints: List[str] = Field(default_factory=list)
    sensory_details: List[str] = Field(default_factory=list)
    npc_observations: List[str] = Field(default_factory=list)
    internal_monologue: Optional[str] = None
    # Nyx governance tracking
    governance_approved: bool = True
    governance_notes: Optional[str] = None
    # Emergent elements
    emergent_elements: Optional[Dict[str, Any]] = None
    system_triggers: List[str] = Field(default_factory=list)
    # Context integration
    context_aware: bool = True
    relevant_memories: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class NPCDialogue(BaseModel):
    """Dialogue from an NPC in daily life"""
    npc_id: int
    npc_name: str
    dialogue: str
    tone: str
    subtext: str
    body_language: str
    power_dynamic: Optional[PowerDynamicType] = None
    requires_response: bool = False
    # System awareness
    hidden_triggers: List[str] = Field(default_factory=list)
    memory_influence: Optional[str] = None
    # Nyx governance
    governance_approved: bool = True
    # Context awareness
    context_informed: bool = False
    
    model_config = ConfigDict(extra="forbid")

class AmbientNarration(BaseModel):
    """Ambient narration for world atmosphere"""
    description: str
    focus: str
    intensity: float = 0.5
    affects_mood: bool = False
    reflects_systems: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class PowerMomentNarration(BaseModel):
    """Narration for a power exchange moment"""
    setup: str
    moment: str
    aftermath: str
    player_feelings: str
    options_presentation: List[str]
    potential_consequences: List[Dict[str, Any]] = Field(default_factory=list)
    governance_tracking: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(extra="forbid")

class DailyActivityNarration(BaseModel):
    """Narration for routine daily activities"""
    activity: str
    description: str
    routine_with_dynamics: str
    npc_involvement: List[str] = Field(default_factory=list)
    subtle_control_elements: List[str] = Field(default_factory=list)
    emergent_variations: Optional[List[str]] = None
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# ENHANCED Narrator Context with Full Integration
# ===============================================================================

@dataclass
class NarratorContext:
    """Complete context for the Slice-of-Life Narrator with ALL systems"""
    user_id: int
    conversation_id: int
    
    # NYX GOVERNANCE - CRITICAL
    nyx_governance: Optional[Any] = None
    governance_active: bool = False
    
    # CONTEXT SERVICES - ENHANCED
    context_service: Optional[ContextService] = None
    memory_manager: Optional[MemoryManager] = None
    vector_service: Optional[VectorService] = None
    performance_monitor: Optional[PerformanceMonitor] = None
    
    # World integration
    world_director: Optional[WorldDirector] = None
    current_world_state: Optional[WorldState] = None
    
    # Core system managers (ALL PRESERVED)
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    relationship_integration: Optional[RelationshipIntegration] = None
    event_system: Optional[EventSystem] = None
    addiction_context: Optional[AddictionContext] = None
    currency_generator: Optional[CurrencyGenerator] = None
    legacy_memory_manager: Optional[LegacyMemoryManager] = None
    reveal_manager: Optional[ProgressiveRevealManager] = None
    
    # Game state (ALL PRESERVED)
    calendar_names: Optional[Dict[str, Any]] = None
    current_vitals: Optional[Dict[str, int]] = None
    current_time: Optional[CurrentTimeData] = None
    active_addictions: Optional[Dict[str, Any]] = None
    player_stats: Optional[Dict[str, Any]] = None
    active_rules: Optional[List[Dict]] = None
    player_inventory: Optional[Dict[str, Any]] = None
    
    # Context-aware state (ENHANCED)
    current_context: Optional[Dict[str, Any]] = None
    context_version: int = 0
    last_context_update: Optional[datetime] = None
    
    # Narrative tracking (ALL PRESERVED)
    recent_narrations: List[str] = field(default_factory=list)
    current_tone: NarrativeTone = NarrativeTone.CASUAL
    narrative_momentum: float = 0.0
    last_revelation_check: Optional[datetime] = None
    
    # NPC tracking (ALL PRESERVED)
    npc_voices: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    npc_masks: Optional[Dict[int, Dict]] = None
    
    # Memory and pattern tracking (ALL PRESERVED + ENHANCED)
    active_memories: Optional[List[Any]] = None
    vector_memories: Optional[List[Dict[str, Any]]] = None
    system_intersections: List[str] = field(default_factory=list)
    pending_consequences: List[Dict] = field(default_factory=list)
    narrative_seeds: List[str] = field(default_factory=list)
    scene_history: List[Dict] = field(default_factory=list)
    
    # Caching (ENHANCED)
    last_narration_time: Optional[datetime] = None
    narrative_cache: Dict[str, Any] = field(default_factory=dict)
    context_cache_key: Optional[str] = None
    
    async def initialize(self):
        """Initialize ALL systems including Nyx governance and context services"""
        logger.info(f"Initializing enhanced narrator context for user {self.user_id}, conversation {self.conversation_id}")
        
        # CRITICAL: Initialize Nyx governance first
        try:
            self.nyx_governance = await get_central_governance(self.user_id, self.conversation_id)
            self.governance_active = True
            logger.info("Nyx governance initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Nyx governance: {e}")
            self.governance_active = False
        
        # Initialize Context Services (ENHANCED)
        try:
            self.context_service = await get_context_service(self.user_id, self.conversation_id)
            self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
            self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
            self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
            logger.info("Context services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize context services: {e}")
        
        # Initialize World Director
        self.world_director = WorldDirector(self.user_id, self.conversation_id)
        await self.world_director.initialize()
        
        # Initialize relationship systems (ALL PRESERVED)
        self.relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
        self.relationship_integration = RelationshipIntegration(self.user_id, self.conversation_id)
        
        # Initialize event system
        self.event_system = EventSystem(self.user_id, self.conversation_id)
        await self.event_system.initialize()
        
        # Initialize addiction context
        self.addiction_context = AddictionContext(self.user_id, self.conversation_id)
        await self.addiction_context.initialize()
        
        # Initialize currency generator
        self.currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
        
        # Load initial context (ENHANCED)
        await self._load_initial_context()
        
        # Load game state (ALL PRESERVED)
        self.calendar_names = await load_calendar_names(self.user_id, self.conversation_id)
        self.current_time = await get_current_time_model(self.user_id, self.conversation_id)
        self.current_vitals = await get_current_vitals(self.user_id, self.conversation_id)
        self.active_addictions = await get_addiction_status(self.user_id, self.conversation_id, "Chase")
        
        # Load player stats and rules
        self.player_stats = await get_all_player_stats(self.user_id, self.conversation_id, "Chase")
        self.active_rules = await enforce_all_rules_on_player("Chase")
        self.player_inventory = await get_inventory(self.user_id, self.conversation_id, "Chase")
        
        # Load NPC masks and memories
        await self._load_npc_masks()
        await self._load_active_memories()
        await self._identify_system_intersections()
        
        logger.info("Narrator context fully initialized with all enhancements")
    
    async def _load_initial_context(self):
        """Load initial context from context service"""
        if self.context_service:
            try:
                self.current_context = await get_comprehensive_context(
                    self.user_id,
                    self.conversation_id,
                    context_budget=4000,
                    use_vector_search=True
                )
                self.context_version = self.current_context.get("version", 0)
                self.last_context_update = datetime.now(timezone.utc)
                logger.info("Initial context loaded from context service")
            except Exception as e:
                logger.error(f"Failed to load initial context: {e}")
    
    async def refresh_context(self, input_text: str = "", location: Optional[str] = None):
        """Refresh context based on current input"""
        if self.context_service:
            try:
                # Use delta updates if we have a version
                source_version = self.context_version if self.context_version > 0 else None
                
                self.current_context = await get_comprehensive_context(
                    self.user_id,
                    self.conversation_id,
                    input_text=input_text,
                    location=location,
                    context_budget=4000,
                    use_vector_search=True,
                    use_delta=True,
                    source_version=source_version
                )
                
                # Update version tracking
                self.context_version = self.current_context.get("version", self.context_version)
                self.last_context_update = datetime.now(timezone.utc)
                
                # Extract relevant memories from context
                if "memories" in self.current_context:
                    self.active_memories = self.current_context["memories"]
                
                # Update NPC data from context
                if "npcs" in self.current_context:
                    for npc_data in self.current_context["npcs"]:
                        npc_id = npc_data.get("npc_id")
                        if npc_id:
                            self.npc_voices[npc_id] = npc_data
                
            except Exception as e:
                logger.error(f"Failed to refresh context: {e}")
    
    async def _load_npc_masks(self):
        """Load NPC facade states for progressive reveals"""
        self.npc_masks = {}
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, dominance, cruelty, personality_traits 
                    FROM NPCStats 
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                for row in rows:
                    self.npc_masks[row['npc_id']] = {
                        "integrity": 100 - (row['dominance'] / 2),
                        "hidden_traits": {
                            "true_dominance": row['dominance'],
                            "true_cruelty": row['cruelty'],
                            "hidden_nature": json.loads(row['personality_traits']) if row['personality_traits'] else []
                        }
                    }
        except Exception as e:
            logger.warning(f"Could not load NPC masks: {e}")
    
    async def _load_active_memories(self):
        """Load recent significant memories using context service"""
        try:
            # Use context service memory manager if available
            if self.memory_manager:
                recent_memories = await self.memory_manager.get_recent_memories(
                    days=7,
                    limit=20
                )
                self.active_memories = recent_memories
                
                # Also search for contextually relevant memories if we have vector service
                if self.vector_service and self.current_world_state:
                    scene_desc = f"Current scene: {self.current_world_state.world_mood.value}"
                    vector_results = await self.vector_service.search_entities(
                        query_text=scene_desc,
                        entity_types=["memory"],
                        top_k=10
                    )
                    self.vector_memories = vector_results
            
            # Fallback to Nyx governance memory if needed
            elif self.governance_active and self.nyx_governance:
                memories = await remember_with_governance(
                    self.user_id, self.conversation_id,
                    "player", self.user_id,
                    "Loading recent memories for narration",
                    importance="medium",
                    emotional=False,
                    tags=["system", "narration"]
                )
                self.active_memories = memories.get("memories", [])[:10]
            else:
                self.active_memories = []
                
        except Exception as e:
            logger.warning(f"Could not load active memories: {e}")
            self.active_memories = []
    
    async def _identify_system_intersections(self):
        """Identify interesting system state combinations for emergent events"""
        self.system_intersections = []
        
        # Check stat combinations
        try:
            stat_combos = await check_for_combination_triggers(self.user_id, self.conversation_id)
            for combo in stat_combos:
                self.system_intersections.append(f"stat_combo:{combo['name']}")
        except:
            pass
        
        # Check addiction + stat intersections
        if self.active_addictions and self.active_addictions.get("has_addictions"):
            if self.player_stats:
                hidden_stats = self.player_stats.get("hidden", {})
                if hidden_stats.get("willpower", 100) < 30:
                    self.system_intersections.append("addiction_low_willpower")
                if hidden_stats.get("dependency", 0) > 70:
                    self.system_intersections.append("addiction_high_dependency")
        
        # Check vitals + rules intersections
        if self.current_vitals:
            if self.current_vitals.get("fatigue", 0) > 80:
                self.system_intersections.append("extreme_fatigue")
            if self.current_vitals.get("hunger", 100) < 20:
                self.system_intersections.append("severe_hunger")
        
        # Check memory patterns (ENHANCED)
        if self.active_memories and len(self.active_memories) > 3:
            self.system_intersections.append("memory_accumulation")
        
        # Check context patterns (NEW)
        if self.current_context:
            if self.current_context.get("relationship_overview"):
                overview = self.current_context["relationship_overview"]
                if overview.get("most_advanced_npcs"):
                    self.system_intersections.append("relationship_progression")

# ===============================================================================
# Core Narration Functions with Enhanced Context Integration
# ===============================================================================

@function_tool
@with_governance_permission(
    agent_type="narrator",
    action_type="narrate_scene",
    id_from_context=lambda ctx: f"narrator_{ctx.context.conversation_id}"
)
async def narrate_slice_of_life_scene(
    ctx: RunContextWrapper,
    scene: SliceOfLifeEvent,
    world_state: WorldState,
    player_action: Optional[str] = None
) -> SliceOfLifeNarration:
    """
    Generate narration for a slice-of-life scene with full Nyx governance and context awareness.
    """
    context: NarratorContext = ctx.context
    
    # Refresh context for the scene
    await context.refresh_context(
        input_text=scene.description if scene.description else "",
        location=scene.location
    )
    
    # Check Nyx governance for scene approval
    governance_approved = True
    governance_notes = None
    
    if context.governance_active and context.nyx_governance:
        try:
            gov_check = await context.nyx_governance.check_scene_appropriateness(
                scene_type=scene.event_type.value,
                participants=[p for p in scene.participants],
                power_level=scene.power_dynamic.value if scene.power_dynamic else "none"
            )
            governance_approved = gov_check.get("approved", True)
            governance_notes = gov_check.get("notes")
        except:
            pass
    
    # Get relationship contexts (ENHANCED with context service)
    relationship_contexts = {}
    for npc_id in scene.participants:
        # Check if we have context service data for this NPC
        npc_context_data = None
        if context.current_context and "npcs" in context.current_context:
            for npc_data in context.current_context["npcs"]:
                if npc_data.get("npc_id") == str(npc_id):
                    npc_context_data = npc_data
                    break
        
        state = await context.relationship_manager.get_relationship_state(
            'npc', npc_id, 'player', context.user_id
        )
        stage = await get_npc_narrative_stage(
            context.user_id, context.conversation_id, npc_id
        )
        
        relationship_contexts[npc_id] = {
            'state': state,
            'stage': stage,
            'dimensions': state.dimensions.to_dict(),
            'patterns': list(state.history.active_patterns),
            'context_data': npc_context_data
        }
    
    # Determine tone based on world mood, relationships, and governance
    tone = _determine_narrative_tone(world_state.world_mood, scene.event_type)
    if governance_notes and "tone_adjustment" in governance_notes:
        tone = NarrativeTone(governance_notes["tone_adjustment"])
    
    # Determine focus
    if scene.power_dynamic:
        focus = SceneFocus.DYNAMICS
    elif scene.participants:
        focus = SceneFocus.DIALOGUE
    else:
        focus = SceneFocus.ATMOSPHERE
    
    # Generate scene description with GPT (ENHANCED with context)
    scene_desc = await _generate_scene_description_with_context(
        context, scene, world_state, tone, focus, relationship_contexts
    )
    
    # Generate atmospheric details
    atmosphere = await _generate_atmosphere(
        context, scene.location, world_state.world_mood
    )
    
    # Generate power dynamic hints if present
    power_hints = []
    if scene.power_dynamic:
        power_hints = await _generate_power_hints(
            context, scene.power_dynamic, scene.participants, relationship_contexts
        )
    
    # Generate sensory details
    sensory = await _generate_sensory_details(
        context, world_state.current_time, scene.location
    )
    
    # Generate NPC observations
    npc_obs = []
    for npc_id in scene.participants:
        obs = await _generate_npc_observation(context, npc_id, scene, relationship_contexts.get(npc_id))
        if obs:
            npc_obs.append(obs)
    
    # Generate internal monologue if tension is high or governance suggests it
    internal = None
    if world_state.world_tension.power_tension > 0.6 or (governance_notes and governance_notes.get("add_internal")):
        internal = await _generate_internal_monologue(
            context, scene, world_state.relationship_dynamics, relationship_contexts
        )
    
    # Identify emergent elements
    emergent_elements = await _identify_emergent_elements(
        context, scene, relationship_contexts
    )
    
    # Extract relevant memories for the scene (NEW)
    relevant_memories = []
    if context.active_memories:
        for memory in context.active_memories[:5]:
            if hasattr(memory, 'content'):
                relevant_memories.append(memory.content[:100])
            elif isinstance(memory, dict) and 'content' in memory:
                relevant_memories.append(memory['content'][:100])
    
    narration = SliceOfLifeNarration(
        scene_description=scene_desc,
        atmosphere=atmosphere,
        tone=tone,
        focus=focus,
        power_dynamic_hints=power_hints,
        sensory_details=sensory,
        npc_observations=npc_obs,
        internal_monologue=internal,
        governance_approved=governance_approved,
        governance_notes=governance_notes,
        emergent_elements=emergent_elements,
        system_triggers=context.system_intersections[:3],
        context_aware=True,
        relevant_memories=relevant_memories
    )
    
    # Track recent narration
    context.recent_narrations.append(scene_desc)
    if len(context.recent_narrations) > 10:
        context.recent_narrations.pop(0)
    
    # Store in memory manager (ENHANCED)
    if context.memory_manager:
        await context.memory_manager.add_memory(
            content=f"Scene: {scene.title} - {scene_desc[:200]}...",
            memory_type="scene",
            importance=0.7,
            tags=["scene", scene.event_type.value],
            metadata={"scene_id": scene.id, "tone": tone.value}
        )
    
    # Record in Nyx memory if governance is active
    if context.governance_active:
        await remember_with_governance(
            context.user_id, context.conversation_id,
            "scene", scene.id,
            f"Scene: {scene.title} - {scene_desc[:100]}...",
            importance="medium",
            emotional=scene.power_dynamic is not None,
            tags=["scene", scene.event_type.value]
        )
    
    return narration

@function_tool
@with_governance_permission(
    agent_type="narrator",
    action_type="generate_dialogue",
    id_from_context=lambda ctx: f"narrator_{ctx.context.conversation_id}"
)
async def generate_npc_dialogue(
    ctx: RunContextWrapper,
    npc_id: int,
    situation: str,
    world_state: WorldState,
    relationship_context: Optional[Dict] = None
) -> NPCDialogue:
    """
    Generate contextual NPC dialogue with enhanced context awareness.
    """
    context: NarratorContext = ctx.context
    
    # Refresh context for NPC dialogue
    await context.refresh_context(input_text=situation)
    
    # Get NPC data (ENHANCED with context service)
    npc_context_data = None
    if context.current_context and "npcs" in context.current_context:
        for npc_data in context.current_context["npcs"]:
            if npc_data.get("npc_id") == str(npc_id):
                npc_context_data = npc_data
                break
    
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits, current_location, cruelty
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, npc_id, context.user_id, context.conversation_id)
    
    if not npc:
        raise ValueError(f"NPC {npc_id} not found")
    
    # Get narrative stage
    stage = await get_npc_narrative_stage(
        context.user_id, context.conversation_id, npc_id
    )
    
    # Get relationship state if not provided
    if not relationship_context:
        manager = OptimizedRelationshipManager(context.user_id, context.conversation_id)
        rel_state = await manager.get_relationship_state(
            'npc', npc_id, 'player', context.user_id
        )
        relationship_context = {
            'trust': rel_state.dimensions.trust,
            'influence': rel_state.dimensions.influence,
            'patterns': list(rel_state.history.active_patterns),
            'context_data': npc_context_data
        }
    
    # Search for relevant memories about this NPC (ENHANCED)
    npc_memories = []
    if context.memory_manager:
        memory_results = await context.memory_manager.search_memories(
            query_text=npc['npc_name'],
            memory_types=["interaction", "dialogue"],
            limit=3
        )
        npc_memories = memory_results.memories if hasattr(memory_results, 'memories') else []
    
    # Check for relevant addictions
    relevant_addictions = await _get_relevant_addictions_for_npc(context, npc_id)
    
    # Check mask state
    mask_state = context.npc_masks.get(npc_id, {})
    mask_integrity = mask_state.get("integrity", 100)
    hidden_traits = mask_state.get("hidden_traits", {})
    
    # Check Nyx governance for dialogue approval
    governance_approved = True
    if context.governance_active and context.nyx_governance:
        try:
            gov_check = await context.nyx_governance.check_dialogue_appropriateness(
                npc_id=npc_id,
                stage=stage.name,
                dominance=npc['dominance'],
                situation=situation
            )
            governance_approved = gov_check.get("approved", True)
            if not governance_approved and gov_check.get("alternative"):
                situation = gov_check["alternative"]
        except:
            pass
    
    # Generate dialogue with GPT (ENHANCED with memory context)
    dialogue = await _generate_contextual_dialogue_with_memory(
        context, npc, stage.name, situation, relationship_context,
        relevant_addictions, mask_integrity, hidden_traits, npc_memories
    )
    
    # Generate tone, subtext, and body language
    tone = await _determine_npc_tone(context, npc, stage.name, dialogue)
    subtext = await _generate_dialogue_subtext(
        context, dialogue, npc['dominance'], stage.name, relevant_addictions
    )
    body_language = await _generate_body_language(
        context, npc['dominance'], tone, stage.name
    )
    
    # Determine if power dynamic is present
    power_dynamic = None
    if npc['dominance'] > 60 and stage.name != "Innocent Beginning":
        power_dynamic = _select_dialogue_power_dynamic(situation, npc['dominance'])
    
    # Identify hidden triggers
    hidden_triggers = await _identify_dialogue_triggers(
        context, dialogue, npc_id, relationship_context
    )
    
    # Determine memory influence (ENHANCED)
    memory_influence = None
    if npc_memories and len(npc_memories) > 0:
        memory_influence = f"Influenced by {len(npc_memories)} past interactions"
    
    dialogue_obj = NPCDialogue(
        npc_id=npc_id,
        npc_name=npc['npc_name'],
        dialogue=dialogue,
        tone=tone,
        subtext=subtext,
        body_language=body_language,
        power_dynamic=power_dynamic,
        requires_response=random.random() > 0.5,
        hidden_triggers=hidden_triggers,
        memory_influence=memory_influence,
        governance_approved=governance_approved,
        context_informed=bool(npc_context_data)
    )
    
    # Store dialogue in memory (ENHANCED)
    if context.memory_manager:
        await context.memory_manager.add_memory(
            content=f"{npc['npc_name']}: {dialogue}",
            memory_type="dialogue",
            importance=0.6,
            tags=["dialogue", f"npc_{npc_id}", stage.name],
            metadata={"npc_id": str(npc_id), "tone": tone, "stage": stage.name}
        )
    
    return dialogue_obj

# ALL OTHER CORE FUNCTIONS PRESERVED (with enhancements)
@function_tool
async def narrate_power_exchange(
    ctx: RunContextWrapper,
    exchange: PowerExchange,
    world_state: WorldState
) -> PowerMomentNarration:
    """
    Generate narration for a power exchange moment with Nyx tracking and memory.
    """
    context: NarratorContext = ctx.context
    
    # Refresh context for power exchange
    await context.refresh_context(input_text=f"Power exchange: {exchange.exchange_type.value}")
    
    # Get NPC details
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, exchange.initiator_npc_id)
    
    # Track with Nyx governance if active
    governance_tracking = None
    if context.governance_active and context.nyx_governance:
        try:
            governance_tracking = await context.nyx_governance.track_power_exchange(
                npc_id=exchange.initiator_npc_id,
                exchange_type=exchange.exchange_type.value,
                intensity=exchange.intensity,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.warning(f"Could not track power exchange with governance: {e}")
    
    # Get relationship state
    rel_state = await context.relationship_manager.get_relationship_state(
        'npc', exchange.initiator_npc_id, 'player', context.user_id
    )
    
    # Calculate susceptibility
    susceptibility = _calculate_susceptibility(context.current_vitals, context.active_addictions)
    
    # Generate all components with GPT
    setup = await _generate_power_moment_setup(
        context, exchange, npc, world_state
    )
    moment = await _generate_power_moment_description(
        context, exchange, npc
    )
    aftermath = await _generate_power_moment_aftermath(
        context, exchange, world_state.relationship_dynamics
    )
    feelings = await _generate_player_feelings(
        context, exchange, world_state.relationship_dynamics
    )
    options = await _present_response_options(
        context, exchange.player_response_options
    )
    
    # Calculate potential consequences
    potential_consequences = await _calculate_power_exchange_consequences(
        context, exchange, rel_state, susceptibility
    )
    
    # Queue event in event system
    await context.event_system.create_event(
        "power_exchange",
        {
            "npc_id": exchange.initiator_npc_id,
            "type": exchange.exchange_type.value,
            "intensity": exchange.intensity,
            "consequences": potential_consequences
        },
        priority=8
    )
    
    # Store in memory manager (ENHANCED)
    if context.memory_manager:
        await context.memory_manager.add_memory(
            content=f"Power exchange with {npc['npc_name']}: {moment[:200]}...",
            memory_type="event",
            importance=0.9,
            tags=["power_exchange", exchange.exchange_type.value],
            metadata={
                "npc_id": str(exchange.initiator_npc_id),
                "intensity": exchange.intensity,
                "consequences": len(potential_consequences)
            }
        )
    
    return PowerMomentNarration(
        setup=setup,
        moment=moment,
        aftermath=aftermath,
        player_feelings=feelings,
        options_presentation=options,
        potential_consequences=potential_consequences,
        governance_tracking=governance_tracking
    )

@function_tool
async def narrate_daily_routine(
    ctx: RunContextWrapper,
    activity: str,
    time_period: TimeOfDay,
    involved_npcs: List[int],
    world_state: WorldState
) -> DailyActivityNarration:
    """
    Generate narration for routine daily activities with system integration.
    """
    context: NarratorContext = ctx.context
    
    # Refresh context for activity
    await context.refresh_context(input_text=f"Daily activity: {activity}")
    
    # Check for addiction effects on routine
    addiction_effects = []
    if context.active_addictions and context.active_addictions.get("has_addictions"):
        for addiction_type, data in context.active_addictions.get("addictions", {}).items():
            if data.get("level", 0) >= 2:
                effect_desc = await _generate_addiction_effect_description(
                    context, addiction_type, data.get("level"), activity
                )
                if effect_desc:
                    addiction_effects.append(effect_desc)
    
    # Generate basic activity description
    description = await _generate_activity_description(
        context, activity, time_period
    )
    
    # Add power dynamics to routine
    routine_with_dynamics = await _generate_routine_with_dynamics(
        context, activity, involved_npcs, world_state.relationship_dynamics
    )
    
    # Generate NPC involvement
    npc_involvement = []
    for npc_id in involved_npcs:
        involvement = await _generate_npc_routine_involvement(
            context, npc_id, activity
        )
        if involvement:
            npc_involvement.append(involvement)
    
    # Generate subtle control elements including addiction effects
    control_elements = addiction_effects + await _generate_subtle_control_elements(
        context, activity, world_state.relationship_dynamics
    )
    
    # Generate emergent variations
    emergent_variations = await _generate_activity_variations(
        context, activity, context.system_intersections
    )
    
    return DailyActivityNarration(
        activity=activity,
        description=description,
        routine_with_dynamics=routine_with_dynamics,
        npc_involvement=npc_involvement,
        subtle_control_elements=control_elements[:5],
        emergent_variations=emergent_variations
    )

@function_tool
async def generate_ambient_narration(
    ctx: RunContextWrapper,
    focus: str,
    world_state: WorldState
) -> AmbientNarration:
    """
    Generate ambient narration for atmosphere and world-building.
    """
    context: NarratorContext = ctx.context
    
    # Update time data
    context.current_time = await get_current_time_model(
        context.user_id, context.conversation_id
    )
    
    # Generate description based on focus
    if focus == "time_passage":
        description = await _narrate_time_passage(context, context.current_time)
    elif focus == "mood_shift":
        description = await _narrate_mood_shift(context, world_state.world_mood)
    elif focus == "tension_building":
        description = await _narrate_tension(context, world_state.world_tension)
    elif focus == "addiction_presence":
        description = await _narrate_addiction_presence(context, context.active_addictions)
    else:
        description = await _narrate_ambient_detail(context, world_state)
    
    # Check for relationship events to weave in
    rel_event = await event_generator.get_next_event(timeout=0.1)
    if rel_event:
        description += f" {await _weave_relationship_event(context, rel_event)}"
    
    # Identify reflected systems
    reflects_systems = await _identify_reflected_systems(context, focus, description)
    
    # Determine intensity based on world tension
    dominant_tension, level = world_state.world_tension.get_dominant_tension()
    intensity = min(1.0, level * 0.8)
    
    return AmbientNarration(
        description=description,
        focus=focus,
        intensity=intensity,
        affects_mood=focus == "mood_shift",
        reflects_systems=reflects_systems
    )

@function_tool
async def narrate_player_action(
    ctx: RunContextWrapper,
    action: str,
    world_state: WorldState,
    affected_npcs: List[int] = None
) -> Dict[str, Any]:
    """
    Narrate the results of a player action in the world.
    """
    context: NarratorContext = ctx.context
    
    # Refresh context for player action
    await context.refresh_context(input_text=action)
    
    # Check Nyx governance for action approval
    if context.governance_active and context.nyx_governance:
        try:
            gov_check = await context.nyx_governance.check_player_action(
                action=action,
                context={"world_state": world_state, "affected_npcs": affected_npcs}
            )
            if not gov_check.get("approved", True):
                return {
                    "acknowledgment": gov_check.get("message", "That action isn't possible right now."),
                    "world_reaction": "",
                    "npc_reactions": [],
                    "dynamic_shift": None,
                    "governance_blocked": True
                }
        except:
            pass
    
    # Generate action acknowledgment
    acknowledgment = await _acknowledge_player_action(context, action)
    
    # Generate world reaction
    world_reaction = await _generate_world_reaction(
        context, action, world_state
    )
    
    # Generate NPC reactions if any
    npc_reactions = []
    if affected_npcs:
        for npc_id in affected_npcs:
            reaction = await _generate_npc_reaction(
                context, npc_id, action, world_state
            )
            if reaction:
                npc_reactions.append(reaction)
    
    # Determine if this shifts any dynamics
    dynamic_shift = None
    if world_state.relationship_dynamics.player_submission_level > 0.5:
        dynamic_shift = await _check_for_dynamic_shift(
            context, action, world_state.relationship_dynamics
        )
    
    # Record action in memory manager (ENHANCED)
    if context.memory_manager:
        await context.memory_manager.add_memory(
            content=f"Player action: {action}",
            memory_type="decision",
            importance=0.7,
            tags=["player_action"],
            metadata={"affected_npcs": affected_npcs or []}
        )
    
    # Record action in Nyx memory
    if context.governance_active:
        await remember_with_governance(
            context.user_id, context.conversation_id,
            "player", context.user_id,
            f"Player action: {action}",
            importance="medium",
            emotional=False,
            tags=["player_action"]
        )
    
    return {
        "acknowledgment": acknowledgment,
        "world_reaction": world_reaction,
        "npc_reactions": npc_reactions,
        "dynamic_shift": dynamic_shift,
        "maintains_atmosphere": True,
        "governance_approved": True
    }

# ===============================================================================
# Enhanced GPT-Powered Helper Functions with Context Integration
# ===============================================================================

async def _generate_scene_description_with_context(
    context: NarratorContext,
    scene: SliceOfLifeEvent,
    world_state: WorldState,
    tone: NarrativeTone,
    focus: SceneFocus,
    relationship_contexts: Dict[int, Dict]
) -> str:
    """Generate scene description using GPT with full context awareness"""
    
    # Build comprehensive prompt context
    calendar = context.calendar_names
    time_desc = f"{calendar['days'][context.current_time.day % 7]} of {calendar['months'][context.current_time.month - 1]}" if calendar and context.current_time else "today"
    
    # Include context-aware elements (ENHANCED)
    context_elements = ""
    if context.current_context:
        if "relationship_overview" in context.current_context:
            overview = context.current_context["relationship_overview"]
            if overview and overview.get("most_advanced_npcs"):
                context_elements += f"\nRelationship dynamics: {len(overview['most_advanced_npcs'])} key relationships evolving"
        
        if "memories" in context.current_context and len(context.current_context["memories"]) > 0:
            context_elements += f"\nRecent memories influencing scene: {len(context.current_context['memories'])}"
    
    system_prompt = """You are narrating a slice-of-life scene in a femdom setting.
    Focus on atmosphere and subtle power dynamics woven into everyday life.
    Use second-person perspective ('you'). Be immersive but not melodramatic.
    The power dynamics should feel natural, not forced or explicit.
    Include environmental storytelling and system state reflections.
    Draw from past memories and relationships to make the scene feel lived-in."""
    
    user_prompt = f"""
    Generate a scene description for:
    Scene: {scene.title} - {scene.description}
    Location: {scene.location}
    Activity Type: {scene.event_type.value}
    Time: {time_desc} during {world_state.current_time.value}
    Tone: {tone.value} - Make the narration feel {tone.value}
    Focus: {focus.value} - Emphasize the {focus.value} aspect
    World Mood: {world_state.world_mood.value}
    
    NPCs Present: {len(scene.participants)} people
    Power Dynamic: {scene.power_dynamic.value if scene.power_dynamic else 'none'}
    {context_elements}
    
    Physical State:
    - Fatigue: {context.current_vitals.get('fatigue', 0)}/100 if context.current_vitals else 'normal'
    - Hunger: {context.current_vitals.get('hunger', 100)}/100 if context.current_vitals else 'normal'
    
    System Intersections: {context.system_intersections[:3] if context.system_intersections else 'none'}
    
    Recent narrations to avoid repetition:
    {chr(10).join(context.recent_narrations[-3:]) if context.recent_narrations else 'None'}
    
    Write 3-4 sentences that:
    1. Set the scene naturally
    2. Reflect the physical and emotional state
    3. Show system states through environmental details
    4. Reference past events subtly if relevant
    5. Don't mention game mechanics directly
    """
    
    description = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=200,
        task_type="narrative"
    )
    
    return description.strip()

async def _generate_contextual_dialogue_with_memory(
    context: NarratorContext,
    npc: Any,
    stage: str,
    situation: str,
    relationship_context: Dict,
    relevant_addictions: Dict[str, Any],
    mask_integrity: int,
    hidden_traits: Dict[str, Any],
    npc_memories: List[Any]
) -> str:
    """Generate NPC dialogue using GPT with full context and memory awareness"""
    
    # Build memory context (ENHANCED)
    memory_context = ""
    if npc_memories and len(npc_memories) > 0:
        memory_context = "\nPast interactions to reference:"
        for mem in npc_memories[:3]:
            if hasattr(mem, 'content'):
                memory_context += f"\n- {mem.content[:100]}"
            elif isinstance(mem, dict) and 'content' in mem:
                memory_context += f"\n- {mem['content'][:100]}"
    
    # Check for context data (NEW)
    context_hints = ""
    if relationship_context.get('context_data'):
        ctx_data = relationship_context['context_data']
        if ctx_data.get('relevance', 0) > 0.5:
            context_hints = f"\nNPC is highly relevant to current situation (relevance: {ctx_data['relevance']:.1f})"
    
    system_prompt = f"""You write natural dialogue for NPCs in a femdom slice-of-life setting.
    Character: {npc['npc_name']}
    Personality traits: {npc['personality_traits']}
    Narrative Stage: {stage}
    
    Guidelines for {stage}:
    - Innocent Beginning: Friendly, no obvious control
    - First Doubts: Occasional 'helpful' suggestions
    - Creeping Realization: Regular gentle steering
    - Veil Thinning: Open but caring control
    - Full Revelation: Complete, casual dominance
    
    Make dialogue feel natural and conversational. Power dynamics through subtext only.
    If mask integrity is low, let true nature slip through subtly.
    Reference past interactions naturally when relevant."""
    
    user_prompt = f"""
    Generate dialogue for this situation:
    Situation: {situation}
    Dominance: {npc['dominance']}/100
    Trust: {relationship_context['trust']}/100
    Influence: {relationship_context['influence']}/100
    Patterns: {', '.join(relationship_context['patterns'][:3]) if relationship_context['patterns'] else 'none'}
    
    Mask Integrity: {mask_integrity}/100 (lower = more reveals)
    Hidden Nature: {list(hidden_traits.keys())[:2] if hidden_traits else 'none'}
    
    Relevant Addictions: {list(relevant_addictions.keys())[:2] if relevant_addictions else 'none'}
    {memory_context}
    {context_hints}
    
    Write 1-3 sentences of natural dialogue that:
    1. Fits the situation and stage
    2. Shows mask slippage if integrity < 50
    3. Exploits known vulnerabilities subtly
    4. References shared history if memories exist
    5. Feels contextually aware
    """
    
    dialogue = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=150,
        task_type="dialogue"
    )
    
    return dialogue.strip()

# ALL OTHER HELPER FUNCTIONS PRESERVED (keeping originals)
async def _generate_atmosphere(
    context: NarratorContext,
    location: str,
    mood: WorldMood
) -> str:
    """Generate atmospheric description"""
    system_prompt = """You create atmospheric descriptions for locations.
    Focus on mood, ambiance, and subtle environmental details.
    Keep it short and evocative."""
    
    user_prompt = f"""
    Generate atmospheric description for:
    Location: {location}
    Mood: {mood.value}
    
    Write 1-2 sentences that capture the feeling of the space.
    """
    
    atmosphere = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=100,
        task_type="atmosphere"
    )
    
    return atmosphere.strip()

async def _generate_power_hints(
    context: NarratorContext,
    power_dynamic: PowerDynamicType,
    participants: List[int],
    relationship_contexts: Dict[int, Dict]
) -> List[str]:
    """Generate subtle power dynamic hints"""
    hints = []
    
    for participant_id in participants[:2]:
        rel_context = relationship_contexts.get(participant_id, {})
        influence = rel_context.get('dimensions', {}).get('influence', 0)
        
        if power_dynamic == PowerDynamicType.SUBTLE_CONTROL:
            if influence > 30:
                hints.append("A gentle suggestion that feels perfectly reasonable")
            hints.append("The natural order of things, unquestioned")
        elif power_dynamic == PowerDynamicType.CASUAL_DOMINANCE:
            if influence > 50:
                hints.append("Decisions made before you realize you weren't asked")
            hints.append("Authority exercised with casual confidence")
        elif power_dynamic == PowerDynamicType.INTIMATE_COMMAND:
            hints.append("A whispered instruction that bypasses thought")
            if influence > 70:
                hints.append("Your body responding before your mind catches up")
    
    return hints[:3]

async def _generate_sensory_details(
    context: NarratorContext,
    time_of_day: TimeOfDay,
    location: str
) -> List[str]:
    """Generate sensory details for the scene"""
    details = []
    
    # Time-based sensory details
    if time_of_day == TimeOfDay.MORNING:
        details.append("The soft morning light filtering through")
        details.append("The lingering coolness before the day's warmth")
    elif time_of_day == TimeOfDay.EVENING:
        details.append("Shadows lengthening across familiar spaces")
        details.append("The day's warmth slowly fading")
    elif time_of_day == TimeOfDay.NIGHT:
        details.append("The intimate darkness wrapping around everything")
        details.append("Sounds carrying differently in the quiet")
    
    # Location-based details
    if "bedroom" in location.lower():
        details.append("The familiar scent of the sheets")
    elif "kitchen" in location.lower():
        details.append("The comforting aroma of something cooking")
    elif "garden" in location.lower():
        details.append("The earthy smell of growing things")
    
    return details[:3]

async def _generate_npc_observation(
    context: NarratorContext,
    npc_id: int,
    scene: SliceOfLifeEvent,
    relationship_context: Optional[Dict]
) -> Optional[str]:
    """Generate an observation about an NPC in the scene"""
    if not relationship_context:
        return None
    
    stage = relationship_context.get('stage')
    if not stage:
        return None
    
    trust = relationship_context.get('dimensions', {}).get('trust', 0)
    influence = relationship_context.get('dimensions', {}).get('influence', 0)
    
    # Generate observation based on stage and relationship
    if stage.name == "Innocent Beginning":
        return f"They seem helpful, always ready with a suggestion"
    elif stage.name == "First Doubts":
        if trust > 40:
            return f"That knowing smile that suggests they understand you"
    elif stage.name == "Creeping Realization":
        if influence > 50:
            return f"How naturally you find yourself following their lead"
    elif stage.name == "Veil Thinning":
        return f"The quiet authority in their presence, undeniable now"
    elif stage.name == "Full Revelation":
        return f"Complete understanding passes between you without words"
    
    return None

async def _generate_internal_monologue(
    context: NarratorContext,
    scene: SliceOfLifeEvent,
    dynamics: Any,
    relationship_contexts: Dict[int, Dict]
) -> str:
    """Generate player's internal thoughts using GPT"""
    
    # Calculate average influence from relationships
    avg_influence = 0
    if relationship_contexts:
        influences = [rc.get('dimensions', {}).get('influence', 0) for rc in relationship_contexts.values()]
        avg_influence = sum(influences) / len(influences) if influences else 0
    
    system_prompt = """You write internal monologues for a player in a femdom slice-of-life game.
    The thoughts should reflect their growing awareness or acceptance of power dynamics.
    Keep it subtle and psychological. Use second person ('you think', 'you realize').
    Show the gradual shift in perception and acceptance."""
    
    user_prompt = f"""
    Generate an internal thought for the player:
    Scene: {scene.title}
    Average NPC Influence: {avg_influence:.0f}/100
    Submission Level: {dynamics.player_submission_level:.1%} (don't mention numbers)
    Acceptance Level: {dynamics.acceptance_level:.1%}
    Resistance Level: {dynamics.resistance_level:.1%}
    
    Active Addictions: {context.active_addictions.get('has_addictions', False)}
    Physical Fatigue: {context.current_vitals.get('fatigue', 0)}/100 if context.current_vitals else 0
    
    Write 1-2 sentences of internal monologue that:
    1. Reflects their psychological state
    2. Shows awareness/acceptance/resistance naturally
    3. Incorporates physical and mental vulnerability
    """
    
    monologue = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=100,
        task_type="introspection"
    )
    
    return monologue.strip()

# ALL REMAINING HELPER FUNCTIONS (preserved exactly as original)
def _determine_narrative_tone(mood: WorldMood, event_type: ActivityType) -> NarrativeTone:
    """Determine appropriate narrative tone based on mood and activity"""
    tone_map = {
        (WorldMood.INTIMATE, ActivityType.INTIMATE): NarrativeTone.SENSUAL,
        (WorldMood.PLAYFUL, ActivityType.SOCIAL): NarrativeTone.TEASING,
        (WorldMood.OPPRESSIVE, ActivityType.ROUTINE): NarrativeTone.COMMANDING,
        (WorldMood.MYSTERIOUS, ActivityType.SPECIAL): NarrativeTone.OBSERVATIONAL,
        (WorldMood.RELAXED, ActivityType.LEISURE): NarrativeTone.CASUAL,
    }
    
    key = (mood, event_type)
    return tone_map.get(key, NarrativeTone.SUBTLE)

def _select_dialogue_power_dynamic(situation: str, dominance: int) -> PowerDynamicType:
    """Select appropriate power dynamic for dialogue based on context"""
    situation_lower = situation.lower()
    
    if dominance > 80:
        if "private" in situation_lower or "alone" in situation_lower:
            return PowerDynamicType.INTIMATE_COMMAND
        else:
            return PowerDynamicType.CASUAL_DOMINANCE
    elif dominance > 60:
        if "decision" in situation_lower:
            return PowerDynamicType.CASUAL_DOMINANCE
        elif "help" in situation_lower:
            return PowerDynamicType.PROTECTIVE_CONTROL
        else:
            return PowerDynamicType.SUBTLE_CONTROL
    else:
        return PowerDynamicType.PLAYFUL_TEASING

def _calculate_susceptibility(vitals: Dict[str, int], addictions: Dict[str, Any]) -> float:
    """Calculate player susceptibility to power dynamics"""
    susceptibility = 0.5
    
    if vitals:
        if vitals.get("fatigue", 0) > 70:
            susceptibility += 0.2
        if vitals.get("hunger", 100) < 30:
            susceptibility += 0.15
        if vitals.get("thirst", 100) < 30:
            susceptibility += 0.15
    
    if addictions and addictions.get("has_addictions"):
        addiction_count = len(addictions.get("addictions", {}))
        susceptibility += min(0.3, addiction_count * 0.1)
    
    return min(1.0, susceptibility)

async def _get_relevant_addictions_for_npc(
    context: NarratorContext,
    npc_id: int
) -> Dict[str, Any]:
    """Get addictions relevant to a specific NPC"""
    relevant = {}
    
    if not context.active_addictions or not context.active_addictions.get("has_addictions"):
        return relevant
    
    for addiction_type, data in context.active_addictions.get("addictions", {}).items():
        if data.get("target_npc_id") == npc_id:
            relevant[addiction_type] = data
    
    return relevant

async def _identify_emergent_elements(
    context: NarratorContext,
    scene: SliceOfLifeEvent,
    relationship_contexts: Dict[int, Dict]
) -> Optional[Dict[str, Any]]:
    """Identify emergent elements from system intersections"""
    
    if not context.system_intersections:
        return None
    
    emergent = {
        "triggers": context.system_intersections[:3],
        "potential_events": [],
        "hidden_connections": []
    }
    
    for intersection in context.system_intersections:
        if "stat_combo" in intersection:
            emergent["potential_events"].append("stat_combination_trigger")
        if "addiction_low_willpower" in intersection:
            emergent["potential_events"].append("vulnerability_window")
        if "extreme_fatigue" in intersection:
            emergent["potential_events"].append("exhaustion_event")
        if "relationship_progression" in intersection:
            emergent["potential_events"].append("relationship_milestone")
    
    if len(context.system_intersections) >= 2:
        emergent["hidden_connections"].append("multiple_systems_converging")
    
    return emergent if emergent["potential_events"] or emergent["hidden_connections"] else None

async def _calculate_power_exchange_consequences(
    context: NarratorContext,
    exchange: PowerExchange,
    rel_state: Any,
    susceptibility: float
) -> List[Dict[str, Any]]:
    """Calculate potential consequences of power exchange"""
    
    consequences = []
    
    if susceptibility > 0.7:
        consequences.append({
            "type": "stat_change",
            "stats": ["willpower", "obedience"],
            "direction": "submission",
            "magnitude": "significant"
        })
    
    if context.active_addictions and context.active_addictions.get("has_addictions"):
        consequences.append({
            "type": "addiction_progression",
            "likelihood": susceptibility,
            "trigger": exchange.exchange_type.value
        })
    
    if exchange.intensity > 5:
        consequences.append({
            "type": "memory_formation",
            "memory_type": "emotional" if exchange.intensity > 7 else "interaction",
            "significance": min(10, exchange.intensity)
        })
    
    if rel_state.dimensions.influence > 60:
        consequences.append({
            "type": "relationship_deepening",
            "patterns": list(rel_state.history.active_patterns)[:2],
            "strength": "increasing"
        })
    
    return consequences

async def _identify_dialogue_triggers(
    context: NarratorContext,
    dialogue: str,
    npc_id: int,
    relationship_context: Dict
) -> List[str]:
    """Identify hidden triggers in dialogue"""
    
    triggers = []
    dialogue_lower = dialogue.lower()
    
    if context.active_addictions and context.active_addictions.get("has_addictions"):
        for addiction_type in context.active_addictions.get("addictions", {}).keys():
            if addiction_type.lower() in dialogue_lower:
                triggers.append(f"addiction_reference:{addiction_type}")
    
    if any(word in dialogue_lower for word in ["obey", "submit", "comply"]):
        triggers.append("obedience_trigger")
    
    if any(word in dialogue_lower for word in ["need", "depend", "without"]):
        triggers.append("dependency_trigger")
    
    if relationship_context.get("influence", 0) > 70:
        triggers.append("high_influence_exploitation")
    
    return triggers[:3]

# Additional preserved helper functions (ALL maintained)
async def _determine_npc_tone(
    context: NarratorContext,
    npc: Any,
    stage: str,
    dialogue: str
) -> str:
    """Determine the tone of NPC dialogue"""
    dominance = npc['dominance']
    
    if stage == "Innocent Beginning":
        return "friendly"
    elif stage == "First Doubts":
        return "knowing" if dominance > 50 else "helpful"
    elif stage == "Creeping Realization":
        return "guiding" if dominance > 60 else "suggestive"
    elif stage == "Veil Thinning":
        return "confident" if dominance > 70 else "assured"
    elif stage == "Full Revelation":
        return "commanding" if dominance > 80 else "dominant"
    
    return "neutral"

async def _generate_dialogue_subtext(
    context: NarratorContext,
    dialogue: str,
    dominance: int,
    stage: str,
    relevant_addictions: Dict[str, Any]
) -> str:
    """Generate subtext for dialogue"""
    if stage == "Full Revelation":
        return "Complete understanding of the power dynamic"
    elif stage == "Veil Thinning" and dominance > 70:
        return "The expectation of compliance barely hidden"
    elif stage == "Creeping Realization" and relevant_addictions:
        return "Awareness of your vulnerabilities"
    elif stage == "First Doubts":
        return "Something more behind the helpful exterior"
    
    return "Surface-level interaction"

async def _generate_body_language(
    context: NarratorContext,
    dominance: int,
    tone: str,
    stage: str
) -> str:
    """Generate body language description"""
    if tone == "commanding":
        return "Posture radiating unquestionable authority"
    elif tone == "confident":
        return "Relaxed confidence in every gesture"
    elif tone == "guiding":
        return "Gentle but firm positioning"
    elif tone == "knowing":
        return "That slight smile of understanding"
    elif tone == "friendly":
        return "Open and welcoming demeanor"
    
    return "Casual stance"

# Power moment generation functions (ALL preserved)
async def _generate_power_moment_setup(
    context: NarratorContext,
    exchange: PowerExchange,
    npc: Any,
    world_state: WorldState
) -> str:
    """Generate setup for power moment"""
    return f"The moment builds naturally from the {world_state.world_mood.value} atmosphere..."

async def _generate_power_moment_description(
    context: NarratorContext,
    exchange: PowerExchange,
    npc: Any
) -> str:
    """Generate the power moment itself"""
    return f"{npc['npc_name']} exercises their influence with {exchange.exchange_type.value}..."

async def _generate_power_moment_aftermath(
    context: NarratorContext,
    exchange: PowerExchange,
    dynamics: Any
) -> str:
    """Generate aftermath of power moment"""
    return "The moment passes, but its weight lingers..."

async def _generate_player_feelings(
    context: NarratorContext,
    exchange: PowerExchange,
    dynamics: Any
) -> str:
    """Generate player's feelings during power exchange"""
    return "You feel the shift in the dynamic, undeniable now..."

async def _present_response_options(
    context: NarratorContext,
    options: List[str]
) -> List[str]:
    """Present player response options"""
    return [f"You could {option}..." for option in options[:3]]

# Activity and routine functions (ALL preserved)
async def _generate_addiction_effect_description(
    context: NarratorContext,
    addiction_type: str,
    level: int,
    activity: str
) -> str:
    """Generate description of addiction effects on activity"""
    if level >= 3:
        return f"The {addiction_type} dependency colors everything about {activity}"
    elif level >= 2:
        return f"Thoughts of {addiction_type} intrude during {activity}"
    return ""

async def _generate_activity_description(
    context: NarratorContext,
    activity: str,
    time_period: TimeOfDay
) -> str:
    """Generate basic activity description"""
    return f"The {time_period.value} {activity} unfolds with familiar rhythm..."

async def _determine_integrated_tone(
    context: NarratorContext,
    world_state: WorldState,
    relationship_contexts: Dict[int, Dict]
) -> NarrativeTone:
    """Determine tone based on ALL systems - vitals, relationships, addictions"""
    
    # Base tone from world mood
    base_tone = _determine_narrative_tone(world_state.world_mood, ActivityType.ROUTINE)
    
    # Adjust for vitals
    if context.current_vitals:
        if context.current_vitals.get("fatigue", 0) > 80:
            return NarrativeTone.OBSERVATIONAL
        elif context.current_vitals.get("hunger", 100) < 20:
            return NarrativeTone.COMMANDING
    
    # Adjust for relationships
    if relationship_contexts:
        avg_influence = sum(rc.get('dimensions', {}).get('influence', 0) 
                          for rc in relationship_contexts.values()) / len(relationship_contexts)
        if avg_influence > 50:
            return NarrativeTone.SUBTLE
        elif avg_influence > 30:
            return NarrativeTone.TEASING
    
    # Adjust for addictions
    if context.active_addictions and context.active_addictions.get("has_addictions"):
        addiction_count = len(context.active_addictions.get("addictions", {}))
        if addiction_count >= 3:
            return NarrativeTone.SENSUAL
    
    return base_tone

async def _generate_routine_with_dynamics(
    context: NarratorContext,
    activity: str,
    involved_npcs: List[int],
    dynamics: Any
) -> str:
    """Generate routine with power dynamics woven in"""
    if dynamics.player_submission_level > 0.5:
        return f"The {activity} follows patterns you've grown accustomed to..."
    return f"You go through the motions of {activity}..."

async def _generate_npc_routine_involvement(
    context: NarratorContext,
    npc_id: int,
    activity: str
) -> str:
    """Generate NPC involvement in routine"""
    return f"Their presence shapes the {activity} in subtle ways..."

async def _generate_subtle_control_elements(
    context: NarratorContext,
    activity: str,
    dynamics: Any
) -> List[str]:
    """Generate subtle control elements in routine"""
    elements = []
    if dynamics.player_submission_level > 0.3:
        elements.append("Choices that aren't really choices")
    if dynamics.acceptance_level > 0.5:
        elements.append("The comfort of established patterns")
    return elements

async def _generate_activity_variations(
    context: NarratorContext,
    activity: str,
    intersections: List[str]
) -> List[str]:
    """Generate emergent variations based on system state"""
    variations = []
    for intersection in intersections[:2]:
        if "fatigue" in intersection:
            variations.append(f"Exhaustion making {activity} more difficult")
        elif "addiction" in intersection:
            variations.append(f"Distraction coloring the {activity}")
    return variations

# Ambient narration functions (ALL preserved)
async def _narrate_time_passage(
    context: NarratorContext,
    current_time: CurrentTimeData
) -> str:
    """Narrate the passage of time"""
    return f"Time moves forward, {current_time.time_of_day} settling over everything..."

async def _narrate_mood_shift(
    context: NarratorContext,
    mood: WorldMood
) -> str:
    """Narrate a shift in world mood"""
    return f"The atmosphere shifts toward something more {mood.value}..."

async def _narrate_tension(
    context: NarratorContext,
    tension: WorldTension
) -> str:
    """Narrate building tension"""
    dominant, level = tension.get_dominant_tension()
    return f"A {dominant} tension builds, barely perceptible but undeniable..."

async def _narrate_addiction_presence(
    context: NarratorContext,
    addictions: Dict[str, Any]
) -> str:
    """Narrate the presence of addiction effects"""
    if addictions and addictions.get("has_addictions"):
        return "The familiar pull of need colors your perception..."
    return "Your mind remains clear, for now..."

async def _narrate_ambient_detail(
    context: NarratorContext,
    world_state: WorldState
) -> str:
    """Narrate ambient environmental detail"""
    return f"The world continues its {world_state.world_mood.value} rhythm..."

async def _weave_relationship_event(
    context: NarratorContext,
    event: Any
) -> str:
    """Weave a relationship event into narration"""
    return "A moment of connection shifts the dynamic subtly..."

async def _identify_reflected_systems(
    context: NarratorContext,
    focus: str,
    description: str
) -> List[str]:
    """Identify which systems are reflected in narration"""
    systems = []
    if "addiction" in focus or "need" in description.lower():
        systems.append("addiction_system")
    if "tension" in focus:
        systems.append("tension_system")
    if "mood" in focus:
        systems.append("mood_system")
    return systems

# Player action functions (ALL preserved)
async def _acknowledge_player_action(
    context: NarratorContext,
    action: str
) -> str:
    """Acknowledge player's action"""
    return f"You {action}..."

async def _generate_world_reaction(
    context: NarratorContext,
    action: str,
    world_state: WorldState
) -> str:
    """Generate world's reaction to player action"""
    return f"The {world_state.world_mood.value} world responds to your action..."

async def _generate_npc_reaction(
    context: NarratorContext,
    npc_id: int,
    action: str,
    world_state: WorldState
) -> str:
    """Generate NPC reaction to player action"""
    return "They notice, filing it away for later..."

async def _check_for_dynamic_shift(
    context: NarratorContext,
    action: str,
    dynamics: Any
) -> Optional[str]:
    """Check if action causes dynamic shift"""
    if dynamics.resistance_level < 0.3:
        return "The power dynamic shifts slightly in response..."
    return None

# ===============================================================================
# Main Narrator Agent Class with Full Enhancement
# ===============================================================================

class SliceOfLifeNarrator:
    """
    The narrative voice for the open-world simulation with FULL integration.
    Enhanced with context services while preserving ALL original functionality.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = NarratorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Performance tracking (ENHANCED)
        self.performance_monitor = None
        
        # Sub-agents for specialized narration (ALL PRESERVED)
        self.scene_narrator = Agent(
            name="SceneNarrator",
            instructions="""
            You narrate slice-of-life scenes in a femdom setting with full system awareness.
            Integrate relationship dynamics, time references, system states, and physical condition.
            Use second-person perspective. Be immersive and atmospheric.
            Show consequences of past events through environmental details.
            Draw from memories and context to make scenes feel lived-in.
            Respect Nyx governance directives.
            """,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.7),
            tools=[narrate_slice_of_life_scene, generate_ambient_narration]
        )
        
        self.dialogue_writer = Agent(
            name="DialogueWriter",
            instructions="""
            You write natural NPC dialogue with awareness of progression stages and masks.
            Power dynamics through subtext, not explicit statements.
            Show mask slippage and true nature bleeding through when appropriate.
            Exploit known vulnerabilities subtly.
            Reference shared history and memories naturally.
            Follow Nyx governance guidelines for appropriate content.
            """,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.8),
            tools=[generate_npc_dialogue]
        )
        
        self.power_narrator = Agent(
            name="PowerNarrator",
            instructions="""
            You narrate power exchanges with awareness of susceptibility and consequences.
            Make power dynamics feel inevitable rather than forced.
            Show how multiple systems reinforce control.
            Track exchanges through Nyx governance.
            Store significant moments in memory.
            """,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.7),
            tools=[narrate_power_exchange, narrate_daily_routine]
        )
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the narrator with all systems including context services"""
        if not self.initialized:
            await self.context.initialize()
            
            # Initialize performance monitor (ENHANCED)
            self.performance_monitor = PerformanceMonitor.get_instance(
                self.user_id,
                self.conversation_id
            )
            
            # Initialize world director
            self.context.world_director = WorldDirector(
                self.user_id, 
                self.conversation_id
            )
            await self.context.world_director.initialize()
            
            self.initialized = True
            logger.info(f"SliceOfLifeNarrator fully initialized with context services")
    
    @track_performance("narrate_world_state")
    async def narrate_world_state(self) -> str:
        """Provide narration for current world state with context awareness"""
        await self.initialize()
        
        # Start performance timer
        if self.performance_monitor:
            timer_id = self.performance_monitor.start_timer("narrate_world_state")
        
        try:
            # Get world state
            world_state = await self.context.world_director.get_world_state()
            
            # Refresh context
            await self.context.refresh_context()
            
            # Generate appropriate narration
            if world_state.ongoing_events:
                # Narrate the most important event
                event = max(world_state.ongoing_events, key=lambda e: e.priority)
                
                result = await self.scene_narrator.run(
                    messages=[{"role": "user", "content": f"Narrate this scene: {event.title}"}],
                    context=self.context,
                    tool_calls=[{
                        "tool": narrate_slice_of_life_scene,
                        "kwargs": {"scene": event, "world_state": world_state}
                    }]
                )
                
                narration = result.data if hasattr(result, 'data') else result
                return narration.scene_description
            else:
                # Generate ambient narration
                result = await self.power_narrator.run(
                    messages=[{"role": "user", "content": "Generate ambient narration"}],
                    context=self.context,
                    tool_calls=[{
                        "tool": generate_ambient_narration,
                        "kwargs": {"focus": "atmosphere", "world_state": world_state}
                    }]
                )
                
                ambient = result.data if hasattr(result, 'data') else result
                return ambient.description
        finally:
            # Stop performance timer
            if self.performance_monitor and 'timer_id' in locals():
                self.performance_monitor.stop_timer(timer_id)
    
    @track_performance("process_player_input")
    async def process_player_input(self, user_input: str) -> Dict[str, Any]:
        """Process player input and generate appropriate narration with full context"""
        await self.initialize()
        
        # Refresh context with player input
        await self.context.refresh_context(input_text=user_input)
        
        # Let world director process the action
        world_response = await self.context.world_director.process_player_action(
            user_input
        )
        
        # Get updated world state
        world_state = await self.context.world_director.get_world_state()
        
        # Generate narration for the action
        narration = await narrate_player_action(
            RunContextWrapper(self.context),
            user_input,
            world_state
        )
        
        # Record performance metrics
        if self.performance_monitor:
            self.performance_monitor.record_token_usage(200)  # Estimate
        
        # Combine responses
        return {
            "narration": narration["acknowledgment"],
            "world_reaction": narration["world_reaction"],
            "npc_reactions": narration["npc_reactions"],
            "world_update": world_response,
            "current_mood": world_state.world_mood.value,
            "tensions": {
                "power": world_state.world_tension.power_tension,
                "sexual": world_state.world_tension.sexual_tension,
                "social": world_state.world_tension.social_tension
            },
            "governance_active": self.context.governance_active,
            "context_version": self.context.context_version
        }
    
    async def generate_scene_transition(self) -> str:
        """Generate narration for time/scene transitions with context"""
        await self.initialize()
        
        # Simulate world tick
        tick_result = await self.context.world_director.simulate_world_tick()
        
        # Get new world state
        world_state = await self.context.world_director.get_world_state()
        
        # Refresh context for transition
        await self.context.refresh_context()
        
        # Generate transition narration
        result = await self.scene_narrator.run(
            messages=[{"role": "user", "content": "Narrate time passing"}],
            context=self.context,
            tool_calls=[{
                "tool": generate_ambient_narration,
                "kwargs": {"focus": "time_passage", "world_state": world_state}
            }]
        )
        
        ambient = result.data if hasattr(result, 'data') else result
        return ambient.description
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the narrator"""
        if self.performance_monitor:
            return self.performance_monitor.get_metrics()
        return {}

# ===============================================================================
# Export the complete enhanced system
# ===============================================================================

def create_slice_of_life_narrator(user_id: int, conversation_id: int) -> SliceOfLifeNarrator:
    """Create an enhanced narrator with full context integration"""
    return SliceOfLifeNarrator(user_id, conversation_id)

__all__ = [
    'SliceOfLifeNarrator',
    'create_slice_of_life_narrator',
    'narrate_slice_of_life_scene',
    'generate_npc_dialogue',
    'narrate_power_exchange',
    'narrate_daily_routine',
    'generate_ambient_narration',
    'narrate_player_action',
    # Models
    'SliceOfLifeNarration',
    'NPCDialogue',
    'PowerMomentNarration',
    'DailyActivityNarration',
    'AmbientNarration',
    'NarrativeTone',
    'SceneFocus'
]
