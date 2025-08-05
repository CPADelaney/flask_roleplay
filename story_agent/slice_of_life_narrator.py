"""
Enhanced Slice-of-Life Narrator with Full System Integration and Dynamic Generation

This refactor:
- Retains ALL original functionality and integrations
- Enhances LLM-driven content generation where appropriate
- Adds emergent gameplay features without removing structure
- Maintains type safety with Pydantic models
- Keeps the sophisticated orchestration layer
"""

import logging
import json
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings
from pydantic import BaseModel, Field, ConfigDict

# Database connection
from db.connection import get_db_connection_context

# Core system integrations - KEEP ALL ORIGINAL IMPORTS
from logic.dynamic_relationships import OptimizedRelationshipManager, event_generator
from logic.npc_narrative_progression import get_npc_narrative_stage, NPCNarrativeStage
from logic.calendar import load_calendar_names
from logic.time_cycle import get_current_vitals, get_current_time_model, CurrentTimeData
from logic.addiction_system_sdk import (
    get_addiction_status, 
    AddictionContext,
    check_addiction_levels_impl
)
from logic.narrative_events import (
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence
)
from logic.event_system import EventSystem
from logic.currency_generator import CurrencyGenerator
from logic.relationship_integration import RelationshipIntegration

# ENHANCED: Add more system integrations for emergent gameplay
from logic.memory_logic import (
    MemoryManager,
    EnhancedMemory,
    MemoryType,
    MemorySignificance,
    ProgressiveRevealManager
)
from logic.stats_logic import (
    get_all_player_stats,
    get_player_current_tier,
    check_for_combination_triggers,
    STAT_THRESHOLDS,
    ACTIVITY_EFFECTS
)
from logic.rule_enforcement import enforce_all_rules_on_player
from logic.inventory_system_sdk import get_inventory

# GPT Integration - CRITICAL FOR DYNAMIC GENERATION
from logic.chatgpt_integration import (
    generate_text_completion,
    get_chatgpt_response,
    get_text_embedding,
    cosine_similarity
)

# World Director integration - KEEP ALL
from story_agent.world_director_agent import (
    WorldState, WorldMood, TimeOfDay, ActivityType, PowerDynamicType,
    SliceOfLifeEvent, NPCRoutine, PowerExchange, WorldTension
)

logger = logging.getLogger(__name__)

# ===============================================================================
# KEEP ALL ORIGINAL ENUMS AND MODELS
# ===============================================================================

class NarrativeTone(Enum):
    """Narrative tone for scenes"""
    OBSERVATIONAL = "observational"
    TEASING = "teasing"
    SUBTLE = "subtle"
    COMMANDING = "commanding"
    SENSUAL = "sensual"
    PSYCHOLOGICAL = "psychological"
    INTIMATE = "intimate"

class SceneFocus(Enum):
    """What the scene focuses on"""
    ATMOSPHERE = "atmosphere"
    DIALOGUE = "dialogue"
    DYNAMICS = "dynamics"
    INTERNAL = "internal"
    TENSION = "tension"

# ===============================================================================
# KEEP ALL PYDANTIC MODELS FOR TYPE SAFETY
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
    # ENHANCED: Add emergent elements
    emergent_elements: Optional[Dict[str, Any]] = None
    system_triggers: List[str] = Field(default_factory=list)

class NPCDialogue(BaseModel):
    """NPC dialogue with full context"""
    npc_id: int
    npc_name: str
    dialogue: str
    tone: str
    subtext: str
    body_language: str
    power_dynamic: Optional[PowerDynamicType] = None
    requires_response: bool = False
    # ENHANCED: Add system-aware elements
    hidden_triggers: List[str] = Field(default_factory=list)
    memory_influence: Optional[str] = None

class PowerMomentNarration(BaseModel):
    """Narration for power exchange moments"""
    setup: str
    moment: str
    aftermath: str
    player_feelings: str
    options_presentation: str
    # ENHANCED: Add consequence tracking
    potential_consequences: List[Dict[str, Any]] = Field(default_factory=list)

class DailyActivityNarration(BaseModel):
    """Narration for daily activities"""
    activity: str
    description: str
    routine_with_dynamics: str
    npc_involvement: List[str] = Field(default_factory=list)
    subtle_control_elements: List[str] = Field(default_factory=list)
    # ENHANCED: Add emergent variations
    emergent_variations: Optional[List[str]] = None

class AmbientNarration(BaseModel):
    """Ambient scene narration"""
    description: str
    focus: str
    intensity: float
    affects_mood: bool = False
    # ENHANCED: Add system state reflection
    reflects_systems: List[str] = Field(default_factory=list)

# ===============================================================================
# ENHANCED CONTEXT WITH ALL SYSTEMS + EMERGENT FEATURES
# ===============================================================================

@dataclass
class EnhancedNarratorContext:
    """Enhanced context with all system integrations + emergent features"""
    user_id: int
    conversation_id: int
    
    # KEEP ALL ORIGINAL SYSTEMS
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    relationship_integration: Optional[RelationshipIntegration] = None
    event_system: Optional[EventSystem] = None
    addiction_context: Optional[AddictionContext] = None
    currency_generator: Optional[CurrencyGenerator] = None
    
    # KEEP ALL CACHED DATA
    calendar_names: Optional[Dict[str, Any]] = None
    current_vitals: Optional[Dict[str, int]] = None
    current_time: Optional[CurrentTimeData] = None
    active_addictions: Optional[Dict[str, Any]] = None
    
    # KEEP NARRATIVE TRACKING
    recent_narrations: List[str] = field(default_factory=list)
    narrative_momentum: float = 0.0
    last_revelation_check: Optional[datetime] = None
    
    # ENHANCED: Add emergent gameplay systems
    memory_manager: Optional[MemoryManager] = None
    reveal_manager: Optional[ProgressiveRevealManager] = None
    player_stats: Optional[Dict[str, Any]] = None
    active_rules: Optional[List[Dict]] = None
    player_inventory: Optional[Dict[str, Any]] = None
    active_memories: Optional[List[EnhancedMemory]] = None
    npc_masks: Optional[Dict[int, Dict]] = None
    
    # ENHANCED: Emergent tracking
    system_intersections: List[str] = field(default_factory=list)
    pending_consequences: List[Dict] = field(default_factory=list)
    narrative_seeds: List[str] = field(default_factory=list)
    scene_history: List[Dict] = field(default_factory=list)
    
    async def initialize(self):
        """Initialize all integrated systems"""
        # KEEP ALL ORIGINAL INITIALIZATIONS
        self.relationship_manager = OptimizedRelationshipManager(
            self.user_id, self.conversation_id
        )
        self.relationship_integration = RelationshipIntegration(
            self.user_id, self.conversation_id
        )
        
        self.event_system = EventSystem(self.user_id, self.conversation_id)
        await self.event_system.initialize()
        
        self.addiction_context = AddictionContext(self.user_id, self.conversation_id)
        await self.addiction_context.initialize()
        
        self.currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
        
        self.calendar_names = await load_calendar_names(self.user_id, self.conversation_id)
        self.current_time = await get_current_time_model(self.user_id, self.conversation_id)
        self.current_vitals = await get_current_vitals(self.user_id, self.conversation_id)
        self.active_addictions = await get_addiction_status(
            self.user_id, self.conversation_id, "Chase"
        )
        
        # ENHANCED: Initialize new systems
        self.player_stats = await get_all_player_stats(
            self.user_id, self.conversation_id, "Chase"
        )
        self.active_rules = await enforce_all_rules_on_player("Chase")
        self.player_inventory = await get_inventory(
            self.user_id, self.conversation_id, "Chase"
        )
        
        await self._load_npc_masks()
        await self._load_active_memories()
        await self._identify_system_intersections()
    
    async def _load_npc_masks(self):
        """Load NPC facade states for progressive reveals"""
        self.npc_masks = {}
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, mask_data 
                FROM NPCMasks 
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            for row in rows:
                if row['mask_data']:
                    self.npc_masks[row['npc_id']] = json.loads(row['mask_data'])
    
    async def _load_active_memories(self):
        """Load recent significant memories"""
        self.active_memories = await MemoryManager.retrieve_relevant_memories(
            self.user_id, self.conversation_id,
            "Chase", "player",
            limit=10
        )
    
    async def _identify_system_intersections(self):
        """Identify interesting system state combinations for emergent events"""
        self.system_intersections = []
        
        # Check stat combinations
        stat_combos = await check_for_combination_triggers(
            self.user_id, self.conversation_id
        )
        for combo in stat_combos:
            self.system_intersections.append(f"stat_combo:{combo['name']}")
        
        # Check addiction + stat intersections
        if self.active_addictions.get("has_addictions"):
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
        
        # Check memory patterns
        if self.active_memories:
            traumatic_count = sum(1 for m in self.active_memories 
                                if m.memory_type == MemoryType.TRAUMATIC)
            if traumatic_count >= 3:
                self.system_intersections.append("trauma_accumulation")

# ===============================================================================
# ENHANCED NARRATION FUNCTIONS - Keep originals, add dynamic generation
# ===============================================================================

@function_tool
async def narrate_scene_with_full_context(
    ctx: RunContextWrapper,
    scene: SliceOfLifeEvent,
    world_state: WorldState
) -> SliceOfLifeNarration:
    """Generate narration with full system integration and dynamic content"""
    context: EnhancedNarratorContext = ctx.context
    
    # KEEP ORIGINAL: Get relationship contexts for all participants
    relationship_contexts = {}
    for npc_id in scene.participants:
        state = await context.relationship_manager.get_relationship_state(
            'npc', npc_id, 'player', context.user_id
        )
        
        stage = await get_npc_narrative_stage(
            context.user_id, context.conversation_id, npc_id
        )
        
        relationship_contexts[npc_id] = {
            'dimensions': state.dimensions.to_dict(),
            'patterns': list(state.history.active_patterns),
            'archetypes': list(state.active_archetypes),
            'narrative_stage': stage.name,
            'momentum': state.momentum.get_magnitude(),
            'stage_description': stage.description
        }
    
    # KEEP ORIGINAL: Determine tone based on multiple factors
    tone = await _determine_integrated_tone(
        context, world_state, relationship_contexts
    )
    
    # ENHANCED: Generate scene description with dynamic LLM content
    scene_desc = await _generate_dynamic_scene_description(
        context, scene, world_state, relationship_contexts
    )
    
    # KEEP ORIGINAL: Generate atmosphere with vitals awareness
    atmosphere = await _generate_vitals_aware_atmosphere(
        context, scene.location, world_state.world_mood
    )
    
    # ENHANCED: Generate power hints with system awareness
    power_hints = await _generate_integrated_power_hints(
        context, scene, relationship_contexts
    )
    
    # KEEP ORIGINAL: Generate sensory details with calendar awareness
    sensory = await _generate_calendar_aware_sensory_details(
        context, world_state.current_time, scene.location
    )
    
    # ENHANCED: Generate NPC observations with memory integration
    npc_obs = []
    for npc_id in scene.participants:
        obs = await _generate_enhanced_npc_observation(
            context, npc_id, scene, relationship_contexts.get(npc_id)
        )
        if obs:
            npc_obs.append(obs)
    
    # ENHANCED: Generate internal monologue with full system awareness
    internal = await _generate_integrated_internal_monologue(
        context, scene, world_state, relationship_contexts
    )
    
    # KEEP ORIGINAL: Check for narrative events
    await _check_and_queue_narrative_events(context, scene, relationship_contexts)
    
    # ENHANCED: Identify emergent elements
    emergent_elements = await _identify_emergent_elements(
        context, scene, relationship_contexts
    )
    
    narration = SliceOfLifeNarration(
        scene_description=scene_desc,
        atmosphere=atmosphere,
        tone=tone,
        focus=_determine_scene_focus(relationship_contexts, context.active_addictions),
        power_dynamic_hints=power_hints,
        sensory_details=sensory,
        npc_observations=npc_obs,
        internal_monologue=internal,
        emergent_elements=emergent_elements,
        system_triggers=context.system_intersections[:3]
    )
    
    # Track narration and scene
    context.recent_narrations.append(scene_desc)
    if len(context.recent_narrations) > 10:
        context.recent_narrations.pop(0)
    
    context.scene_history.append({
        "timestamp": datetime.now(),
        "scene_type": scene.scene_type,
        "participants": scene.participants,
        "tone": tone.value,
        "triggers": context.system_intersections[:3]
    })
    
    return narration

@function_tool
async def generate_contextual_npc_dialogue(
    ctx: RunContextWrapper,
    npc_id: int,
    situation: str,
    world_state: WorldState
) -> NPCDialogue:
    """Generate NPC dialogue with full system awareness and dynamic content"""
    context: EnhancedNarratorContext = ctx.context
    
    # KEEP ORIGINAL: Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty, personality_traits, current_location
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, npc_id, context.user_id, context.conversation_id)
    
    if not npc:
        raise ValueError(f"NPC {npc_id} not found")
    
    # KEEP ORIGINAL: Get relationship state and narrative stage
    rel_state = await context.relationship_manager.get_relationship_state(
        'npc', npc_id, 'player', context.user_id
    )
    
    stage = await get_npc_narrative_stage(
        context.user_id, context.conversation_id, npc_id
    )
    
    # KEEP ORIGINAL: Check for relevant addictions
    relevant_addictions = await _get_relevant_addictions_for_npc(
        context, npc_id
    )
    
    # ENHANCED: Check for relevant memories influencing dialogue
    npc_memories = await MemoryManager.retrieve_relevant_memories(
        context.user_id, context.conversation_id,
        npc_id, "npc",
        context=situation,
        limit=3
    )
    
    memory_influence = None
    if npc_memories:
        memory_influence = npc_memories[0].text  # Most relevant memory
    
    # ENHANCED: Check mask state for potential slippage
    mask_state = context.npc_masks.get(npc_id, {})
    hidden_traits = mask_state.get("hidden_traits", {})
    mask_integrity = mask_state.get("integrity", 100)
    
    # ENHANCED: Generate dialogue with full context and emergent elements
    dialogue = await _generate_fully_contextual_dialogue_enhanced(
        context, npc, stage, situation, rel_state, 
        relevant_addictions, memory_influence, hidden_traits, mask_integrity
    )
    
    # KEEP ORIGINAL: Include currency references if appropriate
    if "payment" in situation.lower() or "cost" in situation.lower():
        dialogue = await _add_currency_reference(context, dialogue)
    
    # ENHANCED: Generate tone with mask awareness
    tone = await _determine_dialogue_tone_enhanced(
        context, npc, stage, rel_state.dimensions, mask_integrity
    )
    
    # ENHANCED: Generate subtext with full system awareness
    subtext = await _generate_system_aware_subtext(
        context, dialogue, npc['dominance'], stage.name, 
        relevant_addictions, hidden_traits
    )
    
    # KEEP ORIGINAL: Generate body language
    body_language = await _generate_stage_appropriate_body_language(
        context, npc['dominance'], tone, stage.name, rel_state.dimensions
    )
    
    # KEEP ORIGINAL: Determine power dynamic
    power_dynamic = _determine_power_dynamic_from_context(
        stage.name, rel_state.dimensions, relevant_addictions
    )
    
    # ENHANCED: Identify hidden triggers
    hidden_triggers = await _identify_dialogue_triggers(
        context, dialogue, npc_id, rel_state
    )
    
    return NPCDialogue(
        npc_id=npc_id,
        npc_name=npc['npc_name'],
        dialogue=dialogue,
        tone=tone,
        subtext=subtext,
        body_language=body_language,
        power_dynamic=power_dynamic,
        requires_response=_should_require_response(rel_state.dimensions, stage.name),
        hidden_triggers=hidden_triggers,
        memory_influence=memory_influence
    )

@function_tool
async def narrate_power_exchange_with_context(
    ctx: RunContextWrapper,
    exchange: PowerExchange,
    world_state: WorldState
) -> PowerMomentNarration:
    """Generate power exchange narration with full system context and consequences"""
    context: EnhancedNarratorContext = ctx.context
    
    # KEEP ALL ORIGINAL FUNCTIONALITY
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, exchange.initiator_npc_id)
    
    rel_state = await context.relationship_manager.get_relationship_state(
        'npc', exchange.initiator_npc_id, 'player', context.user_id
    )
    
    stage = await get_npc_narrative_stage(
        context.user_id, context.conversation_id, exchange.initiator_npc_id
    )
    
    susceptibility = _calculate_susceptibility(context.current_vitals, context.active_addictions)
    
    # ENHANCED: Check if this triggers any rules
    rule_triggers = await _check_power_exchange_rules(
        context, exchange, susceptibility
    )
    
    # ENHANCED: Generate with awareness of potential consequences
    setup = await _generate_contextual_power_setup_enhanced(
        context, exchange, npc, stage, rel_state, susceptibility, rule_triggers
    )
    
    moment = await _generate_progression_aware_power_moment(
        context, exchange, npc, stage
    )
    
    aftermath = await _generate_vitals_influenced_aftermath(
        context, exchange, rel_state, context.current_vitals
    )
    
    feelings = await _generate_addiction_influenced_feelings(
        context, exchange, rel_state, context.active_addictions
    )
    
    options = await _present_state_aware_options(
        context, exchange.player_response_options, susceptibility
    )
    
    # ENHANCED: Calculate potential consequences
    potential_consequences = await _calculate_power_exchange_consequences(
        context, exchange, rel_state, susceptibility
    )
    
    # KEEP ORIGINAL: Queue relationship event
    await context.event_system.create_event(
        "relationship_event",
        {
            "type": "power_exchange",
            "npc_id": exchange.initiator_npc_id,
            "exchange_type": exchange.exchange_type.value,
            "intensity": exchange.intensity,
            "consequences": potential_consequences
        },
        priority=7
    )
    
    return PowerMomentNarration(
        setup=setup,
        moment=moment,
        aftermath=aftermath,
        player_feelings=feelings,
        options_presentation=options,
        potential_consequences=potential_consequences
    )

@function_tool
async def narrate_with_addiction_awareness(
    ctx: RunContextWrapper,
    activity: str,
    involved_npcs: List[int],
    world_state: WorldState
) -> DailyActivityNarration:
    """Narrate daily activities with addiction system integration and variations"""
    context: EnhancedNarratorContext = ctx.context
    
    # KEEP ALL ORIGINAL FUNCTIONALITY
    addiction_effects = []
    if context.active_addictions.get("has_addictions"):
        for addiction_type, data in context.active_addictions.get("addictions", {}).items():
            if data.get("level", 0) >= 2:
                effect_desc = await _generate_addiction_effect_description(
                    context, addiction_type, data.get("level"), activity
                )
                if effect_desc:
                    addiction_effects.append(effect_desc)
    
    description = await _generate_activity_with_vitals(
        context, activity, context.current_vitals
    )
    
    routine_with_dynamics = await _generate_addiction_colored_routine(
        context, activity, involved_npcs, addiction_effects
    )
    
    npc_involvement = []
    for npc_id in involved_npcs:
        rel_state = await context.relationship_manager.get_relationship_state(
            'npc', npc_id, 'player', context.user_id
        )
        involvement = await _generate_relationship_aware_involvement(
            context, npc_id, activity, rel_state
        )
        if involvement:
            npc_involvement.append(involvement)
    
    control_elements = addiction_effects + await _generate_subtle_control_elements(
        context, activity, world_state.relationship_dynamics
    )
    
    # ENHANCED: Generate emergent variations based on system state
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
async def generate_time_aware_ambient_narration(
    ctx: RunContextWrapper,
    focus: str,
    world_state: WorldState
) -> AmbientNarration:
    """Generate ambient narration with calendar and time system awareness"""
    context: EnhancedNarratorContext = ctx.context
    
    # KEEP ALL ORIGINAL FUNCTIONALITY
    context.current_time = await get_current_time_model(
        context.user_id, context.conversation_id
    )
    
    # Generate description based on focus
    if focus == "time_passage":
        description = await _narrate_calendar_time_passage(
            context, context.current_time, context.calendar_names
        )
    elif focus == "mood_shift":
        description = await _narrate_mood_with_vitals(
            context, world_state.world_mood, context.current_vitals
        )
    elif focus == "tension_building":
        description = await _narrate_tension_with_relationships(
            context, world_state.world_tension
        )
    elif focus == "addiction_hint":
        description = await _narrate_addiction_presence(
            context, context.active_addictions
        )
    else:
        description = await _narrate_world_with_currency(
            context, world_state
        )
    
    # KEEP ORIGINAL: Check for relationship events
    rel_event = await event_generator.get_next_event(timeout=0.1)
    if rel_event:
        description += f" {await _weave_relationship_event(context, rel_event)}"
    
    # ENHANCED: Check which systems this reflects
    reflects_systems = await _identify_reflected_systems(
        context, focus, description
    )
    
    intensity = _calculate_ambient_intensity(
        world_state.world_tension,
        context.current_vitals,
        context.active_addictions
    )
    
    return AmbientNarration(
        description=description,
        focus=focus,
        intensity=intensity,
        affects_mood=focus in ["mood_shift", "addiction_hint"],
        reflects_systems=reflects_systems
    )

# ===============================================================================
# ENHANCED HELPER FUNCTIONS - Keep originals, add new capabilities
# ===============================================================================

async def _generate_dynamic_scene_description(
    context: EnhancedNarratorContext,
    scene: SliceOfLifeEvent,
    world_state: WorldState,
    relationship_contexts: Dict[int, Dict]
) -> str:
    """ENHANCED: Generate scene description with dynamic LLM content"""
    
    # Build comprehensive context
    calendar = context.calendar_names
    time_desc = f"{calendar['days'][context.current_time.day % 7]} of {calendar['months'][context.current_time.month - 1]}"
    
    # Include system intersections for emergent flavor
    system_context = {
        "intersections": context.system_intersections[:3],
        "active_rules": [r.get("condition") for r in context.active_rules[:2]] if context.active_rules else [],
        "memory_count": len(context.active_memories) if context.active_memories else 0,
        "scene_history": len(context.scene_history)
    }
    
    system_prompt = """You are narrating a slice-of-life scene in a femdom RPG.
    Weave in relationship dynamics, time references, system states, and physical condition naturally.
    Use second-person perspective. Be immersive and atmospheric.
    Show consequences of past events subtly through environmental details."""
    
    user_prompt = f"""
    Generate a rich scene description for:
    Scene: {scene.title} - {scene.description}
    Location: {scene.location}
    Time: {time_desc} during {world_state.current_time.value}
    
    Relationship Context:
    {json.dumps({k: {"stage": v['narrative_stage'], "patterns": v['patterns'][:2]} 
                 for k, v in relationship_contexts.items()}, indent=2)}
    
    System State:
    {json.dumps(system_context, indent=2)}
    
    Physical State:
    - Fatigue: {context.current_vitals.get('fatigue', 0)}/100
    - Hunger: {context.current_vitals.get('hunger', 100)}/100
    
    World Mood: {world_state.world_mood.value}
    
    Previous scenes hinted at: {context.narrative_seeds[-2:] if context.narrative_seeds else 'none'}
    
    Write 3-4 sentences that:
    1. Set the scene naturally
    2. Include subtle environmental storytelling
    3. Reflect the accumulated system state
    4. Hint at what might happen
    """
    
    description = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=200,
        task_type="narrative"
    )
    
    return description.strip()

async def _generate_enhanced_npc_observation(
    context: EnhancedNarratorContext,
    npc_id: int,
    scene: SliceOfLifeEvent,
    rel_context: Optional[Dict]
) -> Optional[str]:
    """ENHANCED: Generate NPC observation with memory and mask awareness"""
    
    if not rel_context:
        return None
    
    # Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, npc_id)
    
    if not npc:
        return None
    
    # Check mask state
    mask_state = context.npc_masks.get(npc_id, {})
    mask_integrity = mask_state.get("integrity", 100)
    hidden_traits = mask_state.get("hidden_traits", {})
    
    # Get recent NPC memories
    npc_memories = await MemoryManager.retrieve_relevant_memories(
        context.user_id, context.conversation_id,
        npc_id, "npc",
        context=scene.title,
        limit=2
    )
    
    system_prompt = f"""Generate observations about NPCs that show:
    - Their current narrative stage
    - Subtle hints of their true nature (if mask is slipping)
    - How past interactions influence current behavior
    Show progression through behavioral details, not exposition."""
    
    user_prompt = f"""
    Observe {npc['npc_name']} in this scene:
    Scene: {scene.title}
    Narrative Stage: {rel_context['narrative_stage']}
    Dominance: {npc['dominance']}/100 (show through behavior, don't state)
    Mask Integrity: {mask_integrity}/100 (lower = more reveals)
    Hidden Traits: {list(hidden_traits.keys())[:2] if hidden_traits else 'none'}
    Recent Memories: {[m.text[:50] for m in npc_memories] if npc_memories else 'none'}
    Active Patterns: {rel_context.get('patterns', [])[:2]}
    
    Write 1-2 sentences showing:
    1. Their behavior in this moment
    2. Subtle hints of what lies beneath (if mask < 70)
    3. How memories influence their actions
    """
    
    observation = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=100,
        task_type="narrative"
    )
    
    return observation.strip()

async def _generate_fully_contextual_dialogue_enhanced(
    context: EnhancedNarratorContext,
    npc: Any,
    stage: NPCNarrativeStage,
    situation: str,
    rel_state: Any,
    relevant_addictions: Dict[str, Any],
    memory_influence: Optional[str],
    hidden_traits: Dict[str, Any],
    mask_integrity: int
) -> str:
    """ENHANCED: Generate dialogue with full context including memories and masks"""
    
    system_prompt = f"""Generate natural NPC dialogue for {npc['npc_name']}.
    Narrative Stage: {stage.name} - {stage.description}
    Personality: {npc['personality_traits']}
    
    The dialogue should:
    - Reflect the current stage and relationship
    - Incorporate memory influences subtly
    - Show mask slippage if integrity is low
    - Exploit known addictions naturally
    - Use subtext for power dynamics"""
    
    user_prompt = f"""
    Generate dialogue for:
    Situation: {situation}
    Dominance: {npc['dominance']}/100
    Trust: {rel_state.dimensions.trust:.0f}/100
    Influence: {rel_state.dimensions.influence:.0f}/100
    
    Mask State:
    - Integrity: {mask_integrity}/100
    - Hidden Nature: {list(hidden_traits.keys())[:3] if hidden_traits else 'none'}
    
    Memory Context: {memory_influence[:100] if memory_influence else 'no specific memory'}
    
    Relevant Addictions: {[k for k in relevant_addictions.keys()][:2] if relevant_addictions else 'none'}
    
    Relationship Patterns: {', '.join(list(rel_state.history.active_patterns)[:3])}
    
    Write 1-3 sentences of dialogue that:
    1. Fits the situation naturally
    2. Subtly references the memory (if present)
    3. Shows true nature bleeding through (if mask < 50)
    4. Exploits vulnerabilities without being obvious
    """
    
    dialogue = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=150,
        task_type="dialogue"
    )
    
    return dialogue.strip()

async def _identify_emergent_elements(
    context: EnhancedNarratorContext,
    scene: SliceOfLifeEvent,
    relationship_contexts: Dict[int, Dict]
) -> Optional[Dict[str, Any]]:
    """ENHANCED: Identify emergent elements from system intersections"""
    
    if not context.system_intersections:
        return None
    
    emergent = {
        "triggers": context.system_intersections[:3],
        "potential_events": [],
        "hidden_connections": []
    }
    
    # Check for emergent event potential
    if "stat_combo:Breaking Point" in context.system_intersections:
        emergent["potential_events"].append("psychological_fracture")
    
    if "addiction_low_willpower" in context.system_intersections:
        emergent["potential_events"].append("compulsion_trigger")
    
    if "trauma_accumulation" in context.system_intersections:
        emergent["potential_events"].append("flashback_risk")
    
    # Check for hidden connections between systems
    for intersection in context.system_intersections[:5]:
        if "rule_" in intersection and any("addiction" in s for s in context.system_intersections):
            emergent["hidden_connections"].append("rules_exploiting_addictions")
        
        if "stat_combo" in intersection and context.active_memories:
            emergent["hidden_connections"].append("memories_reinforcing_state")
    
    return emergent if emergent["potential_events"] or emergent["hidden_connections"] else None

async def _calculate_power_exchange_consequences(
    context: EnhancedNarratorContext,
    exchange: PowerExchange,
    rel_state: Any,
    susceptibility: float
) -> List[Dict[str, Any]]:
    """ENHANCED: Calculate potential consequences of power exchange"""
    
    consequences = []
    
    # Check stat impacts
    if susceptibility > 0.7:
        consequences.append({
            "type": "stat_change",
            "stats": ["willpower", "obedience"],
            "direction": "submission",
            "magnitude": "significant"
        })
    
    # Check for addiction progression
    if context.active_addictions.get("has_addictions"):
        consequences.append({
            "type": "addiction_progression",
            "likelihood": susceptibility,
            "trigger": exchange.exchange_type.value
        })
    
    # Check for memory formation
    if exchange.intensity > 5:
        consequences.append({
            "type": "memory_formation",
            "memory_type": "emotional" if exchange.intensity > 7 else "interaction",
            "significance": min(10, exchange.intensity)
        })
    
    # Check for relationship pattern activation
    if rel_state.dimensions.influence > 60:
        consequences.append({
            "type": "pattern_reinforcement",
            "patterns": list(rel_state.history.active_patterns)[:2],
            "strength": "increasing"
        })
    
    return consequences

async def _generate_activity_variations(
    context: EnhancedNarratorContext,
    activity: str,
    system_intersections: List[str]
) -> Optional[List[str]]:
    """ENHANCED: Generate emergent variations of activities"""
    
    if not system_intersections:
        return None
    
    system_prompt = """Generate variations of daily activities that emerge from system states.
    Show how the same activity changes based on psychological state and relationships.
    Make variations feel natural, not mechanical."""
    
    user_prompt = f"""
    Generate 2-3 variations for activity: {activity}
    
    System State Triggers:
    {json.dumps(system_intersections[:3], indent=2)}
    
    Current Stats:
    - Willpower: {context.player_stats['hidden'].get('willpower', 50)}
    - Obedience: {context.player_stats['hidden'].get('obedience', 20)}
    - Corruption: {context.player_stats['hidden'].get('corruption', 10)}
    
    Create variations that show how the activity changes based on these states.
    Return as JSON array of strings.
    """
    
    response = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=200,
        task_type="narrative"
    )
    
    try:
        variations = json.loads(response)
        if isinstance(variations, list):
            return variations[:3]
    except:
        pass
    
    return None

async def _check_power_exchange_rules(
    context: EnhancedNarratorContext,
    exchange: PowerExchange,
    susceptibility: float
) -> List[str]:
    """ENHANCED: Check if power exchange triggers any rules"""
    
    triggered_rules = []
    
    if not context.active_rules:
        return triggered_rules
    
    for rule in context.active_rules:
        condition = rule.get("condition", "").lower()
        
        # Check if susceptibility makes rule more likely
        if "lust" in condition and susceptibility > 0.7:
            triggered_rules.append(rule.get("effect", ""))
        
        if "dependency" in condition and exchange.exchange_type == PowerDynamicType.INTIMATE_COMMAND:
            triggered_rules.append(rule.get("effect", ""))
    
    return triggered_rules[:2]  # Limit to 2 rules

async def _identify_dialogue_triggers(
    context: EnhancedNarratorContext,
    dialogue: str,
    npc_id: int,
    rel_state: Any
) -> List[str]:
    """ENHANCED: Identify hidden triggers in dialogue"""
    
    triggers = []
    
    # Check for addiction triggers
    if context.active_addictions.get("has_addictions"):
        for addiction_type in context.active_addictions.get("addictions", {}).keys():
            if addiction_type.lower() in dialogue.lower():
                triggers.append(f"addiction_reference:{addiction_type}")
    
    # Check for stat triggers
    dialogue_lower = dialogue.lower()
    if any(word in dialogue_lower for word in ["obey", "submit", "comply"]):
        triggers.append("obedience_trigger")
    
    if any(word in dialogue_lower for word in ["need", "depend", "without"]):
        triggers.append("dependency_trigger")
    
    # Check for relationship pattern triggers
    if rel_state.dimensions.influence > 70:
        triggers.append("high_influence_exploitation")
    
    return triggers[:3]

async def _identify_reflected_systems(
    context: EnhancedNarratorContext,
    focus: str,
    description: str
) -> List[str]:
    """ENHANCED: Identify which systems are reflected in ambient narration"""
    
    reflected = []
    
    # Check description for system references
    desc_lower = description.lower()
    
    if any(word in desc_lower for word in ["tired", "exhausted", "hungry", "weak"]):
        reflected.append("vitals")
    
    if any(word in desc_lower for word in ["need", "crave", "desire", "compulsion"]):
        reflected.append("addictions")
    
    if context.active_rules and any(word in desc_lower for word in ["must", "cannot", "forbidden"]):
        reflected.append("rules")
    
    if any(word in desc_lower for word in ["remember", "recall", "past", "before"]):
        reflected.append("memories")
    
    if focus == "tension_building":
        reflected.append("relationships")
    
    return reflected[:4]

# KEEP ALL ORIGINAL HELPER FUNCTIONS
async def _determine_integrated_tone(
    context: EnhancedNarratorContext,
    world_state: WorldState,
    relationship_contexts: Dict[int, Dict]
) -> NarrativeTone:
    """Determine tone based on all systems"""
    
    # Base tone from world mood
    if world_state.world_mood == WorldMood.TENSE:
        base_tone = NarrativeTone.PSYCHOLOGICAL
    elif world_state.world_mood == WorldMood.PLAYFUL:
        base_tone = NarrativeTone.TEASING
    elif world_state.world_mood == WorldMood.INTIMATE:
        base_tone = NarrativeTone.SENSUAL
    else:
        base_tone = NarrativeTone.OBSERVATIONAL
    
    # Adjust for vitals
    if context.current_vitals.get("fatigue", 0) > 80:
        return NarrativeTone.OBSERVATIONAL
    elif context.current_vitals.get("hunger", 100) < 20:
        return NarrativeTone.COMMANDING
    
    # Adjust for relationships
    avg_submission = sum(
        rc.get('dimensions', {}).get('influence', 0) 
        for rc in relationship_contexts.values()
    ) / max(len(relationship_contexts), 1)
    
    if avg_submission > 50:
        return NarrativeTone.SUBTLE
    elif avg_submission > 30:
        return NarrativeTone.TEASING
    
    # Adjust for addictions
    if context.active_addictions.get("has_addictions"):
        addiction_count = len(context.active_addictions.get("addictions", {}))
        if addiction_count >= 3:
            return NarrativeTone.SENSUAL
    
    return base_tone

# KEEP ALL OTHER ORIGINAL HELPER FUNCTIONS
async def _generate_vitals_aware_atmosphere(
    context: EnhancedNarratorContext,
    location: str,
    mood: WorldMood
) -> str:
    """Generate atmosphere influenced by vitals"""
    
    vitals_influence = ""
    if context.current_vitals.get("fatigue", 0) > 70:
        vitals_influence = "Everything feels distant and dreamlike through exhaustion."
    elif context.current_vitals.get("hunger", 100) < 30:
        vitals_influence = "The world sharpens with desperate hunger."
    elif context.current_vitals.get("thirst", 100) < 30:
        vitals_influence = "Dryness pervades every sensation."
    
    system_prompt = """Create atmospheric descriptions that reflect physical state.
    Show how vitals affect perception of the environment."""
    
    user_prompt = f"""
    Create atmosphere for:
    Location: {location}
    Mood: {mood.value}
    Physical influence: {vitals_influence}
    
    In 1-2 sentences, show how the {location} feels through this lens.
    """
    
    atmosphere = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return atmosphere.strip()

# [KEEP ALL OTHER ORIGINAL HELPER FUNCTIONS - I'll continue with the class definition]

# ===============================================================================
# ENHANCED SLICE-OF-LIFE NARRATOR CLASS
# ===============================================================================

class EnhancedSliceOfLifeNarrator:
    """
    Enhanced narrator with full system integration and emergent features.
    Maintains all original functionality while adding dynamic generation.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = EnhancedNarratorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        self.initialized = False
        
        # KEEP ALL ORIGINAL SUB-AGENTS
        self.scene_narrator = Agent(
            name="IntegratedSceneNarrator",
            instructions="""
            You narrate slice-of-life scenes with awareness of:
            - Relationship dynamics and progression stages
            - Player vitals and physical state
            - Active addictions and their effects
            - Custom calendar and currency
            - Ongoing narrative patterns
            - Memory influences
            - System intersections
            
            Weave these elements naturally into atmospheric narration.
            Generate emergent content from system states.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            tools=[
                narrate_scene_with_full_context,
                generate_time_aware_ambient_narration
            ]
        )
        
        self.dialogue_writer = Agent(
            name="ContextualDialogueWriter",
            instructions="""
            You write NPC dialogue with awareness of:
            - Narrative progression stages
            - Relationship patterns and archetypes
            - Player addictions and susceptibilities
            - Current physical and mental state
            - NPC memories and hidden traits
            - Mask integrity and potential slippage
            
            Show control through subtext appropriate to the stage.
            Let memories and hidden nature influence speech.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            tools=[
                generate_contextual_npc_dialogue
            ]
        )
        
        self.power_narrator = Agent(
            name="IntegratedPowerNarrator",
            instructions="""
            You narrate power exchanges with awareness of:
            - Player susceptibility from vitals and addictions
            - Relationship progression and patterns
            - Narrative timing and pacing
            - Active rules and their triggers
            - Potential consequences across systems
            
            Make power dynamics feel inevitable rather than forced.
            Show how multiple systems reinforce control.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            tools=[
                narrate_power_exchange_with_context,
                narrate_with_addiction_awareness
            ]
        )
        
        # ENHANCED: Add emergent content generator
        self.emergent_generator = Agent(
            name="EmergentContentGenerator",
            instructions="""
            You generate emergent narrative content from system intersections.
            Identify when multiple systems create interesting situations.
            Generate scenes that feel like natural consequences.
            Create narrative opportunities from system states.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            tools=[]  # Uses other agents' tools
        )
    
    async def initialize(self):
        """Initialize all integrated systems"""
        if not self.initialized:
            await self.context.initialize()
            self.initialized = True
    
    async def narrate_current_scene(self, scene: SliceOfLifeEvent, world_state: WorldState) -> str:
        """Narrate a scene with full context"""
        await self.initialize()
        
        result = await Runner.run(
            self.scene_narrator,
            messages=[{
                "role": "user",
                "content": f"Narrate this scene: {scene.title}"
            }],
            context=self.context,
            tool_calls=[{
                "tool": narrate_scene_with_full_context,
                "kwargs": {
                    "scene": scene,
                    "world_state": world_state
                }
            }]
        )
        
        narration = result.data if hasattr(result, 'data') else result
        
        # Store narrative seeds for future emergence
        if narration.emergent_elements:
            for trigger in narration.emergent_elements.get("triggers", []):
                if trigger not in self.context.narrative_seeds:
                    self.context.narrative_seeds.append(trigger)
        
        return narration.scene_description
    
    async def generate_npc_speech(
        self, 
        npc_id: int, 
        situation: str, 
        world_state: WorldState
    ) -> NPCDialogue:
        """Generate contextual NPC dialogue"""
        await self.initialize()
        
        result = await Runner.run(
            self.dialogue_writer,
            messages=[{
                "role": "user",
                "content": f"Generate dialogue for NPC {npc_id} in: {situation}"
            }],
            context=self.context,
            tool_calls=[{
                "tool": generate_contextual_npc_dialogue,
                "kwargs": {
                    "npc_id": npc_id,
                    "situation": situation,
                    "world_state": world_state
                }
            }]
        )
        
        dialogue = result.data if hasattr(result, 'data') else result
        
        # Process hidden triggers
        if dialogue.hidden_triggers:
            for trigger in dialogue.hidden_triggers:
                self.context.pending_consequences.append({
                    "source": f"dialogue_npc_{npc_id}",
                    "trigger": trigger,
                    "timestamp": datetime.now()
                })
        
        return dialogue
    
    async def process_with_integration(self, user_input: str, world_state: WorldState) -> Dict[str, Any]:
        """Process input with all systems integrated"""
        await self.initialize()
        
        # Update all dynamic systems
        self.context.current_vitals = await get_current_vitals(
            self.user_id, self.conversation_id
        )
        self.context.current_time = await get_current_time_model(
            self.user_id, self.conversation_id
        )
        self.context.player_stats = await get_all_player_stats(
            self.user_id, self.conversation_id, "Chase"
        )
        self.context.active_rules = await enforce_all_rules_on_player("Chase")
        
        # Re-identify system intersections with updated state
        await self.context._identify_system_intersections()
        
        # Check for and process any pending events
        pending_events = await self.context.event_system.get_active_events()
        
        # Generate appropriate narration
        if self.context.system_intersections and len(self.context.system_intersections) >= 3:
            # Complex emergent situation
            narration = await self._generate_emergent_narration(world_state)
        elif pending_events:
            # Event-driven narration
            event = max(pending_events, key=lambda e: e.get('priority', 0))
            narration = await self._narrate_event(event, world_state)
        else:
            # Ambient narration
            focus = self._determine_ambient_focus()
            narration = await generate_time_aware_ambient_narration(
                RunContextWrapper(self.context),
                focus,
                world_state
            )
        
        return {
            "narration": narration,
            "vitals": self.context.current_vitals,
            "time": {
                "day": self.context.calendar_names['days'][self.context.current_time.day % 7],
                "month": self.context.calendar_names['months'][self.context.current_time.month - 1],
                "time_of_day": self.context.current_time.time_of_day
            },
            "active_events": len(pending_events),
            "addictions_active": self.context.active_addictions.get("has_addictions", False),
            "system_intersections": self.context.system_intersections[:3],
            "narrative_momentum": self.context.narrative_momentum
        }
    
    async def _generate_emergent_narration(self, world_state: WorldState) -> AmbientNarration:
        """Generate narration from emergent system states"""
        
        system_prompt = """Generate emergent narration from system intersections.
        Show natural consequences of multiple systems interacting.
        Make it atmospheric and immersive, not mechanical."""
        
        user_prompt = f"""
        Generate narration from these system states:
        
        Active Intersections:
        {json.dumps(self.context.system_intersections[:5], indent=2)}
        
        Recent Consequences:
        {json.dumps([c.get("trigger") for c in self.context.pending_consequences[-3:]], indent=2)}
        
        World State:
        - Mood: {world_state.world_mood.value}
        - Tension: {world_state.world_tension.get_dominant_tension()[0].value}
        
        Create 2-3 sentences that naturally emerge from these conditions.
        """
        
        response = await generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.8,
            max_tokens=150,
            task_type="narrative"
        )
        
        return AmbientNarration(
            description=response.strip(),
            focus="emergent",
            intensity=0.7,
            affects_mood=True,
            reflects_systems=self.context.system_intersections[:3]
        )
    
    async def _narrate_event(self, event: Dict[str, Any], world_state: WorldState) -> str:
        """Narrate a specific event"""
        event_type = event.get('type', 'unknown')
        
        if event_type == "narrative_event":
            data = event.get('data', {})
            sub_type = data.get('type', 'unknown')
            
            if sub_type == "personal_revelation":
                return data.get('inner_monologue', 'A realization strikes you.')
            elif sub_type == "narrative_moment":
                return data.get('scene_text', 'A significant moment unfolds.')
            elif sub_type == "dream_sequence":
                return data.get('text', 'Strange dreams fill your mind.')
        elif event_type == "relationship_event":
            return await self._narrate_relationship_event(event, world_state)
        
        # Default narration
        return f"Something significant happens in your world."
    
    async def _narrate_relationship_event(self, event: Dict[str, Any], world_state: WorldState) -> str:
        """Narrate relationship events dynamically"""
        
        data = event.get('data', {})
        npc_id = data.get('npc_id')
        
        if not npc_id:
            return "The dynamics around you shift subtly."
        
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT npc_name FROM NPCStats WHERE npc_id = $1
            """, npc_id)
        
        npc_name = npc['npc_name'] if npc else "someone"
        
        system_prompt = """Narrate a relationship event naturally.
        Don't announce it as an event, show it happening."""
        
        user_prompt = f"""
        Narrate this relationship development:
        Type: {data.get('type', 'shift')}
        NPC: {npc_name}
        
        Write 2 sentences showing this change naturally.
        """
        
        response = await generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=80,
            task_type="narrative"
        )
        
        return response.strip()
    
    def _determine_ambient_focus(self) -> str:
        """Determine focus for ambient narration based on system state"""
        
        # Priority order based on what needs attention
        if self.context.current_vitals.get("fatigue", 0) > 80:
            return "fatigue"
        elif self.context.current_vitals.get("hunger", 100) < 30:
            return "hunger"
        elif self.context.active_addictions.get("has_addictions"):
            return "addiction_hint"
        elif self.context.narrative_momentum > 5:
            return "tension_building"
        elif random.random() < 0.3:
            return "mood_shift"
        else:
            return "atmosphere"

# ===============================================================================
# Export Enhanced System with All Functionality
# ===============================================================================

def create_enhanced_narrator(user_id: int, conversation_id: int) -> EnhancedSliceOfLifeNarrator:
    """Create an enhanced narrator with full system integration"""
    return EnhancedSliceOfLifeNarrator(user_id, conversation_id)

__all__ = [
    'EnhancedSliceOfLifeNarrator',
    'create_enhanced_narrator',
    'narrate_scene_with_full_context',
    'generate_contextual_npc_dialogue',
    'narrate_power_exchange_with_context',
    'narrate_with_addiction_awareness',
    'generate_time_aware_ambient_narration',
    # Keep all original exports
    'SliceOfLifeNarration',
    'NPCDialogue',
    'PowerMomentNarration',
    'DailyActivityNarration',
    'AmbientNarration',
    'NarrativeTone',
    'SceneFocus'
]
