"""
Slice-of-Life Narrator Agent - Enhanced with full system integration

This system provides:
- Atmospheric narration with relationship and progression awareness
- Dynamic tone based on vitals and addictions
- Calendar-aware time references
- Event system integration
- Currency and world-specific details
"""

import logging
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings
from pydantic import BaseModel, Field, ConfigDict

# Database connection
from db.connection import get_db_connection_context

# Core system integrations
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

# GPT Integration
from logic.chatgpt_integration import generate_text_completion

# World Director integration
from story_agent.world_director_agent import (
    WorldState, WorldMood, TimeOfDay, ActivityType, PowerDynamicType,
    SliceOfLifeEvent, NPCRoutine, PowerExchange, WorldTension
)

logger = logging.getLogger(__name__)

# ===============================================================================
# Enhanced Narrative Context with System Integration
# ===============================================================================

@dataclass
class EnhancedNarratorContext:
    """Enhanced context with all system integrations"""
    user_id: int
    conversation_id: int
    
    # Core systems
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    relationship_integration: Optional[RelationshipIntegration] = None
    event_system: Optional[EventSystem] = None
    addiction_context: Optional[AddictionContext] = None
    currency_generator: Optional[CurrencyGenerator] = None
    
    # Cached data
    calendar_names: Optional[Dict[str, Any]] = None
    current_vitals: Optional[Dict[str, int]] = None
    current_time: Optional[CurrentTimeData] = None
    active_addictions: Optional[Dict[str, Any]] = None
    
    # Narrative tracking
    recent_narrations: List[str] = field(default_factory=list)
    narrative_momentum: float = 0.0
    last_revelation_check: Optional[datetime] = None
    
    async def initialize(self):
        """Initialize all integrated systems"""
        # Initialize relationship systems
        self.relationship_manager = OptimizedRelationshipManager(
            self.user_id, self.conversation_id
        )
        self.relationship_integration = RelationshipIntegration(
            self.user_id, self.conversation_id
        )
        
        # Initialize event system
        self.event_system = EventSystem(self.user_id, self.conversation_id)
        await self.event_system.initialize()
        
        # Initialize addiction context
        self.addiction_context = AddictionContext(self.user_id, self.conversation_id)
        await self.addiction_context.initialize()
        
        # Initialize currency generator
        self.currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
        
        # Load calendar names
        self.calendar_names = await load_calendar_names(self.user_id, self.conversation_id)
        
        # Get current time
        self.current_time = await get_current_time_model(self.user_id, self.conversation_id)
        
        # Get current vitals
        self.current_vitals = await get_current_vitals(self.user_id, self.conversation_id)
        
        # Get addiction status
        self.active_addictions = await get_addiction_status(
            self.user_id, self.conversation_id, "Chase"
        )

# ===============================================================================
# Enhanced Narration Functions with System Integration
# ===============================================================================

@function_tool
async def narrate_scene_with_full_context(
    ctx: RunContextWrapper,
    scene: SliceOfLifeEvent,
    world_state: WorldState
) -> SliceOfLifeNarration:
    """Generate narration with full system integration"""
    context: EnhancedNarratorContext = ctx.context
    
    # Get relationship contexts for all participants
    relationship_contexts = {}
    for npc_id in scene.participants:
        state = await context.relationship_manager.get_relationship_state(
            'npc', npc_id, 'player', context.user_id
        )
        
        # Get narrative stage
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
    
    # Determine tone based on multiple factors
    tone = await _determine_integrated_tone(
        context, world_state, relationship_contexts
    )
    
    # Generate scene description with full context
    scene_desc = await _generate_contextual_scene_description(
        context, scene, world_state, relationship_contexts
    )
    
    # Generate atmosphere with vitals awareness
    atmosphere = await _generate_vitals_aware_atmosphere(
        context, scene.location, world_state.world_mood
    )
    
    # Generate power hints based on relationships and addictions
    power_hints = await _generate_integrated_power_hints(
        context, scene, relationship_contexts
    )
    
    # Generate sensory details with calendar awareness
    sensory = await _generate_calendar_aware_sensory_details(
        context, world_state.current_time, scene.location
    )
    
    # Generate NPC observations with progression awareness
    npc_obs = []
    for npc_id in scene.participants:
        obs = await _generate_progression_aware_npc_observation(
            context, npc_id, scene, relationship_contexts.get(npc_id)
        )
        if obs:
            npc_obs.append(obs)
    
    # Generate internal monologue based on vitals, addictions, and tensions
    internal = await _generate_integrated_internal_monologue(
        context, scene, world_state, relationship_contexts
    )
    
    # Check for narrative events
    await _check_and_queue_narrative_events(context, scene, relationship_contexts)
    
    narration = SliceOfLifeNarration(
        scene_description=scene_desc,
        atmosphere=atmosphere,
        tone=tone,
        focus=_determine_scene_focus(relationship_contexts, context.active_addictions),
        power_dynamic_hints=power_hints,
        sensory_details=sensory,
        npc_observations=npc_obs,
        internal_monologue=internal
    )
    
    # Track narration
    context.recent_narrations.append(scene_desc)
    if len(context.recent_narrations) > 10:
        context.recent_narrations.pop(0)
    
    return narration

@function_tool
async def generate_contextual_npc_dialogue(
    ctx: RunContextWrapper,
    npc_id: int,
    situation: str,
    world_state: WorldState
) -> NPCDialogue:
    """Generate NPC dialogue with full system awareness"""
    context: EnhancedNarratorContext = ctx.context
    
    # Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty, personality_traits, current_location
            FROM NPCStats
            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
        """, npc_id, context.user_id, context.conversation_id)
    
    if not npc:
        raise ValueError(f"NPC {npc_id} not found")
    
    # Get relationship state
    rel_state = await context.relationship_manager.get_relationship_state(
        'npc', npc_id, 'player', context.user_id
    )
    
    # Get narrative stage
    stage = await get_npc_narrative_stage(
        context.user_id, context.conversation_id, npc_id
    )
    
    # Check for relevant addictions
    relevant_addictions = await _get_relevant_addictions_for_npc(
        context, npc_id
    )
    
    # Generate dialogue with full context
    dialogue = await _generate_fully_contextual_dialogue(
        context, npc, stage, situation, rel_state, relevant_addictions
    )
    
    # Include currency references if appropriate
    if "payment" in situation.lower() or "cost" in situation.lower():
        dialogue = await _add_currency_reference(context, dialogue)
    
    # Generate tone based on stage and relationship
    tone = await _determine_dialogue_tone(
        context, npc, stage, rel_state.dimensions
    )
    
    # Generate subtext with addiction awareness
    subtext = await _generate_addiction_aware_subtext(
        context, dialogue, npc['dominance'], stage.name, relevant_addictions
    )
    
    # Generate body language
    body_language = await _generate_stage_appropriate_body_language(
        context, npc['dominance'], tone, stage.name, rel_state.dimensions
    )
    
    # Determine power dynamic
    power_dynamic = _determine_power_dynamic_from_context(
        stage.name, rel_state.dimensions, relevant_addictions
    )
    
    return NPCDialogue(
        npc_id=npc_id,
        npc_name=npc['npc_name'],
        dialogue=dialogue,
        tone=tone,
        subtext=subtext,
        body_language=body_language,
        power_dynamic=power_dynamic,
        requires_response=_should_require_response(rel_state.dimensions, stage.name)
    )

@function_tool
async def narrate_power_exchange_with_context(
    ctx: RunContextWrapper,
    exchange: PowerExchange,
    world_state: WorldState
) -> PowerMomentNarration:
    """Generate power exchange narration with full system context"""
    context: EnhancedNarratorContext = ctx.context
    
    # Get NPC details and relationship
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
    
    # Check if vitals make player more susceptible
    susceptibility = _calculate_susceptibility(context.current_vitals, context.active_addictions)
    
    # Generate components with context
    setup = await _generate_contextual_power_setup(
        context, exchange, npc, stage, rel_state, susceptibility
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
    
    # Present options based on player's state
    options = await _present_state_aware_options(
        context, exchange.player_response_options, susceptibility
    )
    
    # Queue relationship event
    await context.event_system.create_event(
        "relationship_event",
        {
            "type": "power_exchange",
            "npc_id": exchange.initiator_npc_id,
            "exchange_type": exchange.exchange_type.value,
            "intensity": exchange.intensity
        },
        priority=7
    )
    
    return PowerMomentNarration(
        setup=setup,
        moment=moment,
        aftermath=aftermath,
        player_feelings=feelings,
        options_presentation=options
    )

@function_tool
async def narrate_with_addiction_awareness(
    ctx: RunContextWrapper,
    activity: str,
    involved_npcs: List[int],
    world_state: WorldState
) -> DailyActivityNarration:
    """Narrate daily activities with addiction system integration"""
    context: EnhancedNarratorContext = ctx.context
    
    # Get addiction effects for narration
    addiction_effects = []
    if context.active_addictions.get("has_addictions"):
        for addiction_type, data in context.active_addictions.get("addictions", {}).items():
            if data.get("level", 0) >= 2:  # Moderate or higher
                effect_desc = await _generate_addiction_effect_description(
                    context, addiction_type, data.get("level"), activity
                )
                if effect_desc:
                    addiction_effects.append(effect_desc)
    
    # Generate base activity description
    description = await _generate_activity_with_vitals(
        context, activity, context.current_vitals
    )
    
    # Add addiction coloring to routine
    routine_with_dynamics = await _generate_addiction_colored_routine(
        context, activity, involved_npcs, addiction_effects
    )
    
    # Generate NPC involvement with relationship awareness
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
    
    # Add subtle control elements from addictions
    control_elements = addiction_effects + await _generate_subtle_control_elements(
        context, activity, world_state.relationship_dynamics
    )
    
    return DailyActivityNarration(
        activity=activity,
        description=description,
        routine_with_dynamics=routine_with_dynamics,
        npc_involvement=npc_involvement,
        subtle_control_elements=control_elements[:5]  # Limit to 5
    )

@function_tool
async def generate_time_aware_ambient_narration(
    ctx: RunContextWrapper,
    focus: str,
    world_state: WorldState
) -> AmbientNarration:
    """Generate ambient narration with calendar and time system awareness"""
    context: EnhancedNarratorContext = ctx.context
    
    # Update current time
    context.current_time = await get_current_time_model(
        context.user_id, context.conversation_id
    )
    
    # Generate description based on focus with calendar names
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
    
    # Check for relationship events
    rel_event = await event_generator.get_next_event(timeout=0.1)
    if rel_event:
        description += f" {await _weave_relationship_event(context, rel_event)}"
    
    # Determine intensity based on multiple factors
    intensity = _calculate_ambient_intensity(
        world_state.world_tension,
        context.current_vitals,
        context.active_addictions
    )
    
    return AmbientNarration(
        description=description,
        focus=focus,
        intensity=intensity,
        affects_mood=focus in ["mood_shift", "addiction_hint"]
    )

# ===============================================================================
# System-Aware Helper Functions
# ===============================================================================

async def _determine_integrated_tone(
    context: EnhancedNarratorContext,
    world_state: WorldState,
    relationship_contexts: Dict[int, Dict]
) -> NarrativeTone:
    """Determine tone based on all systems"""
    
    # Base tone from world mood
    base_tone = _determine_narrative_tone(world_state.world_mood, ActivityType.ROUTINE)
    
    # Adjust for vitals
    if context.current_vitals.fatigue > 80:
        return NarrativeTone.OBSERVATIONAL  # Too tired for complex emotions
    elif context.current_vitals.hunger < 20:
        return NarrativeTone.COMMANDING  # Desperation makes control easier
    
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

async def _generate_contextual_scene_description(
    context: EnhancedNarratorContext,
    scene: SliceOfLifeEvent,
    world_state: WorldState,
    relationship_contexts: Dict[int, Dict]
) -> str:
    """Generate scene description with full context"""
    
    # Build calendar-aware time reference
    calendar = context.calendar_names
    time_desc = f"{calendar['days'][context.current_time.day % 7]} of {calendar['months'][context.current_time.month - 1]}"
    
    # Build relationship prompts
    rel_prompts = []
    for npc_id, rel_context in relationship_contexts.items():
        stage = rel_context['narrative_stage']
        patterns = rel_context.get('patterns', [])
        
        if stage == 'Full Revelation':
            rel_prompts.append(f"Complete openness about control dynamics")
        elif stage == 'Veil Thinning':
            rel_prompts.append(f"Control barely hidden beneath pleasantries")
        elif stage == 'Creeping Realization':
            rel_prompts.append(f"Patterns becoming visible through repetition")
        
        if 'push_pull' in patterns:
            rel_prompts.append(f"The familiar dance of closeness and distance")
        if 'toxic_bond' in patterns:
            rel_prompts.append(f"An unhealthy attachment pulsing beneath")
    
    # Build vitals prompt
    vitals_prompt = ""
    if context.current_vitals.fatigue > 60:
        vitals_prompt = "Exhaustion weighs on every movement."
    if context.current_vitals.hunger < 40:
        vitals_prompt += " Hunger gnaws as a constant distraction."
    
    system_prompt = """You are narrating a slice-of-life scene with deep context awareness.
    Weave in relationship dynamics, time references, and physical state naturally.
    Use second-person perspective. Be immersive but not melodramatic."""
    
    user_prompt = f"""
    Generate a scene description for:
    Scene: {scene.title} - {scene.description}
    Location: {scene.location}
    Time: {time_desc} during {world_state.current_time.value}
    
    Relationship Context:
    {chr(10).join(rel_prompts)}
    
    Physical State: {vitals_prompt}
    
    World Mood: {world_state.world_mood.value}
    
    Write 2-3 sentences that naturally incorporate these elements.
    Don't explicitly state game mechanics.
    """
    
    description = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=150,
        task_type="narrative"
    )
    
    return description.strip()

async def _generate_vitals_aware_atmosphere(
    context: EnhancedNarratorContext,
    location: str,
    mood: WorldMood
) -> str:
    """Generate atmosphere influenced by vitals"""
    
    vitals_influence = ""
    if context.current_vitals.fatigue > 70:
        vitals_influence = "Everything feels distant and dreamlike through exhaustion."
    elif context.current_vitals.hunger < 30:
        vitals_influence = "The world sharpens with desperate hunger."
    elif context.current_vitals.thirst < 30:
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

async def _generate_integrated_power_hints(
    context: EnhancedNarratorContext,
    scene: SliceOfLifeEvent,
    relationship_contexts: Dict[int, Dict]
) -> List[str]:
    """Generate power hints from relationships and addictions"""
    hints = []
    
    # Relationship-based hints
    for npc_id, rel_context in relationship_contexts.items():
        patterns = rel_context.get('patterns', [])
        archetypes = rel_context.get('archetypes', [])
        
        if 'push_pull' in patterns:
            hints.append("The familiar rhythm of advance and retreat continues")
        if 'slow_burn' in patterns:
            hints.append("A gradual shift, almost imperceptible but constant")
        if 'toxic_bond' in archetypes:
            hints.append("The unhealthy dynamic thrums beneath every interaction")
        if 'mentor_student' in archetypes:
            hints.append("Guidance that shapes more than just knowledge")
    
    # Addiction-based hints
    if context.active_addictions.get("has_addictions"):
        for addiction_type, data in context.active_addictions.get("addictions", {}).items():
            if data.get("level", 0) >= 3:  # Heavy or extreme
                if "feet" in addiction_type:
                    hints.append("Your gaze drops involuntarily, drawn downward")
                elif "scent" in addiction_type:
                    hints.append("Every breath carries meaning you can't ignore")
                elif "humiliation" in addiction_type:
                    hints.append("The familiar warmth of shame feels almost comfortable")
    
    return hints[:4]  # Limit to 4 hints

async def _generate_calendar_aware_sensory_details(
    context: EnhancedNarratorContext,
    time: TimeOfDay,
    location: str
) -> List[str]:
    """Generate sensory details with calendar awareness"""
    
    calendar = context.calendar_names
    month_name = calendar['months'][context.current_time.month - 1]
    
    system_prompt = """Create sensory details that reference custom calendar names.
    Make the world feel unique through its temporal references."""
    
    user_prompt = f"""
    Generate sensory details for:
    Time: {time.value} in the month of {month_name}
    Location: {location}
    Year Name: {calendar['year_name']}
    
    Create 2-3 sensory observations that subtly reference the custom calendar.
    Return as a JSON array of strings.
    """
    
    response = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=120,
        task_type="narrative"
    )
    
    try:
        details = json.loads(response)
        if isinstance(details, list):
            return details[:3]
    except:
        return [f"The {month_name} air carries its own signature"]
    
    return details

async def _generate_progression_aware_npc_observation(
    context: EnhancedNarratorContext,
    npc_id: int,
    scene: SliceOfLifeEvent,
    rel_context: Optional[Dict]
) -> Optional[str]:
    """Generate NPC observation based on narrative progression"""
    
    if not rel_context:
        return None
    
    stage = rel_context['narrative_stage']
    stage_desc = rel_context.get('stage_description', '')
    
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, npc_id)
    
    if not npc:
        return None
    
    system_prompt = f"""Generate observations about NPCs at different narrative stages.
    Current stage: {stage} - {stage_desc}
    Show progression through subtle behavioral changes."""
    
    user_prompt = f"""
    Observe {npc['npc_name']} in this scene:
    Scene: {scene.title}
    Dominance: {npc['dominance']}/100 (don't mention numbers)
    Stage: {stage}
    Active patterns: {', '.join(rel_context.get('patterns', [])[:2])}
    
    Write one sentence showing their behavior at this stage.
    """
    
    observation = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=60,
        task_type="narrative"
    )
    
    return observation.strip()

async def _generate_integrated_internal_monologue(
    context: EnhancedNarratorContext,
    scene: SliceOfLifeEvent,
    world_state: WorldState,
    relationship_contexts: Dict[int, Dict]
) -> Optional[str]:
    """Generate internal monologue based on all systems"""
    
    # Check if we should have internal monologue
    triggers = []
    
    # Vitals trigger
    if context.current_vitals.fatigue > 70:
        triggers.append("exhaustion")
    if context.current_vitals.hunger < 30:
        triggers.append("hunger")
    
    # Addiction trigger
    if context.active_addictions.get("has_addictions"):
        addiction_count = len(context.active_addictions.get("addictions", {}))
        if addiction_count >= 2:
            triggers.append("cravings")
    
    # Relationship trigger
    for rel_context in relationship_contexts.values():
        if rel_context['narrative_stage'] in ['Creeping Realization', 'Veil Thinning']:
            triggers.append("realization")
            break
    
    if not triggers:
        return None
    
    system_prompt = """Generate internal monologue reflecting multiple influences.
    Show how physical state, psychological dependencies, and growing awareness interact.
    Use second person perspective."""
    
    user_prompt = f"""
    Generate internal thoughts influenced by:
    Triggers: {', '.join(triggers)}
    Scene: {scene.title}
    Submission Level: {world_state.relationship_dynamics.player_submission_level:.0%}
    
    Physical State:
    - Fatigue: {context.current_vitals.fatigue}/100
    - Hunger: {context.current_vitals.hunger}/100
    
    Write 1-2 sentences of internal monologue showing these influences.
    """
    
    monologue = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="introspection"
    )
    
    return monologue.strip()

async def _check_and_queue_narrative_events(
    context: EnhancedNarratorContext,
    scene: SliceOfLifeEvent,
    relationship_contexts: Dict[int, Dict]
) -> None:
    """Check for and queue narrative events based on current state"""
    
    # Check if enough time has passed since last check
    now = datetime.now()
    if context.last_revelation_check and (now - context.last_revelation_check).seconds < 300:
        return
    
    context.last_revelation_check = now
    
    # Check for personal revelations
    revelation = await check_for_personal_revelations(
        context.user_id, context.conversation_id
    )
    if revelation:
        await context.event_system.create_event(
            "narrative_event",
            {
                "type": "personal_revelation",
                "data": revelation
            },
            priority=8
        )
    
    # Check for narrative moments
    moment = await check_for_narrative_moments(
        context.user_id, context.conversation_id
    )
    if moment:
        await context.event_system.create_event(
            "narrative_event",
            {
                "type": "narrative_moment",
                "data": moment
            },
            priority=7
        )
    
    # Check for dream sequences if tired
    if context.current_vitals.fatigue > 80:
        dream = await add_dream_sequence(
            context.user_id, context.conversation_id
        )
        if dream:
            await context.event_system.create_event(
                "narrative_event",
                {
                    "type": "dream_sequence",
                    "data": dream
                },
                priority=6
            )

async def _get_relevant_addictions_for_npc(
    context: EnhancedNarratorContext,
    npc_id: int
) -> Dict[str, Any]:
    """Get addictions relevant to a specific NPC"""
    
    relevant = {}
    if not context.active_addictions.get("has_addictions"):
        return relevant
    
    for addiction_key, data in context.active_addictions.get("addictions", {}).items():
        # Check if this is an NPC-specific addiction
        if data.get("type") == "npc_specific" and data.get("npc_id") == npc_id:
            relevant[addiction_key] = data
        # Or if it's a general addiction that might apply
        elif data.get("type") == "general" and data.get("level", 0) >= 2:
            relevant[addiction_key] = data
    
    return relevant

async def _generate_fully_contextual_dialogue(
    context: EnhancedNarratorContext,
    npc: Any,
    stage: NPCNarrativeStage,
    situation: str,
    rel_state: Any,
    relevant_addictions: Dict[str, Any]
) -> str:
    """Generate dialogue with full context awareness"""
    
    # Build addiction context
    addiction_prompts = []
    for addiction_type, data in relevant_addictions.items():
        level = data.get("level", 0)
        if level >= 3:
            addiction_prompts.append(f"Player has strong {addiction_type} addiction")
    
    system_prompt = f"""Generate natural dialogue for {npc['npc_name']}.
    Narrative Stage: {stage.name} - {stage.description}
    Personality: {npc['personality_traits']}
    
    The dialogue should reflect the current stage and any relevant addictions.
    Power dynamics through subtext only."""
    
    user_prompt = f"""
    Generate dialogue for:
    Situation: {situation}
    Dominance: {npc['dominance']}/100
    Trust: {rel_state.dimensions.trust:.0f}/100
    Influence: {rel_state.dimensions.influence:.0f}/100
    Patterns: {', '.join(list(rel_state.history.active_patterns)[:3])}
    
    Relevant context:
    {chr(10).join(addiction_prompts)}
    
    Write 1-2 sentences of natural dialogue.
    """
    
    dialogue = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=100,
        task_type="dialogue"
    )
    
    return dialogue.strip()

async def _add_currency_reference(
    context: EnhancedNarratorContext,
    dialogue: str
) -> str:
    """Add currency references to dialogue"""
    
    currency_system = await context.currency_generator.get_currency_system()
    
    # Simple replacement of generic money references
    dialogue = dialogue.replace("dollars", currency_system.get("currency_plural", "coins"))
    dialogue = dialogue.replace("dollar", currency_system.get("currency_name", "coin"))
    dialogue = dialogue.replace("$", currency_system.get("currency_symbol", ""))
    
    return dialogue

def _calculate_susceptibility(
    vitals: Dict[str, int],
    addictions: Dict[str, Any]
) -> float:
    """Calculate player susceptibility to control"""
    
    susceptibility = 0.5  # Base
    
    # Vitals effects
    if vitals.get("fatigue", 0) > 70:
        susceptibility += 0.2
    if vitals.get("hunger", 100) < 30:
        susceptibility += 0.15
    if vitals.get("thirst", 100) < 30:
        susceptibility += 0.1
    
    # Addiction effects
    if addictions.get("has_addictions"):
        addiction_count = len(addictions.get("addictions", {}))
        susceptibility += min(0.3, addiction_count * 0.1)
    
    return min(1.0, susceptibility)

def _determine_power_dynamic_from_context(
    stage: str,
    dimensions: Any,
    addictions: Dict[str, Any]
) -> Optional[PowerDynamicType]:
    """Determine power dynamic from full context"""
    
    if stage == "Full Revelation":
        return PowerDynamicType.COMPLETE_CONTROL
    elif stage == "Veil Thinning":
        if dimensions.influence > 60:
            return PowerDynamicType.INTIMATE_COMMAND
        else:
            return PowerDynamicType.CASUAL_DOMINANCE
    elif stage == "Creeping Realization":
        if addictions:
            return PowerDynamicType.PROTECTIVE_CONTROL
        else:
            return PowerDynamicType.SUBTLE_CONTROL
    elif stage == "First Doubts":
        return PowerDynamicType.PLAYFUL_TEASING
    
    return None

def _should_require_response(dimensions: Any, stage: str) -> bool:
    """Determine if dialogue requires response based on context"""
    
    if stage in ["Full Revelation", "Veil Thinning"]:
        return dimensions.influence > 50
    elif stage == "Creeping Realization":
        return dimensions.trust > 60
    else:
        return random.random() > 0.6

def _determine_scene_focus(
    relationship_contexts: Dict[int, Dict],
    addictions: Dict[str, Any]
) -> SceneFocus:
    """Determine scene focus based on active elements"""
    
    # Check for high-intensity patterns
    for rel_context in relationship_contexts.values():
        if 'toxic_bond' in rel_context.get('archetypes', []):
            return SceneFocus.TENSION
        if rel_context.get('momentum', 0) > 5:
            return SceneFocus.DYNAMICS
    
    # Check for addiction influence
    if addictions.get("has_addictions"):
        max_level = max(
            data.get("level", 0) 
            for data in addictions.get("addictions", {}).values()
        )
        if max_level >= 3:
            return SceneFocus.INTERNAL
    
    # Default based on participants
    if len(relationship_contexts) > 1:
        return SceneFocus.DIALOGUE
    elif relationship_contexts:
        return SceneFocus.DYNAMICS
    else:
        return SceneFocus.ATMOSPHERE

def _calculate_ambient_intensity(
    tension: WorldTension,
    vitals: Dict[str, int],
    addictions: Dict[str, Any]
) -> float:
    """Calculate ambient narration intensity"""
    
    # Base from tension
    dominant_tension, level = tension.get_dominant_tension()
    intensity = level * 0.5
    
    # Modify for vitals
    if vitals.get("fatigue", 0) > 80:
        intensity += 0.2
    if vitals.get("hunger", 100) < 20:
        intensity += 0.15
    
    # Modify for addictions
    if addictions.get("has_addictions"):
        intensity += 0.1
    
    return min(1.0, intensity)

# Additional helper functions for specific integrations...

async def _narrate_calendar_time_passage(
    context: EnhancedNarratorContext,
    current_time: CurrentTimeData,
    calendar: Dict[str, Any]
) -> str:
    """Narrate time passage with calendar names"""
    
    day_name = calendar['days'][current_time.day % 7]
    month_name = calendar['months'][current_time.month - 1]
    
    system_prompt = """Narrate time transitions using custom calendar names.
    Make the passage of time feel unique to this world."""
    
    user_prompt = f"""
    Narrate transition to:
    {current_time.time_of_day} on {day_name} of {month_name}
    Year: {calendar['year_name']}
    
    Write 1-2 sentences about time passing in this world.
    """
    
    narration = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return narration.strip()

async def _narrate_addiction_presence(
    context: EnhancedNarratorContext,
    addictions: Dict[str, Any]
) -> str:
    """Narrate subtle hints of active addictions"""
    
    if not addictions.get("has_addictions"):
        return "The world continues its subtle influence."
    
    # Get the strongest addiction
    strongest = max(
        addictions.get("addictions", {}).items(),
        key=lambda x: x[1].get("level", 0),
        default=(None, {"level": 0})
    )
    
    if not strongest[0]:
        return "Familiar cravings pulse at the edge of awareness."
    
    addiction_type = strongest[0]
    level = strongest[1].get("level", 0)
    
    system_prompt = """Narrate subtle hints of psychological addiction.
    Never explicitly state the addiction, only its effects on perception."""
    
    user_prompt = f"""
    Hint at a {addiction_type} addiction at level {level}/4.
    
    Write 1-2 sentences showing how this colors perception.
    Be subtle and atmospheric.
    """
    
    narration = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return narration.strip()

async def _weave_relationship_event(
    context: EnhancedNarratorContext,
    event: Dict[str, Any]
) -> str:
    """Weave a relationship event into ambient narration"""
    
    event_type = event.get('event', {}).get('type', 'unknown')
    
    system_prompt = """Weave relationship events into ambient narration.
    Make them feel like natural observations, not game announcements."""
    
    user_prompt = f"""
    Naturally mention this relationship event:
    Type: {event_type}
    
    Write one sentence that hints at this development.
    """
    
    narration = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=60,
        task_type="narrative"
    )
    
    return narration.strip()

# ===============================================================================
# Enhanced Slice-of-Life Narrator Class
# ===============================================================================

class EnhancedSliceOfLifeNarrator:
    """
    Enhanced narrator with full system integration.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = EnhancedNarratorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        self.initialized = False
        
        # Create integrated sub-agents
        self.scene_narrator = Agent(
            name="IntegratedSceneNarrator",
            instructions="""
            You narrate slice-of-life scenes with awareness of:
            - Relationship dynamics and progression stages
            - Player vitals and physical state
            - Active addictions and their effects
            - Custom calendar and currency
            - Ongoing narrative patterns
            
            Weave these elements naturally into atmospheric narration.
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
            
            Show control through subtext appropriate to the stage.
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
            
            Make power dynamics feel inevitable rather than forced.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            tools=[
                narrate_power_exchange_with_context,
                narrate_with_addiction_awareness
            ]
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
        
        return result.data if hasattr(result, 'data') else result
    
    async def process_with_integration(self, user_input: str, world_state: WorldState) -> Dict[str, Any]:
        """Process input with all systems integrated"""
        await self.initialize()
        
        # Update vitals and time
        self.context.current_vitals = await get_current_vitals(
            self.user_id, self.conversation_id
        )
        self.context.current_time = await get_current_time_model(
            self.user_id, self.conversation_id
        )
        
        # Check for and process any pending events
        pending_events = await self.context.event_system.get_active_events()
        
        # Generate appropriate narration
        if pending_events:
            # Narrate the highest priority event
            event = max(pending_events, key=lambda e: e.get('priority', 0))
            narration = await self._narrate_event(event, world_state)
        else:
            # Generate ambient narration
            narration = await generate_time_aware_ambient_narration(
                RunContextWrapper(self.context),
                "atmosphere",
                world_state
            )
        
        return {
            "narration": narration,
            "vitals": self.context.current_vitals.to_dict(),
            "time": {
                "day": self.context.calendar_names['days'][self.context.current_time.day % 7],
                "month": self.context.calendar_names['months'][self.context.current_time.month - 1],
                "time_of_day": self.context.current_time.time_of_day
            },
            "active_events": len(pending_events),
            "addictions_active": self.context.active_addictions.get("has_addictions", False)
        }
    
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
        
        # Default narration
        return f"Something significant happens in your world."

# ===============================================================================
# Export Enhanced System
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
    'generate_time_aware_ambient_narration'
]
