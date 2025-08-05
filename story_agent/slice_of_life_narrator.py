"""
Slice-of-Life Narrator Agent - The narrative voice for the open-world femdom simulation.

This system provides:
- Atmospheric narration for world events
- Slice-of-life scenes with subtle femdom dynamics
- NPC voices in everyday situations
- Immersive descriptions of routine activities
- Response to emergent world states rather than driving plot
"""

import logging
import json
import asyncio
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

from agents import Agent, Runner, function_tool, handoff, ModelSettings, RunContextWrapper
from pydantic import BaseModel, Field, ConfigDict

# Database connection
from db.connection import get_db_connection_context

# GPT Integration
from logic.chatgpt_integration import (
    generate_text_completion,
    get_chatgpt_response,
    TEMPERATURE_SETTINGS
)

# World Director integration
from story_agent.world_director_agent import (
    WorldState, WorldMood, TimeOfDay, ActivityType, PowerDynamicType,
    SliceOfLifeEvent, NPCRoutine, PowerExchange, WorldTension,
    WorldDirector
)

# Import world simulation tools
from story_agent.tools import (
    DailyLifeDirector, AmbientDialogueWriter, 
    PowerDynamicsOrchestrator, PlayerAgencyManager
)

# Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission
from nyx.integrate import get_central_governance, remember_with_governance

# Context system integration
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager
from context.vector_service import get_vector_service
from context.context_performance import PerformanceMonitor, track_performance

# NPC systems
from logic.npc_narrative_progression import get_npc_narrative_stage
from logic.dynamic_relationships import OptimizedRelationshipManager

logger = logging.getLogger(__name__)

# ===============================================================================
# Narrative Tone and Style Enums
# ===============================================================================

class NarrativeTone(Enum):
    """Tone for slice-of-life narration"""
    CASUAL = "casual"           # Everyday, relaxed narration
    INTIMATE = "intimate"        # Close, personal moments
    OBSERVATIONAL = "observational"  # Detached, voyeuristic
    SENSUAL = "sensual"         # Emphasizing physical/emotional sensation
    TEASING = "teasing"         # Playful, slightly mocking
    COMMANDING = "commanding"    # Direct, authoritative
    SUBTLE = "subtle"           # Understated power dynamics

class SceneFocus(Enum):
    """What to emphasize in scene narration"""
    ATMOSPHERE = "atmosphere"    # Environmental details
    DIALOGUE = "dialogue"        # Character interactions
    INTERNAL = "internal"        # Player's thoughts/feelings
    DYNAMICS = "dynamics"        # Power relationships
    ROUTINE = "routine"         # Everyday activities
    TENSION = "tension"         # Underlying conflicts

# ===============================================================================
# Pydantic Models for Narration
# ===============================================================================

class SliceOfLifeNarration(BaseModel):
    """Narration for a slice-of-life scene"""
    scene_description: str
    atmosphere: str
    tone: NarrativeTone
    focus: SceneFocus
    power_dynamic_hints: List[str] = Field(default_factory=list)
    sensory_details: List[str] = Field(default_factory=list)
    npc_observations: List[str] = Field(default_factory=list)
    internal_monologue: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

class NPCDialogue(BaseModel):
    """Dialogue from an NPC in daily life"""
    npc_id: int
    npc_name: str
    dialogue: str
    tone: str  # How they say it
    subtext: str  # What they really mean
    body_language: str
    power_dynamic: Optional[PowerDynamicType] = None
    requires_response: bool = False
    
    model_config = ConfigDict(extra="forbid")

class AmbientNarration(BaseModel):
    """Ambient narration for world atmosphere"""
    description: str
    focus: str  # What we're describing (time passing, mood shift, etc.)
    intensity: float = 0.5  # How prominent this is
    affects_mood: bool = False
    
    model_config = ConfigDict(extra="forbid")

class PowerMomentNarration(BaseModel):
    """Narration for a power exchange moment"""
    setup: str  # Building to the moment
    moment: str  # The actual exchange
    aftermath: str  # Immediate reaction
    player_feelings: str  # Internal reaction
    options_presentation: List[str]  # How choices are presented
    
    model_config = ConfigDict(extra="forbid")

class DailyActivityNarration(BaseModel):
    """Narration for routine daily activities"""
    activity: str
    description: str
    routine_with_dynamics: str  # How power dynamics color routine
    npc_involvement: List[str] = Field(default_factory=list)
    subtle_control_elements: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Narrator Context
# ===============================================================================

@dataclass
class NarratorContext:
    """Context for the Slice-of-Life Narrator"""
    user_id: int
    conversation_id: int
    
    # World integration
    world_director: Optional[WorldDirector] = None
    current_world_state: Optional[WorldState] = None
    
    # Narrative tracking
    recent_narrations: List[str] = field(default_factory=list)
    current_tone: NarrativeTone = NarrativeTone.CASUAL
    narrative_momentum: float = 0.0  # How intense narration has been
    
    # NPC voice tracking
    npc_voices: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Context management
    context_service: Optional[Any] = None
    memory_manager: Optional[Any] = None
    performance_monitor: Optional[Any] = None
    
    # Caching
    last_narration_time: Optional[datetime] = None
    narrative_cache: Dict[str, Any] = field(default_factory=dict)

# ===============================================================================
# Core Narration Functions with GPT Generation
# ===============================================================================

@function_tool
async def narrate_slice_of_life_scene(
    ctx: RunContextWrapper,
    scene: SliceOfLifeEvent,
    world_state: WorldState,
    player_action: Optional[str] = None
) -> SliceOfLifeNarration:
    """
    Generate narration for a slice-of-life scene using GPT.
    Focus on atmosphere and subtle dynamics rather than plot.
    """
    context = ctx.context
    
    # Determine tone based on world mood and scene type
    tone = _determine_narrative_tone(world_state.world_mood, scene.event_type)
    
    # Determine focus based on what's happening
    if scene.power_dynamic:
        focus = SceneFocus.DYNAMICS
    elif scene.participants:
        focus = SceneFocus.DIALOGUE
    else:
        focus = SceneFocus.ATMOSPHERE
    
    # Generate scene description with GPT
    scene_desc = await _generate_scene_description(
        context, scene, world_state, tone, focus
    )
    
    # Generate atmospheric details
    atmosphere = await _generate_atmosphere(
        context, scene.location, world_state.world_mood
    )
    
    # Generate power dynamic hints if present
    power_hints = []
    if scene.power_dynamic:
        power_hints = await _generate_power_hints(
            context, scene.power_dynamic, scene.participants
        )
    
    # Generate sensory details
    sensory = await _generate_sensory_details(
        context, world_state.current_time, scene.location
    )
    
    # Generate NPC observations
    npc_obs = []
    for npc_id in scene.participants:
        obs = await _generate_npc_observation(context, npc_id, scene)
        if obs:
            npc_obs.append(obs)
    
    # Generate internal monologue if tension is high
    internal = None
    if world_state.world_tension.power_tension > 0.6:
        internal = await _generate_internal_monologue(
            context, scene, world_state.relationship_dynamics
        )
    
    narration = SliceOfLifeNarration(
        scene_description=scene_desc,
        atmosphere=atmosphere,
        tone=tone,
        focus=focus,
        power_dynamic_hints=power_hints,
        sensory_details=sensory,
        npc_observations=npc_obs,
        internal_monologue=internal
    )
    
    # Track recent narration
    if hasattr(context, 'recent_narrations'):
        context.recent_narrations.append(scene_desc)
        if len(context.recent_narrations) > 10:
            context.recent_narrations.pop(0)
    
    return narration

@function_tool
async def generate_npc_dialogue(
    ctx: RunContextWrapper,
    npc_id: int,
    situation: str,
    world_state: WorldState,
    relationship_context: Optional[Dict] = None
) -> NPCDialogue:
    """
    Generate contextual NPC dialogue for daily situations using GPT.
    Focus on natural conversation with subtle power dynamics.
    """
    context = ctx.context
    
    # Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits, current_location
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
            'patterns': list(rel_state.history.active_patterns)
        }
    
    # Generate dialogue with GPT
    dialogue = await _generate_contextual_dialogue(
        context, npc, stage.name, situation, relationship_context
    )
    
    # Generate tone, subtext, and body language with GPT
    tone = await _determine_npc_tone(context, npc, stage.name, dialogue)
    subtext = await _generate_dialogue_subtext(
        context, dialogue, npc['dominance'], stage.name
    )
    body_language = await _generate_body_language(
        context, npc['dominance'], tone, stage.name
    )
    
    # Determine if power dynamic is present
    power_dynamic = None
    if npc['dominance'] > 60 and stage.name != "Innocent Beginning":
        power_dynamic = _select_dialogue_power_dynamic(situation, npc['dominance'])
    
    return NPCDialogue(
        npc_id=npc_id,
        npc_name=npc['npc_name'],
        dialogue=dialogue,
        tone=tone,
        subtext=subtext,
        body_language=body_language,
        power_dynamic=power_dynamic,
        requires_response=random.random() > 0.5
    )

@function_tool
async def narrate_power_exchange(
    ctx: RunContextWrapper,
    exchange: PowerExchange,
    world_state: WorldState
) -> PowerMomentNarration:
    """
    Generate narration for a power exchange moment using GPT.
    Make it feel natural and integrated into daily life.
    """
    context = ctx.context
    
    # Get NPC details
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, exchange.initiator_npc_id)
    
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
    
    return PowerMomentNarration(
        setup=setup,
        moment=moment,
        aftermath=aftermath,
        player_feelings=feelings,
        options_presentation=options
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
    Generate narration for routine daily activities using GPT.
    Weave power dynamics naturally into everyday tasks.
    """
    context = ctx.context
    
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
    
    # Generate subtle control elements
    control_elements = await _generate_subtle_control_elements(
        context, activity, world_state.relationship_dynamics
    )
    
    return DailyActivityNarration(
        activity=activity,
        description=description,
        routine_with_dynamics=routine_with_dynamics,
        npc_involvement=npc_involvement,
        subtle_control_elements=control_elements
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
    context = ctx.context
    
    # Generate description based on focus
    if focus == "time_passage":
        description = await _narrate_time_passage(context, world_state.current_time)
    elif focus == "mood_shift":
        description = await _narrate_mood_shift(context, world_state.world_mood)
    elif focus == "tension_building":
        description = await _narrate_tension(context, world_state.world_tension)
    else:
        description = await _narrate_ambient_detail(context, world_state)
    
    # Determine intensity based on world tension
    dominant_tension, level = world_state.world_tension.get_dominant_tension()
    intensity = min(1.0, level * 0.8)
    
    return AmbientNarration(
        description=description,
        focus=focus,
        intensity=intensity,
        affects_mood=focus == "mood_shift"
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
    context = ctx.context
    
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
    
    return {
        "acknowledgment": acknowledgment,
        "world_reaction": world_reaction,
        "npc_reactions": npc_reactions,
        "dynamic_shift": dynamic_shift,
        "maintains_atmosphere": True
    }

# ===============================================================================
# GPT-Powered Helper Functions
# ===============================================================================

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

async def _generate_scene_description(
    context: NarratorContext,
    scene: SliceOfLifeEvent,
    world_state: WorldState,
    tone: NarrativeTone,
    focus: SceneFocus
) -> str:
    """Generate scene description using GPT"""
    
    system_prompt = """You are narrating a slice-of-life scene in a femdom setting.
    Focus on atmosphere and subtle power dynamics woven into everyday life.
    Use second-person perspective ('you'). Be immersive but not melodramatic.
    The power dynamics should feel natural, not forced or explicit."""
    
    user_prompt = f"""
    Generate a scene description for:
    Scene: {scene.title} - {scene.description}
    Location: {scene.location}
    Activity Type: {scene.event_type.value}
    Tone: {tone.value} - Make the narration feel {tone.value}
    Focus: {focus.value} - Emphasize the {focus.value} aspect
    World Mood: {world_state.world_mood.value}
    
    NPCs Present: {len(scene.participants)} people
    Power Dynamic: {scene.power_dynamic.value if scene.power_dynamic else 'none'}
    
    Recent narrations to avoid repetition:
    {chr(10).join(context.recent_narrations[-3:]) if context.recent_narrations else 'None'}
    
    Write 2-3 sentences that set the scene naturally. Don't mention game mechanics.
    """
    
    description = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=150,
        task_type="narrative"
    )
    
    return description.strip()

async def _generate_atmosphere(
    context: NarratorContext,
    location: str,
    mood: WorldMood
) -> str:
    """Generate atmospheric description using GPT"""
    
    system_prompt = """You create atmospheric descriptions for a slice-of-life simulation.
    Focus on sensory details and emotional atmosphere. Keep it brief and evocative."""
    
    user_prompt = f"""
    Create an atmospheric description for:
    Location: {location}
    Mood: {mood.value}
    
    In 1-2 sentences, describe how the {location} feels with a {mood.value} atmosphere.
    Focus on subtle details that convey the mood without stating it directly.
    """
    
    atmosphere = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return atmosphere.strip()

async def _generate_power_hints(
    context: NarratorContext,
    power_dynamic: PowerDynamicType,
    participants: List[int]
) -> List[str]:
    """Generate subtle hints about power dynamics using GPT"""
    
    system_prompt = """You write subtle hints about power dynamics in everyday situations.
    Never be explicit about dominance/submission. Use subtext, body language, and atmosphere.
    Make it feel like natural social dynamics with an undercurrent of control."""
    
    user_prompt = f"""
    Generate 2-3 subtle hints for this power dynamic:
    Type: {power_dynamic.value.replace('_', ' ')}
    Number of NPCs: {len(participants)}
    
    Create brief, subtle observations that hint at the power dynamic without stating it.
    Focus on:
    - Body language
    - Unspoken expectations
    - Subtle shifts in atmosphere
    - Natural social dynamics
    
    Return as a JSON array of strings.
    """
    
    response = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=150,
        task_type="narrative"
    )
    
    try:
        hints = json.loads(response)
        if isinstance(hints, list):
            return hints[:3]
    except:
        # Fallback: split by newlines if not valid JSON
        return [line.strip() for line in response.split('\n') if line.strip()][:3]
    
    return ["The dynamic shifts subtly"]

async def _generate_sensory_details(
    context: NarratorContext,
    time: TimeOfDay,
    location: str
) -> List[str]:
    """Generate sensory details using GPT"""
    
    system_prompt = """You create immersive sensory details for different times and places.
    Focus on smell, sound, temperature, light, and texture. Be specific and evocative."""
    
    user_prompt = f"""
    Generate 2-3 sensory details for:
    Time: {time.value}
    Location: {location}
    
    Create brief sensory observations appropriate for this time and place.
    Make them specific and atmospheric.
    
    Return as a JSON array of strings.
    """
    
    response = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=120,
        task_type="narrative"
    )
    
    try:
        details = json.loads(response)
        if isinstance(details, list):
            return details[:3]
    except:
        return [line.strip() for line in response.split('\n') if line.strip()][:3]
    
    return ["The world continues around you"]

async def _generate_npc_observation(
    context: NarratorContext,
    npc_id: int,
    scene: SliceOfLifeEvent
) -> Optional[str]:
    """Generate observation about an NPC using GPT"""
    
    # Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, npc_id)
    
    if not npc:
        return None
    
    system_prompt = """You write brief observations about NPCs in slice-of-life scenes.
    Focus on body language, presence, and subtle character details.
    Never explicitly state dominance levels - show through behavior."""
    
    user_prompt = f"""
    Write a brief observation about {npc['npc_name']}:
    Dominance: {npc['dominance']}/100 (don't mention numbers)
    Scene: {scene.title}
    
    In one sentence, describe what you notice about them in this moment.
    Focus on their presence, body language, or a subtle detail.
    """
    
    observation = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=60,
        task_type="narrative"
    )
    
    return observation.strip()

async def _generate_internal_monologue(
    context: NarratorContext,
    scene: SliceOfLifeEvent,
    dynamics: Any
) -> str:
    """Generate player's internal thoughts using GPT"""
    
    system_prompt = """You write internal monologues for a player in a femdom slice-of-life game.
    The thoughts should reflect their growing awareness or acceptance of power dynamics.
    Keep it subtle and psychological. Use second person ('you think', 'you realize')."""
    
    user_prompt = f"""
    Generate an internal thought for the player:
    Scene: {scene.title}
    Submission Level: {dynamics.player_submission_level:.1%} (don't mention numbers)
    Acceptance Level: {dynamics.acceptance_level:.1%}
    Resistance Level: {dynamics.resistance_level:.1%}
    
    Write 1-2 sentences of internal monologue that reflects their psychological state.
    Show their awareness/acceptance/resistance to the power dynamics naturally.
    """
    
    monologue = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="introspection"
    )
    
    return monologue.strip()

async def _generate_contextual_dialogue(
    context: NarratorContext,
    npc: Any,
    stage: str,
    situation: str,
    relationship_context: Dict
) -> str:
    """Generate NPC dialogue using GPT"""
    
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
    
    Make dialogue feel natural and conversational. Power dynamics through subtext only."""
    
    user_prompt = f"""
    Generate dialogue for this situation:
    Situation: {situation}
    Dominance: {npc['dominance']}/100
    Trust: {relationship_context['trust']}/100
    Influence: {relationship_context['influence']}/100
    Patterns: {', '.join(relationship_context['patterns'][:3]) if relationship_context['patterns'] else 'none'}
    
    Write 1-2 sentences of natural dialogue appropriate for the stage and situation.
    """
    
    dialogue = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=100,
        task_type="dialogue"
    )
    
    return dialogue.strip()

async def _determine_npc_tone(
    context: NarratorContext,
    npc: Any,
    stage: str,
    dialogue: str
) -> str:
    """Determine tone of NPC dialogue using GPT"""
    
    system_prompt = """You analyze dialogue tone in a slice-of-life context.
    Identify the emotional tone and delivery style of the dialogue."""
    
    user_prompt = f"""
    Analyze the tone of this dialogue:
    Speaker: {npc['npc_name']}
    Dominance: {npc['dominance']}/100
    Stage: {stage}
    Dialogue: "{dialogue}"
    
    In 1-3 words, describe HOW this is said (e.g., "warmly commanding", "playfully firm", "casually assertive").
    """
    
    tone = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
        max_tokens=20,
        task_type="analysis"
    )
    
    return tone.strip()

async def _generate_dialogue_subtext(
    context: NarratorContext,
    dialogue: str,
    dominance: int,
    stage: str
) -> str:
    """Generate subtext for dialogue using GPT"""
    
    system_prompt = """You reveal the hidden meaning behind dialogue in a power dynamic context.
    The subtext should hint at control, influence, or expectations without being explicit."""
    
    user_prompt = f"""
    What's the subtext of this dialogue?
    Dialogue: "{dialogue}"
    Speaker Dominance: {dominance}/100
    Relationship Stage: {stage}
    
    In one brief sentence, explain what they really mean or want.
    Focus on hidden expectations, subtle control, or unspoken dynamics.
    """
    
    subtext = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=60,
        task_type="analysis"
    )
    
    return subtext.strip()

async def _generate_body_language(
    context: NarratorContext,
    dominance: int,
    tone: str,
    stage: str
) -> str:
    """Generate body language description using GPT"""
    
    system_prompt = """You describe subtle body language that conveys power dynamics.
    Focus on posture, eye contact, gestures, and spatial positioning.
    Never explicitly state dominance - show it through physical presence."""
    
    user_prompt = f"""
    Describe body language for:
    Dominance Level: {dominance}/100 (don't mention numbers)
    Tone: {tone}
    Stage: {stage}
    
    In one sentence, describe their physical presence and body language.
    Make it subtle and natural.
    """
    
    body_language = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=60,
        task_type="narrative"
    )
    
    return body_language.strip()

async def _generate_power_moment_setup(
    context: NarratorContext,
    exchange: PowerExchange,
    npc: Any,
    world_state: WorldState
) -> str:
    """Generate setup for power moment using GPT"""
    
    system_prompt = """You narrate the build-up to subtle power exchange moments.
    Create tension and anticipation without being explicit about what's coming.
    Focus on atmosphere, body language, and subtle shifts in dynamic."""
    
    user_prompt = f"""
    Set up this power exchange:
    Type: {exchange.exchange_type.value.replace('_', ' ')}
    Initiator: {npc['npc_name']}
    Intensity: {exchange.intensity:.1%}
    Setting: {world_state.world_mood.value} mood
    
    Write 2-3 sentences that build toward the moment without revealing it.
    Focus on subtle changes in atmosphere or behavior.
    """
    
    setup = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=120,
        task_type="narrative"
    )
    
    return setup.strip()

async def _generate_power_moment_description(
    context: NarratorContext,
    exchange: PowerExchange,
    npc: Any
) -> str:
    """Generate the moment itself using GPT"""
    
    system_prompt = """You narrate subtle power exchanges in everyday situations.
    Make it feel natural and integrated into normal interaction.
    The control should feel inevitable rather than forced."""
    
    user_prompt = f"""
    Narrate this power exchange:
    Type: {exchange.exchange_type.value.replace('_', ' ')}
    Description: {exchange.description}
    Character: {npc['npc_name']}
    Intensity: {exchange.intensity:.1%}
    Public: {exchange.is_public}
    
    Write 2-3 sentences describing the moment itself.
    Make it feel like a natural part of daily interaction.
    """
    
    moment = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=120,
        task_type="narrative"
    )
    
    return moment.strip()

async def _generate_power_moment_aftermath(
    context: NarratorContext,
    exchange: PowerExchange,
    dynamics: Any
) -> str:
    """Generate aftermath description using GPT"""
    
    system_prompt = """You describe the immediate aftermath of power exchanges.
    Focus on lingering atmosphere, unspoken understanding, and subtle shifts.
    Show how the dynamic has subtly changed without stating it."""
    
    user_prompt = f"""
    Describe the aftermath of:
    Exchange Type: {exchange.exchange_type.value}
    Player Submission: {dynamics.player_submission_level:.1%} (don't mention numbers)
    
    Write 1-2 sentences about what lingers after the moment.
    Focus on atmosphere and unspoken changes.
    """
    
    aftermath = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return aftermath.strip()

async def _generate_player_feelings(
    context: NarratorContext,
    exchange: PowerExchange,
    dynamics: Any
) -> str:
    """Generate player's internal reaction using GPT"""
    
    system_prompt = """You describe internal emotional reactions to power dynamics.
    Focus on complex, conflicting feelings. Show gradual acceptance or resistance.
    Use second person perspective."""
    
    user_prompt = f"""
    Describe the player's internal reaction to:
    Exchange: {exchange.exchange_type.value}
    Intensity: {exchange.intensity:.1%}
    Acceptance Level: {dynamics.acceptance_level:.1%}
    Resistance Level: {dynamics.resistance_level:.1%}
    
    Write 1-2 sentences about what they feel internally.
    Show complexity and ambivalence.
    """
    
    feelings = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="introspection"
    )
    
    return feelings.strip()

async def _present_response_options(
    context: NarratorContext,
    options: List[str]
) -> List[str]:
    """Present response options naturally using GPT"""
    
    system_prompt = """You present choices in a narrative way for a slice-of-life game.
    Make options feel like natural impulses or thoughts rather than menu items.
    Use second person perspective."""
    
    presented_options = []
    for option in options[:4]:  # Limit to 4 options
        user_prompt = f"""
        Present this choice naturally:
        Option: {option}
        
        Write a brief sentence that presents this as a natural impulse or thought.
        Start with "You could..." or "Part of you wants to..." or similar.
        """
        
        presented = await generate_text_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.6,
            max_tokens=40,
            task_type="narrative"
        )
        
        presented_options.append(presented.strip())
    
    return presented_options

async def _acknowledge_player_action(context: NarratorContext, action: str) -> str:
    """Acknowledge player action using GPT"""
    
    system_prompt = """You acknowledge player actions in a slice-of-life simulation.
    Be brief and natural. Don't judge or evaluate, just acknowledge what they're doing."""
    
    user_prompt = f"""
    Acknowledge this player action:
    "{action}"
    
    Write 1-2 sentences acknowledging what they're doing.
    Use second person perspective.
    """
    
    acknowledgment = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=60,
        task_type="narrative"
    )
    
    return acknowledgment.strip()

async def _generate_world_reaction(
    context: NarratorContext,
    action: str,
    world_state: WorldState
) -> str:
    """Generate world's reaction to player action using GPT"""
    
    system_prompt = """You describe how the world responds to player actions.
    Focus on environmental and atmospheric changes. Keep it subtle and natural."""
    
    user_prompt = f"""
    Describe the world's reaction to:
    Action: "{action}"
    Current Mood: {world_state.world_mood.value}
    Time: {world_state.current_time.value}
    
    Write 1-2 sentences about how the environment or atmosphere responds.
    """
    
    reaction = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return reaction.strip()

async def _generate_activity_description(
    context: NarratorContext,
    activity: str,
    time_period: TimeOfDay
) -> str:
    """Generate description of a daily activity using GPT"""
    
    system_prompt = """You describe routine daily activities in a slice-of-life setting.
    Make mundane activities feel atmospheric and meaningful."""
    
    user_prompt = f"""
    Describe this activity:
    Activity: {activity}
    Time: {time_period.value}
    
    Write 1-2 sentences describing the activity naturally.
    Use second person perspective.
    """
    
    description = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return description.strip()

async def _generate_routine_with_dynamics(
    context: NarratorContext,
    activity: str,
    involved_npcs: List[int],
    dynamics: Any
) -> str:
    """Generate routine colored by power dynamics using GPT"""
    
    system_prompt = """You describe how power dynamics subtly color routine activities.
    The control should feel woven into everyday life, not forced or explicit."""
    
    user_prompt = f"""
    Show how power dynamics affect this routine:
    Activity: {activity}
    NPCs Involved: {len(involved_npcs)} people
    Player Submission: {dynamics.player_submission_level:.1%} (don't mention numbers)
    
    Write 2-3 sentences showing how the activity is subtly different due to dynamics.
    Focus on small details and unspoken expectations.
    """
    
    routine = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=120,
        task_type="narrative"
    )
    
    return routine.strip()

async def _generate_npc_routine_involvement(
    context: NarratorContext,
    npc_id: int,
    activity: str
) -> Optional[str]:
    """Generate how an NPC is involved in routine using GPT"""
    
    # Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, npc_id)
    
    if not npc:
        return None
    
    system_prompt = """You describe how NPCs participate in daily activities.
    Show their personality and dominance through their involvement."""
    
    user_prompt = f"""
    How is {npc['npc_name']} involved in:
    Activity: {activity}
    Dominance: {npc['dominance']}/100 (don't mention numbers)
    Personality: {npc['personality_traits']}
    
    Write one sentence about their involvement.
    Show character through action.
    """
    
    involvement = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=60,
        task_type="narrative"
    )
    
    return involvement.strip()

async def _generate_subtle_control_elements(
    context: NarratorContext,
    activity: str,
    dynamics: Any
) -> List[str]:
    """Generate subtle control elements in routine using GPT"""
    
    system_prompt = """You identify subtle control elements in everyday activities.
    Focus on small details that show power dynamics without stating them."""
    
    user_prompt = f"""
    Find subtle control elements in:
    Activity: {activity}
    Submission Level: {dynamics.player_submission_level:.1%} (don't mention numbers)
    
    List 2-3 small details that show control woven into the routine.
    Focus on:
    - Unspoken rules
    - Small permissions
    - Assumed expectations
    - Natural deference
    
    Return as a JSON array of strings.
    """
    
    response = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=150,
        task_type="narrative"
    )
    
    try:
        elements = json.loads(response)
        if isinstance(elements, list):
            return elements[:3]
    except:
        return [line.strip() for line in response.split('\n') if line.strip()][:3]
    
    return []

async def _narrate_time_passage(context: NarratorContext, current_time: TimeOfDay) -> str:
    """Narrate the passage of time using GPT"""
    
    system_prompt = """You narrate time transitions in a slice-of-life simulation.
    Focus on how the changing time affects atmosphere and daily rhythms.
    Be brief and evocative."""
    
    user_prompt = f"""
    Narrate a transition to {current_time.value}.
    
    Write 1-2 sentences about time passing and the new period beginning.
    Focus on sensory and atmospheric changes.
    """
    
    narration = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return narration.strip()

async def _narrate_mood_shift(context: NarratorContext, new_mood: WorldMood) -> str:
    """Narrate a shift in world mood using GPT"""
    
    system_prompt = """You narrate subtle shifts in emotional atmosphere.
    Show mood changes through environmental and social cues, not explicit statements."""
    
    user_prompt = f"""
    Narrate a shift to a {new_mood.value} mood.
    
    Write 1-2 sentences showing how the atmosphere is changing.
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

async def _narrate_tension(context: NarratorContext, tension: WorldTension) -> str:
    """Narrate building tension using GPT"""
    
    dominant_tension, level = tension.get_dominant_tension()
    
    system_prompt = """You narrate subtle tension building in a slice-of-life setting.
    Focus on unspoken dynamics and atmospheric pressure."""
    
    user_prompt = f"""
    Narrate {dominant_tension} tension at {level:.0%} intensity.
    Power: {tension.power_tension:.0%}
    Sexual: {tension.sexual_tension:.0%}
    Social: {tension.social_tension:.0%}
    
    Write 1-2 sentences showing this tension subtly.
    Don't mention the type of tension explicitly.
    """
    
    narration = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return narration.strip()

async def _narrate_ambient_detail(context: NarratorContext, world_state: WorldState) -> str:
    """Generate ambient world detail using GPT"""
    
    system_prompt = """You create ambient details for a living world.
    Focus on small, atmospheric observations that make the world feel real."""
    
    user_prompt = f"""
    Create an ambient detail for:
    Time: {world_state.current_time.value}
    Mood: {world_state.world_mood.value}
    Activity Level: {len(world_state.ongoing_events)} events happening
    
    Write 1-2 sentences of atmospheric detail.
    Focus on something small but evocative.
    """
    
    detail = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=80,
        task_type="narrative"
    )
    
    return detail.strip()

async def _generate_npc_reaction(
    context: NarratorContext,
    npc_id: int,
    action: str,
    world_state: WorldState
) -> Optional[str]:
    """Generate NPC reaction to player action using GPT"""
    
    # Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, personality_traits
            FROM NPCStats
            WHERE npc_id = $1
        """, npc_id)
    
    if not npc:
        return None
    
    system_prompt = """You describe how NPCs react to player actions.
    Reactions should be subtle and character-appropriate.
    Show personality through response."""
    
    user_prompt = f"""
    How does {npc['npc_name']} react to:
    Player Action: "{action}"
    Character Dominance: {npc['dominance']}/100
    Personality: {npc['personality_traits']}
    Current Mood: {world_state.world_mood.value}
    
    Write 1-2 sentences showing their reaction.
    Focus on behavior and body language.
    """
    
    reaction = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=80,
        task_type="narrative"
    )
    
    return reaction.strip()

async def _check_for_dynamic_shift(
    context: NarratorContext,
    action: str,
    dynamics: Any
) -> Optional[str]:
    """Check if action causes a dynamic shift using GPT"""
    
    system_prompt = """You identify subtle shifts in power dynamics.
    Determine if a player action changes the relationship dynamic."""
    
    user_prompt = f"""
    Does this action shift the power dynamic?
    Action: "{action}"
    Current Submission: {dynamics.player_submission_level:.0%}
    Current Acceptance: {dynamics.acceptance_level:.0%}
    
    If yes, write 1 sentence about the subtle shift.
    If no, return "None".
    """
    
    shift = await generate_text_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=60,
        task_type="analysis"
    )
    
    shift_text = shift.strip()
    return shift_text if shift_text.lower() != "none" else None

# ===============================================================================
# Main Narrator Agent
# ===============================================================================

class SliceOfLifeNarrator:
    """
    The narrative voice for the open-world simulation.
    Works with the World Director to provide immersive narration.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = NarratorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Sub-agents for specialized narration
        self.scene_narrator = Agent(
            name="SceneNarrator",
            instructions="""
            You narrate slice-of-life scenes in a femdom setting.
            Focus on atmosphere, subtle power dynamics, and everyday moments.
            Your tone should be immersive but not overly dramatic.
            Make the power dynamics feel natural and woven into daily life.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
        
        self.dialogue_writer = Agent(
            name="DialogueWriter",
            instructions="""
            You write natural dialogue for NPCs in daily situations.
            Power dynamics should be shown through subtext, not explicit statements.
            Early stages: Friendly, no obvious control
            Late stages: Casual commands, assumed obedience
            Always maintain character voice and personality.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the narrator with world director"""
        if not self.initialized:
            # Initialize world director
            self.context.world_director = WorldDirector(
                self.user_id, 
                self.conversation_id
            )
            await self.context.world_director.initialize()
            
            # Initialize context components
            self.context.context_service = await get_context_service(
                self.user_id, 
                self.conversation_id
            )
            self.context.memory_manager = await get_memory_manager(
                self.user_id, 
                self.conversation_id
            )
            
            self.initialized = True
    
    async def narrate_world_state(self) -> str:
        """Provide narration for current world state"""
        await self.initialize()
        
        # Get world state
        world_state = await self.context.world_director.get_world_state()
        
        # Generate appropriate narration
        if world_state.ongoing_events:
            # Narrate the most important event
            event = max(world_state.ongoing_events, key=lambda e: e.priority)
            narration = await narrate_slice_of_life_scene(
                RunContextWrapper(self.context),
                event,
                world_state
            )
            return narration.scene_description
        else:
            # Generate ambient narration
            ambient = await generate_ambient_narration(
                RunContextWrapper(self.context),
                "atmosphere",
                world_state
            )
            return ambient.description
    
    async def process_player_input(self, user_input: str) -> Dict[str, Any]:
        """Process player input and generate appropriate narration"""
        await self.initialize()
        
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
            }
        }
    
    async def generate_scene_transition(self) -> str:
        """Generate narration for time/scene transitions"""
        await self.initialize()
        
        # Simulate world tick
        tick_result = await self.context.world_director.simulate_world_tick()
        
        # Get new world state
        world_state = await self.context.world_director.get_world_state()
        
        # Generate transition narration
        ambient = await generate_ambient_narration(
            RunContextWrapper(self.context),
            "time_passage",
            world_state
        )
        
        return ambient.description

# ===============================================================================
# Export the refactored system
# ===============================================================================

def create_slice_of_life_narrator(user_id: int, conversation_id: int) -> SliceOfLifeNarrator:
    """Create a narrator for the slice-of-life simulation"""
    return SliceOfLifeNarrator(user_id, conversation_id)

__all__ = [
    'SliceOfLifeNarrator',
    'create_slice_of_life_narrator',
    'narrate_slice_of_life_scene',
    'generate_npc_dialogue',
    'narrate_power_exchange',
    'narrate_daily_routine',
    'generate_ambient_narration',
    'narrate_player_action'
]
