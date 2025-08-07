# story_agent/world_simulation_tools.py

"""
World Simulation Tools for the open-world femdom slice-of-life game.
Fully dynamic and agent-driven for emergent gameplay and daily life simulation.
"""

# Standard library imports
import logging
import json
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Literal
from enum import Enum

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict

# Agent SDK imports
from agents import Agent, function_tool, FunctionTool, RunContextWrapper, Runner, ModelSettings

# Database imports
from db.connection import get_db_connection_context

# Local application imports - Core systems
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    progress_npc_narrative_stage,
    check_for_npc_revelation,
    NPC_NARRATIVE_STAGES,
    NPCNarrativeStage
)

from logic.narrative_events import (
    get_relationship_overview,
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence,
    add_moment_of_clarity
)

from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    get_relationship_summary_tool,
    process_relationship_interaction_tool,
    poll_relationship_events_tool
)

# Context system imports
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager, MemorySearchRequest
from context.vector_service import get_vector_service
from context.context_performance import PerformanceMonitor, track_performance

# Time cycle integration
from logic.time_cycle import (
    get_current_time_model,
    get_current_vitals,
    VitalsData,
    CurrentTimeData,
    ActivityType
)

# Initialize logger
logger = logging.getLogger(__name__)

# ============= ENUMS FOR WORLD SIMULATION =============

class DailyRoutinePhase(Enum):
    """Phases of daily routine"""
    WAKE_UP = "wake_up"
    MORNING_ROUTINE = "morning_routine"
    WORK_HOURS = "work_hours"
    LUNCH_BREAK = "lunch_break"
    AFTERNOON_WORK = "afternoon_work"
    EVENING_COMMUTE = "evening_commute"
    DINNER_TIME = "dinner_time"
    EVENING_LEISURE = "evening_leisure"
    BEDTIME_ROUTINE = "bedtime_routine"
    SLEEP = "sleep"

class LocationType(Enum):
    """Types of locations in the world"""
    HOME = "home"
    WORKPLACE = "workplace"
    SOCIAL_VENUE = "social_venue"
    PUBLIC_SPACE = "public_space"
    PRIVATE_SPACE = "private_space"
    COMMERCIAL = "commercial"
    RECREATIONAL = "recreational"

class InteractionIntensity(Enum):
    """Intensity levels for slice-of-life interactions"""
    CASUAL = "casual"          # Normal daily interaction
    FRIENDLY = "friendly"      # Warm, positive interaction
    INTIMATE = "intimate"      # Close, personal interaction
    TENSE = "tense"           # Underlying tension
    DOMINANT = "dominant"      # Clear power dynamic
    SUBMISSIVE = "submissive" # Player in submissive position

class MoodType(Enum):
    """NPC mood types for daily interactions"""
    RELAXED = "relaxed"
    PLAYFUL = "playful"
    ASSERTIVE = "assertive"
    CARING = "caring"
    STRICT = "strict"
    TEASING = "teasing"
    INTIMATE = "intimate"
    MYSTERIOUS = "mysterious"

class PowerDynamicType(Enum):
    """Types of power dynamics in daily life"""
    SUBTLE_CONTROL = "subtle_control"
    CASUAL_DOMINANCE = "casual_dominance"
    PROTECTIVE_CONTROL = "protective_control"
    PLAYFUL_TEASING = "playful_teasing"
    RITUAL_SUBMISSION = "ritual_submission"
    FINANCIAL_CONTROL = "financial_control"
    SOCIAL_HIERARCHY = "social_hierarchy"
    INTIMATE_COMMAND = "intimate_command"

# ============= PYDANTIC MODELS =============

class DailyRoutineEvent(BaseModel):
    """An event in someone's daily routine"""
    phase: DailyRoutinePhase
    activity: str
    location: LocationType
    participants: List[int] = Field(default_factory=list)
    start_time: str  # "HH:MM" format
    duration_minutes: int
    is_interruptible: bool = True
    power_dynamic_present: bool = False
    mood: MoodType = MoodType.RELAXED
    
    model_config = ConfigDict(extra="forbid")

class AmbientInteraction(BaseModel):
    """A small, ambient interaction in daily life"""
    interaction_type: str
    initiator_id: int
    target_id: Optional[int] = None
    dialogue: str
    intensity: InteractionIntensity
    requires_response: bool = False
    possible_responses: List[str] = Field(default_factory=list)
    emotional_context: Optional[str] = None
    power_dynamic: Optional[PowerDynamicType] = None
    
    model_config = ConfigDict(extra="forbid")

class PowerDynamicMoment(BaseModel):
    """A subtle power dynamic moment in daily life"""
    moment_type: str
    description: str
    npc_id: int
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    context: str
    player_options: List[str] = Field(default_factory=list)
    acceptance_increases_dynamic: bool = True
    narrative_stage_appropriate: bool = True
    
    model_config = ConfigDict(extra="forbid")

class SliceOfLifeScene(BaseModel):
    """A complete slice-of-life scene"""
    scene_id: str
    title: str
    setting: LocationType
    time_of_day: DailyRoutinePhase
    participants: List[int]
    atmosphere: str
    primary_activity: str
    power_dynamics: List[PowerDynamicMoment] = Field(default_factory=list)
    ambient_interactions: List[AmbientInteraction] = Field(default_factory=list)
    scene_mood: MoodType
    can_leave_early: bool = True
    natural_duration_minutes: int = 30
    emergent_narrative: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")

class NPCScheduleEntry(BaseModel):
    """An entry in an NPC's schedule"""
    time_phase: DailyRoutinePhase
    activity: str
    location: LocationType
    availability: str  # "busy", "available", "interruptible"
    mood: MoodType
    power_tendency: Optional[PowerDynamicType] = None
    
    model_config = ConfigDict(extra="forbid")

class WorldStateContext(BaseModel):
    """Context about the current world state"""
    current_phase: DailyRoutinePhase
    location: LocationType
    present_npcs: List[int] = Field(default_factory=list)
    recent_activities: List[str] = Field(default_factory=list)
    active_power_dynamics: List[PowerDynamicType] = Field(default_factory=list)
    overall_mood: MoodType = MoodType.RELAXED
    tension_level: float = Field(0.0, ge=0.0, le=1.0)
    
    model_config = ConfigDict(extra="forbid")


class NPCInfo(BaseModel):
    """Minimal information about an NPC needed for interactions"""
    id: int
    dominance: int
    stage: str

    model_config = ConfigDict(extra="ignore")


class PowerMomentData(BaseModel):
    """Data describing a power dynamic moment"""
    opportunity: Optional[str] = None
    approach: Optional[str] = None
    intensity: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


class RelationshipImpacts(BaseModel):
    """Changes to relationship dimensions"""
    influence: Optional[int] = None
    dependence: Optional[int] = None
    trust: Optional[int] = None
    volatility: Optional[int] = None
    unresolved_conflict: Optional[int] = None
    hidden_agendas: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


class PlayerResponseResult(BaseModel):
    """Result of processing a player response"""
    response_type: Optional[str] = None
    relationship_impacts: RelationshipImpacts = Field(default_factory=RelationshipImpacts)
    narrative_flavor: Optional[str] = None
    interaction_processed: bool = True
    error: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class VitalRequirement(BaseModel):
    """Requirements related to player vitals"""
    fatigue_max: Optional[int] = None
    hunger_min: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


class DailyActivity(BaseModel):
    """An available daily activity option"""
    name: str
    type: str
    vital_requirement: Optional[VitalRequirement] = None
    priority: Optional[str] = None
    npc_id: Optional[int] = None
    power_dynamic: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class TransitionResult(BaseModel):
    """Result of transitioning to a new daily phase"""
    old_phase: Optional[str] = None
    new_phase: Optional[str] = None
    transition_scene: Optional[str] = None
    time_updated: bool = False
    npcs_relocated: int = 0
    error: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

# ============= DYNAMIC AGENT SYSTEM =============

# Agent for generating contextual daily life scenes
@function_tool
def generate_daily_scene(
    time_phase: str,
    location: str,
    participating_npcs: str,  # JSON string of NPC data
    recent_context: str,
    player_vitals: str  # JSON string of vitals
) -> str:
    """Generate a dynamic daily life scene based on context."""
    # This is called by the agent
    return json.dumps({
        "scene": "A dynamically generated scene",
        "atmosphere": "contextual",
        "power_dynamics": []
    })

DailyLifeDirector = Agent(
    name="DailyLifeDirector",
    instructions="""You are the director of daily life in a femdom slice-of-life simulation.
    
    Your role is to create natural, emergent scenes from everyday activities where power dynamics 
    arise organically rather than being forced. Consider:
    
    1. Time of day and typical activities for that phase
    2. NPC personalities and their narrative stages (how open they are about control)
    3. Player's current state (tired, hungry, stressed, etc.)
    4. Recent interactions and their emotional residue
    5. The location and its social dynamics
    
    Power dynamics should emerge from normal situations:
    - Morning: Who decides breakfast, clothing, schedule?
    - Work: Subtle hierarchies, "helpful" guidance
    - Meals: Control over food choices, eating pace
    - Evening: Relaxation on whose terms?
    
    Generate scenes that feel like real life with an undercurrent of power exchange.
    The more advanced an NPC's narrative stage, the more overt their control can be.
    
    Remember: This is slice-of-life. Most moments are mundane with subtle dynamics.""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.7),
    tools=[generate_daily_scene]
)

# Agent for generating contextual ambient dialogue
@function_tool
def create_ambient_dialogue(
    npc_data: str,  # JSON with NPC personality, stage, mood
    context: str,  # Current situation
    relationship_state: str,  # JSON of relationship dynamics
    power_dynamic: Optional[str] = None
) -> str:
    """Create natural dialogue that reflects relationship dynamics."""
    return json.dumps({
        "dialogue": "Contextual dialogue",
        "subtext": "Hidden meaning",
        "responses": ["Option 1", "Option 2", "Option 3"]
    })

AmbientDialogueWriter = Agent(
    name="AmbientDialogueWriter",
    instructions="""You write natural, contextual dialogue for NPCs in daily situations.
    
    Guidelines:
    - Dialogue should feel conversational and realistic
    - Power dynamics are shown through subtext, not explicit statements
    - Match the NPC's personality and current narrative stage
    - Early stages: Friendly, no obvious control
    - Middle stages: Subtle suggestions, gentle steering
    - Late stages: Casual commands, assumed obedience
    
    Consider the specific context:
    - Morning: "I laid out your clothes. The blue shirt suits you better."
    - Meals: "You're having the salad. You need to eat healthier."
    - Evening: "Come here. You look tense." (then taking control of their relaxation)
    
    Include 2-3 natural response options for the player that range from:
    - Acceptance (increases dynamic)
    - Mild resistance (maintains tension)
    - Deflection (avoids the dynamic)
    
    Never be heavy-handed. Even dominant NPCs speak naturally.""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.8),
    tools=[create_ambient_dialogue]
)

# Agent for detecting and creating power dynamic moments
@function_tool
def identify_power_moment(
    scene_context: str,
    npc_stage: str,
    recent_interactions: str,
    player_state: str
) -> str:
    """Identify organic power dynamic opportunities in daily life."""
    return json.dumps({
        "opportunity": "A power dynamic opportunity",
        "approach": "How to implement it",
        "intensity": 0.5
    })

PowerDynamicsOrchestrator = Agent(
    name="PowerDynamicsOrchestrator",
    instructions="""You identify and orchestrate subtle power dynamics in everyday situations.
    
    Look for natural opportunities where control can emerge:
    - Decision points (what to eat, wear, do)
    - Moments of player uncertainty or fatigue
    - Routine activities that can establish patterns
    - Social situations with hierarchy implications
    
    Based on NPC narrative stage:
    - Innocent Beginning: No deliberate control, just personality
    - First Doubts: Occasional "helpful" decisions
    - Creeping Realization: Regular gentle steering
    - Veil Thinning: Open but caring control
    - Full Revelation: Complete, casual dominance
    
    Consider player state:
    - High fatigue: More susceptible to suggestion
    - Low vitals: Opportunities for "caretaking" control
    - Recent submission: Build on established patterns
    
    Power dynamics should feel natural, not forced. They emerge from:
    - Personality differences
    - Confidence disparities  
    - Established relationship patterns
    - Contextual authority (expertise, ownership, seniority)
    
    Avoid cartoon villainy. Even controlling NPCs believe they're being helpful/caring.""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.6),
    tools=[identify_power_moment]
)

# Agent for generating emergent narrative threads
@function_tool
def detect_emergent_narrative(
    recent_events: str,  # JSON of recent scenes/interactions
    relationship_patterns: str,  # Active patterns from dynamic system
    world_state: str
) -> str:
    """Detect emergent narrative threads from accumulated interactions."""
    return json.dumps({
        "narrative_thread": "An emergent story",
        "key_npcs": [],
        "next_development": "What might happen next"
    })

EmergentNarrativeDetector = Agent(
    name="EmergentNarrativeDetector",
    instructions="""You detect emergent narratives from the accumulation of daily interactions.
    
    Look for patterns that suggest larger stories:
    - Multiple NPCs showing similar behaviors (coordination?)
    - Escalating control across different contexts
    - Player habits being subtly shaped over time
    - Relationship dynamics creating dependencies
    
    These aren't pre-planned stories but emerge from:
    - NPC personalities interacting
    - Power dynamics accumulating
    - Player choices creating patterns
    - Daily routines establishing expectations
    
    When you detect a narrative thread:
    - Identify the key players
    - Recognize the pattern
    - Suggest how it might develop
    - Keep it grounded in daily life
    
    Remember: Emergent narratives should feel discovered, not authored.""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.7),
    tools=[detect_emergent_narrative]
)

# Agent for managing player agency within power dynamics
@function_tool
def generate_player_agency(
    current_situation: str,
    power_dynamic: str,
    relationship_context: str,
    player_history: str
) -> str:
    """Generate meaningful player choices that respect both agency and dynamics."""
    return json.dumps({
        "choices": [],
        "hidden_implications": {},
        "long_term_effects": {}
    })

PlayerAgencyManager = Agent(
    name="PlayerAgencyManager",
    instructions="""You ensure player agency remains meaningful within power dynamics.
    
    Generate choices that:
    - Feel natural to the situation
    - Respect established relationship dynamics
    - Allow for resistance, acceptance, or negotiation
    - Have subtle long-term implications
    
    Choice categories:
    1. Compliance choices (degrees of acceptance)
       - Eager compliance (deepens dynamic)
       - Resigned acceptance (maintains status quo)
       - Reluctant obedience (internal resistance)
    
    2. Resistance choices (pushing back)
       - Playful deflection (maintains relationship)
       - Firm boundary (serious resistance)
       - Counter-proposal (negotiation)
    
    3. Subversion choices (working within constraints)
       - Malicious compliance (following letter not spirit)
       - Finding loopholes (creative interpretation)
       - Delayed resistance (comply now, resist later)
    
    Every situation should have at least one choice from each category.
    Make implications clear through subtext, not explicit warnings.
    
    Long-term effects to track:
    - Relationship pattern reinforcement
    - Behavioral conditioning
    - Emotional dependencies
    - Power dynamic escalation/reduction""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.7),
    tools=[generate_player_agency]
)

# ============= HELPER FUNCTIONS =============

def _ensure_tool(obj):
    """Ensure object is a FunctionTool"""
    if obj is None:
        return None
    if isinstance(obj, FunctionTool):
        return obj
    if callable(obj):
        return function_tool(obj)
    logger.warning(f"Object {obj} is not a valid tool - skipping")
    return None

async def _get_current_world_context(ctx: RunContextWrapper) -> WorldStateContext:
    """Get comprehensive world state context"""
    context = ctx.context
    
    # Get time and location
    current_time = await get_current_time_model(context.user_id, context.conversation_id)
    phase = _map_time_to_phase(current_time.time_of_day)
    
    # Get current location from context
    async with get_db_connection_context() as conn:
        location_val = await conn.fetchval("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentLocation'
        """, context.user_id, context.conversation_id)
    
    location = _map_location_string(location_val or "home")
    
    # Get present NPCs
    async with get_db_connection_context() as conn:
        present_npcs = await conn.fetch("""
            SELECT npc_id FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
            AND introduced=true AND current_location=$3
        """, context.user_id, context.conversation_id, location.value)
    
    npc_ids = [row['npc_id'] for row in present_npcs]
    
    # Get recent activities from memory
    memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
    recent_memories = await memory_manager.search_memories(
        query_text="activity action",
        limit=5,
        memory_types=["activity", "interaction"],
        use_vector=False
    )
    
    recent_activities = [mem.content for mem in recent_memories[:3]] if hasattr(recent_memories[0], 'content') else []
    
    # Determine overall mood and tension
    relationship_overview = await get_relationship_overview(context.user_id, context.conversation_id)
    avg_corruption = relationship_overview.get('aggregate_stats', {}).get('average_corruption', 0)
    tension = min(1.0, avg_corruption / 100.0)
    
    mood = MoodType.RELAXED
    if tension > 0.7:
        mood = MoodType.INTIMATE
    elif tension > 0.5:
        mood = MoodType.TEASING
    elif tension > 0.3:
        mood = MoodType.PLAYFUL
    
    return WorldStateContext(
        current_phase=phase,
        location=location,
        present_npcs=npc_ids,
        recent_activities=recent_activities,
        active_power_dynamics=[],
        overall_mood=mood,
        tension_level=tension
    )

def _map_time_to_phase(time_of_day: str) -> DailyRoutinePhase:
    """Map time string to daily routine phase"""
    mapping = {
        "Morning": DailyRoutinePhase.MORNING_ROUTINE,
        "Afternoon": DailyRoutinePhase.AFTERNOON_WORK,
        "Evening": DailyRoutinePhase.EVENING_LEISURE,
        "Night": DailyRoutinePhase.BEDTIME_ROUTINE
    }
    return mapping.get(time_of_day, DailyRoutinePhase.MORNING_ROUTINE)

def _map_location_string(location: str) -> LocationType:
    """Map location string to LocationType"""
    location_lower = location.lower()
    if "home" in location_lower or "apartment" in location_lower:
        return LocationType.HOME
    elif "work" in location_lower or "office" in location_lower:
        return LocationType.WORKPLACE
    elif "bar" in location_lower or "club" in location_lower:
        return LocationType.SOCIAL_VENUE
    elif "store" in location_lower or "shop" in location_lower:
        return LocationType.COMMERCIAL
    elif "park" in location_lower or "gym" in location_lower:
        return LocationType.RECREATIONAL
    else:
        return LocationType.PUBLIC_SPACE

# ============= MAIN TOOL FUNCTIONS =============

@function_tool
@track_performance("generate_daily_life_scene")
async def generate_daily_life_scene(
    ctx: RunContextWrapper,
    forced_participants: Optional[List[int]] = None,
    forced_activity: Optional[str] = None,
    include_power_dynamics: bool = True
) -> SliceOfLifeScene:
    """
    Generate a complete daily life scene using AI agents.
    
    Args:
        forced_participants: Specific NPCs to include
        forced_activity: Specific activity to center scene around
        include_power_dynamics: Whether to include power dynamics
        
    Returns:
        Dynamically generated slice-of-life scene
    """
    context = ctx.context
    
    try:
        # Get world context
        world_context = await _get_current_world_context(ctx)
        
        # Get participating NPCs data
        if forced_participants:
            npc_ids = forced_participants
        else:
            npc_ids = world_context.present_npcs[:2]  # Limit to 2 for focus
        
        npc_data = []
        for npc_id in npc_ids:
            async with get_db_connection_context() as conn:
                npc = await conn.fetchrow("""
                    SELECT npc_name, dominance, personality_traits
                    FROM NPCStats WHERE npc_id=$1
                """, npc_id)
            
            if npc:
                stage = await get_npc_narrative_stage(
                    context.user_id, context.conversation_id, npc_id
                )
                npc_data.append({
                    "id": npc_id,
                    "name": npc['npc_name'],
                    "dominance": npc['dominance'],
                    "stage": stage.name,
                    "traits": json.loads(npc['personality_traits']) if isinstance(npc['personality_traits'], str) else npc['personality_traits']
                })
        
        # Get player vitals for context
        vitals = await get_current_vitals(context.user_id, context.conversation_id)
        
        # Call the Daily Life Director
        messages = [{
            "role": "user",
            "content": f"Generate a scene for {world_context.current_phase.value}"
        }]
        
        result = await Runner.run(
            DailyLifeDirector,
            messages=messages,
            calls=[{
                "name": "generate_daily_scene",
                "kwargs": {
                    "time_phase": world_context.current_phase.value,
                    "location": world_context.location.value,
                    "participating_npcs": json.dumps(npc_data),
                    "recent_context": json.dumps(world_context.recent_activities),
                    "player_vitals": json.dumps(vitals.to_dict())
                }
            }]
        )
        
        # Parse agent output
        scene_data = json.loads(result.output) if result.output else {}
        
        # Create scene ID
        scene_id = f"scene_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        
        # Build the scene
        scene = SliceOfLifeScene(
            scene_id=scene_id,
            title=scene_data.get("title", "Daily Life"),
            setting=world_context.location,
            time_of_day=world_context.current_phase,
            participants=npc_ids,
            atmosphere=scene_data.get("atmosphere", "A typical moment in daily life"),
            primary_activity=forced_activity or scene_data.get("activity", "routine"),
            scene_mood=world_context.overall_mood,
            natural_duration_minutes=30
        )
        
        # Add power dynamics if requested
        if include_power_dynamics and npc_ids:
            for npc_info in npc_data[:1]:  # One main dynamic per scene
                power_moment = await generate_organic_power_moment(
                    ctx, npc_info["id"], world_context
                )
                if power_moment:
                    scene.power_dynamics.append(power_moment)
        
        # Generate ambient interactions
        for npc_info in npc_data:
            interaction = await generate_contextual_interaction(
                ctx, NPCInfo(**npc_info), world_context
            )
            if interaction:
                scene.ambient_interactions.append(interaction)
        
        # Check for emergent narrative
        emergent = await detect_emergent_story(ctx, scene, world_context)
        if emergent:
            scene.emergent_narrative = emergent
        
        return scene
        
    except Exception as e:
        logger.error(f"Error generating daily life scene: {e}", exc_info=True)
        # Return a basic scene as fallback
        return SliceOfLifeScene(
            scene_id=f"fallback_{int(datetime.now().timestamp())}",
            title="A Quiet Moment",
            setting=LocationType.HOME,
            time_of_day=DailyRoutinePhase.MORNING_ROUTINE,
            participants=[],
            atmosphere="The day continues quietly",
            primary_activity="routine",
            scene_mood=MoodType.RELAXED
        )

@function_tool
async def generate_organic_power_moment(
    ctx: RunContextWrapper,
    npc_id: int,
    world_context: WorldStateContext
) -> Optional[PowerDynamicMoment]:
    """
    Generate an organic power dynamic moment using AI.
    
    Args:
        npc_id: NPC creating the dynamic
        world_context: Current world state
        
    Returns:
        Dynamically generated power moment or None
    """
    context = ctx.context
    
    try:
        # Get NPC details and stage
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT npc_name, dominance, personality_traits
                FROM NPCStats WHERE npc_id=$1
            """, npc_id)
        
        if not npc:
            return None
        
        stage = await get_npc_narrative_stage(
            context.user_id, context.conversation_id, npc_id
        )
        
        # Get relationship state
        manager = OptimizedRelationshipManager(context.user_id, context.conversation_id)
        rel_state = await manager.get_relationship_state(
            'npc', npc_id, 'player', context.user_id
        )
        
        # Get player state
        vitals = await get_current_vitals(context.user_id, context.conversation_id)
        player_state = {
            "fatigue": vitals.fatigue,
            "stress": world_context.tension_level,
            "recent_submission": any("accept" in act.lower() for act in world_context.recent_activities)
        }
        
        # Call Power Dynamics Orchestrator
        messages = [{
            "role": "user",
            "content": "Identify a natural power dynamic opportunity"
        }]
        
        result = await Runner.run(
            PowerDynamicsOrchestrator,
            messages=messages,
            calls=[{
                "name": "identify_power_moment",
                "kwargs": {
                    "scene_context": json.dumps({
                        "phase": world_context.current_phase.value,
                        "location": world_context.location.value,
                        "mood": world_context.overall_mood.value
                    }),
                    "npc_stage": stage.name,
                    "recent_interactions": json.dumps(world_context.recent_activities),
                    "player_state": json.dumps(player_state)
                }
            }]
        )
        
        if not result.output:
            return None

        moment_data = PowerMomentData(**json.loads(result.output))

        # Generate player choices
        choices = await generate_player_choices_for_moment(
            ctx, moment_data, npc_id, stage.name
        )

        return PowerDynamicMoment(
            moment_type=moment_data.approach or "subtle_control",
            description=moment_data.opportunity or "A subtle shift in dynamics",
            npc_id=npc_id,
            intensity=moment_data.intensity or 0.5,
            context=world_context.current_phase.value,
            player_options=choices,
            acceptance_increases_dynamic=True,
            narrative_stage_appropriate=True
        )
        
    except Exception as e:
        logger.error(f"Error generating power moment: {e}", exc_info=True)
        return None

@function_tool
async def generate_contextual_interaction(
    ctx: RunContextWrapper,
    npc_info: NPCInfo,
    world_context: WorldStateContext
) -> Optional[AmbientInteraction]:
    """
    Generate contextual ambient dialogue using AI.
    
    Args:
        npc_info: NPC data including personality and stage
        world_context: Current world state
        
    Returns:
        Contextual interaction or None
    """
    context = ctx.context
    
    try:
        # Get relationship dynamics
        manager = OptimizedRelationshipManager(context.user_id, context.conversation_id)
        rel_state = await manager.get_relationship_state(
            'npc', npc_info.id, 'player', context.user_id
        )
        
        relationship_data = {
            "trust": rel_state.dimensions.trust,
            "affection": rel_state.dimensions.affection,
            "influence": rel_state.dimensions.influence,
            "patterns": list(rel_state.history.active_patterns),
            "momentum": rel_state.momentum.get_magnitude()
        }
        
        # Determine if power dynamic should be present
        power_dynamic = None
        if npc_info.dominance > 60 and npc_info.stage != "Innocent Beginning":
            if rel_state.dimensions.influence > 70:
                power_dynamic = PowerDynamicType.INTIMATE_COMMAND.value
            elif rel_state.dimensions.influence > 50:
                power_dynamic = PowerDynamicType.CASUAL_DOMINANCE.value
            else:
                power_dynamic = PowerDynamicType.SUBTLE_CONTROL.value
        
        # Call Ambient Dialogue Writer
        messages = [{
            "role": "user",
            "content": f"Create dialogue for {world_context.current_phase.value}"
        }]
        
        result = await Runner.run(
            AmbientDialogueWriter,
            messages=messages,
            calls=[{
                "name": "create_ambient_dialogue",
                "kwargs": {
                    "npc_data": json.dumps(npc_info),
                    "context": json.dumps({
                        "phase": world_context.current_phase.value,
                        "location": world_context.location.value,
                        "recent_activities": world_context.recent_activities
                    }),
                    "relationship_state": json.dumps(relationship_data),
                    "power_dynamic": power_dynamic
                }
            }]
        )
        
        if not result.output:
            return None
        
        dialogue_data = json.loads(result.output)
        
        # Determine intensity based on context
        intensity = InteractionIntensity.CASUAL
        if power_dynamic:
            if PowerDynamicType.INTIMATE_COMMAND.value in power_dynamic:
                intensity = InteractionIntensity.DOMINANT
            elif rel_state.dimensions.affection > 70:
                intensity = InteractionIntensity.INTIMATE
        
        return AmbientInteraction(
            interaction_type="contextual_dialogue",
            initiator_id=npc_info["id"],
            dialogue=dialogue_data.get("dialogue", "..."),
            intensity=intensity,
            requires_response=len(dialogue_data.get("responses", [])) > 0,
            possible_responses=dialogue_data.get("responses", []),
            emotional_context=dialogue_data.get("subtext"),
            power_dynamic=PowerDynamicType[power_dynamic] if power_dynamic else None
        )
        
    except Exception as e:
        logger.error(f"Error generating contextual interaction: {e}", exc_info=True)
        return None

@function_tool
async def generate_player_choices_for_moment(
    ctx: RunContextWrapper,
    moment_data: PowerMomentData,
    npc_id: int,
    npc_stage: str
) -> List[str]:
    """
    Generate meaningful player choices using AI.
    
    Args:
        moment_data: Power moment data
        npc_id: NPC involved
        npc_stage: NPC's narrative stage
        
    Returns:
        List of player choice options
    """
    context = ctx.context
    
    try:
        # Get player history with this NPC
        memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
        npc_memories = await memory_manager.search_memories(
            query_text=f"npc_{npc_id} choice decision",
            limit=5,
            tags=[f"npc_{npc_id}", "player_choice"],
            use_vector=True
        )
        
        player_history = [mem.content for mem in npc_memories[:3]] if npc_memories else []
        
        # Get relationship context
        manager = OptimizedRelationshipManager(context.user_id, context.conversation_id)
        rel_state = await manager.get_relationship_state(
            'npc', npc_id, 'player', context.user_id
        )
        
        # Call Player Agency Manager
        messages = [{
            "role": "user",
            "content": "Generate player choices for this situation"
        }]
        
        result = await Runner.run(
            PlayerAgencyManager,
            messages=messages,
            calls=[{
                "name": "generate_player_agency",
                "kwargs": {
                    "current_situation": moment_data.model_dump_json(),
                    "power_dynamic": moment_data.approach or "subtle",
                    "relationship_context": json.dumps({
                        "stage": npc_stage,
                        "influence": rel_state.dimensions.influence,
                        "trust": rel_state.dimensions.trust,
                        "patterns": list(rel_state.history.active_patterns)
                    }),
                    "player_history": json.dumps(player_history)
                }
            }]
        )
        
        if not result.output:
            # Fallback choices
            return [
                "Accept naturally",
                "Question gently",
                "Deflect playfully",
                "Resist firmly"
            ]
        
        choice_data = json.loads(result.output)
        return choice_data.get("choices", [])[:5]  # Max 5 choices
        
    except Exception as e:
        logger.error(f"Error generating player choices: {e}", exc_info=True)
        return ["Continue", "Say something", "Do something else"]

@function_tool
async def detect_emergent_story(
    ctx: RunContextWrapper,
    current_scene: SliceOfLifeScene,
    world_context: WorldStateContext
) -> Optional[str]:
    """
    Detect emergent narrative threads from accumulated interactions.
    
    Args:
        current_scene: The current scene
        world_context: World state context
        
    Returns:
        Emergent narrative description or None
    """
    context = ctx.context
    
    try:
        # Get recent events from memory
        memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
        recent_events = await memory_manager.search_memories(
            query_text="scene interaction power",
            limit=10,
            memory_types=["interaction", "power_moment", "scene"],
            use_vector=False
        )
        
        # Get relationship patterns
        relationship_overview = await get_relationship_overview(
            context.user_id, context.conversation_id
        )
        
        # Collect active patterns across all relationships
        all_patterns = set()
        for rel in relationship_overview.get('relationships', []):
            # Assuming relationships have patterns from the dynamic system
            manager = OptimizedRelationshipManager(context.user_id, context.conversation_id)
            state = await manager.get_relationship_state(
                'npc', rel['npc_id'], 'player', context.user_id
            )
            all_patterns.update(state.history.active_patterns)
        
        # Call Emergent Narrative Detector
        messages = [{
            "role": "user",
            "content": "Detect any emergent narratives"
        }]
        
        result = await Runner.run(
            EmergentNarrativeDetector,
            messages=messages,
            calls=[{
                "name": "detect_emergent_narrative",
                "kwargs": {
                    "recent_events": json.dumps([
                        mem.content if hasattr(mem, 'content') else str(mem)
                        for mem in recent_events[:5]
                    ]),
                    "relationship_patterns": json.dumps(list(all_patterns)),
                    "world_state": json.dumps({
                        "tension": world_context.tension_level,
                        "mood": world_context.overall_mood.value,
                        "active_npcs": len(world_context.present_npcs)
                    })
                }
            }]
        )
        
        if not result.output:
            return None
        
        narrative_data = json.loads(result.output)
        
        # Only return if there's a meaningful narrative thread
        if narrative_data.get("narrative_thread"):
            return narrative_data["narrative_thread"]
        
        return None
        
    except Exception as e:
        logger.error(f"Error detecting emergent narrative: {e}", exc_info=True)
        return None

@function_tool
async def process_player_response_to_interaction(
    ctx: RunContextWrapper,
    interaction: AmbientInteraction,
    player_response: str
) -> PlayerResponseResult:
    """
    Process how a player responds to an interaction and update relationships.
    
    Args:
        interaction: The interaction being responded to
        player_response: The player's chosen response
        
    Returns:
        Results of processing the response
    """
    context = ctx.context
    
    try:
        # Determine response type
        response_type = "neutral"
        if any(word in player_response.lower() for word in ["yes", "okay", "sure", "of course"]):
            response_type = "accept"
        elif any(word in player_response.lower() for word in ["no", "don't", "stop", "won't"]):
            response_type = "resist"
        elif any(word in player_response.lower() for word in ["maybe", "later", "busy"]):
            response_type = "deflect"
        
        # Calculate relationship impacts based on response
        impacts = {}
        if interaction.power_dynamic:
            if response_type == "accept":
                impacts = {
                    "influence": 3,
                    "dependence": 2,
                    "trust": 1
                }
            elif response_type == "resist":
                impacts = {
                    "influence": -1,
                    "volatility": 2,
                    "unresolved_conflict": 1
                }
            else:  # deflect
                impacts = {
                    "hidden_agendas": 1
                }
        
        # Process relationship interaction
        if impacts:
            result = await process_relationship_interaction_tool(
                ctx,
                entity1_type="npc",
                entity1_id=interaction.initiator_id,
                entity2_type="player",
                entity2_id=context.user_id,
                interaction_type="dialogue_response",
                context=f"{response_type}: {player_response}",
                check_for_event=False
            )
        
        # Store in memory
        memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
        await memory_manager.add_memory(
            content=f"Responded to interaction: {response_type} - {player_response[:50]}",
            memory_type="player_choice",
            importance=0.4,
            tags=["interaction", f"npc_{interaction.initiator_id}", response_type],
            metadata={
                "response_type": response_type,
                "power_dynamic": interaction.power_dynamic.value if interaction.power_dynamic else None
            }
        )
        
        # Generate narrative flavor
        flavor = "The moment passes."
        if response_type == "accept" and interaction.power_dynamic:
            flavor = "You feel the subtle shift in dynamics, another thread in the growing pattern."
        elif response_type == "resist":
            flavor = "A brief tension hangs in the air before dissolving."
        elif response_type == "deflect":
            flavor = "You navigate around the moment, maintaining your space."
        
        return PlayerResponseResult(
            response_type=response_type,
            relationship_impacts=RelationshipImpacts(**impacts) if impacts else RelationshipImpacts(),
            narrative_flavor=flavor,
            interaction_processed=True
        )
        
    except Exception as e:
        logger.error(f"Error processing player response: {e}", exc_info=True)
        return PlayerResponseResult(error=str(e), interaction_processed=False)

@function_tool
async def simulate_npc_daily_routine(
    ctx: RunContextWrapper,
    npc_id: int,
    time_phase: Optional[DailyRoutinePhase] = None
) -> NPCScheduleEntry:
    """
    Simulate an NPC's activity for a time phase based on personality and stage.
    
    Args:
        npc_id: NPC to simulate
        time_phase: Phase to simulate (current if None)
        
    Returns:
        NPC's schedule entry for the phase
    """
    context = ctx.context
    
    try:
        # Get NPC data
        async with get_db_connection_context() as conn:
            npc = await conn.fetchrow("""
                SELECT npc_name, dominance, personality_traits, current_location
                FROM NPCStats WHERE npc_id=$1
            """, npc_id)
        
        if not npc:
            raise ValueError(f"NPC {npc_id} not found")
        
        # Get narrative stage
        stage = await get_npc_narrative_stage(
            context.user_id, context.conversation_id, npc_id
        )
        
        # Determine time phase
        if not time_phase:
            current_time = await get_current_time_model(context.user_id, context.conversation_id)
            time_phase = _map_time_to_phase(current_time.time_of_day)
        
        # Simulate based on personality and phase
        traits = json.loads(npc['personality_traits']) if isinstance(npc['personality_traits'], str) else npc['personality_traits']
        
        # Default activities by phase
        phase_activities = {
            DailyRoutinePhase.MORNING_ROUTINE: "morning preparations",
            DailyRoutinePhase.WORK_HOURS: "working",
            DailyRoutinePhase.LUNCH_BREAK: "having lunch",
            DailyRoutinePhase.AFTERNOON_WORK: "afternoon tasks",
            DailyRoutinePhase.EVENING_LEISURE: "relaxing",
            DailyRoutinePhase.BEDTIME_ROUTINE: "evening routine"
        }
        
        activity = phase_activities.get(time_phase, "daily activities")
        
        # Determine location
        location = LocationType.HOME
        if time_phase in [DailyRoutinePhase.WORK_HOURS, DailyRoutinePhase.AFTERNOON_WORK]:
            location = LocationType.WORKPLACE
        elif time_phase == DailyRoutinePhase.LUNCH_BREAK:
            location = LocationType.COMMERCIAL
        
        # Determine availability based on dominance and stage
        availability = "available"
        if time_phase in [DailyRoutinePhase.WORK_HOURS, DailyRoutinePhase.AFTERNOON_WORK]:
            availability = "busy"
        elif npc['dominance'] > 70 and stage.name != "Innocent Beginning":
            availability = "interruptible"  # Dominant NPCs make time for player
        
        # Determine mood
        mood = MoodType.RELAXED
        if stage.name in ["Veil Thinning", "Full Revelation"]:
            mood = MoodType.ASSERTIVE
        elif stage.name == "Creeping Realization":
            mood = MoodType.PLAYFUL
        
        # Determine power tendency
        power_tendency = None
        if npc['dominance'] > 60 and stage.name != "Innocent Beginning":
            if time_phase == DailyRoutinePhase.MORNING_ROUTINE:
                power_tendency = PowerDynamicType.CASUAL_DOMINANCE
            elif time_phase in [DailyRoutinePhase.EVENING_LEISURE, DailyRoutinePhase.BEDTIME_ROUTINE]:
                power_tendency = PowerDynamicType.INTIMATE_COMMAND
            else:
                power_tendency = PowerDynamicType.SUBTLE_CONTROL
        
        return NPCScheduleEntry(
            time_phase=time_phase,
            activity=activity,
            location=location,
            availability=availability,
            mood=mood,
            power_tendency=power_tendency
        )
        
    except Exception as e:
        logger.error(f"Error simulating NPC routine: {e}", exc_info=True)
        return NPCScheduleEntry(
            time_phase=time_phase or DailyRoutinePhase.MORNING_ROUTINE,
            activity="unknown",
            location=LocationType.HOME,
            availability="unknown",
            mood=MoodType.RELAXED
        )

@function_tool
async def get_available_daily_activities(
    ctx: RunContextWrapper,
    include_npcs: bool = True,
    filter_by_vitals: bool = True
) -> List[DailyActivity]:
    """
    Get available activities based on time, location, NPCs, and player state.
    
    Args:
        include_npcs: Include NPC-initiated activities
        filter_by_vitals: Filter based on player vitals
        
    Returns:
        List of available activities
    """
    context = ctx.context
    
    try:
        # Get world context
        world_context = await _get_current_world_context(ctx)
        
        # Get player vitals
        vitals = await get_current_vitals(context.user_id, context.conversation_id)
        
        activities: List[DailyActivity] = []
        
        # Base activities for current phase
        phase_activities = {
            DailyRoutinePhase.MORNING_ROUTINE: [
                {"name": "Have breakfast", "type": "routine", "vital_requirement": None},
                {"name": "Get ready for the day", "type": "routine", "vital_requirement": None},
                {"name": "Check messages", "type": "routine", "vital_requirement": None}
            ],
            DailyRoutinePhase.WORK_HOURS: [
                {"name": "Focus on work", "type": "work", "vital_requirement": {"fatigue_max": 80}},
                {"name": "Take a break", "type": "rest", "vital_requirement": None},
                {"name": "Chat with colleagues", "type": "social", "vital_requirement": None}
            ],
            DailyRoutinePhase.LUNCH_BREAK: [
                {"name": "Eat lunch", "type": "routine", "vital_requirement": None},
                {"name": "Go for a walk", "type": "leisure", "vital_requirement": {"fatigue_max": 70}},
                {"name": "Socialize", "type": "social", "vital_requirement": None}
            ],
            DailyRoutinePhase.EVENING_LEISURE: [
                {"name": "Relax at home", "type": "leisure", "vital_requirement": None},
                {"name": "Go out", "type": "social", "vital_requirement": {"fatigue_max": 60}},
                {"name": "Personal time", "type": "personal", "vital_requirement": None}
            ],
            DailyRoutinePhase.BEDTIME_ROUTINE: [
                {"name": "Prepare for bed", "type": "routine", "vital_requirement": None},
                {"name": "Wind down", "type": "rest", "vital_requirement": None},
                {"name": "Go to sleep", "type": "sleep", "vital_requirement": None}
            ]
        }
        
        base_activities = phase_activities.get(world_context.current_phase, [])
        
        # Filter by vitals if requested
        for activity in base_activities:
            if filter_by_vitals and activity["vital_requirement"]:
                req = activity["vital_requirement"]
                if "fatigue_max" in req and vitals.fatigue > req["fatigue_max"]:
                    continue
                if "hunger_min" in req and vitals.hunger < req["hunger_min"]:
                    continue
            activities.append(DailyActivity(**activity))
        
        # Add vital-critical activities
        if vitals.hunger < 30:
            activities.insert(0, DailyActivity(name="Find food", type="vital", priority="high"))
        if vitals.thirst < 30:
            activities.insert(0, DailyActivity(name="Get water", type="vital", priority="high"))
        if vitals.fatigue > 80:
            activities.insert(0, DailyActivity(name="Rest urgently", type="vital", priority="high"))
        
        # Add NPC-initiated activities if present
        if include_npcs and world_context.present_npcs:
            for npc_id in world_context.present_npcs[:2]:
                # Check NPC's power tendency for this phase
                npc_schedule = await simulate_npc_daily_routine(ctx, npc_id, world_context.current_phase)
                
                if npc_schedule.power_tendency and npc_schedule.availability != "busy":
                    async with get_db_connection_context() as conn:
                        npc_name = await conn.fetchval(
                            "SELECT npc_name FROM NPCStats WHERE npc_id=$1",
                            npc_id
                        )

                        activities.append(
                            DailyActivity(
                                name=f"Spend time with {npc_name}",
                                type="npc_initiated",
                                npc_id=npc_id,
                                power_dynamic=npc_schedule.power_tendency.value
                            )
                        )
        
        return activities
        
    except Exception as e:
        logger.error(f"Error getting available activities: {e}", exc_info=True)
        return [DailyActivity(name="Continue with routine", type="default")]

@function_tool
async def transition_daily_phase(
    ctx: RunContextWrapper,
    skip_to: Optional[DailyRoutinePhase] = None
) -> TransitionResult:
    """
    Transition to the next daily phase with appropriate scene generation.
    
    Args:
        skip_to: Optional phase to skip to
        
    Returns:
        Transition results including new phase and generated scene
    """
    context = ctx.context
    
    try:
        # Get current phase
        world_context = await _get_current_world_context(ctx)
        old_phase = world_context.current_phase
        
        # Determine new phase
        if skip_to:
            new_phase = skip_to
        else:
            # Natural progression
            phase_order = list(DailyRoutinePhase)
            current_idx = phase_order.index(old_phase)
            new_idx = (current_idx + 1) % len(phase_order)
            new_phase = phase_order[new_idx]
        
        # Update time in database (integrate with time_cycle)
        from logic.time_cycle import advance_time_and_update
        await advance_time_and_update(context.user_id, context.conversation_id, 1)
        
        # Update NPC locations for new phase
        async with get_db_connection_context() as conn:
            npcs = await conn.fetch("""
                SELECT npc_id FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """, context.user_id, context.conversation_id)
        
        for npc in npcs:
            schedule = await simulate_npc_daily_routine(ctx, npc['npc_id'], new_phase)
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE NPCStats
                    SET current_location=$1
                    WHERE npc_id=$2
                """, schedule.location.value, npc['npc_id'])
        
        # Generate transition scene
        transition_scene = await generate_daily_life_scene(ctx)
        
        # Check for narrative developments
        if new_phase == DailyRoutinePhase.MORNING_ROUTINE and old_phase == DailyRoutinePhase.SLEEP:
            # New day - check for revelations
            revelation = await check_for_personal_revelations(context.user_id, context.conversation_id)
            if revelation:
                transition_scene.emergent_narrative = revelation.get("inner_monologue")
        
        # Store transition in memory
        memory_manager = await get_memory_manager(context.user_id, context.conversation_id)
        await memory_manager.add_memory(
            content=f"Transitioned from {old_phase.value} to {new_phase.value}",
            memory_type="phase_transition",
            importance=0.3,
            tags=["daily_routine", new_phase.value]
        )
        
        return TransitionResult(
            old_phase=old_phase.value,
            new_phase=new_phase.value,
            transition_scene=transition_scene,
            time_updated=True,
            npcs_relocated=len(npcs)
        )
        
    except Exception as e:
        logger.error(f"Error transitioning phase: {e}", exc_info=True)
        return TransitionResult(error=str(e), time_updated=False)

# ============= INTEGRATION FUNCTIONS =============

async def integrate_with_narrative_progression(
    ctx: RunContextWrapper,
    scene: SliceOfLifeScene
) -> Dict[str, Any]:
    """
    Integrate a scene with the NPC narrative progression system.
    
    Args:
        scene: The scene to process
        
    Returns:
        Integration results including any stage changes
    """
    context = ctx.context
    results = {"stage_changes": [], "revelations": []}
    
    try:
        for npc_id in scene.participants:
            # Check for stage progression based on power dynamics
            if scene.power_dynamics:
                power_moment = next((pd for pd in scene.power_dynamics if pd.npc_id == npc_id), None)
                if power_moment and power_moment.acceptance_increases_dynamic:
                    # Small progression based on acceptance
                    progression = await progress_npc_narrative_stage(
                        context.user_id,
                        context.conversation_id,
                        npc_id,
                        corruption_change=2,
                        dependency_change=3,
                        realization_change=1
                    )
                    
                    if progression.get("stage_changed"):
                        results["stage_changes"].append({
                            "npc_id": npc_id,
                            "new_stage": progression["new_stage"]
                        })
            
            # Check for revelations
            revelation = await check_for_npc_revelation(
                context.user_id,
                context.conversation_id,
                npc_id
            )
            if revelation:
                results["revelations"].append(revelation)
        
        return results
        
    except Exception as e:
        logger.error(f"Error integrating with narrative progression: {e}", exc_info=True)
        return results

# ============= EXPORT TOOL LISTS =============

# Core daily life tools
daily_life_tools = [_ensure_tool(t) for t in [
    generate_daily_life_scene,
    generate_organic_power_moment,
    generate_contextual_interaction,
    generate_player_choices_for_moment,
    detect_emergent_story,
    process_player_response_to_interaction
] if t]

# NPC routine tools
npc_routine_tools = [_ensure_tool(t) for t in [
    simulate_npc_daily_routine,
    get_available_daily_activities,
    transition_daily_phase
] if t]

# All world simulation tools
all_world_simulation_tools = daily_life_tools + npc_routine_tools

# Export the main tool lists
__all__ = [
    'all_world_simulation_tools',
    'daily_life_tools',
    'npc_routine_tools',
    'DailyLifeDirector',
    'AmbientDialogueWriter',
    'PowerDynamicsOrchestrator',
    'EmergentNarrativeDetector',
    'PlayerAgencyManager'
]
