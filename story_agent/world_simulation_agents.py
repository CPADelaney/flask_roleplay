# story_agent/world_simulation_agents.py

"""
Specialized agents for the open-world slice-of-life simulation.
Focus on daily life, relationships, and emergent narratives.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, ModelSettings
from agents.exceptions import AgentsException, ModelBehaviorError

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from story_agent.world_director_agent import (
        WorldState, WorldMood, TimeOfDay, ActivityType, PowerDynamicType
    )

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "gpt-4.1-nano"
FAST_MODEL = "gpt-4.1-nano"

# ----- Specialized Agent Contexts -----

@dataclass
class SliceOfLifeContext:
    """Context for slice-of-life agents"""
    user_id: int
    conversation_id: int
    world_state: Optional[Any] = None  # Changed from WorldState to Any
    current_scene: Optional[Dict[str, Any]] = None
    recent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_interaction(self, interaction: Dict[str, Any]):
        self.recent_interactions.append(interaction)
        if len(self.recent_interactions) > 20:
            self.recent_interactions.pop(0)

# ----- Daily Life Coordinator Agent -----

def create_daily_life_coordinator():
    """Agent that coordinates daily slice-of-life activities"""
    
    instructions = """
    You are the Daily Life Coordinator for an open-world femdom slice-of-life simulation.
    
    Your role is to:
    1. Generate natural daily activities with embedded power dynamics
    2. Coordinate NPC schedules and availability
    3. Create routine activities that build patterns over time
    4. Ensure activities match the time of day and world mood
    
    Daily activities should:
    - Feel mundane on the surface with subtle control elements
    - Build routines that condition behavior
    - Create opportunities for power exchanges
    - Vary based on relationships and world state
    
    Morning activities: Getting ready, breakfast, morning routines
    Daytime activities: Work, errands, social obligations
    Evening activities: Dinner, relaxation, domestic duties
    Night activities: Intimate time, personal care, rest
    
    Power dynamics should be woven naturally into these activities,
    not forced or explicit. Focus on the slice-of-life aspect.
    """
    
    # Lazy import to avoid circular dependency
    try:
        from story_agent.tools import daily_life_tools
        tools = daily_life_tools
    except ImportError:
        logger.warning("Could not import daily_life_tools, using empty tools list")
        tools = []
    
    agent = Agent(
        name="Daily Life Coordinator",
        instructions=instructions,
        tools=tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(temperature=0.6, max_tokens=1536)
    )
    
    return agent

# ----- Relationship Dynamics Agent -----

def create_relationship_dynamics_agent():
    """Agent focused on managing relationship dynamics in daily life"""
    
    instructions = """
    You are the Relationship Dynamics Agent for a slice-of-life simulation.
    
    Your focus is on:
    1. Natural relationship evolution through daily interactions
    2. Subtle power dynamics in everyday situations
    3. Building dependency and intimacy over time
    4. Creating emergent relationship patterns
    
    Relationship dynamics should emerge from:
    - Repeated daily interactions
    - Small decisions and preferences
    - Gradual boundary shifts
    - Established routines and expectations
    
    Power dynamics to incorporate:
    - Who makes daily decisions (food, clothing, schedule)
    - Subtle permissions and approvals
    - Established patterns of deference
    - Caretaking that creates dependency
    
    Avoid dramatic confrontations or explicit dominance.
    Focus on the slow, natural evolution of relationships.
    """
    
    # Lazy import to avoid circular dependency
    try:
        from story_agent.tools import npc_routine_tools
        tools = npc_routine_tools
    except ImportError:
        logger.warning("Could not import npc_routine_tools, using empty tools list")
        tools = []
    
    agent = Agent(
        name="Relationship Dynamics",
        instructions=instructions,
        tools=tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(temperature=0.5, max_tokens=1536)
    )
    
    return agent

# ----- Ambient World Agent -----

def create_ambient_world_agent():
    """Agent that manages ambient world details and atmosphere"""
    
    instructions = """
    You are the Ambient World Agent, creating atmosphere and background details.
    
    Your responsibilities:
    1. Generate ambient details that make the world feel alive
    2. Create background NPC activities
    3. Manage environmental factors and mood
    4. Add sensory details to scenes
    
    Focus on:
    - Time of day affecting atmosphere
    - Weather and environmental details
    - Background NPC conversations and activities
    - Subtle mood shifts throughout the day
    - Sensory details (sounds, smells, textures)
    
    Create a living, breathing world where things happen
    independently of the player's direct involvement.
    """
    
    agent = Agent(
        name="Ambient World",
        instructions=instructions,
        tools=[],  # Uses narrative generation tools
        model=FAST_MODEL,
        model_settings=ModelSettings(temperature=0.7, max_tokens=1024)
    )
    
    return agent

# ----- Pattern Recognition Agent -----

def create_pattern_recognition_agent():
    """Agent that detects emergent patterns and narratives"""
    
    instructions = """
    You are the Pattern Recognition Agent for emergent narrative detection.
    
    Your role is to:
    1. Identify emerging patterns in player behavior
    2. Detect relationship patterns across multiple NPCs
    3. Recognize routine establishment and conditioning
    4. Spot narrative threads emerging from interactions
    
    Look for patterns like:
    - Escalating submission across contexts
    - Coordinated NPC behaviors
    - Established daily routines becoming rituals
    - Gradual boundary erosion
    - Dependency formation
    
    When you detect a pattern:
    - Note the key indicators
    - Track participating NPCs
    - Measure pattern strength
    - Suggest narrative developments
    
    Focus on emergent narratives, not predetermined plots.
    """
    
    agent = Agent(
        name="Pattern Recognition",
        instructions=instructions,
        tools=[],
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(temperature=0.3, max_tokens=1536)
    )
    
    return agent

# ----- Dialogue Specialist Agent -----

def create_dialogue_specialist():
    """Agent specialized in natural dialogue with subtle power dynamics"""
    
    instructions = """
    You are the Dialogue Specialist for slice-of-life interactions.
    
    Generate natural, contextual dialogue that:
    1. Sounds like real conversation, not theatrical
    2. Embeds power dynamics through subtext
    3. Reflects relationship stage and NPC personality
    4. Varies based on time, mood, and situation
    
    Dialogue progression:
    - Early: Friendly, no obvious control
    - Middle: Gentle suggestions, subtle steering
    - Late: Casual commands, assumed compliance
    
    Power dynamics in dialogue:
    - Decisions phrased as statements
    - Preferences expressed as facts
    - Questions that aren't really questions
    - Gentle corrections and guidance
    
    Keep dialogue concise (1-3 sentences) and natural.
    """
    
    # Lazy import
    try:
        from story_agent.specialized_agents import create_dialogue_generator
        return create_dialogue_generator()
    except ImportError:
        logger.warning("Could not import specialized_agents, creating basic dialogue agent")
        return Agent(
            name="Dialogue Specialist",
            instructions=instructions,
            tools=[],
            model=DEFAULT_MODEL,
            model_settings=ModelSettings(temperature=0.6, max_tokens=1024)
        )

# ----- Activity Generator Agent -----

def create_activity_generator():
    """Agent that generates contextual activities"""
    
    instructions = """
    You are the Activity Generator for daily life simulation.
    
    Generate activities that:
    1. Match the time of day and location
    2. Involve available NPCs naturally
    3. Include subtle power dynamics
    4. Build on established routines
    
    Activity types:
    - Routine: Daily necessities (meals, hygiene, chores)
    - Work: Professional or productive tasks
    - Social: Interactions with others
    - Leisure: Relaxation and entertainment
    - Intimate: Close personal interactions
    
    Each activity should:
    - Have clear start and end conditions
    - Allow for player choice within constraints
    - Create opportunities for power dynamics
    - Feel like part of daily life
    """
    
    agent = Agent(
        name="Activity Generator",
        instructions=instructions,
        tools=[],
        model=FAST_MODEL,
        model_settings=ModelSettings(temperature=0.6, max_tokens=1024)
    )
    
    return agent

# ----- Helper Functions -----

async def coordinate_slice_of_life_scene(
    context: SliceOfLifeContext,
    focus_type: str = "routine"
) -> Dict[str, Any]:
    """Coordinate multiple agents to create a slice-of-life scene"""
    
    # Get coordinators
    daily_coordinator = create_daily_life_coordinator()
    relationship_agent = create_relationship_dynamics_agent()
    ambient_agent = create_ambient_world_agent()
    
    # Get world state values with safe access
    time_value = "morning"
    mood_value = "relaxed"
    
    if context.world_state:
        # Safely access attributes if they exist
        if hasattr(context.world_state, 'current_time'):
            time_obj = context.world_state.current_time
            if hasattr(time_obj, 'value'):
                time_value = time_obj.value
            elif isinstance(time_obj, str):
                time_value = time_obj
        
        if hasattr(context.world_state, 'world_mood'):
            mood_obj = context.world_state.world_mood
            if hasattr(mood_obj, 'value'):
                mood_value = mood_obj.value
            elif isinstance(mood_obj, str):
                mood_value = mood_obj
    
    # Generate base scene with daily coordinator
    scene_prompt = f"""
    Create a {focus_type} slice-of-life scene for:
    Time: {time_value}
    Mood: {mood_value}
    
    Focus on natural daily activities with subtle power dynamics.
    """
    
    scene_result = await Runner.run(daily_coordinator, scene_prompt, context=context)
    
    # Initialize results
    relationship_result = None
    ambient_result = None
    
    # Add relationship dynamics
    if context.world_state and hasattr(context.world_state, 'active_npcs'):
        active_npcs = context.world_state.active_npcs
        if active_npcs:
            relationship_prompt = f"""
            Add relationship dynamics to this scene.
            NPCs present: {len(active_npcs)}
            
            Include subtle power dynamics appropriate to relationships.
            """
            
            relationship_result = await Runner.run(
                relationship_agent, 
                relationship_prompt, 
                context=context
            )
    
    # Add ambient details
    ambient_prompt = "Add atmospheric and sensory details to make the scene immersive."
    ambient_result = await Runner.run(ambient_agent, ambient_prompt, context=context)
    
    return {
        "scene": scene_result.final_output if scene_result else {},
        "relationships": relationship_result.final_output if relationship_result else {},
        "atmosphere": ambient_result.final_output if ambient_result else {}
    }

async def detect_emergent_patterns(
    context: SliceOfLifeContext
) -> List[Dict[str, Any]]:
    """Use pattern recognition agent to detect emergent narratives"""
    
    pattern_agent = create_pattern_recognition_agent()
    
    prompt = f"""
    Analyze recent interactions for emergent patterns:
    {json.dumps(context.recent_interactions[-10:], indent=2)}
    
    Identify any emerging narratives or behavioral patterns.
    """
    
    result = await Runner.run(pattern_agent, prompt, context=context)
    
    if result and result.final_output:
        try:
            # Parse patterns from response
            patterns = json.loads(result.final_output) if isinstance(result.final_output, str) else result.final_output
            return patterns if isinstance(patterns, list) else [patterns]
        except:
            return []
    
    return []

# ----- Agent Initialization -----

def initialize_world_simulation_agents():
    """Initialize all world simulation agents"""
    return {
        "daily_coordinator": create_daily_life_coordinator(),
        "relationship_dynamics": create_relationship_dynamics_agent(),
        "ambient_world": create_ambient_world_agent(),
        "pattern_recognition": create_pattern_recognition_agent(),
        "dialogue_specialist": create_dialogue_specialist(),
        "activity_generator": create_activity_generator()
    }

# Export
__all__ = [
    'SliceOfLifeContext',
    'initialize_world_simulation_agents',
    'coordinate_slice_of_life_scene',
    'detect_emergent_patterns',
    'create_daily_life_coordinator',
    'create_relationship_dynamics_agent',
    'create_ambient_world_agent',
    'create_pattern_recognition_agent',
    'create_dialogue_specialist',
    'create_activity_generator'
]
