# story_agent/agent_interaction.py

"""
Agent Interaction Module for Open-World Slice-of-Life Simulation.
Coordinates specialized agents for emergent gameplay and dynamic world simulation.
"""

import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from agents import Agent, Runner, trace, handoff
from agents.exceptions import AgentsException, ModelBehaviorError

# Lazy loading to avoid circular imports
if TYPE_CHECKING:
    from story_agent.world_director_agent import (
        CompleteWorldDirector, CompleteWorldState, WorldMood, ActivityType
    )

# Import world simulation components
from story_agent.world_simulation_agents import (
    initialize_world_simulation_agents,
    SliceOfLifeContext,
    coordinate_slice_of_life_scene,
    detect_emergent_patterns
)

# Context system integration
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager
from context.vector_service import get_vector_service
from context.context_performance import PerformanceMonitor, track_performance

# Database connection
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# Local Enums (to avoid circular import)
# ===============================================================================

class PowerDynamicType(Enum):
    """Types of power dynamics in interactions"""
    CASUAL_DOMINANCE = "casual_dominance"
    INTIMATE_COMMAND = "intimate_command"
    SOCIAL_HIERARCHY = "social_hierarchy"
    SUBTLE_CONTROL = "subtle_control"
    RITUAL_SUBMISSION = "ritual_submission"

class TimeOfDay(Enum):
    """Time of day for scene context"""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    LATE_NIGHT = "late_night"

# ===============================================================================
# Local Data Classes
# ===============================================================================

@dataclass
class SliceOfLifeEvent:
    """Represents a slice-of-life event"""
    event_type: str
    title: str
    description: str
    involved_npcs: List[int] = field(default_factory=list)
    location: Optional[str] = None
    mood: Optional[str] = None
    choices: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PowerExchange:
    """Represents a power exchange interaction"""
    exchange_type: PowerDynamicType
    initiator_npc_id: int
    intensity: float
    is_public: bool = False
    description: Optional[str] = None

# ===============================================================================
# Agent Coordination Context with Lazy Loading
# ===============================================================================

@dataclass
class AgentCoordinationContext:
    """Context for coordinating multiple agents in slice-of-life simulation"""
    user_id: int
    conversation_id: int
    
    # World state tracking - lazy loaded
    world_director: Optional[Any] = None  # Will be CompleteWorldDirector
    current_world_state: Optional[Any] = None  # Will be CompleteWorldState
    active_scene: Optional[SliceOfLifeEvent] = None
    
    # Agent references
    simulation_agents: Dict[str, Agent] = field(default_factory=dict)
    
    # Interaction tracking
    recent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    emergent_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context components
    context_service: Optional[Any] = None
    memory_manager: Optional[Any] = None
    performance_monitor: Optional[Any] = None
    
    async def initialize(self):
        """Initialize all components with lazy loading"""
        # Lazy import to avoid circular dependency
        from story_agent.world_director_agent import CompleteWorldDirector
        
        # Initialize world director
        self.world_director = CompleteWorldDirector(self.user_id, self.conversation_id)
        await self.world_director.initialize()
        
        # Get current world state
        if self.world_director.context:
            self.current_world_state = self.world_director.context.current_world_state
        
        # Initialize simulation agents
        self.simulation_agents = initialize_world_simulation_agents()
        
        # Initialize context components
        self.context_service = await get_context_service(self.user_id, self.conversation_id)
        self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)

# ===============================================================================
# Helper function to get WorldMood enum
# ===============================================================================

def _get_world_mood_enum():
    """Lazy load WorldMood enum"""
    from story_agent.world_director_agent import WorldMood
    return WorldMood

# ===============================================================================
# Scene Orchestration Functions
# ===============================================================================

@track_performance("orchestrate_daily_scene")
async def orchestrate_daily_scene(
    user_id: int,
    conversation_id: int,
    scene_focus: str = "routine",
    involved_npcs: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Orchestrate a complete daily life scene with multiple agents.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        scene_focus: Type of scene (routine, social, intimate, etc.)
        involved_npcs: Optional list of NPCs to involve
        
    Returns:
        Complete scene with dialogue, atmosphere, and interactions
    """
    start_time = time.time()
    
    # Initialize coordination context
    context = AgentCoordinationContext(user_id, conversation_id)
    await context.initialize()
    
    timer_id = context.performance_monitor.start_timer("daily_scene_orchestration")
    
    try:
        # Create slice-of-life context
        sol_context = SliceOfLifeContext(
            user_id=user_id,
            conversation_id=conversation_id,
            world_state=context.current_world_state
        )
        
        # Generate base scene
        scene = await coordinate_slice_of_life_scene(sol_context, scene_focus)
        
        # Select NPCs if not specified
        if not involved_npcs and context.current_world_state:
            available_npcs = []
            if hasattr(context.current_world_state, 'active_npcs'):
                for npc in context.current_world_state.active_npcs:
                    # Handle both dict and object representations
                    if isinstance(npc, dict):
                        npc_id = npc.get('npc_id')
                    else:
                        npc_id = getattr(npc, 'npc_id', None)
                    
                    if npc_id:
                        available_npcs.append(npc_id)
            
            involved_npcs = available_npcs[:2]  # Limit to 2 for focused interaction
        
        # Generate dialogue for each NPC
        dialogues = []
        if involved_npcs:
            dialogue_agent = context.simulation_agents.get("dialogue_specialist")
            
            for npc_id in involved_npcs:
                # Get NPC data
                async with get_db_connection_context() as conn:
                    npc = await conn.fetchrow("""
                        SELECT npc_name, dominance, closeness
                        FROM NPCStats WHERE npc_id=$1
                    """, npc_id)
                
                if npc and dialogue_agent:
                    # Get current time safely
                    time_value = "afternoon"  # default
                    if context.current_world_state:
                        if hasattr(context.current_world_state, 'current_time'):
                            current_time = context.current_world_state.current_time
                            if hasattr(current_time, 'time_of_day'):
                                time_value = str(current_time.time_of_day.value)
                    
                    prompt = f"""
                    Generate natural dialogue for {npc['npc_name']} in a {scene_focus} scene.
                    Dominance: {npc['dominance']}, Closeness: {npc['closeness']}
                    Time: {time_value}
                    Keep it slice-of-life, 1-2 sentences.
                    """
                    
                    result = await Runner.run(dialogue_agent, prompt)
                    if result:
                        dialogues.append({
                            "npc_id": npc_id,
                            "npc_name": npc['npc_name'],
                            "dialogue": result.final_output
                        })
        
        # Check for power dynamics opportunities
        power_dynamics = []
        if context.current_world_state:
            # Check tension safely
            tension_value = 0.0
            if hasattr(context.current_world_state, 'tension_factors'):
                tension_value = context.current_world_state.tension_factors.get('power', 0.0)
            
            if tension_value > 0.3:
                for npc_id in involved_npcs[:1]:  # One main dynamic per scene
                    power_exchange = await generate_contextual_power_exchange(
                        context, npc_id, scene_focus
                    )
                    if power_exchange:
                        power_dynamics.append(power_exchange)
        
        # Detect emergent patterns
        sol_context.record_interaction({
            "type": "scene",
            "focus": scene_focus,
            "npcs": involved_npcs,
            "time": datetime.now().isoformat()
        })
        
        patterns = await detect_emergent_patterns(sol_context)
        
        # Store scene in memory
        scene_description = f"{scene_focus} scene with {len(involved_npcs)} NPCs"
        await context.memory_manager.add_memory(
            content=scene_description,
            memory_type="daily_scene",
            importance=0.5,
            tags=["scene", scene_focus]
        )
        
        execution_time = time.time() - start_time
        context.performance_monitor.stop_timer(timer_id)
        
        # Get world mood safely
        world_mood_value = "relaxed"
        if context.current_world_state and hasattr(context.current_world_state, 'world_mood'):
            world_mood = context.current_world_state.world_mood
            if hasattr(world_mood, 'value'):
                world_mood_value = world_mood.value
        
        return {
            "scene": scene,
            "dialogues": dialogues,
            "power_dynamics": power_dynamics,
            "emergent_patterns": patterns,
            "world_mood": world_mood_value,
            "execution_time": execution_time
        }
        
    except Exception as e:
        logger.error(f"Error orchestrating daily scene: {e}", exc_info=True)
        if 'timer_id' in locals():
            context.performance_monitor.stop_timer(timer_id)
        return {"error": str(e), "execution_time": time.time() - start_time}

@track_performance("process_power_exchange")
async def process_power_exchange_with_agents(
    user_id: int,
    conversation_id: int,
    exchange: PowerExchange,
    player_response: str
) -> Dict[str, Any]:
    """
    Process a power exchange using multiple agents to determine outcomes.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        exchange: The power exchange to process
        player_response: How the player responded
        
    Returns:
        Outcomes and narrative consequences
    """
    context = AgentCoordinationContext(user_id, conversation_id)
    await context.initialize()
    
    try:
        # Analyze response type
        response_type = analyze_response_type(player_response)
        
        # Get relationship dynamics agent
        relationship_agent = context.simulation_agents.get("relationship_dynamics")
        
        # Process relationship impact
        relationship_prompt = f"""
        Player responded to {exchange.exchange_type.value} with: {player_response}
        Response type: {response_type}
        Intensity: {exchange.intensity}
        
        Determine relationship impacts and progression.
        """
        
        relationship_result = None
        if relationship_agent:
            relationship_result = await Runner.run(relationship_agent, relationship_prompt)
        
        # Update relationship in database
        relationship_impacts = await update_relationship_from_exchange(
            context, exchange, response_type
        )
        
        # Generate narrative outcome
        pattern_agent = context.simulation_agents.get("pattern_recognition")
        narrative_outcome = "The moment passes."
        
        if pattern_agent:
            pattern_prompt = f"""
            Analyze this power exchange for narrative significance:
            Type: {exchange.exchange_type.value}
            Response: {response_type}
            
            What patterns or narrative threads emerge?
            """
            
            pattern_result = await Runner.run(pattern_agent, pattern_prompt)
            if pattern_result:
                narrative_outcome = pattern_result.final_output
        
        # Adjust world tensions
        await adjust_world_tensions_from_exchange(context, exchange, response_type)
        
        # Store in memory
        memory_content = f"Power exchange ({exchange.exchange_type.value}): {response_type} response"
        await context.memory_manager.add_memory(
            content=memory_content,
            memory_type="power_exchange",
            importance=0.6 + (exchange.intensity * 0.2),
            tags=["power_exchange", exchange.exchange_type.value, response_type],
            metadata={
                "npc_id": exchange.initiator_npc_id,
                "intensity": exchange.intensity,
                "public": exchange.is_public
            }
        )
        
        return {
            "response_type": response_type,
            "relationship_impacts": relationship_impacts,
            "narrative_outcome": narrative_outcome,
            "world_tension_changes": {
                "power": 0.1 if response_type == "submit" else -0.05,
                "sexual": 0.05 if exchange.intensity > 0.7 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing power exchange: {e}", exc_info=True)
        return {"error": str(e)}

@track_performance("generate_ambient_world")
async def generate_ambient_world_details(
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    """
    Generate ambient world details using specialized agents.
    
    Returns:
        Ambient details including background NPCs, atmosphere, etc.
    """
    context = AgentCoordinationContext(user_id, conversation_id)
    await context.initialize()
    
    try:
        ambient_agent = context.simulation_agents.get("ambient_world")
        
        if not ambient_agent:
            return {"error": "Ambient agent not available"}
        
        # Get world state values safely
        time_value = "afternoon"
        mood_value = "relaxed"
        active_npc_count = 0
        
        if context.current_world_state:
            if hasattr(context.current_world_state, 'current_time'):
                current_time = context.current_world_state.current_time
                if hasattr(current_time, 'time_of_day'):
                    time_value = str(current_time.time_of_day.value)
            
            if hasattr(context.current_world_state, 'world_mood'):
                world_mood = context.current_world_state.world_mood
                if hasattr(world_mood, 'value'):
                    mood_value = world_mood.value
            
            if hasattr(context.current_world_state, 'active_npcs'):
                active_npc_count = len(context.current_world_state.active_npcs)
        
        prompt = f"""
        Generate ambient details for:
        Time: {time_value}
        Mood: {mood_value}
        Location: current
        Active NPCs: {active_npc_count}
        
        Include sensory details, background activity, and atmosphere.
        """
        
        result = await Runner.run(ambient_agent, prompt)
        
        return {
            "ambient_details": result.final_output if result else "",
            "world_mood": mood_value,
            "time_of_day": time_value
        }
        
    except Exception as e:
        logger.error(f"Error generating ambient details: {e}", exc_info=True)
        return {"error": str(e)}

# ===============================================================================
# Inter-Agent Communication
# ===============================================================================

@track_performance("agent_handoff")
async def coordinate_agent_handoff(
    user_id: int,
    conversation_id: int,
    from_agent: str,
    to_agent: str,
    handoff_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Coordinate handoff between specialized agents.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        from_agent: Source agent name
        to_agent: Target agent name
        handoff_data: Data to pass between agents
        
    Returns:
        Result of the handoff
    """
    context = AgentCoordinationContext(user_id, conversation_id)
    await context.initialize()
    
    try:
        source = context.simulation_agents.get(from_agent)
        target = context.simulation_agents.get(to_agent)
        
        if not source or not target:
            return {"error": f"Agent not found: {from_agent if not source else to_agent}"}
        
        # Get world state values safely
        time_value = "afternoon"
        mood_value = "relaxed"
        
        if context.current_world_state:
            if hasattr(context.current_world_state, 'current_time'):
                current_time = context.current_world_state.current_time
                if hasattr(current_time, 'time_of_day'):
                    time_value = str(current_time.time_of_day.value)
            
            if hasattr(context.current_world_state, 'world_mood'):
                world_mood = context.current_world_state.world_mood
                if hasattr(world_mood, 'value'):
                    mood_value = world_mood.value
        
        # Format handoff message
        handoff_message = f"""
        Handoff from {from_agent}:
        
        Context: {json.dumps(handoff_data, indent=2)}
        World State: Time={time_value}, 
                    Mood={mood_value}
        
        Please continue with your specialized processing.
        """
        
        with trace(workflow_name="AgentHandoff", group_id=f"user_{user_id}"):
            result = await Runner.run(target, handoff_message)
        
        return {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "result": result.final_output if result else None,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in agent handoff: {e}", exc_info=True)
        return {"error": str(e), "success": False}

# ===============================================================================
# Helper Functions
# ===============================================================================

def analyze_response_type(response: str) -> str:
    """Analyze player response to categorize it"""
    response_lower = response.lower()
    
    if any(word in response_lower for word in ["yes", "okay", "sure", "of course", "gladly"]):
        return "submit"
    elif any(word in response_lower for word in ["no", "don't", "won't", "refuse", "never"]):
        return "resist"
    elif any(word in response_lower for word in ["maybe", "later", "busy", "hmm"]):
        return "deflect"
    elif any(word in response_lower for word in ["why", "what", "how", "explain"]):
        return "question"
    else:
        return "neutral"

async def generate_contextual_power_exchange(
    context: AgentCoordinationContext,
    npc_id: int,
    scene_focus: str
) -> Optional[Dict[str, Any]]:
    """Generate a contextual power exchange for the scene"""
    
    # Get NPC data
    async with get_db_connection_context() as conn:
        npc = await conn.fetchrow("""
            SELECT npc_name, dominance, closeness, intensity
            FROM NPCStats WHERE npc_id=$1
        """, npc_id)
    
    if not npc:
        return None
    
    # Determine exchange type based on scene and NPC
    if scene_focus == "intimate" and npc['closeness'] > 60:
        exchange_type = PowerDynamicType.INTIMATE_COMMAND
    elif scene_focus == "routine" and npc['dominance'] > 60:
        exchange_type = PowerDynamicType.CASUAL_DOMINANCE
    elif scene_focus == "social":
        exchange_type = PowerDynamicType.SOCIAL_HIERARCHY
    else:
        exchange_type = PowerDynamicType.SUBTLE_CONTROL
    
    return {
        "type": exchange_type.value,
        "npc_id": npc_id,
        "npc_name": npc['npc_name'],
        "intensity": min(1.0, npc['intensity'] / 100.0),
        "description": f"{npc['npc_name']} subtly asserts control"
    }

async def update_relationship_from_exchange(
    context: AgentCoordinationContext,
    exchange: PowerExchange,
    response_type: str
) -> Dict[str, float]:
    """Update relationship based on power exchange outcome"""
    
    impacts = {}
    
    # Determine impacts based on response
    if response_type == "submit":
        impacts = {
            "submission": 0.02 * exchange.intensity,
            "trust": 0.01,
            "dependency": 0.015 * exchange.intensity
        }
    elif response_type == "resist":
        impacts = {
            "submission": -0.01,
            "tension": 0.02,
            "conflict": 0.01
        }
    elif response_type == "deflect":
        impacts = {
            "tension": 0.01,
            "mystery": 0.01
        }
    
    # Apply impacts (would integrate with relationship system)
    # This is a placeholder for the actual relationship update
    
    return impacts

async def adjust_world_tensions_from_exchange(
    context: AgentCoordinationContext,
    exchange: PowerExchange,
    response_type: str
):
    """Adjust world tensions based on power exchange"""
    
    if context.current_world_state and hasattr(context.current_world_state, 'tension_factors'):
        # Adjust power tension
        if response_type == "submit":
            if 'power' in context.current_world_state.tension_factors:
                context.current_world_state.tension_factors['power'] = min(
                    1.0,
                    context.current_world_state.tension_factors['power'] + (0.1 * exchange.intensity)
                )
        elif response_type == "resist":
            if 'conflict' in context.current_world_state.tension_factors:
                context.current_world_state.tension_factors['conflict'] = min(
                    1.0,
                    context.current_world_state.tension_factors.get('conflict', 0) + 0.05
                )
        
        # Adjust sexual tension for intimate exchanges
        if exchange.exchange_type in [PowerDynamicType.INTIMATE_COMMAND, PowerDynamicType.RITUAL_SUBMISSION]:
            if 'sexual' in context.current_world_state.tension_factors:
                context.current_world_state.tension_factors['sexual'] = min(
                    1.0,
                    context.current_world_state.tension_factors.get('sexual', 0) + (0.05 * exchange.intensity)
                )

# ===============================================================================
# Public API Functions
# ===============================================================================

async def get_agent_analysis(
    user_id: int,
    conversation_id: int,
    analysis_type: str = "patterns"
) -> Dict[str, Any]:
    """
    Get agent analysis of current game state.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        analysis_type: Type of analysis (patterns, relationships, tensions)
        
    Returns:
        Analysis results from specialized agents
    """
    context = AgentCoordinationContext(user_id, conversation_id)
    await context.initialize()
    
    results = {}
    
    if analysis_type == "patterns":
        # Use pattern recognition agent
        agent = context.simulation_agents.get("pattern_recognition")
        if agent:
            prompt = "Analyze current patterns in player behavior and relationships"
            result = await Runner.run(agent, prompt)
            results["patterns"] = result.final_output if result else "No patterns detected"
    
    elif analysis_type == "relationships":
        # Use relationship dynamics agent
        agent = context.simulation_agents.get("relationship_dynamics")
        if agent:
            prompt = "Analyze current relationship dynamics and power structures"
            result = await Runner.run(agent, prompt)
            results["relationships"] = result.final_output if result else "No analysis available"
    
    elif analysis_type == "tensions":
        # Analyze world tensions
        if context.current_world_state and hasattr(context.current_world_state, 'tension_factors'):
            tensions = context.current_world_state.tension_factors
            
            # Find dominant tension
            dominant_tension = "none"
            max_level = 0
            for tension_type, level in tensions.items():
                if level > max_level:
                    max_level = level
                    dominant_tension = tension_type
            
            results["tensions"] = {
                "dominant": dominant_tension,
                "level": max_level,
                "all_tensions": tensions
            }
    
    return results

async def simulate_autonomous_world(
    user_id: int,
    conversation_id: int,
    hours: int = 1
) -> Dict[str, Any]:
    """
    Simulate autonomous world progression without player input.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        hours: Hours to simulate
        
    Returns:
        Summary of autonomous world events
    """
    context = AgentCoordinationContext(user_id, conversation_id)
    await context.initialize()
    
    events = []
    
    for hour in range(hours):
        # Advance time
        if context.world_director:
            await context.world_director.advance_time(1)
        
        # Generate NPC autonomous actions
        if context.current_world_state and hasattr(context.current_world_state, 'active_npcs'):
            for npc in context.current_world_state.active_npcs[:3]:  # Limit to 3 NPCs
                activity_agent = context.simulation_agents.get("activity_generator")
                if activity_agent:
                    npc_name = npc.get('npc_name', 'Unknown') if isinstance(npc, dict) else getattr(npc, 'npc_name', 'Unknown')
                    activity = await Runner.run(
                        activity_agent,
                        f"Generate autonomous activity for NPC {npc_name}"
                    )
                    if activity:
                        events.append({
                            "hour": hour,
                            "npc": npc_name,
                            "activity": activity.final_output if activity else "Unknown activity"
                        })
    
    # Get final world state safely
    final_state = None
    if context.current_world_state:
        if hasattr(context.current_world_state, 'model_dump'):
            final_state = context.current_world_state.model_dump()
        elif hasattr(context.current_world_state, '__dict__'):
            final_state = context.current_world_state.__dict__
    
    return {
        "simulated_hours": hours,
        "events": events,
        "final_world_state": final_state
    }
