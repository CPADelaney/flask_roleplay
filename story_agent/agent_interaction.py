# story_agent/agent_interaction.py

"""
Agent Interaction Module for Open-World Slice-of-Life Simulation.
Coordinates specialized agents for emergent gameplay and dynamic world simulation.
"""

import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from agents import Agent, Runner, trace, handoff
from agents.exceptions import AgentsException, ModelBehaviorError

# Import world simulation components
from story_agent.world_director_agent import (
    WorldDirector, WorldState, SliceOfLifeEvent,
    PowerExchange, WorldMood, TimeOfDay, 
    ActivityType, PowerDynamicType
)

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
# Agent Coordination Context
# ===============================================================================

@dataclass
class AgentCoordinationContext:
    """Context for coordinating multiple agents in slice-of-life simulation"""
    user_id: int
    conversation_id: int
    
    # World state tracking
    world_director: Optional[WorldDirector] = None
    current_world_state: Optional[WorldState] = None
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
        """Initialize all components"""
        # Initialize world director
        self.world_director = WorldDirector(self.user_id, self.conversation_id)
        await self.world_director.initialize()
        
        # Get current world state
        self.current_world_state = await self.world_director.get_world_state()
        
        # Initialize simulation agents
        self.simulation_agents = initialize_world_simulation_agents()
        
        # Initialize context components
        self.context_service = await get_context_service(self.user_id, self.conversation_id)
        self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)

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
            available_npcs = [
                npc.npc_id for npc in context.current_world_state.active_npcs
                if npc.availability in ["available", "eager", "commanding"]
            ]
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
                    prompt = f"""
                    Generate natural dialogue for {npc['npc_name']} in a {scene_focus} scene.
                    Dominance: {npc['dominance']}, Closeness: {npc['closeness']}
                    Time: {context.current_world_state.current_time.value}
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
        if context.current_world_state and context.current_world_state.world_tension.power_tension > 0.3:
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
            tags=["scene", scene_focus, f"time_{context.current_world_state.current_time.value}"]
        )
        
        execution_time = time.time() - start_time
        context.performance_monitor.stop_timer(timer_id)
        
        return {
            "scene": scene,
            "dialogues": dialogues,
            "power_dynamics": power_dynamics,
            "emergent_patterns": patterns,
            "world_mood": context.current_world_state.world_mood.value,
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
        
        prompt = f"""
        Generate ambient details for:
        Time: {context.current_world_state.current_time.value}
        Mood: {context.current_world_state.world_mood.value}
        Location: current
        Active NPCs: {len(context.current_world_state.active_npcs)}
        
        Include sensory details, background activity, and atmosphere.
        """
        
        result = await Runner.run(ambient_agent, prompt)
        
        return {
            "ambient_details": result.final_output if result else "",
            "world_mood": context.current_world_state.world_mood.value,
            "time_of_day": context.current_world_state.current_time.value
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
        
        # Format handoff message
        handoff_message = f"""
        Handoff from {from_agent}:
        
        Context: {json.dumps(handoff_data, indent=2)}
        World State: Time={context.current_world_state.current_time.value}, 
                    Mood={context.current_world_state.world_mood.value}
        
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
    
    if context.current_world_state:
        # Adjust power tension
        if response_type == "submit":
            context.current_world_state.world_tension.power_tension = min(
                1.0,
                context.current_world_state.world_tension.power_tension + (0.1 * exchange.intensity)
            )
        elif response_type == "resist":
            context.current_world_state.world_tension.conflict_tension = min(
                1.0,
                context.current_world_state.world_tension.conflict_tension + 0.05
            )
        
        # Adjust sexual tension for intimate exchanges
        if exchange.exchange_type in [PowerDynamicType.INTIMATE_COMMAND, PowerDynamicType.RITUAL_SUBMISSION]:
            context.current_world_state.world_tension.sexual_tension = min(
                1.0,
                context.current_world_state.world_tension.sexual_tension + (0.05 * exchange.intensity)
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
        if context.current_world_state:
            dominant_tension, level = context.current_world_state.world_tension.get_dominant_tension()
            results["tensions"] = {
                "dominant": dominant_tension,
                "level": level,
                "all_tensions": {
                    "social": context.current_world_state.world_tension.social_tension,
                    "sexual": context.current_world_state.world_tension.sexual_tension,
                    "power": context.current_world_state.world_tension.power_tension,
                    "mystery": context.current_world_state.world_tension.mystery_tension,
                    "conflict": context.current_world_state.world_tension.conflict_tension
                }
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
        await context.world_director.simulate_world_tick()
        
        # Generate NPC autonomous actions
        if context.current_world_state:
            for npc in context.current_world_state.active_npcs[:3]:  # Limit to 3 NPCs
                activity = await context.simulation_agents["activity_generator"].run(
                    f"Generate autonomous activity for NPC {npc.npc_name}"
                )
                if activity:
                    events.append({
                        "hour": hour,
                        "npc": npc.npc_name,
                        "activity": activity
                    })
    
    return {
        "simulated_hours": hours,
        "events": events,
        "final_world_state": context.current_world_state.model_dump() if context.current_world_state else None
    }
