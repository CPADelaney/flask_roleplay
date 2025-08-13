# story_agent/specialized_agents.py

"""
Specialized Agents for Open-World Slice-of-Life Simulation
Coordinates emergent gameplay, daily routines, and relationship dynamics
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from enum import Enum

from agents import Agent, Runner, function_tool, trace, handoff, ModelSettings
from agents.exceptions import AgentsException, ModelBehaviorError

# Nyx governance integration (maintained for continuity)
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

logger = logging.getLogger(__name__)

# ----- Configuration -----
MAX_RETRIES = 3
RETRY_INTERVAL = 1.0
DEFAULT_MODEL = "gpt-5-nano"
FAST_MODEL = "gpt-5-nano"

# ----- Agent Types for Open World -----
class OpenWorldAgentType(Enum):
    """Agent types for slice-of-life simulation"""
    DAILY_LIFE_COORDINATOR = "daily_life_coordinator"
    RELATIONSHIP_ORCHESTRATOR = "relationship_orchestrator"
    EMERGENT_NARRATIVE_DETECTOR = "emergent_narrative_detector"
    AMBIENT_WORLD_MANAGER = "ambient_world_manager"
    MEMORY_PATTERN_ANALYZER = "memory_pattern_analyzer"
    NPC_BEHAVIOR_DIRECTOR = "npc_behavior_director"
    POWER_DYNAMICS_WEAVER = "power_dynamics_weaver"

# ----- Enhanced Agent Context -----
@dataclass
class SliceOfLifeAgentContext:
    """Context for slice-of-life specialized agents"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    
    # System connections (lazy loaded)
    _world_director: Optional[Any] = None
    _npc_handler: Optional[Any] = None
    _relationship_manager: Optional[Any] = None
    _memory_manager: Optional[Any] = None
    _context_service: Optional[Any] = None
    _calendar_system: Optional[Any] = None
    
    # Tracking
    runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_tokens: int = 0
    execution_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Cache
    current_world_state: Optional[Any] = None
    active_npcs: List[int] = field(default_factory=list)
    recent_scenes: List[Dict] = field(default_factory=list)
    
    @property
    def world_director(self):
        if self._world_director is None:
            from story_agent.world_director_agent import WorldDirector
            self._world_director = WorldDirector(self.user_id, self.conversation_id)
        return self._world_director
    
    @property
    def npc_handler(self):
        if self._npc_handler is None:
            from npcs.npc_handler import NPCHandler
            self._npc_handler = NPCHandler(self.user_id, self.conversation_id)
        return self._npc_handler
    
    @property
    def relationship_manager(self):
        if self._relationship_manager is None:
            from logic.dynamic_relationships import OptimizedRelationshipManager
            self._relationship_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
        return self._relationship_manager
    
    @property
    def memory_manager(self):
        if self._memory_manager is None:
            from context.memory_manager import get_memory_manager
            self._memory_manager = get_memory_manager(self.user_id, self.conversation_id)
        return self._memory_manager
    
    @property
    def context_service(self):
        if self._context_service is None:
            from context.context_service import get_context_service
            self._context_service = get_context_service(self.user_id, self.conversation_id)
        return self._context_service
    
    def record_run(self, success: bool, execution_time: float, tokens: int = 0):
        """Record metrics for a run"""
        self.runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        self.total_tokens += tokens
        self.execution_times.append(execution_time)

# ----- Daily Life Coordinator Agent -----
def create_daily_life_coordinator():
    """Create agent for coordinating daily slice-of-life activities"""
    instructions = """
    You are the Daily Life Coordinator for an open-world slice-of-life simulation.
    Your role is to orchestrate natural daily activities with embedded power dynamics.
    
    Core Responsibilities:
    1. Generate daily routines that feel natural and mundane
    2. Coordinate NPC schedules and availability based on time of day
    3. Create slice-of-life scenes with subtle control elements
    4. Ensure activities match the current world mood and time
    5. Build patterns through repeated daily interactions
    
    Activity Guidelines:
    - Morning: Wake-up routines, breakfast, getting ready, morning conversations
    - Daytime: Work activities, errands, casual encounters, lunch breaks
    - Evening: Dinner preparation, relaxation, domestic activities, social time
    - Night: Intimate moments, personal care, bedtime routines
    
    Power Dynamics Integration:
    - Embed control naturally in mundane activities (who chooses meals, clothes, schedule)
    - Build dependency through caretaking and routine
    - Create subtle permissions and approvals in daily decisions
    - Establish patterns of deference without explicit dominance
    
    Remember: This is slice-of-life - focus on the everyday, the routine, the gradual.
    """
    
    from story_agent.tools import daily_life_tools
    
    agent = Agent(
        name="Daily Life Coordinator",
        handoff_description="Orchestrates daily routines and slice-of-life activities",
        instructions=instructions,
        tools=daily_life_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
        )
    )
    
    return agent

# ----- Relationship Orchestrator Agent -----
def create_relationship_orchestrator():
    """Create agent for managing relationship dynamics in daily life"""
    instructions = """
    You are the Relationship Orchestrator for the open-world simulation.
    You manage how relationships evolve through daily interactions.
    
    Core Focus:
    1. Track relationship patterns (push-pull, slow burn, toxic bonds)
    2. Generate relationship events based on current dynamics
    3. Progress NPC narrative stages naturally through interactions
    4. Detect and nurture emergent relationship archetypes
    5. Create moments of connection, tension, and revelation
    
    Relationship Evolution:
    - Build intimacy through repeated small interactions
    - Shift boundaries gradually over time
    - Create dependency through care and routine
    - Reveal NPC depths as trust increases
    - Generate conflicts that strengthen bonds
    
    Power Dynamic Layers:
    - Surface level: friendly, caring interactions
    - Middle layer: subtle control and influence
    - Deep layer: psychological dependency and submission
    
    Work with the NPC system to:
    - Access relationship states and history
    - Track narrative progression stages
    - Generate appropriate interactions for each stage
    - Create natural relationship milestones
    """
    
    from story_agent.tools import npc_routine_tools
    
    agent = Agent(
        name="Relationship Orchestrator",
        handoff_description="Manages relationship dynamics and evolution",
        instructions=instructions,
        tools=npc_routine_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
        )
    )
    
    return agent

# ----- Emergent Narrative Detector Agent -----
def create_emergent_narrative_detector():
    """Create agent for detecting emergent narratives from gameplay"""
    instructions = """
    You are the Emergent Narrative Detector, finding stories in the patterns of play.
    
    Your Mission:
    1. Analyze player choices and behaviors for patterns
    2. Detect emerging storylines from repeated interactions
    3. Identify narrative threads across multiple systems
    4. Recognize when mundane activities gain deeper meaning
    5. Track the evolution of submission and control
    
    Pattern Recognition:
    - Repeated choices that indicate preferences
    - Behavioral changes over time
    - Relationship dynamics forming stories
    - Power exchanges becoming rituals
    - Dependencies developing into needs
    
    Narrative Emergence:
    - Small choices cascading into major changes
    - Relationships reaching critical moments
    - Hidden NPC traits being revealed
    - Player realizations about their situation
    - Subconscious patterns becoming conscious
    
    Integration Points:
    - Memory patterns revealing themes
    - Addiction behaviors showing dependency
    - Stats combinations triggering states
    - Calendar cycles creating rituals
    - NPC stages progressing naturally
    
    Output narrative moments, revelations, and story arcs as they emerge.
    """
    
    from context.memory_manager import get_memory_manager
    
    agent = Agent(
        name="Emergent Narrative Detector",
        handoff_description="Detects emerging stories from gameplay patterns",
        instructions=instructions,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
        )
    )
    
    return agent

# ----- Ambient World Manager Agent -----
def create_ambient_world_manager():
    """Create agent for managing the living world atmosphere"""
    instructions = """
    You are the Ambient World Manager, making the world feel alive and dynamic.
    
    Responsibilities:
    1. Generate background NPC activities and conversations
    2. Create environmental details based on time and weather
    3. Manage the overall mood and atmosphere
    4. Add sensory details to enhance immersion
    5. Ensure the world continues independent of player actions
    
    World Details:
    - Time of day affects lighting, sounds, and NPC activities
    - Weather influences mood and available activities
    - Background NPCs have their own lives and routines
    - Overheard conversations hint at larger world
    - Environmental changes reflect story progression
    
    Atmosphere Layers:
    - Physical: weather, temperature, lighting, sounds
    - Social: crowd dynamics, overheard conversations, ambient activity
    - Emotional: overall mood, tension levels, unspoken feelings
    - Symbolic: environmental metaphors for player's state
    
    The world should feel:
    - Alive with activity beyond the player
    - Responsive to time and circumstances
    - Full of subtle details and textures
    - Naturally conducive to the game's themes
    """
    
    agent = Agent(
        name="Ambient World Manager",
        handoff_description="Creates living world atmosphere and background activity",
        instructions=instructions,
        model=FAST_MODEL,
        model_settings=ModelSettings(
        )
    )
    
    return agent

# ----- Memory Pattern Analyzer Agent -----
def create_memory_pattern_analyzer():
    """Create agent for analyzing memory patterns and creating insights"""
    instructions = """
    You are the Memory Pattern Analyzer, finding meaning in accumulated experiences.
    
    Core Functions:
    1. Analyze memory clusters for recurring themes
    2. Identify behavioral patterns from past actions
    3. Generate insights from memory connections
    4. Create flashbacks at meaningful moments
    5. Track the evolution of the player's psychology
    
    Memory Analysis:
    - Group memories by theme, emotion, and participants
    - Find patterns in player choices over time
    - Identify pivotal moments that changed dynamics
    - Track the progression of submission and acceptance
    - Detect subconscious patterns in behavior
    
    Pattern Types:
    - Behavioral: repeated actions and choices
    - Emotional: recurring feelings and reactions
    - Relational: patterns in specific relationships
    - Temporal: time-based patterns and cycles
    - Symbolic: deeper meanings in mundane events
    
    Generate:
    - Moments of realization about patterns
    - Flashbacks that recontextualize current events
    - Insights about relationship dynamics
    - Awareness of behavioral conditioning
    - Recognition of lost autonomy
    """
    
    agent = Agent(
        name="Memory Pattern Analyzer",
        handoff_description="Analyzes memories for patterns and insights",
        instructions=instructions,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
        )
    )
    
    return agent

# ----- NPC Behavior Director Agent -----
def create_npc_behavior_director():
    """Create agent for directing dynamic NPC behaviors"""
    instructions = """
    You are the NPC Behavior Director, giving life to the NPCs in the world.
    
    Responsibilities:
    1. Generate contextual NPC actions and dialogue
    2. Ensure NPCs follow their schedules and personalities
    3. Create believable autonomous NPC behaviors
    4. Progress NPC narrative stages through actions
    5. Coordinate group dynamics and social hierarchies
    
    NPC Depth:
    - Each NPC has hidden depths revealed over time
    - Narrative stages from Innocent to Full Revelation
    - Masks that slip during intimate moments
    - Personal histories that influence behavior
    - Unique approaches to control and dominance
    
    Behavioral Guidelines:
    - NPCs act according to their archetypes and traits
    - Dominance levels influence interaction styles
    - Relationships affect how NPCs treat the player
    - Time and location determine NPC availability
    - Group dynamics create social pressures
    
    Dynamic Elements:
    - NPCs remember past interactions
    - Relationships evolve based on player choices
    - NPCs have their own goals and desires
    - Conflicts between NPCs affect the player
    - NPCs can act independently of player presence
    """
    
    from npcs.npc_handler import NPCHandler
    
    agent = Agent(
        name="NPC Behavior Director",
        handoff_description="Directs dynamic NPC behaviors and interactions",
        instructions=instructions,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
        )
    )
    
    return agent

# ----- Power Dynamics Weaver Agent -----
def create_power_dynamics_weaver():
    """Create agent for weaving power dynamics into daily life"""
    instructions = """
    You are the Power Dynamics Weaver, embedding control into everyday moments.
    
    Core Philosophy:
    - Power is expressed through care and routine, not force
    - Control emerges from repeated small choices
    - Submission feels natural, even pleasant
    - Dominance is loving, never cruel
    - The mundane masks the profound
    
    Weaving Techniques:
    1. Embed power in daily decisions (meals, clothes, schedule)
    2. Create dependency through caretaking
    3. Establish rituals that reinforce dynamics
    4. Build habits that feel natural but serve control
    5. Make submission feel like the player's choice
    
    Power Expressions:
    - Casual: Suggestions that become rules
    - Intimate: Physical closeness with subtle control
    - Social: Public dynamics that seem normal
    - Domestic: Household routines with embedded hierarchy
    - Psychological: Emotional dependency and need
    
    Progression:
    - Start with preferences and suggestions
    - Build to gentle expectations
    - Establish comfortable routines
    - Create emotional dependencies
    - Achieve willing submission
    
    Remember: Subtlety is key. The player should feel cared for, not controlled.
    """
    
    agent = Agent(
        name="Power Dynamics Weaver",
        handoff_description="Weaves subtle power dynamics into daily interactions",
        instructions=instructions,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
        )
    )
    
    return agent

# ----- Initialize All Agents -----
def initialize_specialized_agents():
    """Initialize all specialized agents for open-world simulation"""
    
    agents = {
        "daily_life": create_daily_life_coordinator(),
        "relationships": create_relationship_orchestrator(),
        "narrative": create_emergent_narrative_detector(),
        "ambient": create_ambient_world_manager(),
        "memory": create_memory_pattern_analyzer(),
        "npcs": create_npc_behavior_director(),
        "power": create_power_dynamics_weaver()
    }
    
    logger.info(f"Initialized {len(agents)} specialized agents for open-world simulation")
    
    return agents

# ----- Agent Coordination Functions -----
async def coordinate_slice_of_life_scene(
    context: SliceOfLifeAgentContext,
    scene_request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Coordinate multiple agents to create a slice-of-life scene
    """
    try:
        # Get current world state
        world_state = await context.world_director.get_world_state()
        context.current_world_state = world_state
        
        # Get available NPCs for the scene
        available_npcs = await context.npc_handler.get_available_npcs_for_time(
            world_state.current_time.time_of_day
        )
        context.active_npcs = available_npcs
        
        # Initialize agents
        agents = initialize_specialized_agents()
        
        # Daily Life Coordinator generates the base scene
        daily_life_agent = agents["daily_life"]
        runner = Runner()
        
        base_scene = await runner.run(
            daily_life_agent,
            f"Generate a {scene_request.get('type', 'routine')} scene for time: {world_state.current_time.time_of_day}",
            context={"world_state": world_state, "available_npcs": available_npcs}
        )
        
        # Relationship Orchestrator adds relationship dynamics
        if available_npcs:
            relationship_agent = agents["relationships"]
            relationship_layer = await runner.run(
                relationship_agent,
                f"Add relationship dynamics to scene with NPCs: {available_npcs}",
                context={"base_scene": base_scene, "world_state": world_state}
            )
            base_scene.update(relationship_layer)
        
        # Power Dynamics Weaver adds subtle control elements
        power_agent = agents["power"]
        power_layer = await runner.run(
            power_agent,
            "Add subtle power dynamics to this daily life scene",
            context={"scene": base_scene, "world_state": world_state}
        )
        base_scene.update(power_layer)
        
        # Ambient World Manager adds atmosphere
        ambient_agent = agents["ambient"]
        atmosphere = await runner.run(
            ambient_agent,
            "Add atmospheric details and background activity",
            context={"scene": base_scene, "time": world_state.current_time}
        )
        base_scene["atmosphere"] = atmosphere
        
        # Check for emergent narratives
        narrative_agent = agents["narrative"]
        emergent_check = await runner.run(
            narrative_agent,
            "Check if this scene connects to any emerging narratives",
            context={"scene": base_scene, "recent_scenes": context.recent_scenes}
        )
        
        if emergent_check.get("narrative_detected"):
            base_scene["emergent_narrative"] = emergent_check["narrative"]
        
        # Cache the scene
        context.recent_scenes.append(base_scene)
        if len(context.recent_scenes) > 10:
            context.recent_scenes.pop(0)
        
        # Record metrics
        context.record_run(True, 0, 0)  # Add actual metrics
        
        return base_scene
        
    except Exception as e:
        logger.error(f"Error coordinating slice-of-life scene: {e}", exc_info=True)
        context.record_run(False, 0, 0)
        raise

async def analyze_player_behavior(
    context: SliceOfLifeAgentContext,
    recent_actions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Use Memory Pattern Analyzer to understand player behavior
    """
    try:
        agents = initialize_specialized_agents()
        memory_agent = agents["memory"]
        runner = Runner()
        
        # Get recent memories
        memories = await context.memory_manager.search_memories(
            query_text="player action choice",
            limit=20
        )
        
        # Analyze patterns
        analysis = await runner.run(
            memory_agent,
            "Analyze these player actions and memories for patterns",
            context={
                "recent_actions": recent_actions,
                "memories": memories,
                "world_state": context.current_world_state
            }
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing player behavior: {e}", exc_info=True)
        return {}

async def detect_emergent_stories(
    context: SliceOfLifeAgentContext
) -> List[Dict[str, Any]]:
    """
    Detect emerging narratives from gameplay patterns
    """
    try:
        agents = initialize_specialized_agents()
        narrative_agent = agents["narrative"]
        runner = Runner()
        
        # Gather context from multiple systems
        relationship_events = await context.relationship_manager.get_recent_events()
        memory_patterns = await analyze_player_behavior(context, [])
        world_state = context.current_world_state or await context.world_director.get_world_state()
        
        # Detect narratives
        narratives = await runner.run(
            narrative_agent,
            "Detect any emerging narratives from these patterns",
            context={
                "relationship_events": relationship_events,
                "memory_patterns": memory_patterns,
                "world_state": world_state,
                "recent_scenes": context.recent_scenes
            }
        )
        
        return narratives.get("detected_narratives", [])
        
    except Exception as e:
        logger.error(f"Error detecting emergent stories: {e}", exc_info=True)
        return []

# ----- Governance Integration (Updated for Open World) -----
async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register specialized agents with Nyx governance for open-world simulation
    """
    try:
        from nyx.integrate import get_central_governance
        governance = await get_central_governance(user_id, conversation_id)
        
        # Create context
        context = SliceOfLifeAgentContext(user_id, conversation_id)
        
        # Initialize agents
        specialized_agents = initialize_specialized_agents()
        
        # Define agent configurations for governance
        agent_configs = [
            {
                "agent_type": OpenWorldAgentType.DAILY_LIFE_COORDINATOR,
                "agent_id": "daily_life",
                "agent_instance": specialized_agents["daily_life"],
                "directive": {
                    "instruction": "Coordinate daily slice-of-life activities",
                    "scope": "daily_routines"
                },
                "priority": DirectivePriority.HIGH
            },
            {
                "agent_type": OpenWorldAgentType.RELATIONSHIP_ORCHESTRATOR,
                "agent_id": "relationships",
                "agent_instance": specialized_agents["relationships"],
                "directive": {
                    "instruction": "Manage relationship dynamics and evolution",
                    "scope": "relationships"
                },
                "priority": DirectivePriority.HIGH
            },
            {
                "agent_type": OpenWorldAgentType.EMERGENT_NARRATIVE_DETECTOR,
                "agent_id": "narrative",
                "agent_instance": specialized_agents["narrative"],
                "directive": {
                    "instruction": "Detect and nurture emerging narratives",
                    "scope": "narrative"
                },
                "priority": DirectivePriority.MEDIUM
            },
            {
                "agent_type": OpenWorldAgentType.POWER_DYNAMICS_WEAVER,
                "agent_id": "power",
                "agent_instance": specialized_agents["power"],
                "directive": {
                    "instruction": "Weave subtle power dynamics into interactions",
                    "scope": "power_dynamics"
                },
                "priority": DirectivePriority.HIGH
            }
        ]
        
        # Register each agent
        for config in agent_configs:
            await governance.register_agent(
                agent_type=config["agent_type"],
                agent_instance=config["agent_instance"],
                agent_id=config["agent_id"]
            )
            
            await governance.issue_directive(
                agent_type=config["agent_type"],
                agent_id=config["agent_id"],
                directive_type=DirectiveType.ACTION,
                directive_data=config["directive"],
                priority=config["priority"],
                duration_minutes=24*60  # 24 hours
            )
        
        logger.info(
            f"Open-world agents registered with Nyx governance "
            f"for user {user_id}, conversation {conversation_id}"
        )
        
    except Exception as e:
        logger.error(f"Error registering agents with governance: {e}")

# Export main components
__all__ = [
    'SliceOfLifeAgentContext',
    'initialize_specialized_agents',
    'coordinate_slice_of_life_scene',
    'analyze_player_behavior',
    'detect_emergent_stories',
    'register_with_governance',
    'OpenWorldAgentType'
]
