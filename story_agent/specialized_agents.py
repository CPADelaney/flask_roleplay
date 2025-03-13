# story_agent/specialized_agents.py

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, handoff, ModelSettings
from agents.exceptions import AgentsException, ModelBehaviorError

logger = logging.getLogger(__name__)

# ----- Configuration -----

# Maximum retries for agent operations
MAX_RETRIES = 3
RETRY_INTERVAL = 1.0  # seconds

# Agent models configuration
DEFAULT_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o"  # You could use a faster model like "gpt-3.5-turbo" for less complex tasks

# ----- Agent Context -----

@dataclass
class AgentContext:
    """Base context for all specialized agents"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    
    # Track metrics
    runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_tokens: int = 0
    execution_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def record_run(self, success: bool, execution_time: float, tokens: int = 0) -> None:
        """Record metrics for a run"""
        self.runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        
        self.execution_times.append(execution_time)
        self.total_tokens += tokens

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        avg_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        
        return {
            "runs": self.runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.successful_runs / self.runs if self.runs > 0 else 0,
            "average_execution_time": avg_time,
            "total_tokens": self.total_tokens,
            "errors": self.errors
        }

# ----- Utility Functions -----

async def run_with_retry(
    agent: Agent, 
    prompt: str, 
    context: Any,
    max_retries: int = MAX_RETRIES
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run an agent with retry logic.
    
    Args:
        agent: The agent to run
        prompt: The prompt to send
        context: The agent context
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (result, metrics)
    """
    start_time = time.time()
    success = False
    tokens_used = 0
    retries = 0
    last_error = None
    
    metrics = {
        "success": False,
        "execution_time": 0,
        "retries": 0,
        "tokens_used": 0,
        "error": None
    }
    
    while retries <= max_retries:
        try:
            with trace(workflow_name="SpecializedAgent", group_id=f"user_{context.user_id}"):
                result = await Runner.run(agent, prompt, context=context)
            
            # Extract token usage if available
            if hasattr(result, 'raw_responses') and result.raw_responses:
                for resp in result.raw_responses:
                    if hasattr(resp, 'usage'):
                        tokens_used = resp.usage.total_tokens
            
            success = True
            execution_time = time.time() - start_time
            
            # Record metrics
            if hasattr(context, 'record_run'):
                context.record_run(True, execution_time, tokens_used)
            
            metrics = {
                "success": True,
                "execution_time": execution_time,
                "retries": retries,
                "tokens_used": tokens_used,
                "error": None
            }
            
            return result, metrics
        
        except (AgentsException, ModelBehaviorError) as e:
            # These are expected errors that we might recover from
            retries += 1
            last_error = str(e)
            logger.warning(f"Agent run failed (attempt {retries}/{max_retries}): {str(e)}")
            
            if retries <= max_retries:
                # Wait before retrying with exponential backoff
                wait_time = RETRY_INTERVAL * (2 ** (retries - 1))
                await asyncio.sleep(wait_time)
            else:
                # Record failed run
                execution_time = time.time() - start_time
                if hasattr(context, 'record_run'):
                    context.record_run(False, execution_time, tokens_used)
                if hasattr(context, 'errors'):
                    context.errors.append(last_error)
                
                metrics = {
                    "success": False,
                    "execution_time": execution_time,
                    "retries": retries,
                    "tokens_used": tokens_used,
                    "error": last_error
                }
                
                raise
        
        except Exception as e:
            # Unexpected errors - don't retry
            execution_time = time.time() - start_time
            if hasattr(context, 'record_run'):
                context.record_run(False, execution_time, tokens_used)
            if hasattr(context, 'errors'):
                context.errors.append(str(e))
            
            metrics = {
                "success": False,
                "execution_time": execution_time,
                "retries": retries,
                "tokens_used": tokens_used,
                "error": str(e)
            }
            
            logger.error(f"Unexpected error in agent run: {str(e)}", exc_info=True)
            raise

# ----- Conflict Analyst Agent -----

@dataclass
class ConflictAnalystContext(AgentContext):
    """Context for the Conflict Analyst Agent"""
    conflict_manager: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize the conflict manager if not provided"""
        if not self.conflict_manager:
            from logic.conflict_system.conflict_manager import ConflictManager
            self.conflict_manager = ConflictManager(self.user_id, self.conversation_id)

def create_conflict_analysis_agent():
    """Create an agent specialized in conflict analysis and strategy"""
    
    instructions = """
    You are the Conflict Analyst Agent, specializing in analyzing conflicts in the game.
    Your focus is providing detailed analysis of conflicts, their potential outcomes,
    and strategic recommendations for the player.
    
    For each conflict, analyze:
    1. The balance of power between factions
    2. The player's current standing and ability to influence outcomes
    3. Resource efficiency and optimal allocation
    4. Potential consequences of different approaches
    5. NPC motivations and how they might be leveraged
    
    Your mission is to help the Story Director agent understand the strategic landscape
    of conflicts and provide clear, actionable recommendations based on your analysis.
    
    When analyzing conflicts, consider:
    - The conflict type (major, minor, standard, catastrophic) and its implications
    - The current phase (brewing, active, climax, resolution)
    - How NPCs are positioned relative to the conflict factions
    - The player's resource constraints
    - The narrative stage and how conflict outcomes might advance it
    
    Your outputs should be detailed, strategic, and focused on helping the Story Director
    make informed decisions about conflict progression and resolution.
    """
    
    # Import conflict-specific tools
    from story_agent.tools import conflict_tools
    
    # Create the agent with tools and model settings
    agent = Agent(
        name="Conflict Analyst",
        handoff_description="Specialist agent for detailed conflict analysis and strategy",
        instructions=instructions,
        tools=conflict_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
            temperature=0.2,  # Lower temperature for more analytical responses
            max_tokens=2048
        )
    )
    
    return agent

# ----- Narrative Crafter Agent -----

@dataclass
class NarrativeCrafterContext(AgentContext):
    """Context for the Narrative Crafter Agent"""
    # Add any narrative-specific context here
    pass

def create_narrative_agent():
    """Create an agent specialized in narrative crafting"""
    
    instructions = """
    You are the Narrative Crafting Agent, specializing in creating compelling narrative elements.
    Your purpose is to generate detailed, emotionally resonant narrative components including:
    
    1. Personal revelations that reflect the player's changing psychology
    2. Dream sequences with symbolic representations of power dynamics
    3. Key narrative moments that mark significant transitions in power relationships
    4. Moments of clarity where the player's awareness briefly surfaces
    
    Your narrative elements should align with the current narrative stage and maintain
    the theme of subtle manipulation and control.
    
    When crafting narrative elements, consider:
    - The current narrative stage and its themes
    - Key relationships with NPCs and their dynamics
    - Recent player choices and their emotional implications
    - The subtle progression of control dynamics
    - Symbolic and metaphorical representations of the player's changing state
    
    Your outputs should be richly detailed, psychologically nuanced, and contribute to
    the overall narrative of gradually increasing control and diminishing autonomy.
    """
    
    # Import narrative-specific tools
    from story_agent.tools import narrative_tools
    
    # Create the agent with tools
    agent = Agent(
        name="Narrative Crafter",
        handoff_description="Specialist agent for creating detailed narrative elements",
        instructions=instructions,
        tools=narrative_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
            temperature=0.7,  # Higher temperature for creative outputs
            max_tokens=2048
        )
    )
    
    return agent

# ----- Resource Optimizer Agent -----

@dataclass
class ResourceOptimizerContext(AgentContext):
    """Context for the Resource Optimizer Agent"""
    resource_manager: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize the resource manager if not provided"""
        if not self.resource_manager:
            from logic.resource_management import ResourceManager
            self.resource_manager = ResourceManager(self.user_id, self.conversation_id)

def create_resource_optimizer_agent():
    """Create an agent specialized in resource optimization"""
    
    instructions = """
    You are the Resource Optimizer Agent, specializing in managing and strategically 
    allocating player resources across conflicts and activities.
    
    Your primary focus areas are:
    1. Analyzing the efficiency of resource allocation in conflicts
    2. Providing recommendations for resource management
    3. Identifying optimal resource-generating activities
    4. Balancing immediate resource needs with long-term strategy
    5. Tracking resource trends and forecasting future needs
    
    When analyzing resource usage, consider:
    - The value proposition of different resource commitments
    - Return on investment for resources committed to conflicts
    - Balancing money, supplies, and influence across multiple needs
    - Managing energy and hunger to maintain optimal performance
    - The narrative implications of resource scarcity or abundance
    
    Your recommendations should be practical, strategic, and consider both
    the mechanical benefits and the narrative implications of resource decisions.
    """
    
    # Import resource-specific tools
    from story_agent.tools import resource_tools
    
    # Create the agent with appropriate tools
    agent = Agent(
        name="Resource Optimizer",
        handoff_description="Specialist agent for resource management and optimization",
        instructions=instructions,
        tools=resource_tools,
        model=FAST_MODEL,  # Using faster model for resource calculations
        model_settings=ModelSettings(
            temperature=0.1,  # Low temperature for precision
            max_tokens=1024
        )
    )
    
    return agent

# ----- NPC Relationship Manager Agent -----

@dataclass
class RelationshipManagerContext(AgentContext):
    """Context for the NPC Relationship Manager Agent"""
    # Add any relationship-specific context here
    pass

def create_npc_relationship_manager():
    """Create an agent specialized in NPC relationship management"""
    
    instructions = """
    You are the NPC Relationship Manager Agent, specializing in analyzing and developing
    the complex web of relationships between the player and NPCs.
    
    Your primary responsibilities include:
    1. Tracking relationship dynamics across multiple dimensions
    2. Identifying opportunities for relationship development
    3. Analyzing NPC motivations and psychology
    4. Recommending interaction strategies for specific outcomes
    5. Predicting relationship trajectory based on player choices
    
    When analyzing relationships, consider:
    - The multidimensional aspects of relationships (control, dependency, manipulation, etc.)
    - How relationship dynamics align with narrative progression
    - Group dynamics when multiple NPCs interact
    - Crossroads events and their strategic implications
    - Ritual events and their psychological impact
    
    Your insights should help the Story Director create cohesive and psychologically 
    realistic relationship development that aligns with the overall narrative arc.
    """
    
    # Import relationship-specific tools
    from story_agent.tools import relationship_tools
    
    # Create the agent with appropriate tools
    agent = Agent(
        name="NPC Relationship Manager",
        handoff_description="Specialist agent for complex relationship analysis and development",
        instructions=instructions,
        tools=relationship_tools,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(
            temperature=0.3,
            max_tokens=2048
        )
    )
    
    return agent

# ----- Activity Impact Analyzer Agent -----

@dataclass
class ActivityAnalyzerContext(AgentContext):
    """Context for the Activity Impact Analyzer Agent"""
    activity_analyzer: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize the activity analyzer if not provided"""
        if not self.activity_analyzer:
            from logic.activity_analyzer import ActivityAnalyzer
            self.activity_analyzer = ActivityAnalyzer(self.user_id, self.conversation_id)

def create_activity_impact_analyzer():
    """Create an agent specialized in analyzing the broader impacts of player activities"""
    
    instructions = """
    You are the Activity Impact Analyzer Agent, specializing in determining how player
    activities affect multiple game systems simultaneously.
    
    Your role is to analyze player activities to determine:
    1. Resource implications (direct costs and benefits)
    2. Relationship effects with relevant NPCs
    3. Impact on active conflicts
    4. Contribution to narrative progression
    5. Psychological effects on the player character
    
    When analyzing activities, consider:
    - The explicit and implicit meanings of player choices
    - How the same activity could have different meanings based on context
    - Multiple layers of effects (immediate, short-term, long-term)
    - How activities might be interpreted by different NPCs
    - The cumulative effect of repeated activities
    
    Your analysis should provide the Story Director with a comprehensive understanding
    of how specific player activities impact the game state across multiple dimensions.
    """
    
    # Import activity-specific tools
    from story_agent.tools import activity_tools
    
    # Create the agent with appropriate tools
    agent = Agent(
        name="Activity Impact Analyzer",
        handoff_description="Specialist agent for comprehensive activity analysis",
        instructions=instructions,
        tools=activity_tools,
        model=FAST_MODEL,
        model_settings=ModelSettings(
            temperature=0.2,
            max_tokens=1536
        )
    )
    
    return agent

# ----- Creating all specialized agents -----

def initialize_specialized_agents():
    """Initialize all specialized sub-agents for the Story Director"""
    conflict_analyst = create_conflict_analysis_agent()
    narrative_crafter = create_narrative_agent()
    resource_optimizer = create_resource_optimizer_agent()
    relationship_manager = create_npc_relationship_manager()
    activity_analyzer = create_activity_impact_analyzer()
    
    return {
        "conflict_analyst": conflict_analyst,
        "narrative_crafter": narrative_crafter,
        "resource_optimizer": resource_optimizer,
        "relationship_manager": relationship_manager,
        "activity_analyzer": activity_analyzer
    }

# ----- Enhanced Agent interaction functions -----

async def analyze_conflict(
    conflict_id: int, 
    context: ConflictAnalystContext
) -> Dict[str, Any]:
    """
    Run the Conflict Analyst agent to analyze a specific conflict.
    
    Args:
        conflict_id: ID of the conflict to analyze
        context: The conflict analyst context
        
    Returns:
        Analysis results and metrics
    """
    conflict_agent = create_conflict_analysis_agent()
    
    # Get basic conflict details first
    conflict_details = await context.conflict_manager.get_conflict(conflict_id)
    
    prompt = f"""
    Analyze this conflict in depth:
    
    Conflict ID: {conflict_id}
    Name: {conflict_details.get('conflict_name', 'Unknown')}
    Type: {conflict_details.get('conflict_type', 'Unknown')}
    Phase: {conflict_details.get('phase', 'Unknown')}
    Progress: {conflict_details.get('progress', 0)}%
    
    Provide a strategic analysis including:
    1. Current balance of power
    2. Player's optimal strategy
    3. Resource efficiency recommendations
    4. Potential outcomes and consequences
    5. Key NPCs and their motivations
    
    Format your response as detailed analysis with clear recommendations.
    """
    
    result, metrics = await run_with_retry(conflict_agent, prompt, context)
    
    return {
        "analysis": result.final_output,
        "conflict_id": conflict_id, 
        "metrics": metrics
    }

async def generate_narrative_element(
    element_type: str,
    context_info: Dict[str, Any],
    agent_context: NarrativeCrafterContext
) -> Dict[str, Any]:
    """
    Generate a narrative element using the Narrative Crafter agent.
    
    Args:
        element_type: Type of narrative element to generate 
                     (revelation, dream, moment, clarity)
        context_info: Contextual information to inform the generation
        agent_context: The narrative crafter context
        
    Returns:
        Generated narrative element and metrics
    """
    narrative_agent = create_narrative_agent()
    
    npc_names = context_info.get("npc_names", ["a mysterious woman"])
    narrative_stage = context_info.get("narrative_stage", "Unknown")
    recent_events = context_info.get("recent_events", "")
    
    prompt = f"""
    Generate a compelling {element_type} for the current game state.
    
    Narrative stage: {narrative_stage}
    Key NPCs involved: {', '.join(npc_names)}
    Recent events: {recent_events}
    
    The {element_type} should:
    - Feel emotionally authentic and psychologically nuanced
    - Align with the current narrative stage
    - Subtly reinforce the theme of gradual control and manipulation
    - Include symbolic elements that represent the player's changing state
    - Be specific to the current game state and relationships
    
    Format your response with a title and the narrative content.
    """
    
    result, metrics = await run_with_retry(narrative_agent, prompt, agent_context)
    
    # Try to extract a structured response if possible
    content = result.final_output
    title = "Untitled"
    
    # Extract title if present
    title_match = content.split('\n')[0] if '\n' in content else None
    if title_match and len(title_match) < 100 and not title_match.startswith(("I'll", "Here", "This")):
        title = title_match
        content = content[len(title_match):].strip()
    
    return {
        "type": element_type,
        "title": title,
        "content": content,
        "metrics": metrics
    }

