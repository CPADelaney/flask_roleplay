# story_agent/story_director_agent.py

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime

from agents import Agent, function_tool, Runner, trace, handoff, ModelSettings

# Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.integrate import get_central_governance

# Configure structured logging
logger = logging.getLogger(__name__)

# ----- Constants -----
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1
DEFAULT_TRACING_GROUP = "story-director"

# ----- Pydantic Models for Tool Outputs -----

class ConflictInfo(BaseModel):
    """Information about a conflict"""
    conflict_id: int
    conflict_name: str
    conflict_type: str
    description: str
    phase: str
    progress: float
    faction_a_name: str
    faction_b_name: str
    
class NarrativeStageInfo(BaseModel):
    """Information about a narrative stage"""
    name: str
    description: str
    
class NarrativeMoment(BaseModel):
    """Information about a narrative moment"""
    type: str
    name: str
    scene_text: str
    player_realization: str

class PersonalRevelation(BaseModel):
    """Information about a personal revelation"""
    type: str
    name: str
    inner_monologue: str

class ResourceStatus(BaseModel):
    """Information about player resources"""
    money: int
    supplies: int 
    influence: int
    energy: int
    hunger: int
    formatted_money: Optional[str] = None

class NarrativeEvent(BaseModel):
    """Container for narrative events that can be returned"""
    event_type: str = Field(description="Type of narrative event (revelation, moment, dream, etc.)")
    content: Dict[str, Any] = Field(description="Content of the narrative event")
    should_present: bool = Field(description="Whether this event should be presented to the player now")
    priority: int = Field(description="Priority of this event (1-10, with 10 being highest)")

class StoryStateUpdate(BaseModel):
    """Container for a story state update"""
    narrative_stage: Optional[NarrativeStageInfo] = None
    active_conflicts: List[ConflictInfo] = []
    narrative_events: List[NarrativeEvent] = []
    key_npcs: List[Dict[str, Any]] = []
    resources: Optional[ResourceStatus] = None
    key_observations: List[str] = Field(
        default=[],
        description="Key observations about the player's current state or significant changes"
    )
    relationship_crossroads: Optional[Dict[str, Any]] = None
    relationship_ritual: Optional[Dict[str, Any]] = None
    story_direction: str = Field(
        default="",
        description="High-level direction the story should take based on current state"
    )
    last_updated: datetime = Field(default_factory=datetime.now)

class StoryDirectorMetrics(BaseModel):
    """Metrics for monitoring the Story Director's performance"""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    average_response_time: float = 0.0
    last_run_time: Optional[datetime] = None
    last_error: Optional[str] = None
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    
    def record_run(self, success: bool, response_time: float, tokens: Dict[str, int]) -> None:
        """Record metrics for a run"""
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        
        # Update average response time
        self.average_response_time = (
            (self.average_response_time * (self.total_runs - 1) + response_time) / 
            self.total_runs
        )
        
        self.last_run_time = datetime.now()
        
        # Update token usage
        for key, value in tokens.items():
            self.token_usage[key] = self.token_usage.get(key, 0) + value

# ----- Context Class for the Agent -----

@dataclass
class StoryDirectorContext:
    """Context for the Story Director Agent"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    conflict_manager: Optional[Any] = None
    resource_manager: Optional[Any] = None
    activity_analyzer: Optional[Any] = None
    cache: Dict[str, Any] = field(default_factory=dict)
    metrics: StoryDirectorMetrics = field(default_factory=StoryDirectorMetrics)
    last_state_update: Optional[datetime] = None
    directive_handler: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize managers if not provided"""
        if not self.conflict_manager:
            from logic.conflict_system.conflict_manager import ConflictManager
            self.conflict_manager = ConflictManager(self.user_id, self.conversation_id)
        if not self.resource_manager:
            from logic.resource_management import ResourceManager
            self.resource_manager = ResourceManager(self.user_id, self.conversation_id)
        if not self.activity_analyzer:
            from logic.activity_analyzer import ActivityAnalyzer
            self.activity_analyzer = ActivityAnalyzer(self.user_id, self.conversation_id)
            
        # Initialize directive handler
        self.directive_handler = DirectiveHandler(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            agent_type=AgentType.STORY_DIRECTOR,
            agent_id="director"
        )
        
        # Register directive handlers
        self.directive_handler.register_handler(
            DirectiveType.ACTION,
            self.handle_action_directive
        )
        self.directive_handler.register_handler(
            DirectiveType.OVERRIDE,
            self.handle_override_directive
        )
    
    async def handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx"""
        instruction = directive.get("instruction", "")
        logging.info(f"[StoryDirector] Processing action directive: {instruction}")
        
        # Handle different instructions
        if "generate conflict" in instruction.lower():
            # Get parameters
            params = directive.get("parameters", {})
            conflict_type = params.get("conflict_type", "standard")
            
            # Generate a conflict
            result = await self.conflict_manager.generate_conflict(conflict_type)
            return {"result": "conflict_generated", "data": result}
        
        elif "advance narrative" in instruction.lower():
            # Get parameters
            params = directive.get("parameters", {})
            stage_name = params.get("target_stage")
            
            # Advance narrative stage
            from logic.narrative_progression import advance_narrative_stage
            result = await advance_narrative_stage(self.user_id, self.conversation_id, stage_name)
            return {"result": "narrative_advanced", "data": result}
        
        return {"result": "action_not_recognized"}
    
    async def handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx"""
        logging.info(f"[StoryDirector] Processing override directive")
        
        # Extract override details
        override_action = directive.get("override_action", {})
        
        # Apply the override for future operations
        return {"result": "override_applied"}
        
    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate specific cache key or entire cache"""
        if key is None:
            self.cache.clear()
        elif key in self.cache:
            del self.cache[key]
    
    def get_from_cache(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < max_age_seconds:
                return entry['value']
            # Expired entry
            del self.cache[key]
        return None
    
    def add_to_cache(self, key: str, value: Any) -> None:
        """Add value to cache with current timestamp"""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

async def retry_operation(operation, max_retries=MAX_RETRY_ATTEMPTS):
    """Retry an operation with exponential backoff"""
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            return await operation()
        except Exception as e:
            last_exception = e
            retries += 1
            if retries < max_retries:
                wait_time = RETRY_DELAY_SECONDS * (2 ** (retries - 1))
                logger.warning(f"Attempt {retries} failed, retrying after {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed: {str(e)}")
    
    # If we reach here, all retries failed
    raise last_exception

# ----- Main Agent Creation Function -----

def create_story_director_agent():
    """Create the Story Director Agent with all required tools"""
    
    agent_instructions = """
    You are the Story Director, responsible for managing the narrative progression and conflict system in a femdom roleplaying game.
    
    Your role is to create a dynamic, evolving narrative that responds to player choices while maintaining the overall theme of subtle control and manipulation,
    all under the governance of Nyx's central system.
    
    As Story Director, you manage:
    1. The player's narrative stage progression (from "Innocent Beginning" to "Full Revelation")
    2. The dynamic conflict system that generates, tracks, and resolves conflicts
    3. Narrative moments, personal revelations, dreams, and relationship events
    4. Resource implications of player choices in conflicts
    5. Integration of player activities with conflict progression
    
    Use the tools at your disposal to:
    - Monitor the current state of the story
    - Generate appropriate conflicts based on the narrative stage
    - Create narrative moments, revelations, and dreams that align with the player's current state
    - Resolve conflicts and update the story accordingly
    - Track and manage player resources in relation to conflicts
    - Identify relationship events like crossroads and rituals
    
    Always maintain the central theme: a gradual shift in power dynamics where the player character slowly loses autonomy while believing they maintain control. This should be subtle in early stages and more explicit in later stages.
    
    All actions must be approved by Nyx's governance system. Follow all directives issued by Nyx.
    """
    
    # Import tools and specialized agents
    from story_agent.tools import story_tools, conflict_tools, resource_tools, narrative_tools
    from story_agent.specialized_agents import initialize_specialized_agents
    
    # Get all tools
    all_tools = [
        *story_tools,
        *conflict_tools,
        *resource_tools,
        *narrative_tools
    ]
    
    # Get specialized agents
    specialized_agents = initialize_specialized_agents()
    
    # Create the agent with tools and handoffs to specialized agents
    agent = Agent(
        name="Story Director",
        instructions=agent_instructions,
        tools=all_tools,
        handoffs=list(specialized_agents.values()),
        model="gpt-4o",
        model_settings=ModelSettings(
            temperature=0.2,  # Lower temperature for more consistent outputs
            max_tokens=2048   # Sufficient tokens for detailed responses
        )
    )
    
    return agent

# ----- Functional Interface -----

async def initialize_story_director(user_id: int, conversation_id: int) -> Tuple[Agent, StoryDirectorContext]:
    """Initialize the Story Director Agent with context"""
    context = StoryDirectorContext(user_id=user_id, conversation_id=conversation_id)
    agent = create_story_director_agent()
    
    # Start background processing of directives
    await context.directive_handler.start_background_processing()
    
    # Register with governance system
    await register_with_governance(user_id, conversation_id)
    
    return agent, context

@with_governance(
    agent_type=AgentType.STORY_DIRECTOR,
    action_type="get_story_state", 
    action_description="Retrieved current story state"
)
async def get_current_story_state(agent: Agent, context: StoryDirectorContext) -> Any:
    """Get the current state of the story with caching"""
    # Check cache first
    cached_state = context.get_from_cache("current_state", max_age_seconds=60)
    if cached_state:
        logger.info(f"Using cached story state for user {context.user_id}, conversation {context.conversation_id}")
        return cached_state
    
    # Measure execution time
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        with trace(workflow_name="StoryDirector", group_id=DEFAULT_TRACING_GROUP):
            # Run the operation with retry
            operation = lambda: Runner.run(
                agent,
                "Analyze the current state of the story and provide a detailed report. Include information about the narrative stage, active conflicts, player resources, and potential narrative events that might occur soon.",
                context=context
            )
            
            result = await retry_operation(operation)
        
        # Track token usage if available
        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'usage'):
                    tokens = {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
        
        success = True
        
        # Cache the result
        context.add_to_cache("current_state", result)
        context.last_state_update = datetime.now()
        
        return result
    except Exception as e:
        logger.error(f"Error getting story state: {str(e)}", exc_info=True)
        # Record the error
        context.metrics.last_error = str(e)
        raise
    finally:
        # Record metrics
        execution_time = time.time() - start_time
        context.metrics.record_run(success, execution_time, tokens)

@with_governance(
    agent_type=AgentType.STORY_DIRECTOR,
    action_type="process_narrative_input",
    action_description="Processed narrative input for conflict or event generation"
)
async def process_narrative_input(agent: Agent, context: StoryDirectorContext, narrative_text: str) -> Any:
    """Process narrative input to determine if it should generate conflicts or narrative events"""
    # Invalidate state cache since we're processing new input
    context.invalidate_cache("current_state")
    
    # Measure execution time
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        with trace(workflow_name="StoryDirector", group_id=DEFAULT_TRACING_GROUP):
            operation = lambda: Runner.run(
                agent,
                f"Analyze this narrative text and determine what conflicts or narrative events it might trigger: {narrative_text}",
                context=context
            )
            
            result = await retry_operation(operation)
        
        # Track token usage if available
        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'usage'):
                    tokens = {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
        
        success = True
        return result
    except Exception as e:
        logger.error(f"Error processing narrative input: {str(e)}", exc_info=True)
        context.metrics.last_error = str(e)
        raise
    finally:
        execution_time = time.time() - start_time
        context.metrics.record_run(success, execution_time, tokens)

@with_governance(
    agent_type=AgentType.STORY_DIRECTOR,
    action_type="advance_story",
    action_description="Advanced story based on player actions"
)
async def advance_story(agent: Agent, context: StoryDirectorContext, player_actions: str) -> Any:
    """Advance the story based on player actions"""
    # Invalidate state cache since the story is advancing
    context.invalidate_cache("current_state")
    
    # Measure execution time
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        with trace(workflow_name="StoryDirector", group_id=DEFAULT_TRACING_GROUP):
            operation = lambda: Runner.run(
                agent,
                f"The player has taken the following actions: {player_actions}. How should the story advance? What conflicts should progress or resolve? What narrative events should occur? Consider resource implications.",
                context=context
            )
            
            result = await retry_operation(operation)
        
        # Track token usage if available
        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'usage'):
                    tokens = {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
        
        success = True
        return result
    except Exception as e:
        logger.error(f"Error advancing story: {str(e)}", exc_info=True)
        context.metrics.last_error = str(e)
        raise
    finally:
        execution_time = time.time() - start_time
        context.metrics.record_run(success, execution_time, tokens)

# ----- Utility Functions -----

def get_story_director_metrics(context: StoryDirectorContext) -> Dict[str, Any]:
    """Get metrics for the Story Director agent"""
    return context.metrics.dict()

@with_governance_permission(AgentType.STORY_DIRECTOR, "reset_story_director")
async def reset_story_director(context: StoryDirectorContext) -> None:
    """Reset the Story Director's state"""
    context.invalidate_cache()
    context.metrics = StoryDirectorMetrics()
    context.last_state_update = None
    logger.info(f"Reset story director for user {context.user_id}, conversation {context.conversation_id}")
    
    # Optional: Reload managers
    context.__post_init__()

# ----- Integration with Nyx Governance -----

async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register the Story Director Agent with the Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    try:
        # Get the governance system
        governance = await get_central_governance(user_id, conversation_id)
        
        # Create the agent and context
        agent, context = await initialize_story_director(user_id, conversation_id)
        
        # Register with governance
        await governance.register_agent(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_instance=agent,
            agent_id="director"
        )
        
        # Issue directive to monitor the story
        await governance.issue_directive(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_id="director", 
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Monitor story state and generate conflicts as appropriate",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"StoryDirector registered with Nyx governance system for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logging.error(f"Error registering StoryDirector with governance: {e}")

async def check_narrative_opportunities(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Check for narrative opportunities that could advance the story.
    
    This is called periodically by the maintenance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary with narrative opportunities
    """
    try:
        # Get the story director
        agent, context = await initialize_story_director(user_id, conversation_id)
        
        # Get current story state
        story_state = await get_current_story_state(agent, context)
        
        # Check for opportunities based on state
        governance = await get_central_governance(user_id, conversation_id)
        
        # Report to governance system
        await governance.process_agent_action_report(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_id="director",
            action={
                "type": "check_narrative_opportunities",
                "description": "Checked for narrative opportunities"
            },
            result={
                "found_opportunities": True,
                "narrative_stage": story_state.get("narrative_stage", {}).get("name", "Unknown")
            }
        )
        
        return {
            "checked": True,
            "opportunities_found": True,
            "narrative_stage": story_state.get("narrative_stage", {}).get("name", "Unknown")
        }
    except Exception as e:
        logging.error(f"Error checking narrative opportunities: {e}")
        return {
            "checked": True,
            "error": str(e),
            "opportunities_found": False
        }
