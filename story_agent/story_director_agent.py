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

# NEW: Context system integration
from context.context_service import get_context_service, get_comprehensive_context
from context.context_config import get_config
from context.memory_manager import get_memory_manager, Memory
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

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
    
    # NEW: Context management components
    context_service: Optional[Any] = None
    memory_manager: Optional[Any] = None
    vector_service: Optional[Any] = None
    performance_monitor: Optional[Any] = None
    context_manager: Optional[Any] = None
    
    # NEW: Version tracking for delta updates
    last_context_version: Optional[int] = None
    
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
            agent_id="director",
            governance=governance  # pass the object here
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
            
        # NEW: Initialize context management components
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        self.context_manager = get_context_manager()
        
        # Register context change handlers
        self.context_manager.subscribe_to_changes("/narrative_stage", self.handle_narrative_stage_change)
        self.context_manager.subscribe_to_changes("/conflicts", self.handle_conflict_change)
    
    async def initialize_context_components(self):
        """Initialize context components that require async initialization"""
        self.context_service = await get_context_service(self.user_id, self.conversation_id)
        self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
    
    async def handle_narrative_stage_change(self, changes: List[ContextDiff]):
        """React to changes in narrative stage"""
        logging.info(f"Narrative stage changed: {changes}")
        for change in changes:
            if change.operation in ("add", "replace"):
                stage_info = change.value
                if isinstance(stage_info, dict) and "name" in stage_info:
                    # Create a memory about the stage change
                    await self.add_narrative_memory(
                        f"Narrative stage progressed to {stage_info['name']}",
                        "narrative_progression",
                        0.8
                    )
                    
                    # Report to governance
                    governance = await get_central_governance(self.user_id, self.conversation_id)
                    await governance.process_agent_action_report(
                        agent_type=AgentType.STORY_DIRECTOR,
                        agent_id="director",
                        action={"type": "narrative_stage_change"},
                        result={"new_stage": stage_info["name"]}
                    )
    
    async def handle_conflict_change(self, changes: List[ContextDiff]):
        """React to changes in conflicts"""
        logging.info(f"Conflict changed: {changes}")
        for change in changes:
            if change.operation == "add":
                conflict_info = change.value
                if isinstance(conflict_info, dict) and "conflict_name" in conflict_info:
                    # Create a memory about the new conflict
                    await self.add_narrative_memory(
                        f"New conflict emerged: {conflict_info['conflict_name']}",
                        "conflict_generation",
                        0.7
                    )
    
    async def add_narrative_memory(self, content: str, memory_type: str, importance: float = 0.5):
        """Add a memory using the memory manager"""
        if not self.memory_manager:
            self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        await self.memory_manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=["story_director", memory_type],
            metadata={"source": "story_director"}
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
        
        elif "retrieve context" in instruction.lower():
            # NEW: Handle context retrieval directive
            params = directive.get("parameters", {})
            input_text = params.get("input_text", "")
            use_vector = params.get("use_vector", True)
            
            if not self.context_service:
                await self.initialize_context_components()
                
            context = await self.context_service.get_context(
                input_text=input_text,
                use_vector_search=use_vector
            )
            
            return {"result": "context_retrieved", "data": context}
        
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
    
    # NEW: Enhanced context management methods
    
    async def get_comprehensive_context(self, input_text: str = "") -> Dict[str, Any]:
        """Get comprehensive context using the context service"""
        # Initialize components if needed
        if not self.context_service:
            await self.initialize_context_components()
        
        # Get context
        config = get_config()
        context_budget = config.get_token_budget("default")
        use_vector = config.is_enabled("use_vector_search")
        
        # If we have a previous version, try delta updates
        if self.last_context_version is not None:
            context = await self.context_service.get_context(
                input_text=input_text,
                context_budget=context_budget,
                use_vector_search=use_vector,
                use_delta=True,
                source_version=self.last_context_version
            )
        else:
            context = await self.context_service.get_context(
                input_text=input_text,
                context_budget=context_budget,
                use_vector_search=use_vector,
                use_delta=False
            )
        
        # Store version for future delta updates
        if "version" in context:
            self.last_context_version = context["version"]
        
        return context
    
    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant memories using vector search"""
        if not self.memory_manager:
            await self.initialize_context_components()
        
        # Use vector search for semantic retrieval
        memories = await self.memory_manager.search_memories(
            query_text=query,
            limit=limit,
            use_vector=True
        )
        
        # Convert to dictionaries
        memory_dicts = []
        for memory in memories:
            if hasattr(memory, 'to_dict'):
                memory_dicts.append(memory.to_dict())
            else:
                memory_dicts.append(memory)
        
        return memory_dicts

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
    
    # NEW: Add context tools
    from story_agent.tools import context_tools
    
    # Get all tools
    all_tools = [
        *story_tools,
        *conflict_tools,
        *resource_tools,
        *narrative_tools,
        *context_tools
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
    
    # Initialize context components
    await context.initialize_context_components()
    
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
@track_performance("get_story_state")  # NEW: Performance tracking
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
        # NEW: Get comprehensive context using the context service
        comprehensive_context = await context.get_comprehensive_context()
        
        # Build a prompt that uses the comprehensive context
        prompt = """
        Analyze the current state of the story and provide a detailed report.
        
        Include information about:
        1. The narrative stage
        2. Active conflicts
        3. Player resources
        4. Potential narrative events that might occur soon
        5. Key NPCs and their relationships
        
        Your analysis should be comprehensive and consider all relevant factors
        in the current game state.
        """
        
        with trace(workflow_name="StoryDirector", group_id=DEFAULT_TRACING_GROUP):
            # Run the operation with retry
            operation = lambda: Runner.run(agent, prompt, context=context)
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
        
        # NEW: Create a memory about this analysis
        await context.add_narrative_memory(
            "Analyzed current story state and identified key elements",
            "story_analysis",
            0.5
        )
        
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
        
        # Record to performance monitor
        context.performance_monitor.record_token_usage(tokens.get("total", 0))

@with_governance(
    agent_type=AgentType.STORY_DIRECTOR,
    action_type="process_narrative_input",
    action_description="Processed narrative input for conflict or event generation"
)
@track_performance("process_narrative_input")  # NEW: Performance tracking
async def process_narrative_input(agent: Agent, context: StoryDirectorContext, narrative_text: str) -> Any:
    """Process narrative input to determine if it should generate conflicts or narrative events"""
    # Invalidate state cache since we're processing new input
    context.invalidate_cache("current_state")
    
    # Measure execution time
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        # NEW: Get comprehensive context with the narrative
        comprehensive_context = await context.get_comprehensive_context(narrative_text)
        
        # NEW: Get relevant memories
        relevant_memories = await context.get_relevant_memories(narrative_text, limit=3)
        memory_text = ""
        if relevant_memories:
            memory_text = "Relevant memories:\n" + "\n".join([
                f"- {memory.get('content', '')[:100]}..." for memory in relevant_memories
            ])
        
        # Build enhanced prompt with comprehensive context
        prompt = f"""
        Analyze this narrative text and determine what conflicts or narrative events it might trigger:
        
        Narrative text:
        {narrative_text}
        
        {memory_text}
        
        Consider:
        1. The current narrative stage
        2. Existing conflicts and their status
        3. Character relationships and dynamics
        4. Recent events and their implications
        5. The overall theme of subtle control and manipulation
        
        Determine if this narrative should:
        - Generate a new conflict
        - Progress an existing conflict
        - Trigger a narrative event (revelation, dream, moment of clarity)
        - Affect relationships between characters
        """
        
        with trace(workflow_name="StoryDirector", group_id=DEFAULT_TRACING_GROUP):
            operation = lambda: Runner.run(agent, prompt, context=context)
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
        
        # NEW: Create a memory about this processing
        await context.add_narrative_memory(
            f"Processed narrative: {narrative_text[:100]}...",
            "narrative_processing",
            0.6
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing narrative input: {str(e)}", exc_info=True)
        context.metrics.last_error = str(e)
        raise
    finally:
        execution_time = time.time() - start_time
        context.metrics.record_run(success, execution_time, tokens)
        
        # Record to performance monitor
        context.performance_monitor.record_token_usage(tokens.get("total", 0))

@with_governance(
    agent_type=AgentType.STORY_DIRECTOR,
    action_type="advance_story",
    action_description="Advanced story based on player actions"
)
@track_performance("advance_story")  # NEW: Performance tracking
async def advance_story(agent: Agent, context: StoryDirectorContext, player_actions: str) -> Any:
    """Advance the story based on player actions"""
    # Invalidate state cache since the story is advancing
    context.invalidate_cache("current_state")
    
    # Measure execution time
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        # NEW: Get comprehensive context with player actions
        comprehensive_context = await context.get_comprehensive_context(player_actions)
        
        # NEW: Get relevant memories related to player actions
        relevant_memories = await context.get_relevant_memories(player_actions, limit=3)
        memory_text = ""
        if relevant_memories:
            memory_text = "Relevant memories:\n" + "\n".join([
                f"- {memory.get('content', '')[:100]}..." for memory in relevant_memories
            ])
        
        # Build enhanced prompt with comprehensive context
        prompt = f"""
        The player has taken the following actions:
        
        {player_actions}
        
        {memory_text}
        
        How should the story advance? Consider:
        1. What conflicts should progress or resolve?
        2. What narrative events should occur?
        3. How should character relationships evolve?
        4. What are the resource implications?
        5. How does this affect the overall narrative progression?
        
        Your response should include concrete recommendations for
        advancing the story based on these player actions.
        """
        
        with trace(workflow_name="StoryDirector", group_id=DEFAULT_TRACING_GROUP):
            operation = lambda: Runner.run(agent, prompt, context=context)
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
        
        # NEW: Create a memory about this story advancement
        await context.add_narrative_memory(
            f"Advanced story based on player actions: {player_actions[:100]}...",
            "story_advancement",
            0.7
        )
        
        return result
    except Exception as e:
        logger.error(f"Error advancing story: {str(e)}", exc_info=True)
        context.metrics.last_error = str(e)
        raise
    finally:
        execution_time = time.time() - start_time
        context.metrics.record_run(success, execution_time, tokens)
        
        # Record to performance monitor
        context.performance_monitor.record_token_usage(tokens.get("total", 0))

# ----- Utility Functions -----

def get_story_director_metrics(context: StoryDirectorContext) -> Dict[str, Any]:
    """Get metrics for the Story Director agent"""
    # NEW: Add performance metrics from context
    base_metrics = context.metrics.dict()
    
    if context.performance_monitor:
        perf_metrics = context.performance_monitor.get_metrics()
        base_metrics["performance"] = perf_metrics
    
    return base_metrics

@with_governance_permission(AgentType.STORY_DIRECTOR, "reset_story_director")
async def reset_story_director(context: StoryDirectorContext) -> None:
    """Reset the Story Director's state"""
    context.invalidate_cache()
    context.metrics = StoryDirectorMetrics()
    context.last_state_update = None
    context.last_context_version = None  # Reset version tracking
    logger.info(f"Reset story director for user {context.user_id}, conversation {context.conversation_id}")
    
    # NEW: Invalidate context cache
    context_cache.invalidate(f"story_state:{context.user_id}:{context.conversation_id}")
    
    # Optional: Reload managers
    context.__post_init__()
    await context.initialize_context_components()

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

@track_performance("check_narrative_opportunities")  # NEW: Performance tracking
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
        
        # NEW: Get context directly from the context service
        context_service = await get_context_service(user_id, conversation_id)
        comprehensive_context = await context_service.get_context(
            input_text="check for narrative opportunities",
            use_vector_search=True
        )
        
        # Add to context for story state analysis
        context.last_context_version = comprehensive_context.get("version")
        
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
            "narrative_stage": story_state.get("narrative_stage", {}).get("name", "Unknown"),
            "context_version": context.last_context_version
        }
    except Exception as e:
        logging.error(f"Error checking narrative opportunities: {e}")
        return {
            "checked": True,
            "error": str(e),
            "opportunities_found": False
        }
