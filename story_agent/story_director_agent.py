# story_agent/story_director_agent.py

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime

from agents import Agent, function_tool, Runner, trace, handoff, ModelSettings, RunContextWrapper, FunctionTool

# Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
# REMOVED: from nyx.integrate import get_central_governance  <-- Causes circular import

# NEW: Context system integration
from context.context_service import get_context_service, get_comprehensive_context
from context.context_config import get_config
from context.memory_manager import get_memory_manager, Memory
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
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
    
    # NEW: Context management components
    context_service: Optional[Any] = None
    memory_manager: Optional[Any] = None
    vector_service: Optional[Any] = None
    performance_monitor: Optional[Any] = None
    context_manager: Optional[Any] = None
    
    # NEW: Version tracking for delta updates
    last_context_version: Optional[int] = None
    
    def __post_init__(self):
        """Synchronous post-init; cannot contain 'await'."""
        if not self.conflict_manager:
            from logic.conflict_system.conflict_integration import ConflictSystemIntegration
            self.conflict_manager = ConflictSystemIntegration(self.user_id, self.conversation_id)
        if not self.resource_manager:
            from logic.resource_management import ResourceManager
            self.resource_manager = ResourceManager(self.user_id, self.conversation_id)
        if not self.activity_analyzer:
            from logic.activity_analyzer import ActivityAnalyzer
            self.activity_analyzer = ActivityAnalyzer(self.user_id, self.conversation_id)
        
        # We'll do governance or further async initialization in an async method below.
    
    async def initialize_context_components(self):
        """Initialize context components that require async calls."""
        self.context_service = await get_context_service(self.user_id, self.conversation_id)
        self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        self.vector_service = await get_vector_service(self.user_id, self.conversation_id)

        # Initialize performance monitor and context manager
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        self.context_manager = get_context_manager()
        
        # Subscribe to changes
        self.context_manager.subscribe_to_changes("/narrative_stage", self.handle_narrative_stage_change)
        self.context_manager.subscribe_to_changes("/conflicts", self.handle_conflict_change)

        # Retrieve governance inside an async method to avoid circular imports
        from nyx.directive_handler import DirectiveHandler  # already imported, but ensuring local usage
        # local import of get_central_governance if needed:
        # from nyx.integrate import get_central_governance
        
        # If you do need governance here, do it locally:
        # governance = await get_central_governance(self.user_id, self.conversation_id)
        
        # Initialize the directive handler
        self.directive_handler = DirectiveHandler(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            agent_type=AgentType.STORY_DIRECTOR,
            agent_id="director",
            governance=None  # Or pass in an instance if needed
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
                    
                    # Local import to avoid top-level circular dependency
                    from nyx.integrate import get_central_governance
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
            
            from logic.narrative_progression import advance_narrative_stage
            result = await advance_narrative_stage(self.user_id, self.conversation_id, stage_name)
            return {"result": "narrative_advanced", "data": result}
        
        elif "retrieve context" in instruction.lower():
            # Handle context retrieval
            params = directive.get("parameters", {})
            input_text = params.get("input_text", "")
            use_vector = params.get("use_vector", True)
            
            if not self.context_service:
                await self.initialize_context_components()
                
            context_data = await self.context_service.get_context(
                input_text=input_text,
                use_vector_search=use_vector
            )
            
            return {"result": "context_retrieved", "data": context_data}
        
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
        if not self.context_service:
            await self.initialize_context_components()
        
        config = await get_config()
        context_budget = config.get_token_budget("default")
        use_vector = config.is_enabled("use_vector_search")
        
        if self.last_context_version is not None:
            context_data = await self.context_service.get_context(
                input_text=input_text,
                context_budget=context_budget,
                use_vector_search=use_vector,
                use_delta=True,
                source_version=self.last_context_version
            )
        else:
            context_data = await self.context_service.get_context(
                input_text=input_text,
                context_budget=context_budget,
                use_vector_search=use_vector,
                use_delta=False
            )
        
        if "version" in context_data:
            self.last_context_version = context_data["version"]
        
        return context_data
    
    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant memories using vector search"""
        if not self.memory_manager:
            await self.initialize_context_components()
        
        memories = await self.memory_manager.search_memories(
            query_text=query,
            limit=limit,
            use_vector=True
        )
        
        memory_dicts = []
        for memory in memories:
            if hasattr(memory, 'to_dict'):
                memory_dicts.append(memory.to_dict())
            else:
                memory_dicts.append(memory)
        
        return memory_dicts

async def retry_operation(operation, max_retries=3):
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
                wait_time = 1 * (2 ** (retries - 1))
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
    
    from story_agent.tools import story_tools, conflict_tools, resource_tools, narrative_tools
    from story_agent.specialized_agents import initialize_specialized_agents
    from story_agent.tools import context_tools
    
    from story_agent.story_director_agent import get_story_state_tool

    all_tools = [
        get_story_state_tool,
        *story_tools,
        *conflict_tools,
        *resource_tools,
        *narrative_tools,
        *context_tools,
    ]

    agent = Agent(
        name="Story Director",
        instructions=agent_instructions,
        tools=all_tools,
        handoffs=list(specialized_agents.values()),
        model="gpt-4o",
        model_settings=ModelSettings(temperature=0.2, max_tokens=2048),
    )
    return agent
    
# ----- Functional Interface -----

async def initialize_story_director(user_id: int, conversation_id: int) -> Tuple[Agent, StoryDirectorContext]:
    """Initialize the Story Director Agent with context"""
    context = StoryDirectorContext(user_id=user_id, conversation_id=conversation_id)
    agent = create_story_director_agent()
    
    # Initialize context components (async)
    await context.initialize_context_components()
    
    # Start background processing of directives
    await context.directive_handler.start_background_processing()
    
    return agent, context

from nyx.governance_helpers import with_governance_permission

@with_governance_permission(AgentType.STORY_DIRECTOR, "reset_story_director")
async def reset_story_director(ctx: RunContextWrapper[StoryDirectorContext]) -> None:
    context = ctx.context
    """Reset the Story Director's state"""
    context.invalidate_cache()
    context.metrics = StoryDirectorMetrics()
    context.last_state_update = None
    context.last_context_version = None
    logger.info(f"Reset story director for user {context.user_id}, conversation {context.conversation_id}")
    
    context_cache.invalidate(f"story_state:{context.user_id}:{context.conversation_id}")
    
    context.__post_init__()  # re-run sync init to wipe references
    await context.initialize_context_components()

async def _invoke_get_story_state(
    ctx: RunContextWrapper[Any],
    args_json: str
) -> str:
    # pull out your StoryDirectorContext
    context = ctx.context
    # build (or re-use) your Story Director agent
    agent = create_story_director_agent()
    # call your existing logic
    result = await get_current_story_state(agent, context)
    # ensure we return a string
    return json.dumps(result) if not isinstance(result, str) else result

# 2) Manual FunctionTool with a fully-declared JSON schema
get_story_state_tool = FunctionTool(
    name="get_story_state",
    description=(
        "Returns the current state of the story for the user "
        "in the conversation as a JSON string."
    ),
    params_json_schema={
        "type": "object",
        "properties": {
            "ctx": {
                "type": "object",
                "description": (
                    "The context wrapper object containing user and "
                    "conversation information"
                )
            }
        },
        "required": ["ctx"],
    },
    on_invoke_tool=_invoke_get_story_state,
)


@function_tool
async def get_story_state(ctx: RunContextWrapper[Any]) -> str:
    """
    Get the current state of the story for the user in the conversation.

    Returns:
        A JSON string summarizing the current narrative stage, conflicts, and other story elements.
    """
    # Extract the inner context
    context = ctx.context

    # Create or reuse your Story Director agent
    agent = create_story_director_agent()

    # Delegate to your existing logic
    result = await get_current_story_state(agent, context)

    # Always return a string
    return json.dumps(result) if not isinstance(result, str) else result

@track_performance("get_story_state")
@with_action_reporting(agent_type=AgentType.STORY_DIRECTOR, action_type="get_story_state")
async def get_current_story_state(agent: Agent, ctx: Union[RunContextWrapper[StoryDirectorContext], StoryDirectorContext]) -> Any:
    """Get the current state of the story with caching"""
    # Extract the actual context from the wrapper or use directly if already a StoryDirectorContext
    if isinstance(ctx, RunContextWrapper):
        context = ctx.context
    else:
        context = ctx
        
    # Rest of the function remains the same
    cached_state = context.get_from_cache("current_state", max_age_seconds=60)
    if cached_state:
        logger.info(f"Using cached story state for user {context.user_id}, conversation {context.conversation_id}")
        return cached_state
    
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        comprehensive_context = await context.get_comprehensive_context()
        
        prompt = """
        Analyze the current state of the story and provide a detailed report.
        
        Include information about:
        1. The narrative stage
        2. Active conflicts
        3. Player resources
        4. Potential narrative events that might occur soon
        5. Key NPCs and their relationships
        """
        
        with trace(workflow_name="StoryDirector", group_id=DEFAULT_TRACING_GROUP):
            operation = lambda: Runner.run(agent, prompt, context=context)
            result = await retry_operation(operation)
        
        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'usage'):
                    tokens = {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
        
        success = True
        
        context.add_to_cache("current_state", result)
        context.last_state_update = datetime.now()
        
        await context.add_narrative_memory(
            "Analyzed current story state and identified key elements",
            "story_analysis",
            0.5
        )
        
        return result
    except Exception as e:
        logger.error(f"Error getting story state: {str(e)}", exc_info=True)
        context.metrics.last_error = str(e)
        raise
    finally:
        execution_time = time.time() - start_time
        context.metrics.record_run(success, execution_time, tokens)
        context.performance_monitor.record_token_usage(tokens.get("total", 0))

@track_performance("process_narrative_input")
@with_action_reporting(agent_type=AgentType.STORY_DIRECTOR, action_type="process_narrative_input")
async def process_narrative_input(agent: Agent, ctx: RunContextWrapper[StoryDirectorContext], narrative_text: str) -> Any:
    context = ctx.context
    """Process narrative input to determine if it should generate conflicts or narrative events"""
    context.invalidate_cache("current_state")
    
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        comprehensive_context = await context.get_comprehensive_context(narrative_text)
        
        relevant_memories = await context.get_relevant_memories(narrative_text, limit=3)
        memory_text = ""
        if relevant_memories:
            memory_text = "Relevant memories:\n" + "\n".join([
                f"- {mem.get('content', '')[:100]}..." for mem in relevant_memories
            ])
        
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
        
        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'usage'):
                    tokens = {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
        
        success = True
        
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
        context.performance_monitor.record_token_usage(tokens.get("total", 0))

@track_performance("advance_story")
@with_action_reporting(agent_type=AgentType.STORY_DIRECTOR, action_type="advance_story")
async def advance_story(agent: Agent, ctx: RunContextWrapper[StoryDirectorContext], player_actions: str) -> Any:
    context = ctx.context
    """Advance the story based on player actions"""
    context.invalidate_cache("current_state")
    
    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    try:
        comprehensive_context = await context.get_comprehensive_context(player_actions)
        
        relevant_memories = await context.get_relevant_memories(player_actions, limit=3)
        memory_text = ""
        if relevant_memories:
            memory_text = "Relevant memories:\n" + "\n".join([
                f"- {mem.get('content', '')[:100]}..." for mem in relevant_memories
            ])
        
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
        
        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'usage'):
                    tokens = {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
        
        success = True
        
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
        context.performance_monitor.record_token_usage(tokens.get("total", 0))

def get_story_director_metrics(context: StoryDirectorContext) -> Dict[str, Any]:
    """Get metrics for the Story Director agent"""
    base_metrics = context.metrics.dict()
    if context.performance_monitor:
        perf_metrics = context.performance_monitor.get_metrics()
        base_metrics["performance"] = perf_metrics
    return base_metrics

async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register the Story Director Agent with the Nyx governance system.
    """
    try:
        # Local import to avoid top-level dependency cycle
        from nyx.integrate import get_central_governance
        
        governance = await get_central_governance(user_id, conversation_id)
        
        # Possibly re-initialize or retrieve existing agent/context
        agent, _ = await initialize_story_director(user_id, conversation_id)
        
        await governance.register_agent(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_instance=agent,
            agent_id="director"
        )
        
        await governance.issue_directive(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_id="director", 
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Monitor story state and generate conflicts as appropriate",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60
        )
        
        logging.info(
            f"StoryDirector registered with Nyx governance system "
            f"for user {user_id}, conversation {conversation_id}"
        )
    except Exception as e:
        logging.error(f"Error registering StoryDirector with governance: {e}")

@track_performance("check_narrative_opportunities")
async def check_narrative_opportunities(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Check for narrative opportunities that could advance the story.
    This is called periodically by the maintenance system.
    """
    try:
        agent, context = await initialize_story_director(user_id, conversation_id)
        
        context_service = await get_context_service(user_id, conversation_id)
        comprehensive_context = await context_service.get_context(
            input_text="check for narrative opportunities",
            use_vector_search=True
        )
        
        context.last_context_version = comprehensive_context.get("version")
        
        story_state = await get_current_story_state(agent, context)
        
        from nyx.integrate import get_central_governance
        governance = await get_central_governance(user_id, conversation_id)
        
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
class StoryDirector:
    """Class that encapsulates Story Director functionality"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent = None
        self.context = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the agent and context"""
        if not self.initialized:
            self.agent, self.context = await initialize_story_director(
                self.user_id, self.conversation_id
            )
            self.initialized = True
        return self
        
    async def get_current_state(self):
        """Get the current story state"""
        if not self.initialized:
            await self.initialize()
        return await get_current_story_state(self.agent, self.context)
    
    async def process_input(self, narrative_text: str):
        """Process narrative input"""
        if not self.initialized:
            await self.initialize()
        return await process_narrative_input(self.agent, self.context, narrative_text)
    
    async def advance_story(self, player_actions: str):
        """Advance the story based on player actions"""
        if not self.initialized:
            await self.initialize()
        return await advance_story(self.agent, self.context, player_actions)
    
    async def reset(self):
        """Reset the story director state"""
        if self.initialized and self.context:
            await reset_story_director(self.context)
        
    def get_metrics(self):
        """Get story director metrics"""
        if self.context:
            return get_story_director_metrics(self.context)
        return {}
