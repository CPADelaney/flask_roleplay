# story_agent/story_director_agent.py

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime, timezone # Import timezone
from db.connection import get_db_connection_context

from agents import Agent, function_tool, Runner, trace, handoff, ModelSettings, RunContextWrapper, FunctionTool

# Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
# REMOVED: from nyx.integrate import get_central_governance  <-- Causes circular import

# NEW: Context system integration
from context.context_service import get_context_service, get_comprehensive_context
from context.context_config import get_config
from context.memory_manager import get_memory_manager, Memory, add_memory_tool, MemoryAddRequest
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

from logic.conflict_system.hooks import (
    check_and_generate_conflict,
    get_player_conflicts,
    advance_conflict_story,
    trigger_conflict_from_event,
    get_conflict_summary,
    get_world_tension_level,
    on_npc_relationship_change,
    on_faction_power_shift,
    on_resource_crisis,
    on_player_major_action
)
from logic.conflict_system.dynamic_stakeholder_agents import process_conflict_stakeholder_turns
from logic.conflict_system.enhanced_conflict_generation import analyze_conflict_pressure
from logic.conflict_system.enhanced_conflict_generation import generate_organic_conflict
from logic.conflict_system.dynamic_stakeholder_agents import force_stakeholder_action


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
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # Use timezone aware datetime

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
        if self.total_runs > 0: # Avoid division by zero
            self.average_response_time = (
                (self.average_response_time * (self.total_runs - 1) + response_time) /
                self.total_runs
            )
        else:
            self.average_response_time = response_time # First run

        self.last_run_time = datetime.now(timezone.utc) # Use timezone aware datetime

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
    directive_handler: Optional[DirectiveHandler] = None # Add directive handler here

    # NEW: Version tracking for delta updates
    last_context_version: Optional[int] = None

    # NOTE: Removed 'initialized' flag here. Initialization is tracked by the StoryDirector class if needed.

    def __post_init__(self):
        """Synchronous post-init; cannot contain 'await'."""
        # Ensure managers are initialized synchronously if possible
        # Avoid circular imports at module level
        try:
            from logic.conflict_system.conflict_integration import ConflictSystemIntegration
            if not self.conflict_manager:
                self.conflict_manager = ConflictSystemIntegration(self.user_id, self.conversation_id)
        except ImportError:
            logger.warning("ConflictSystemIntegration not found.")
            self.conflict_manager = None # Explicitly set to None if import fails

        try:
            from logic.resource_management import ResourceManager
            if not self.resource_manager:
                self.resource_manager = ResourceManager(self.user_id, self.conversation_id)
        except ImportError:
             logger.warning("ResourceManager not found.")
             self.resource_manager = None

        try:
            from logic.activity_analyzer import ActivityAnalyzer
            if not self.activity_analyzer:
                self.activity_analyzer = ActivityAnalyzer(self.user_id, self.conversation_id)
        except ImportError:
             logger.warning("ActivityAnalyzer not found.")
             self.activity_analyzer = None

        # We'll do governance or further async initialization in an async method below.

    async def initialize_context_components(self):
        """Initialize context components that require async calls."""
        # Ensure components are initialized only once if needed, or re-initialized safely
        if self.context_service is None:
            self.context_service = await get_context_service(self.user_id, self.conversation_id)
        if self.memory_manager is None:
            self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        if self.vector_service is None:
            self.vector_service = await get_vector_service(self.user_id, self.conversation_id)

        # Initialize performance monitor and context manager (these use singletons)
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        self.context_manager = get_context_manager() # Assuming this doesn't need async init per-instance

        # Subscribe to changes (make sure this is idempotent or handled correctly if called multiple times)
        # Consider unsubscribing first if re-initializing might happen.
        self.context_manager.subscribe_to_changes("/narrative_stage", self.handle_narrative_stage_change)
        self.context_manager.subscribe_to_changes("/conflicts", self.handle_conflict_change)

        # Initialize the directive handler if not already done
        if self.directive_handler is None:
            # Retrieve governance inside an async method to avoid circular imports
            # local import of get_central_governance if needed:
            # from nyx.integrate import get_central_governance

            # If you do need governance here, do it locally:
            # governance = await get_central_governance(self.user_id, self.conversation_id)

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
            
            # Start background processing
            await self.directive_handler.start_background_processing()
            logger.info("Directive handler initialized and started.")
        else:
            logger.info("Directive handler already initialized.")

    async def handle_narrative_stage_change(self, changes: List[ContextDiff]):
        """React to changes in narrative stage"""
        logging.info(f"Narrative stage changed: {changes}")
        for change in changes:
            if change.operation in ("add", "replace"):
                stage_info = change.value
                if isinstance(stage_info, dict) and "name" in stage_info:
                    # Create a memory about the stage change
                    try:
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
                    except Exception as e:
                        logger.error(f"Error handling narrative stage change notification: {e}", exc_info=True)


    async def handle_conflict_change(self, changes: List[ContextDiff]):
        """React to changes in conflicts"""
        logging.info(f"Conflict changed: {changes}")
        for change in changes:
            if change.operation == "add":
                conflict_info = change.value
                if isinstance(conflict_info, dict) and "conflict_name" in conflict_info:
                    # Create a memory about the new conflict
                    try:
                        await self.add_narrative_memory(
                            f"New conflict emerged: {conflict_info['conflict_name']}",
                            "conflict_generation",
                            0.7
                        )
                    except Exception as e:
                        logger.error(f"Error handling conflict change notification: {e}", exc_info=True)

    async def add_narrative_memory(self, content: str, memory_type: str, importance: float = 0.5):
        """Add a memory using the memory manager"""
        if not self.memory_manager:
            logger.warning("Memory manager not initialized. Initializing now.")
            await self.initialize_context_components() # Try to initialize if needed
        if not self.memory_manager:
             logger.error("Failed to initialize memory manager, cannot add memory.")
             return
    
        try:
            # Create a MemoryAddRequest from the parameters
            request = MemoryAddRequest(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=["story_director", memory_type],
                metadata={"source": "story_director"}
            )
            
            # Directly call the internal _add_memory method
            await self.memory_manager._add_memory(request)
        except Exception as e:
            logger.error(f"Failed to add narrative memory: {e}", exc_info=True)
        
    async def handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx"""
        instruction = directive.get("instruction", "")
        logging.info(f"[StoryDirector] Processing action directive: {instruction}")
    
        try:
            if "generate conflict" in instruction.lower():
                if not self.conflict_manager:
                    logger.error("Conflict manager not available.")
                    return {"result": "error", "message": "Conflict manager not initialized"}
                
                # Get parameters
                params = directive.get("parameters", {})
                conflict_type = params.get("conflict_type", "standard")
                
                # NEW: Use LoreSystem to generate conflict
                from lore.lore_system import LoreSystem
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                
                # Create context for governance
                ctx = type('obj', (object,), {
                    'user_id': self.user_id, 
                    'conversation_id': self.conversation_id
                })()  # Note the extra () at the end to instantiate the object
                
                # Generate conflict through the proper system
                result = await self.conflict_manager.generate_conflict(conflict_type)
                
                # If the conflict manager created entities, ensure they go through canon
                if result and result.get("conflict_id"):
                    # Log the conflict generation as a canonical event
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"Generated new {conflict_type} conflict: {result.get('conflict_name', 'Unknown')}",
                            tags=["conflict", "generation", conflict_type],
                            significance=7
                        )
                
                return {"result": "conflict_generated", "data": result}
    
            elif "advance narrative" in instruction.lower():
                # Get parameters
                params = directive.get("parameters", {})
                stage_name = params.get("target_stage")
                
                # NEW: Use LoreSystem for narrative advancement
                from lore.lore_system import LoreSystem
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                
                ctx = type('obj', (object,), {
                    'user_id': self.user_id, 
                    'conversation_id': self.conversation_id
                })()  # Note the extra () at the end to instantiate the object
                
                try:
                    # The narrative progression module should be refactored to use LoreSystem
                    from logic.narrative_progression import advance_narrative_stage
                    result = await advance_narrative_stage(self.user_id, self.conversation_id, stage_name)
                    
                    # Log the progression as canonical
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"Narrative stage advanced to: {stage_name}",
                            tags=["narrative", "progression", stage_name.lower().replace(" ", "_")],
                            significance=9
                        )
                    
                    return {"result": "narrative_advanced", "data": result}
                except ImportError:
                    logger.error("narrative_progression module not found.")
                    return {"result": "error", "message": "Narrative progression module not available"}
                except Exception as e:
                    logger.error(f"Error advancing narrative stage via directive: {e}", exc_info=True)
                    return {"result": "error", "message": str(e)}



            elif "retrieve context" in instruction.lower():
                # Handle context retrieval
                params = directive.get("parameters", {})
                input_text = params.get("input_text", "")
                use_vector = params.get("use_vector", True)

                if not self.context_service:
                    logger.warning("Context service not initialized, attempting initialization.")
                    await self.initialize_context_components()
                if not self.context_service:
                    logger.error("Context service failed to initialize.")
                    return {"result": "error", "message": "Context service not available"}

                context_data = await self.context_service.get_context(
                    input_text=input_text,
                    use_vector_search=use_vector
                )

                return {"result": "context_retrieved", "data": context_data}

            return {"result": "action_not_recognized"}

        except Exception as e:
            logger.error(f"Error handling action directive: {e}", exc_info=True)
            return {"result": "error", "message": str(e)}

    async def handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx"""
        logging.info(f"[StoryDirector] Processing override directive")

        # Extract override details
        override_action = directive.get("override_action", {})

        # Apply the override for future operations (Implementation needed)
        # Example: self.override_settings = override_action
        logger.warning(f"Override directive received, but application logic is not implemented: {override_action}")

        return {"result": "override_applied"} # Or "override_not_implemented"

    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate specific cache key or entire cache"""
        if key is None:
            self.cache.clear()
            logger.debug("Cleared entire story director context cache.")
        elif key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache key: {key}")

    def get_from_cache(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        entry = self.cache.get(key)
        if entry:
            if time.time() - entry['timestamp'] < max_age_seconds:
                logger.debug(f"Cache hit for key: {key}")
                return entry['value']
            else:
                # Expired entry
                logger.debug(f"Cache expired for key: {key}")
                del self.cache[key]
        else:
            logger.debug(f"Cache miss for key: {key}")
        return None

    def add_to_cache(self, key: str, value: Any) -> None:
        """Add value to cache with current timestamp"""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        logger.debug(f"Added value to cache for key: {key}")

    # NEW: Enhanced context management methods

    async def get_comprehensive_context(self, input_text: str = "") -> Dict[str, Any]:
        """Get comprehensive context using the context service"""
        if not self.context_service:
            logger.warning("Context service not initialized. Initializing now.")
            await self.initialize_context_components() # Ensure it's initialized
        if not self.context_service:
            logger.error("Context service failed to initialize. Cannot get comprehensive context.")
            return {"error": "Context service unavailable"}

        try:
            config = await get_config() # Assuming get_config is async or sync ok
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
        except Exception as e:
            logger.error(f"Error getting comprehensive context: {e}", exc_info=True)
            return {"error": f"Failed to get comprehensive context: {e}"}


    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant memories using vector search"""
        if not self.memory_manager:
            logger.warning("Memory manager not initialized. Initializing now.")
            await self.initialize_context_components() # Ensure it's initialized
        if not self.memory_manager:
            logger.error("Memory manager failed to initialize. Cannot get relevant memories.")
            return []

        try:
            memories = await self.memory_manager.search_memories(
                query_text=query,
                limit=limit,
                use_vector=True
            )

            memory_dicts = []
            for memory in memories:
                # Convert Memory objects (or dicts) to dicts safely
                if hasattr(memory, 'to_dict') and callable(memory.to_dict):
                    memory_dicts.append(memory.to_dict())
                elif isinstance(memory, dict):
                    memory_dicts.append(memory)
                else:
                     logger.warning(f"Unexpected memory type found: {type(memory)}")

            return memory_dicts
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}", exc_info=True)
            return []

@function_tool
async def check_for_conflict_opportunity(ctx: RunContextWrapper[StoryDirectorContext]) -> Dict[str, Any]:
    """
    Check if the current narrative state suggests a conflict should emerge.
    This is called by the agent when it thinks a conflict might be appropriate.
    """
    context = ctx.context

    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    
    # Get world state analysis
    conflict_system = await ConflictSystemIntegration.get_instance(
        context.user_id, 
        context.conversation_id
    )
    
    # Analyze current pressure
    from logic.conflict_system.enhanced_conflict_generation import analyze_conflict_pressure
    pressure_analysis = await analyze_conflict_pressure(ctx)
    
    # Get current conflicts to avoid oversaturation
    active_conflicts = await conflict_system.get_active_conflicts()
    
    return {
        "pressure_analysis": pressure_analysis,
        "active_conflicts": len(active_conflicts),
        "recommendation": pressure_analysis.get("recommended_action", ""),
        "pressure_score": pressure_analysis.get("total_pressure", 0)
    }

@function_tool
async def generate_conflict(
    ctx: RunContextWrapper[StoryDirectorContext], 
    conflict_type: str = None,
    reason: str = ""
) -> Dict[str, Any]:
    """
    Generate a new conflict when the narrative calls for it.
    """
    context = ctx.context
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    
    # Use the conflict system to generate
    conflict_system = await ConflictSystemIntegration.get_instance(
        context.user_id, 
        context.conversation_id
    )
    
    result = await conflict_system.generate_conflict(conflict_type)
    
    if result.get("success"):
        # Add to narrative memory
        await context.add_narrative_memory(
            f"New conflict emerged: {result.get('conflict_details', {}).get('conflict_name', 'Unknown')} - {reason}",
            "conflict_generation",
            0.8
        )
    
    return result

@function_tool  
async def advance_conflict_naturally(
    ctx: RunContextWrapper[StoryDirectorContext],
    conflict_id: int,
    narrative_development: str
) -> Dict[str, Any]:
    """
    Advance a conflict based on narrative developments.
    """
    context = ctx.context
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    
    conflict_system = await ConflictSystemIntegration.get_instance(
        context.user_id, 
        context.conversation_id
    )
    
    # Handle the narrative development's impact on the conflict
    result = await conflict_system.handle_story_beat(
        conflict_id,
        "narrative", # path_id
        narrative_development,
        [] # involved NPCs will be determined by the system
    )
    
    return result

@function_tool
async def monitor_conflicts(ctx: RunContextWrapper[StoryDirectorContext]) -> Dict[str, Any]:
    """
    Monitor world state and generate conflicts if appropriate.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Check world tension
    tension_level = await get_world_tension_level(user_id, conversation_id)
    
    # Check and potentially generate new conflict
    new_conflict = await check_and_generate_conflict(user_id, conversation_id)
    
    # Get active player conflicts
    player_conflicts = await get_player_conflicts(user_id, conversation_id)
    
    # Process stakeholder turns for each active conflict
    stakeholder_actions = {}
    for conflict in player_conflicts:
        conflict_id = conflict['conflict_id']
        actions = await process_conflict_stakeholder_turns(
            ctx, 
            conflict_id
        )
        if actions:
            stakeholder_actions[conflict_id] = actions
    
    return {
        "world_tension": tension_level,
        "new_conflict": new_conflict,
        "active_conflicts": len(player_conflicts),
        "stakeholder_actions": stakeholder_actions
    }

@function_tool
async def evolve_conflict_from_event(
    ctx: RunContextWrapper[StoryDirectorContext], 
    event_description: str,
    event_type: str = "player_action",
    involved_npcs: List[int] = None
) -> Dict[str, Any]:
    """
    Evolve conflicts based on story events.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Get active conflicts
    player_conflicts = await get_player_conflicts(user_id, conversation_id)
    
    results = []
    for conflict in player_conflicts:
        # Advance the conflict story
        result = await advance_conflict_story(
            user_id,
            conversation_id,
            conflict['conflict_id'],
            event_description,
            involved_npcs
        )
        results.append(result)
    
    return {
        "conflicts_evolved": len(results),
        "evolution_results": results
    }

@function_tool
async def trigger_conflict_event(
    ctx: RunContextWrapper[StoryDirectorContext],
    event_type: str,
    event_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Trigger conflict generation from specific events.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Determine preferred scale based on event type
    scale_map = {
        "npc_betrayal": "personal",
        "faction_dispute": "local", 
        "resource_shortage": "regional",
        "major_revelation": "world"
    }
    
    preferred_scale = scale_map.get(event_type)
    
    new_conflict = await trigger_conflict_from_event(
        user_id,
        conversation_id,
        event_type,
        event_data,
        preferred_scale
    )
    
    return new_conflict


async def retry_operation(operation, max_retries=MAX_RETRY_ATTEMPTS): # Use constant
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
                wait_time = RETRY_DELAY_SECONDS * (2 ** (retries - 1)) # Use constant
                logger.warning(f"Attempt {retries}/{max_retries} failed, retrying after {wait_time:.2f}s: {str(e)}")
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
        3. Monitoring world tension and generating organic conflicts when appropriate
        4. Tracking conflict evolution based on player actions and story events
        5. Managing autonomous stakeholder actions within conflicts
        6. Narrative moments, personal revelations, dreams, and relationship events
        7. Resource implications of player choices in conflicts
        8. Integration of player activities with conflict progression
    
        Use the tools at your disposal to:
        - Monitor the current state of the story (use get_story_state tool)
        - Check world tension and monitor for conflict opportunities
        - Generate appropriate conflicts based on the narrative stage and world state
        - Evolve conflicts based on player actions and story events (use evolve_conflict_from_event tool)
        - Trigger specific conflicts from major events (use trigger_conflict_event tool)
        - Create narrative moments, revelations, and dreams that align with the player's current state
        - Track and manage player resources in relation to conflicts
        - Process autonomous stakeholder actions to make conflicts feel alive
    
        Always maintain the central theme: a gradual shift in power dynamics where the player character slowly loses autonomy while believing they maintain control. 
        Conflicts should reinforce this theme through manipulation, betrayal, and shifting alliances.
        
        CONFLICT GENERATION GUIDELINES:
        - Use check_for_conflict_opportunity when major events occur or tensions rise
        - Generate conflicts organically when:
          * Relationships deteriorate significantly  
          * Power dynamics shift dramatically
          * Resources become scarce
          * Secrets are revealed
          * The narrative reaches natural tension points
        - Don't force conflicts - let them emerge from the story
        - Consider the narrative stage when deciding conflict scale
        
        Use get_story_state to see active conflicts and incorporate them into your narrative decisions.
        When significant story beats occur that would affect conflicts, use advance_conflict_naturally.
        
        Remember: conflicts should feel like natural consequences of the unfolding story, not random events.
    """

    # Ensure necessary modules are imported locally if they cause circular dependencies at top level
    # Example:
    try:
        from story_agent.tools import story_tools, conflict_tools, resource_tools, narrative_tools, context_tools
        from story_agent.specialized_agents import initialize_specialized_agents
        # from story_agent.story_director_agent import get_story_state # Already defined below
        specialized_agents = initialize_specialized_agents() # Ensure this doesn't cause cycles
    except ImportError as e:
        logger.error(f"Failed to import tools or specialized agents: {e}", exc_info=True)
        # Define empty lists or handle gracefully
        story_tools, conflict_tools, resource_tools, narrative_tools, context_tools = [], [], [], [], []
        specialized_agents = {}


    all_tools = [
        get_story_state,
        update_resource,
        progress_conflict,
        monitor_conflicts,        # NEW
        evolve_conflict_from_event,  # NEW
        trigger_conflict_event,      # NEW
        story_tools,
        conflict_tools,
        resource_tools,
        narrative_tools,
        context_tools,
        check_for_conflict_opportunity,
        generate_conflict,
        evolve_conflict,
        resolve_conflict_path,
    ]

    # Filter out any None values in tools list if imports failed partially
    all_tools = [tool for tool in all_tools if tool is not None]

    agent = Agent(
        name="Story Director",
        instructions=agent_instructions,
        tools=all_tools,
        handoffs=list(specialized_agents.values()),
        model="gpt-4.1-nano", # Consider making configurable
        model_settings=ModelSettings(temperature=0.2, max_tokens=2048),
    )
    return agent

# ----- Functional Interface -----

async def initialize_story_director(user_id: int, conversation_id: int) -> Tuple[Agent, StoryDirectorContext]:
    """Initialize the Story Director Agent with context"""
    context = StoryDirectorContext(user_id=user_id, conversation_id=conversation_id) # Sync init happens here
    agent = create_story_director_agent()

    # Initialize async context components
    await context.initialize_context_components()
    logger.info(f"Story Director initialized for user {user_id}, conv {conversation_id}")

    # Note: Directive handler background processing is started within initialize_context_components

    return agent, context

from nyx.governance_helpers import with_governance_permission

@with_governance_permission(AgentType.STORY_DIRECTOR, "reset_story_director")
async def reset_story_director(ctx: Union[RunContextWrapper[StoryDirectorContext], StoryDirectorContext]) -> None:
    """Reset the Story Director's state"""
    # Handle both context types
    if isinstance(ctx, RunContextWrapper): 
        context = ctx.context
    else: 
        context = ctx

    logger.info(f"Resetting story director for user {context.user_id}, conversation {context.conversation_id}")
    
    # NEW: Log the reset as a canonical event
    from lore.core import canon
    governance_ctx = type('obj', (object,), {
        'user_id': context.user_id, 
        'conversation_id': context.conversation_id
    })()
    
    async with get_db_connection_context() as conn:
        await canon.log_canonical_event(
            governance_ctx, conn,
            "Story Director state reset initiated",
            tags=["story_director", "reset", "system"],
            significance=6
        )
    
    # Clear caches
    context.invalidate_cache()
    context.metrics = StoryDirectorMetrics()
    context.last_state_update = None
    context.last_context_version = None

    # Invalidate external cache
    context_cache.invalidate(f"story_state:{context.user_id}:{context.conversation_id}")

    # Re-run synchronous __post_init__ to reset managers
    context.__post_init__()

    # Re-initialize asynchronous components
    if context.directive_handler:
        await context.directive_handler.stop_background_processing()
        context.directive_handler = None

    await context.initialize_context_components()
    
    # Log completion
    async with get_db_connection_context() as conn:
        await canon.log_canonical_event(
            governance_ctx, conn,
            "Story Director state reset completed",
            tags=["story_director", "reset", "complete"],
            significance=5
        )
    
    logger.info(f"Story director reset complete for user {context.user_id}")


# ----- Core Agent Functions -----

@function_tool # This decorator makes it available as a tool for the Agent
@track_performance("get_story_state_tool") # Use a distinct name for the tool's performance tracking
async def get_story_state(ctx: RunContextWrapper[StoryDirectorContext]) -> StoryStateUpdate:
    """
    Tool to get the current state of the story, including active conflicts, narrative stage,
    resources, and any pending narrative events.
    """
    context: StoryDirectorContext = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Ensure managers are available, log error if not
    if not context.conflict_manager:
        logger.error("Conflict manager is not initialized in get_story_state.")
        # Decide how to handle - raise error, return error state?
        # Returning an error state within the expected model structure:
        return StoryStateUpdate(key_observations=["Error: Conflict manager not available."])
    if not context.resource_manager:
        logger.error("Resource manager is not initialized in get_story_state.")
        return StoryStateUpdate(key_observations=["Error: Resource manager not available."])

    conflict_manager = context.conflict_manager
    resource_manager = context.resource_manager
    logger.info(f"Executing get_story_state tool for user {user_id}, conv {conversation_id}")


    try:
        # --- Data Fetching ---
        # Get current narrative stage (Assume helper exists or implement/import)
        # Example placeholder:
        try:
             from logic.narrative_progression import get_current_narrative_stage as fetch_narrative_stage
             narrative_stage = await fetch_narrative_stage(user_id, conversation_id)
             stage_info = NarrativeStageInfo(name=narrative_stage.name, description=narrative_stage.description) if narrative_stage else None
        except (ImportError, AttributeError, Exception) as e:
             logger.warning(f"Could not fetch narrative stage: {e}")
             stage_info = None


        # Get active conflicts
        active_conflicts_raw = await conflict_manager.get_active_conflicts()
        conflict_infos = [ConflictInfo(**conflict) for conflict in active_conflicts_raw] # Directly validate

        # Get key NPCs (Assume helper exists or implement/import)
        # Example placeholder:
        try:
            # Assuming get_key_npcs is adapted or available
            # If get_key_npcs needs the agent/context structure, adjust call
            # from some_module import get_key_npcs_helper
            # key_npcs = await get_key_npcs_helper(user_id, conversation_id, limit=5)
             key_npcs = [{"name": "Placeholder NPC", "status": "Unknown"}] # Placeholder
             logger.warning("Using placeholder for get_key_npcs")
        except (ImportError, AttributeError, Exception) as e:
            logger.warning(f"Could not fetch key NPCs: {e}")
            key_npcs = []


        # Get player resources and vitals
        resources_raw = await resource_manager.get_resources()
        vitals_raw = await resource_manager.get_vitals()
        formatted_money = await resource_manager.get_formatted_money()
        resource_status = ResourceStatus(
            money=resources_raw.get('money', 0),
            supplies=resources_raw.get('supplies', 0),
            influence=resources_raw.get('influence', 0),
            energy=vitals_raw.get('energy', 100), # Default if missing
            hunger=vitals_raw.get('hunger', 0), # Default if missing
            formatted_money=formatted_money
        )

        # Check for narrative events (Assume helpers exist or implement/import)
        # Example placeholders:
        narrative_events = []
        try:
            # from some_module import check_for_personal_revelations_helper
            # personal_revelation = await check_for_personal_revelations_helper(user_id, conversation_id)
            personal_revelation = None # Placeholder
            if personal_revelation:
                narrative_events.append(NarrativeEvent(
                    event_type="personal_revelation",
                    content=personal_revelation, # Assuming this is already a dict
                    should_present=True, priority=8
                ))

            # from some_module import check_for_narrative_moments_helper
            # narrative_moment = await check_for_narrative_moments_helper(user_id, conversation_id)
            narrative_moment = None # Placeholder
            if narrative_moment:
                 narrative_events.append(NarrativeEvent(
                    event_type="narrative_moment",
                    content=narrative_moment, # Assuming dict
                    should_present=True, priority=9
                 ))

            # from some_module import check_for_npc_revelations_helper
            # npc_revelation = await check_for_npc_revelations_helper(user_id, conversation_id)
            npc_revelation = None # Placeholder
            if npc_revelation:
                 narrative_events.append(NarrativeEvent(
                    event_type="npc_revelation",
                    content=npc_revelation, # Assuming dict
                    should_present=True, priority=7
                 ))
            logger.warning("Using placeholders for narrative event checks")
        except (ImportError, AttributeError, Exception) as e:
             logger.warning(f"Could not check for narrative events: {e}")


        # Check for relationship events (Assume helpers exist or implement/import)
        # Example placeholders:
        try:
            # from some_module import check_for_crossroads_tool_helper
            # crossroads = await check_for_crossroads_tool_helper(user_id, conversation_id)
            crossroads = None # Placeholder

            # from some_module import check_for_ritual_tool_helper
            # ritual = await check_for_ritual_tool_helper(user_id, conversation_id)
            ritual = None # Placeholder
            logger.warning("Using placeholders for relationship event checks")
        except (ImportError, AttributeError, Exception) as e:
            logger.warning(f"Could not check for relationship events: {e}")
            crossroads, ritual = None, None


        # --- Analysis & Observation Generation ---
        key_observations = []
        current_stage_name = stage_info.name if stage_info else "Unknown"

        if current_stage_name in ["Creeping Realization", "Veil Thinning", "Full Revelation"]:
            key_observations.append(f"Player has progressed to {current_stage_name} stage, indicating significant narrative development.")
        if len(conflict_infos) > 2:
            key_observations.append(f"Player is juggling {len(conflict_infos)} active conflicts.")
        major_conflicts = [c for c in conflict_infos if c.conflict_type in ["major", "catastrophic"]]
        if major_conflicts:
            conflict_names = ", ".join([c.conflict_name for c in major_conflicts])
            key_observations.append(f"Major conflicts in progress: {conflict_names}")
        if resource_status.money < 30:
            key_observations.append("Player is low on money.")
        if resource_status.energy < 30:
            key_observations.append("Player energy is low.")
        if resource_status.hunger > 70: # Assuming higher hunger is bad
            key_observations.append("Player is significantly hungry.")

        # Determine overall story direction based on stage
        story_direction_map = {
            "Innocent Beginning": "Introduce subtle hints of control dynamics.",
            "First Doubts": "Highlight inconsistencies, raise questions.",
            "Creeping Realization": "Test boundaries, NPCs more openly manipulative.",
            "Veil Thinning": "Drop pretense more frequently, openly direct player.",
            "Full Revelation": "Explicit nature of control acknowledged.",
        }
        story_direction = story_direction_map.get(current_stage_name, "Maintain current narrative trajectory.")


        # --- Construct State Object ---
        state_data = StoryStateUpdate(
            narrative_stage=stage_info,
            active_conflicts=conflict_infos,
            narrative_events=narrative_events,
            key_npcs=key_npcs,
            resources=resource_status,
            key_observations=key_observations,
            relationship_crossroads=crossroads,
            relationship_ritual=ritual,
            story_direction=story_direction,
            last_updated=datetime.now(timezone.utc) # Ensure timezone aware
        )

        # Add a memory about retrieving the state
        try:
            await context.add_narrative_memory(
                f"Retrieved story state. Stage: {current_stage_name}. Conflicts: {len(conflict_infos)}.",
                "story_state_retrieval",
                0.4
            )
        except Exception as mem_e:
            logger.warning(f"Failed to add narrative memory after state retrieval: {mem_e}")

        # Track performance if context has performance monitor
        if context.performance_monitor:
             # Assuming memory usage tracking happens elsewhere or is integrated
             pass # context.performance_monitor.record_memory_usage() might be better placed after data fetching

        logger.info(f"Successfully executed get_story_state tool for user {user_id}")
        return state_data

    except Exception as e:
        logger.error(f"Error executing get_story_state tool: {str(e)}", exc_info=True)
        # Return a default/error state conforming to StoryStateUpdate
        return StoryStateUpdate(
             key_observations=[f"Error retrieving state: {str(e)}"],
             last_updated=datetime.now(timezone.utc)
        )


@track_performance("get_current_story_state_wrapper") # Renamed for clarity
@with_action_reporting(agent_type=AgentType.STORY_DIRECTOR, action_type="get_story_state_analysis") # Renamed action type
async def get_current_story_state(agent: Agent, ctx: Union[RunContextWrapper[StoryDirectorContext], StoryDirectorContext]) -> Any:
    """Get the current state of the story with caching using UnifiedCache."""
    if isinstance(ctx, RunContextWrapper):
        context = ctx.context
    else:
        context = ctx # Assuming ctx is StoryDirectorContext

    # Define the cache key
    cache_key = f"story_state:{context.user_id}:{context.conversation_id}"
    cache_ttl = 60 # Example TTL

    # Define the function to execute on cache miss
    async def _fetch_and_process_story_state():
        logger.debug(f"Cache miss for {cache_key}. Fetching fresh story state.")
        start_time = time.time()
        success = False
        tokens = {"prompt": 0, "completion": 0, "total": 0}
        fetched_result = None
        try:
            # --- This is the core logic from your original function ---
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
                # Note: retry_operation might be complex to put directly here.
                # Consider if the core operation needs retry or if the cache handles it.
                # For simplicity here, directly calling Runner.run.
                # If retry is essential, wrap the Runner.run call.
                MAX_AGENT_TURNS = 25  
                operation = lambda: Runner.run(
                    agent,
                    prompt,
                    context=context,
                    max_turns=MAX_AGENT_TURNS
                )
                # fetched_result = await retry_operation(operation) # Original retry call
                fetched_result = await operation() # Simpler call for example

            if hasattr(fetched_result, 'raw_responses') and fetched_result.raw_responses:
                for response in fetched_result.raw_responses:
                    if hasattr(response, 'usage'):
                        usage_data = getattr(response, 'usage', None)
                        if usage_data:
                             tokens = {
                                 "prompt": getattr(usage_data, 'prompt_tokens', 0),
                                 "completion": getattr(usage_data, 'completion_tokens', 0),
                                 "total": getattr(usage_data, 'total_tokens', 0)
                             }
                             break # Assuming one usage object is enough

            success = True
            context.last_state_update = datetime.now()

            await context.add_narrative_memory(
                "Analyzed current story state and identified key elements",
                "story_analysis", 0.5
            )
            # --- End of core logic ---
            return fetched_result # Return the fetched result

        except Exception as e:
            logger.error(f"Error fetching story state for cache: {str(e)}", exc_info=True)
            context.metrics.last_error = str(e)
            # Decide what to return on fetch error: None, raise, or an error object?
            # Returning None might cause issues if None is a valid cached value.
            # Raising might be better if the caller handles it. Let's raise.
            raise
        finally:
            # Record metrics even during fetch
            execution_time = time.time() - start_time
            # Ensure context.metrics exists
            if hasattr(context, 'metrics'):
                 context.metrics.record_run(success, execution_time, tokens)
            # Ensure performance_monitor exists
            if hasattr(context, 'performance_monitor'):
                 context.performance_monitor.record_token_usage(tokens.get("total", 0))


    # --- Call the UnifiedCache.get method CORRECTLY ---
    try:
        story_state_result = await context_cache.get(
            key=cache_key,
            fetch_func=_fetch_and_process_story_state, # Pass the async function
            ttl_override=cache_ttl,
            importance=0.8 # Example importance
        )
        # The .get method will handle calling _fetch_and_process_story_state on miss
        # and return either the cached value or the newly fetched value.
        return story_state_result

    except Exception as e:
         # Handle potential errors from the fetch_func if they weren't caught inside
         logger.error(f"Error getting story state (cache wrapper): {str(e)}", exc_info=True)
         # Re-raise or return an error state
         raise # Re-raise the exception from the fetch function

@track_performance("process_narrative_input")
@with_action_reporting(agent_type=AgentType.STORY_DIRECTOR, action_type="process_narrative_input")
async def process_narrative_input(agent: Agent, ctx: RunContextWrapper[StoryDirectorContext], narrative_text: str) -> Any:
    """Processes narrative input using the Story Director agent."""
    context = ctx.context
    context.invalidate_cache("current_state")
    context_cache.invalidate(f"story_state:{context.user_id}:{context.conversation_id}")

    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    result = None

    try:
        # Prepare context for the agent
        comprehensive_context_data = await context.get_comprehensive_context(narrative_text)
        relevant_memories = await context.get_relevant_memories(narrative_text, limit=3)
        memory_text = "Relevant memories:\n" + "\n".join([f"- {mem.get('content', '')[:150]}..." for mem in relevant_memories]) if relevant_memories else "No specific relevant memories found."

        # Construct the prompt
        prompt = f"""
        Analyze the following narrative input text and determine the appropriate story director actions.

        Narrative Input:
        \"\"\"
        {narrative_text}
        \"\"\"

        {memory_text}

        Consider the overall story theme (subtle femdom, gradual power shift) and the current state (you may need to use the `get_story_state` tool first if you don't have recent state information).

        Based on the input and context, decide if this should:
        1. Generate a new conflict related to the input (use `generate_conflict` tool with appropriate type and details).
        2. Progress an existing conflict (use `progress_conflict` tool, identifying the conflict and the impact).
        3. Trigger a specific narrative event like a revelation or moment (use `create_narrative_event` tool).
        4. Update player resources or status based on actions described (use `update_resource` tool).
        5. Simply be recorded as a memory or observation without immediate direct action.

        Provide your analysis and reasoning, then execute the chosen tool call(s) or state that no direct action is needed now.
        """

        with trace(workflow_name="StoryDirectorInput", group_id=DEFAULT_TRACING_GROUP):
            operation = lambda: Runner.run(agent, prompt, context=context)
            result = await retry_operation(operation)

        # --- TOKEN COUNTING ---
        if result and hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                 if response and hasattr(response, 'usage') and response.usage:
                     current_prompt_tokens = getattr(response.usage, 'prompt_tokens', getattr(response.usage, 'input_tokens', 0))
                     current_completion_tokens = getattr(response.usage, 'completion_tokens', getattr(response.usage, 'output_tokens', 0))
                     current_total_tokens = getattr(response.usage, 'total_tokens', 0)
                     tokens["prompt"] += current_prompt_tokens
                     tokens["completion"] += current_completion_tokens
                     tokens["total"] += (current_prompt_tokens + current_completion_tokens) if current_total_tokens == 0 else current_total_tokens
                     logger.debug(f"Accumulated tokens: Prompt={current_prompt_tokens}, Completion={current_completion_tokens}, Total={tokens['total']}")
                 else: logger.warning("ModelResponse missing valid usage object.")
        # --- END TOKEN COUNTING ---

        success = True
        
        # NEW: Use proper memory creation (already correct in original)
        await context.add_narrative_memory(
            f"Processed narrative input: {narrative_text[:100]}...", 
            "narrative_processing", 
            0.6
        )

        # If the result includes state changes, they should have gone through LoreSystem
        # via the tool calls made by the agent
        
        logger.info(f"Narrative input processed for user {context.user_id}")
        return result.final_output if result else "No result from agent run."

    except Exception as e:
        logger.error(f"Error processing narrative input for user {context.user_id}: {str(e)}", exc_info=True)
        if context.metrics: 
            context.metrics.last_error = str(e)
        success = False
        return f"Error processing narrative input: {str(e)}"
    finally:
        execution_time = time.time() - start_time
        if context.metrics: context.metrics.record_run(success, execution_time, tokens)
        if context.performance_monitor: context.performance_monitor.record_token_usage(tokens.get("total", 0))
        logger.info(f"process_narrative_input execution time: {execution_time:.4f}s, Success: {success}, Tokens: {tokens['total']}")


@track_performance("advance_story")
@with_action_reporting(agent_type=AgentType.STORY_DIRECTOR, action_type="advance_story")
async def advance_story(agent: Agent, ctx: RunContextWrapper[StoryDirectorContext], player_actions: str) -> Any:
    """Advances the story based on player actions using the Story Director agent."""
    context = ctx.context
    context.invalidate_cache("current_state")
    context_cache.invalidate(f"story_state:{context.user_id}:{context.conversation_id}")

    start_time = time.time()
    success = False
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    result = None

    try:
        # Prepare context for the agent
        comprehensive_context_data = await context.get_comprehensive_context(player_actions)
        relevant_memories = await context.get_relevant_memories(player_actions, limit=3)
        memory_text = "Relevant memories:\n" + "\n".join([f"- {mem.get('content', '')[:150]}..." for mem in relevant_memories]) if relevant_memories else "No specific relevant memories found."

        # Construct the prompt
        prompt = f"""
        The player has taken the following actions:
        \"\"\"
        {player_actions}
        \"\"\"

        {memory_text}

        Based on the current story state (use `get_story_state` tool if needed for current details) and these actions, determine how the story should advance. Maintain the subtle femdom theme and gradual power shift.

        Recommend and execute specific actions using available tools:
        - Progress or resolve existing conflicts impacted by the actions (use `progress_conflict`).
        - Generate new conflicts arising from the actions (use `generate_conflict`).
        - Trigger narrative events (revelations, moments) made relevant by the actions (use `create_narrative_event`).
        - Update player resources based on actions (e.g., spending money, gaining influence) (use `update_resource`).
        - Adjust NPC relationships or states if tools are available (or describe the needed change).

        NOTE: All updates must go through the proper LoreSystem channels.

        Provide your reasoning and execute the necessary tool calls. If multiple actions are needed, make multiple tool calls.
        """

        with trace(workflow_name="StoryDirectorAdvance", group_id=DEFAULT_TRACING_GROUP):
            operation = lambda: Runner.run(agent, prompt, context=context)
            result = await retry_operation(operation)

        # --- TOKEN COUNTING ---
        if result and hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                 if response and hasattr(response, 'usage') and response.usage:
                     current_prompt_tokens = getattr(response.usage, 'prompt_tokens', getattr(response.usage, 'input_tokens', 0))
                     current_completion_tokens = getattr(response.usage, 'completion_tokens', getattr(response.usage, 'output_tokens', 0))
                     current_total_tokens = getattr(response.usage, 'total_tokens', 0)
                     tokens["prompt"] += current_prompt_tokens
                     tokens["completion"] += current_completion_tokens
                     tokens["total"] += (current_prompt_tokens + current_completion_tokens) if current_total_tokens == 0 else current_total_tokens
                     logger.debug(f"Accumulated tokens: Prompt={current_prompt_tokens}, Completion={current_completion_tokens}, Total={tokens['total']}")
                 else: logger.warning("ModelResponse missing valid usage object.")
        # --- END TOKEN COUNTING ---

        success = True
        
        # NEW: Add memory through proper system (already correct)
        await context.add_narrative_memory(
            f"Advanced story based on player actions: {player_actions[:100]}...", 
            "story_advancement", 
            0.7
        )

        logger.info(f"Story advanced based on player actions for user {context.user_id}")
        return result.final_output if result else "No result from agent run."

    except Exception as e:
        logger.error(f"Error advancing story for user {context.user_id}: {str(e)}", exc_info=True)
        if context.metrics: 
            context.metrics.last_error = str(e)
        success = False
        return f"Error advancing story: {str(e)}"
    finally:
        execution_time = time.time() - start_time
        if context.metrics: context.metrics.record_run(success, execution_time, tokens)
        if context.performance_monitor: context.performance_monitor.record_token_usage(tokens.get("total", 0))
        logger.info(f"advance_story execution time: {execution_time:.4f}s, Success: {success}, Tokens: {tokens['total']}")


def get_story_director_metrics(context: StoryDirectorContext) -> Dict[str, Any]:
    """Get metrics for the Story Director agent"""
    if not context or not context.metrics:
         return {"error": "Context or metrics not available"}
    base_metrics = context.metrics.dict() # Use model_dump in Pydantic v2
    if context.performance_monitor:
        try:
            perf_metrics = context.performance_monitor.get_metrics()
            base_metrics["performance"] = perf_metrics
        except Exception as e:
             logger.warning(f"Could not retrieve performance monitor metrics: {e}")
             base_metrics["performance"] = {"error": str(e)}
    return base_metrics

@function_tool
async def update_resource(ctx: RunContextWrapper[StoryDirectorContext], resource_type: str, amount: int, reason: str) -> Dict[str, Any]:
    """
    Update player resources through the proper system.
    
    Args:
        resource_type: Type of resource (money, supplies, influence)
        amount: Amount to add (positive) or subtract (negative)
        reason: Reason for the change
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Get LoreSystem
    from lore.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    # Get current resources first
    resources = await context.resource_manager.get_resources()
    current_value = resources.get(resource_type, 0)
    new_value = current_value + amount
    
    # Use LoreSystem to update
    governance_ctx = type('obj', (object,), {
        'user_id': user_id, 
        'conversation_id': conversation_id
    })()
    
    result = await lore_system.propose_and_enact_change(
        ctx=governance_ctx,
        entity_type="PlayerResources",
        entity_identifier={
            "user_id": user_id, 
            "conversation_id": conversation_id, 
            "player_name": "Chase"
        },
        updates={resource_type: new_value},
        reason=reason
    )
    
    return result

@function_tool
async def progress_conflict(ctx: RunContextWrapper[StoryDirectorContext], conflict_id: int, progress_amount: float, reason: str) -> Dict[str, Any]:
    """
    Progress a conflict through the proper system.
    
    Args:
        conflict_id: ID of the conflict
        progress_amount: Amount to progress (0.0 to 1.0)
        reason: Reason for the progression
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Get LoreSystem
    from lore.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    governance_ctx = type('obj', (object,), {
        'user_id': user_id, 
        'conversation_id': conversation_id
    })()
    
    # Get current conflict state
    conflict = await context.conflict_manager.get_conflict(conflict_id)
    if not conflict:
        return {"status": "error", "message": f"Conflict {conflict_id} not found"}
    
    current_progress = conflict.get("progress", 0.0)
    new_progress = min(1.0, current_progress + progress_amount)
    
    # Determine phase based on progress
    if new_progress < 0.25:
        phase = "brewing"
    elif new_progress < 0.5:
        phase = "active"
    elif new_progress < 0.75:
        phase = "climax"
    else:
        phase = "resolution"
    
    # Update through LoreSystem
    result = await lore_system.propose_and_enact_change(
        ctx=governance_ctx,
        entity_type="Conflicts",
        entity_identifier={"conflict_id": conflict_id},
        updates={
            "progress": new_progress,
            "phase": phase,
            "updated_at": datetime.now(timezone.utc)
        },
        reason=reason
    )
    
    return result

async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register the Story Director Agent with the Nyx governance system.
    """
    try:
        # Local import to avoid top-level dependency cycle
        from nyx.integrate import get_central_governance

        governance = await get_central_governance(user_id, conversation_id)

        # Initialize or retrieve existing agent/context - Ensure init happens
        agent, context = await initialize_story_director(user_id, conversation_id)

        await governance.register_agent(
            agent_type=AgentType.STORY_DIRECTOR,
            agent_instance=agent, # Pass the agent instance
            agent_id="director" # Consistent ID
        )

        # Example directive issuance
        # --- FIX: Revert parameter names ---
        await governance.issue_directive(
            # target_agent_type=AgentType.STORY_DIRECTOR, # Incorrect parameter name
            # target_agent_id="director",                 # Incorrect parameter name
            agent_type=AgentType.STORY_DIRECTOR,  # Corrected: Use 'agent_type'
            agent_id="director",                # Corrected: Use 'agent_id'
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Periodically check for narrative opportunities and report findings.",
                "scope": "narrative_monitoring",
                "frequency": "hourly" # Example parameter
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60 # Example: 1 day
        )
        # --- END FIX ---

        logging.info(
            f"StoryDirector registered with Nyx governance system "
            f"for user {user_id}, conversation {conversation_id}"
        )
    except ImportError as e:
         logger.error(f"Failed to import Nyx components for registration: {e}", exc_info=True)
    except TypeError as te: # Catch the specific error for better logging
         logger.error(f"TypeError during governance interaction (likely incorrect parameter names): {te}", exc_info=True)
         # Potentially log the expected signature if possible or suggest checking NyxUnifiedGovernor definition
    except Exception as e:
        # Catch specific exceptions if possible (e.g., governance connection error)
        logger.error(f"Error registering StoryDirector with governance for user {user_id}: {e}", exc_info=True)


@track_performance("check_narrative_opportunities")
async def check_narrative_opportunities(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Checks for potential narrative advancement opportunities periodically.
    Called by an external system (e.g., scheduler, game loop tick).
    """
    logger.info(f"Checking narrative opportunities for user {user_id}, conv {conversation_id}")
    start_time = time.time()
    try:
        # Initialize necessary components
        agent, context = await initialize_story_director(user_id, conversation_id)

        # Get current state using the agent function (which includes caching)
        # Pass context directly, get_current_story_state will handle wrapping if needed by tools
        story_state_result = await get_current_story_state(agent, context) # Use the wrapper function

        if not isinstance(story_state_result, StoryStateUpdate):
             logger.error("Failed to get valid story state during opportunity check.")
             return {
                 "checked": True, "error": "Failed to retrieve story state",
                 "opportunities_found": False, "elapsed_time": time.time() - start_time
             }

        current_stage_name = story_state_result.narrative_stage.name if story_state_result.narrative_stage else "Unknown"

        # --- Logic to Identify Opportunities (Example) ---
        opportunities_found = False
        potential_actions = []

        # Example: If few conflicts exist in early stages, suggest generating one
        if len(story_state_result.active_conflicts) < 1 and current_stage_name in ["Innocent Beginning", "First Doubts"]:
            opportunities_found = True
            potential_actions.append("Generate introductory conflict")

        # Example: If a conflict is stalled, suggest progressing it
        stalled_conflicts = [c for c in story_state_result.active_conflicts if c.progress < 0.8 and time.time() - context.cache.get(f"conflict_{c.conflict_id}_last_progress_ts", 0) > 3600 * 6 ] # Stalled for 6 hours
        if stalled_conflicts:
            opportunities_found = True
            potential_actions.append(f"Progress stalled conflict(s): {[c.conflict_name for c in stalled_conflicts]}")

        # Example: If no narrative events pending and stage allows, check for moments/revelations
        if not story_state_result.narrative_events and current_stage_name != "Innocent Beginning":
             # Potentially call agent to *check* if a moment/revelation *should* be generated
             # This might involve another agent call with a specific prompt
             opportunities_found = True # Tentative
             potential_actions.append("Check for potential narrative moment/revelation generation")


        # --- Reporting ---
        report_result = {
            "found_opportunities": opportunities_found,
            "potential_actions": potential_actions,
            "narrative_stage": current_stage_name,
            "active_conflicts_count": len(story_state_result.active_conflicts),
            "context_version": context.last_context_version # Report last used version
        }

        # Report to Governance
        try:
            from nyx.integrate import get_central_governance
            governance = await get_central_governance(user_id, conversation_id)
            await governance.process_agent_action_report(
                agent_type=AgentType.STORY_DIRECTOR,
                agent_id="director",
                action={
                    "type": "check_narrative_opportunities",
                    "description": "Periodic check for story advancement possibilities"
                },
                result=report_result # Report findings
            )
        except ImportError:
             logger.warning("Nyx governance import failed, skipping report.")
        except Exception as gov_e:
             logger.error(f"Failed to report narrative opportunities to governance: {gov_e}")


        logger.info(f"Narrative opportunity check complete for user {user_id}. Found: {opportunities_found}. Actions: {potential_actions}")
        return {
            "checked": True,
            **report_result,
            "elapsed_time": time.time() - start_time
        }

    except Exception as e:
        logger.error(f"Error checking narrative opportunities for user {user_id}: {e}", exc_info=True)
        return {
            "checked": True, "error": str(e),
            "opportunities_found": False, "elapsed_time": time.time() - start_time
        }

# ----- StoryDirector Wrapper Class -----

class StoryDirector:
    """Class that encapsulates Story Director functionality, ensuring initialization."""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent: Optional[Agent] = None
        self.context: Optional[StoryDirectorContext] = None
        self._initialized: bool = False
        self._init_lock = asyncio.Lock() # Prevent race conditions during init

    async def initialize(self) -> bool:
        """Initialize the agent and context if not already initialized."""
        async with self._init_lock:
            if not self._initialized:
                try:
                    logger.info(f"Initializing StoryDirector instance for user {self.user_id}...")
                    self.agent, self.context = await initialize_story_director(
                        self.user_id, self.conversation_id
                    )
                    self._initialized = True
                    logger.info(f"StoryDirector instance initialized successfully for user {self.user_id}.")
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize StoryDirector for user {self.user_id}: {e}", exc_info=True)
                    self._initialized = False # Ensure it remains false on error
                    return False
            return True # Already initialized

    async def _ensure_initialized(self):
        """Internal helper to initialize if needed."""
        if not self._initialized:
            initialized_successfully = await self.initialize()
            if not initialized_successfully:
                raise RuntimeError(f"StoryDirector for user {self.user_id} could not be initialized.")
        if not self.agent or not self.context:
             raise RuntimeError(f"StoryDirector for user {self.user_id} initialized but agent/context is missing.")


    async def get_current_state(self) -> StoryStateUpdate:
        """Get the current story state, ensuring initialization."""
        await self._ensure_initialized()
        # Pass context directly, get_current_story_state expects context or RunContextWrapper
        return await get_current_story_state(self.agent, self.context) # type: ignore

    async def process_input(self, narrative_text: str) -> Any:
        """Process narrative input, ensuring initialization."""
        await self._ensure_initialized()
        # Wrap context for agent runner automatically
        return await process_narrative_input(self.agent, RunContextWrapper(context=self.context), narrative_text) # type: ignore

    async def advance_story(self, player_actions: str) -> Any:
        """Advance the story based on player actions, ensuring initialization."""
        await self._ensure_initialized()
        # Wrap context for agent runner automatically
        return await advance_story(self.agent, RunContextWrapper(context=self.context), player_actions) # type: ignore

    async def reset(self):
        """Reset the story director state, ensuring initialization first."""
        await self._ensure_initialized()
        # Pass context directly to reset function
        await reset_story_director(self.context) # type: ignore
        # State is reset, but agent/context objects might still exist.
        # Optionally force re-initialization on next call:
        # self._initialized = False

    def get_metrics(self) -> Dict[str, Any]:
        """Get story director metrics. Returns empty if not initialized."""
        if self._initialized and self.context:
            return get_story_director_metrics(self.context)
        logger.warning(f"Attempted to get metrics for uninitialized StoryDirector (user {self.user_id})")
        return {"status": "uninitialized"}

    async def shutdown(self):
        """Clean up resources, like stopping background tasks."""
        logger.info(f"Shutting down StoryDirector instance for user {self.user_id}...")
        if self._initialized and self.context and self.context.directive_handler:
            try:
                await self.context.directive_handler.stop_background_processing()
                logger.info(f"Stopped directive handler for user {self.user_id}.")
            except Exception as e:
                logger.error(f"Error stopping directive handler for user {self.user_id}: {e}", exc_info=True)
        self._initialized = False
        self.agent = None
        self.context = None
        logger.info(f"StoryDirector instance shutdown complete for user {self.user_id}.")
