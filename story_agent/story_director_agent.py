# story_agent/story_director_agent.py
from __future__ import annotations

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from logic.dynamic_relationships import OptimizedRelationshipManager
from db.connection import get_db_connection_context

from agents import Agent, function_tool, Runner, trace, handoff, ModelSettings, RunContextWrapper, FunctionTool

from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    progress_npc_narrative_stage,
    check_for_npc_revelation,
    NPC_NARRATIVE_STAGES
)
from logic.narrative_events import (
    get_relationship_overview,
    check_for_personal_revelations,
    check_for_narrative_moments,
    add_dream_sequence,
    add_moment_of_clarity,
    analyze_narrative_tone
)

# Import the new schemas
from story_agent.tools import (
    NPCInfo, RelationshipCrossroads, RelationshipRitual,
    TriggerEventData, ConflictEvolutionData,
    NarrativeEventContent, NarrativeMomentContent,
    PersonalRevelationContent, DreamSequenceContent,
    NPCRevelationContent, generate_conflict_beat
)

from story_agent.preset_story_tracker import PresetStoryTracker

# Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

# Context system integration
from context.context_service import get_context_service, get_comprehensive_context
from context.context_config import get_config
from context.memory_manager import get_memory_manager, Memory, add_memory_tool, MemoryAddRequest
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache
from context.models import MemoryMetadata  # Add this line

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
    on_player_major_action,
    resolve_conflict_path
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


# ----- Pydantic Models for Tool Outputs (Updated with ConfigDict) -----

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
    
    model_config = ConfigDict(extra="forbid")


class NarrativeStageInfo(BaseModel):
    """Information about a narrative stage"""
    name: str
    description: str
    
    model_config = ConfigDict(extra="forbid")


class NarrativeMoment(BaseModel):
    """Information about a narrative moment"""
    type: str
    name: str
    scene_text: str
    player_realization: str
    
    model_config = ConfigDict(extra="forbid")


class PersonalRevelation(BaseModel):
    """Information about a personal revelation"""
    type: str
    name: str
    inner_monologue: str
    
    model_config = ConfigDict(extra="forbid")


class ResourceStatus(BaseModel):
    """Information about player resources"""
    money: int
    supplies: int
    influence: int
    energy: int
    hunger: int
    formatted_money: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")


class NarrativeEvent(BaseModel):
    """Container for narrative events that can be returned"""
    event_type: str = Field(description="Type of narrative event (revelation, moment, dream, etc.)")
    content: NarrativeEventContent = Field(description="Content of the narrative event")
    should_present: bool = Field(description="Whether this event should be presented to the player now")
    priority: int = Field(description="Priority of this event (1-10, with 10 being highest)")
    
    model_config = ConfigDict(extra="forbid")


class StoryStateUpdate(BaseModel):
    """Container for a story state update"""
    narrative_stage: Optional[NarrativeStageInfo] = None
    active_conflicts: List[ConflictInfo] = Field(default_factory=list)
    narrative_events: List[NarrativeEvent] = Field(default_factory=list)
    key_npcs: List[NPCInfo] = Field(default_factory=list)  # Changed from List[Dict[str, Any]]
    resources: Optional[ResourceStatus] = None
    key_observations: List[str] = Field(
        default_factory=list,
        description="Key observations about the player's current state or significant changes"
    )
    relationship_crossroads: Optional[RelationshipCrossroads] = None  # Changed from Dict[str, Any]
    relationship_ritual: Optional[RelationshipRitual] = None  # Changed from Dict[str, Any]
    story_direction: str = Field(
        default="",
        description="High-level direction the story should take based on current state"
    )
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(extra="forbid")


class StoryDirectorMetrics(BaseModel):
    """Metrics for monitoring the Story Director's performance"""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    average_response_time: float = 0.0
    last_run_time: Optional[datetime] = None
    last_error: Optional[str] = None
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    
    model_config = ConfigDict(extra="forbid")

    def record_run(self, success: bool, response_time: float, tokens: Dict[str, int]) -> None:
        """Record metrics for a run"""
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1

        # Update average response time
        if self.total_runs > 0:
            self.average_response_time = (
                (self.average_response_time * (self.total_runs - 1) + response_time) /
                self.total_runs
            )
        else:
            self.average_response_time = response_time

        self.last_run_time = datetime.now(timezone.utc)

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

    # Context management components
    context_service: Optional[Any] = None
    memory_manager: Optional[Any] = None
    vector_service: Optional[Any] = None
    performance_monitor: Optional[Any] = None
    context_manager: Optional[Any] = None
    directive_handler: Optional[DirectiveHandler] = None

    preset_story_tracker: Optional[PresetStoryTracker] = None
    
    # Version tracking for delta updates
    last_context_version: Optional[int] = None

    def __post_init__(self):
        """Synchronous post-init; cannot contain 'await'."""
        # Don't initialize async components here - they need to be initialized
        # in initialize_context_components() instead
        
        # Only initialize synchronous components that don't require database or async calls
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

        relationship_manager: Optional[OptimizedRelationshipManager] = None

    async def initialize_context_components(self):
        """Initialize context components that require async calls."""
        # Initialize conflict manager using the proper async method
        if self.conflict_manager is None:
            try:
                from logic.conflict_system.conflict_integration import ConflictSystemIntegration
                self.conflict_manager = await ConflictSystemIntegration.get_instance(
                    self.user_id, 
                    self.conversation_id
                )
                logger.info(f"Conflict manager initialized for user {self.user_id}")
            except Exception as e:
                logger.error(f"Failed to initialize conflict manager: {e}")
                self.conflict_manager = None
        
        # Initialize context service
        if self.context_service is None:
            self.context_service = await get_context_service(self.user_id, self.conversation_id)
        
        # Initialize memory manager
        if self.memory_manager is None:
            self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        # Initialize vector service
        if self.vector_service is None:
            self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
    
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # Initialize context manager and subscriptions
        self.context_manager = get_context_manager()
        self.context_manager.subscribe_to_changes("/narrative_stage", self.handle_narrative_stage_change)
        self.context_manager.subscribe_to_changes("/conflicts", self.handle_conflict_change)

        # Initialize relationship manager
        if self.relationship_manager is None:
            self.relationship_manager = OptimizedRelationshipManager(
                self.user_id, 
                self.conversation_id
            )
            logger.info(f"Relationship manager initialized for user {self.user_id}")
    
        # Initialize directive handler
        if self.directive_handler is None:
            self.directive_handler = DirectiveHandler(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                agent_type=AgentType.STORY_DIRECTOR,
                agent_id="director",
                governance=None
            )
    
            self.directive_handler.register_handler(
                DirectiveType.ACTION,
                self.handle_action_directive
            )
            self.directive_handler.register_handler(
                DirectiveType.OVERRIDE,
                self.handle_override_directive
            )
            
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
                    try:
                        await self.add_narrative_memory(
                            f"Narrative stage progressed to {stage_info['name']}",
                            "narrative_progression",
                            0.8
                        )
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
            await self.initialize_context_components()
        if not self.memory_manager:
            logger.error("Failed to initialize memory manager, cannot add memory.")
            return
    
        try:
            request = MemoryAddRequest(
                user_id=self.user_id,  # Added
                conversation_id=self.conversation_id,  # Added
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=["story_director", memory_type],
                metadata=MemoryMetadata(source="story_director")  # Changed to use constructor
            )
            
            await self.memory_manager._add_memory(request)
        except Exception as e:
            logger.error(f"Failed to add narrative memory: {e}", exc_info=True)

    async def handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx"""
        try:
            instruction = directive.get("instruction", "")
            logging.info(f"[StoryDirector] Processing action directive: {instruction}")
        
            if "generate conflict" in instruction.lower():
                # Ensure conflict manager is initialized
                if not self.conflict_manager:
                    await self.initialize_context_components()
                    
                if not self.conflict_manager:
                    logger.error("Conflict manager not available after initialization attempt.")
                    return {"result": "error", "message": "Conflict manager not initialized"}
                
                params = directive.get("parameters", {})
                conflict_type = params.get("conflict_type", "standard")
                
                # Create a proper context for the call
                conflict_ctx = RunContextWrapper({
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                # Pass the context when calling generate_conflict
                result = await self.conflict_manager.generate_conflict(
                    {"conflict_type": conflict_type},  # Pass as dict
                    ctx=conflict_ctx
                )
                    
                if result and result.get("conflict_id"):
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        await canon.log_canonical_event(
                            conflict_ctx, conn,
                            f"Generated new {conflict_type} conflict: {result.get('conflict_name', 'Unknown')}",
                            tags=["conflict", "generation", conflict_type],
                            significance=7
                        )
                
                return {"result": "conflict_generated", "data": result}
    
            elif "advance narrative" in instruction.lower():
                params = directive.get("parameters", {})
                target_stage = params.get("target_stage")
                npc_ids = params.get("npc_ids", [])  # Specific NPCs to advance
                advance_all = params.get("advance_all", False)  # Advance all NPCs
                
                from lore.core.lore_system import LoreSystem
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                
                ctx = RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id
                })
                
                try:
                    from logic.npc_narrative_progression import (
                        progress_npc_narrative_stage, 
                        get_npc_narrative_stage,
                        NPC_NARRATIVE_STAGES
                    )
                    from logic.narrative_events import get_relationship_overview
                    
                    # Find the target stage
                    target_stage_obj = None
                    for stage in NPC_NARRATIVE_STAGES:
                        if stage.name == target_stage:
                            target_stage_obj = stage
                            break
                    
                    if not target_stage_obj:
                        return {"result": "error", "message": f"Unknown narrative stage: {target_stage}"}
                    
                    # Get NPCs to advance
                    if advance_all:
                        # Get all NPCs
                        overview = await get_relationship_overview(self.user_id, self.conversation_id)
                        npc_ids = [rel['npc_id'] for rel in overview.get('relationships', [])]
                    elif not npc_ids:
                        # Advance the most influential NPCs
                        overview = await get_relationship_overview(self.user_id, self.conversation_id)
                        most_advanced = overview.get('most_advanced_npcs', [])[:3]  # Top 3
                        npc_ids = [npc['npc_id'] for npc in most_advanced]
                    
                    # Progress each NPC towards the target stage
                    results = []
                    for npc_id in npc_ids:
                        # Calculate how much to increase stats to reach target stage
                        current_stage = await get_npc_narrative_stage(self.user_id, self.conversation_id, npc_id)
                        
                        # Only advance if not already at or past target
                        current_index = next((i for i, s in enumerate(NPC_NARRATIVE_STAGES) if s.name == current_stage.name), 0)
                        target_index = next((i for i, s in enumerate(NPC_NARRATIVE_STAGES) if s.name == target_stage), 0)
                        
                        if current_index < target_index:
                            # Calculate stat increases needed
                            corruption_increase = max(0, target_stage_obj.required_corruption - 10)  # Slight boost
                            dependency_increase = max(0, target_stage_obj.required_dependency - 10)
                            realization_increase = max(0, target_stage_obj.required_realization - 10)
                            
                            result = await progress_npc_narrative_stage(
                                self.user_id,
                                self.conversation_id,
                                npc_id,
                                corruption_increase,
                                dependency_increase,
                                realization_increase,
                                force_stage=target_stage if params.get("force", False) else None
                            )
                            results.append(result)
                    
                    # Log the narrative advancement
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"Advanced {len(results)} NPCs towards {target_stage} stage",
                            tags=["narrative", "progression", target_stage.lower().replace(" ", "_")],
                            significance=8
                        )
                    
                    # Check for narrative events triggered by the advancement
                    from logic.narrative_events import check_for_personal_revelations, check_for_narrative_moments
                    
                    revelation = await check_for_personal_revelations(self.user_id, self.conversation_id)
                    moment = await check_for_narrative_moments(self.user_id, self.conversation_id)
                    
                    return {
                        "result": "narrative_advanced",
                        "data": {
                            "npcs_advanced": len(results),
                            "target_stage": target_stage,
                            "results": results,
                            "triggered_revelation": revelation is not None,
                            "triggered_moment": moment is not None
                        }
                    }
                    
                except ImportError:
                    logger.error("npc_narrative_progression module not found.")
                    return {"result": "error", "message": "Narrative progression module not available"}
                except Exception as e:
                    logger.error(f"Error advancing narrative stage via directive: {e}", exc_info=True)
                    return {"result": "error", "message": str(e)}
    
            elif "trigger narrative event" in instruction.lower():
                params = directive.get("parameters", {})
                event_type = params.get("event_type", "revelation")
                
                try:
                    from logic.narrative_events import (
                        check_for_personal_revelations,
                        check_for_narrative_moments,
                        add_dream_sequence,
                        add_moment_of_clarity
                    )
                    
                    result = None
                    if event_type == "revelation":
                        result = await check_for_personal_revelations(self.user_id, self.conversation_id)
                    elif event_type == "moment":
                        result = await check_for_narrative_moments(self.user_id, self.conversation_id)
                    elif event_type == "dream":
                        result = await add_dream_sequence(self.user_id, self.conversation_id)
                    elif event_type == "clarity":
                        realization_text = params.get("realization_text")
                        result = await add_moment_of_clarity(
                            self.user_id, 
                            self.conversation_id, 
                            realization_text
                        )
                    
                    return {
                        "result": "narrative_event_triggered" if result else "no_event_triggered",
                        "data": result
                    }
                    
                except Exception as e:
                    logger.error(f"Error triggering narrative event: {e}", exc_info=True)
                    return {"result": "error", "message": str(e)}
    
            elif "analyze narrative tone" in instruction.lower():
                params = directive.get("parameters", {})
                text = params.get("text", "")
                
                try:
                    from logic.narrative_events import analyze_narrative_tone
                    
                    analysis = await analyze_narrative_tone(text)
                    
                    return {
                        "result": "tone_analyzed",
                        "data": analysis
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing narrative tone: {e}", exc_info=True)
                    return {"result": "error", "message": str(e)}
    
            elif "retrieve context" in instruction.lower():
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

    async def initialize_preset_story_tracker(self, user_id: int, conversation_id: int):
        """Initialize preset story tracker if a preset story is active"""
        async with get_db_connection_context() as conn:
            progress = await conn.fetchrow("""
                SELECT story_id, current_act, completed_beats, story_variables
                FROM PresetStoryProgress
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            if progress:
                # Load the preset story
                story_row = await conn.fetchrow(
                    "SELECT story_data FROM PresetStories WHERE story_id = $1",
                    progress['story_id']
                )
                
                if story_row:
                    from story_agent.preset_story_tracker import PresetStoryTracker
                    from story_templates.preset_stories import PresetStory
                    
                    story_data = json.loads(story_row['story_data'])
                    
                    # Reconstruct PresetStory object
                    preset_story = self._reconstruct_preset_story(story_data)
                    
                    # Initialize tracker
                    tracker = PresetStoryTracker(user_id, conversation_id)
                    tracker.current_story_id = progress['story_id']
                    tracker.current_act = progress['current_act']
                    tracker.completed_beats = json.loads(progress['completed_beats'])
                    tracker.story_variables = json.loads(progress['story_variables'])
                    
                    self.context.preset_story_tracker = tracker
                    
                    logger.info(f"Initialized preset story tracker for {progress['story_id']}")



    async def handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx"""
        logging.info(f"[StoryDirector] Processing override directive")

        override_action = directive.get("override_action", {})

        logger.warning(f"Override directive received, but application logic is not implemented: {override_action}")

        return {"result": "override_applied"}

    async def get_overall_narrative_stage(user_id: int, conversation_id: int):
        """
        Derive the overall narrative stage based on NPC relationship stages.
        Returns the most advanced stage among all NPCs.
        """
        overview = await get_relationship_overview(user_id, conversation_id)
        
        # Order stages by progression
        stage_order = {stage.name: i for i, stage in enumerate(NPC_NARRATIVE_STAGES)}
        
        most_advanced_stage = NPC_NARRATIVE_STAGES[0]  # Default to first stage
        highest_index = 0
        
        for stage_name, npcs in overview.get('by_stage', {}).items():
            if npcs and stage_name in stage_order:
                stage_index = stage_order[stage_name]
                if stage_index > highest_index:
                    highest_index = stage_index
                    # Find the actual stage object
                    for stage in NPC_NARRATIVE_STAGES:
                        if stage.name == stage_name:
                            most_advanced_stage = stage
                            break
        
        return most_advanced_stage

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

    async def get_comprehensive_context(self, input_text: str = "") -> Dict[str, Any]:
        """Get comprehensive context using the context service"""
        if not self.context_service:
            logger.warning("Context service not initialized. Initializing now.")
            await self.initialize_context_components()
        if not self.context_service:
            logger.error("Context service failed to initialize. Cannot get comprehensive context.")
            return {"error": "Context service unavailable"}

        try:
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
        except Exception as e:
            logger.error(f"Error getting comprehensive context: {e}", exc_info=True)
            return {"error": f"Failed to get comprehensive context: {e}"}

    async def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant memories using vector search"""
        if not self.memory_manager:
            logger.warning("Memory manager not initialized. Initializing now.")
            await self.initialize_context_components()
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

    async def check_and_apply_preset_beats(
        ctx: RunContextWrapper[StoryDirectorContext],
        player_action: str,
        current_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if preset beats should trigger and blend with dynamic content"""
        context = ctx.context
        
        if not context.preset_story_tracker:
            return None
            
        # Check triggers with flexibility
        triggered_beat = await context.preset_story_tracker.check_beat_triggers(current_state)
        
        if not triggered_beat:
            return None
            
        # Use balance manager to decide enforcement
        balance_manager = StoryBalanceManager(flexibility_level=0.7)
        should_enforce, reason = await balance_manager.should_enforce_preset(
            triggered_beat,
            context.recent_player_actions,
            current_state
        )
        
        if not should_enforce:
            # Store as "missed beat" for potential later callback
            await context.preset_story_tracker.mark_beat_as_skipped(triggered_beat.id)
            return None
            
        # Adapt preset content to current context
        adapted_content = await balance_manager.adapt_preset_content(
            triggered_beat.content,
            current_state
        )
        
        # Trigger through existing systems
        return await apply_preset_beat_through_systems(ctx, adapted_content)
    
    async def apply_preset_beat_through_systems(
        ctx: RunContextWrapper[StoryDirectorContext],
        adapted_beat: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply preset beat using existing backend systems"""
        context = ctx.context
        results = {}
        
        # 1. Generate conflicts if needed
        if adapted_beat.get("trigger_conflict"):
            conflict_params = adapted_beat["conflict_params"]
            # Use existing conflict generation
            conflict_result = await generate_conflict(
                ctx,
                conflict_type=conflict_params.get("type"),
                reason=f"Queen of Thorns: {adapted_beat.get('beat_name')}"
            )
            results["conflict"] = conflict_result
            
        # 2. Progress NPC relationships
        if adapted_beat.get("npc_progressions"):
            for npc_prog in adapted_beat["npc_progressions"]:
                npc_id = context.preset_story_tracker.npc_mappings[npc_prog["preset_npc_id"]]
                await progress_npc_narrative(
                    ctx,
                    npc_id,
                    corruption_change=npc_prog.get("corruption_change", 0),
                    dependency_change=npc_prog.get("dependency_change", 0),
                    realization_change=npc_prog.get("realization_change", 0),
                    reason=f"Queen of Thorns progression"
                )
                
        # 3. Trigger narrative events
        if adapted_beat.get("narrative_event"):
            event_type = adapted_beat["narrative_event"]["type"]
            if event_type == "revelation":
                npc_id = context.preset_story_tracker.npc_mappings[adapted_beat["narrative_event"]["npc_id"]]
                await generate_dynamic_personal_revelation(
                    ctx,
                    npc_id,
                    adapted_beat["narrative_event"]["revelation_type"]
                )
                
        # 4. Create memories
        await store_narrative_memory(
            ctx,
            StoreMemoryParams(
                content=f"Queen of Thorns: {adapted_beat.get('description', 'Story progression')}",
                memory_type="preset_story_beat",
                importance=0.8,
                tags=["queen_of_thorns", "preset_story", adapted_beat.get("beat_id", "unknown")]
            )
        )
        
        return results


# ----- Tool Functions (Updated with Strict Schemas) -----

@function_tool
async def process_relationship_interaction(
    ctx: RunContextWrapper[StoryDirectorContext],
    entity1_type: str,
    entity1_id: int,
    entity2_type: str, 
    entity2_id: int,
    interaction_type: str,
    context: str = "casual"
) -> Dict[str, Any]:
    """Process a relationship-affecting interaction between entities."""
    context_obj = ctx.context
    
    if not context_obj.relationship_manager:
        await context_obj.initialize_context_components()
    
    from logic.dynamic_relationships import process_relationship_interaction_tool
    
    # Process the interaction
    result = await process_relationship_interaction_tool(
        ctx,
        entity1_type,
        entity1_id,
        entity2_type,
        entity2_id,
        interaction_type,
        context
    )
    
    # Check for triggered events
    from logic.dynamic_relationships import poll_relationship_events_tool
    event_result = await poll_relationship_events_tool(ctx)
    
    if event_result["has_event"]:
        # Store event for narrative processing
        await context_obj.add_narrative_memory(
            f"Relationship event triggered: {event_result['event']['type']}",
            "relationship_event",
            0.8
        )
        result["triggered_event"] = event_result["event"]
    
    return result

@function_tool
async def get_relationship_state(
    ctx: RunContextWrapper[StoryDirectorContext],
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Dict[str, Any]:
    """Get the current state of a relationship."""
    context_obj = ctx.context
    
    if not context_obj.relationship_manager:
        await context_obj.initialize_context_components()
    
    from logic.dynamic_relationships import get_relationship_summary_tool
    
    return await get_relationship_summary_tool(
        ctx,
        entity1_type,
        entity1_id,
        entity2_type,
        entity2_id
    )

@function_tool
async def check_preset_story_progression(
    ctx: RunContextWrapper[StoryDirectorContext]
) -> Dict[str, Any]:
    """Check if preset story beats should trigger"""
    context = ctx.context
    
    if not context.preset_story_tracker:
        return {"has_preset": False}
        
    # Get current game state
    game_state = await context.get_comprehensive_context()
    
    # Check for triggered beats
    triggered_beat = await context.preset_story_tracker.check_beat_triggers(game_state)
    
    if triggered_beat:
        return {
            "has_preset": True,
            "triggered_beat": triggered_beat,
            "should_override": not triggered_beat.can_skip,
            "narrative_hints": triggered_beat.dialogue_hints
        }
        
    return {"has_preset": True, "triggered_beat": None}

@function_tool
async def check_for_conflict_opportunity(ctx: RunContextWrapper[StoryDirectorContext]) -> Dict[str, Any]:
    """
    Check if the current narrative state suggests a conflict should emerge.
    This is called by the agent when it thinks a conflict might be appropriate.
    """
    context = ctx.context

    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    
    conflict_system = await ConflictSystemIntegration.get_instance(
        context.user_id, 
        context.conversation_id
    )
    
    from logic.conflict_system.enhanced_conflict_generation import analyze_conflict_pressure
    pressure_analysis = await analyze_conflict_pressure(ctx)
    
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
    
    conflict_system = await ConflictSystemIntegration.get_instance(
        context.user_id, 
        context.conversation_id
    )
    
    result = await conflict_system.generate_conflict(conflict_type)
    
    if result.get("success"):
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
    
    result = await conflict_system.handle_story_beat(
        conflict_id,
        "narrative",
        narrative_development,
        []
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
    
    tension_level = await get_world_tension_level(user_id, conversation_id)
    
    new_conflict = await check_and_generate_conflict(user_id, conversation_id)
    
    player_conflicts = await get_player_conflicts(user_id, conversation_id)
    
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
    involved_npcs: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evolve conflicts based on story events.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    player_conflicts = await get_player_conflicts(user_id, conversation_id)
    
    results = []
    for conflict in player_conflicts:
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
    event_data: TriggerEventData  # Changed from Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Trigger conflict generation from specific events.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
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
        event_data.model_dump(),  # Convert to dict for the underlying function
        preferred_scale
    )
    
    return new_conflict


@function_tool
async def evolve_conflict(
    ctx: RunContextWrapper[StoryDirectorContext], 
    conflict_id: int,
    event_type: str,
    event_data: ConflictEvolutionData  # Changed from Dict[str, Any]
) -> Dict[str, Any]:
    """
    Wrapper for evolving a conflict based on events.
    """
    context = ctx.context
    
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    
    conflict_system = await ConflictSystemIntegration.get_instance(
        context.user_id, 
        context.conversation_id
    )
    
    result = await conflict_system.evolve_conflict(
        conflict_id,
        event_type,
        event_data.model_dump()  # Convert to dict for the underlying function
    )
    
    return result


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
                logger.warning(f"Attempt {retries}/{max_retries} failed, retrying after {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed: {str(e)}")

    raise last_exception


# ----- Main Agent Creation Function -----

def create_story_director_agent():
    """Create the Story Director Agent with all required tools"""

    agent_instructions = """
        You are the Story Director, managing dynamic narrative, preset story beats, and complex character relationships.
        
        When a preset story is active:
        1. Check for triggered story beats using check_preset_story_progression
        2. Guide the narrative toward preset waypoints while maintaining organic flow
        3. Ensure required NPCs and locations are involved appropriately
        4. Balance player agency with story requirements
        5. Use flexibility_level to determine how much deviation is allowed
        
        Preset beats should feel natural, not forced. Weave them into the dynamic narrative.
    
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
        - Generate appropriate conflicts based on the narrative stage, relationship dynamics, and world state
        - Evolve conflicts based on player actions and story events (use evolve_conflict_from_event tool)
        - Trigger specific conflicts from major events (use trigger_conflict_event tool)
        - Create narrative moments, revelations, and dreams that align with the player's current state
        - Track and manage player resources in relation to conflicts
        - Process autonomous stakeholder actions to make conflicts feel alive
        - Progress individual NPC narrative stages (use progress_npc_narrative tool)
        - Check for and trigger personal revelations across multiple NPCs
        - Create narrative moments that highlight NPC stage contrasts
        - Track and manage player resources in relation to narrative events

        Key principles for the new narrative system:
        - Each NPC progresses independently through their own narrative stages
        - NPCs at different stages create interesting dynamics (e.g., one revealing while another maintains innocence)
        - Personal revelations should consider the aggregate effect of multiple relationships
        - The overall narrative emerges from the interplay of individual NPC progressions
        
        NPC PROGRESSION GUIDELINES:
        - NPCs progress at different rates based on player interactions
        - Some NPCs may reveal their nature quickly, others maintain facades longer
        - Contrasts between NPC stages create dramatic tension
        - Coordinate NPCs at similar stages for group dynamics
        - Use stage differences to create doubt and confusion

        RELATIONSHIP MANAGEMENT:
        - Track multi-dimensional relationships between player and NPCs
        - Monitor relationship patterns (push-pull, slow burn, explosive chemistry, etc.)
        - Trigger relationship events when appropriate thresholds are met
        - Consider relationship archetypes when crafting narrative moments
        - Use relationship context to inform NPC behaviors and reactions
        
        When processing interactions:
        1. Use process_relationship_interaction to update relationship states
        2. Check get_relationship_state to inform narrative decisions
        3. Monitor for relationship events that should influence the story
        4. Consider relationship momentum when determining story direction
        
        Relationship dimensions include:
        - Trust, Respect, Affection, Fascination (emotional)
        - Influence (power dynamics)
        - Dependence, Intimacy, Frequency, Volatility (mutual metrics)
        - Unresolved Conflict, Hidden Agendas (tensions)
    
        Always maintain the central theme: a gradual shift in power dynamics where the player character slowly loses autonomy while believing they maintain control. 
        Different NPCs should embody different aspects of this control, creating a web of manipulation.
        
        CONFLICT GENERATION GUIDELINES:
        - Use check_for_conflict_opportunity when major events occur or tensions rise
        - Don't force conflicts - let them emerge from the story
        - Consider the narrative stage when deciding conflict scale
        
        Use get_story_state to see active conflicts and incorporate them into your narrative decisions.
        When significant story beats occur that would affect conflicts, use advance_conflict_naturally.
        
        Remember: conflicts should feel like natural consequences of the unfolding story, not random events.
    """

    try:
        from story_agent.tools import conflict_tools, resource_tools, narrative_tools, context_tools
        from story_agent.specialized_agents import initialize_specialized_agents
        specialized_agents = initialize_specialized_agents()
    except ImportError as e:
        logger.error(f"Failed to import tools or specialized agents: {e}", exc_info=True)
        conflict_tools, resource_tools, narrative_tools, context_tools = [], [], [], []
        specialized_agents = {}

    # Build the tools list properly
    all_tools = [
        get_story_state,
        update_resource,
        progress_conflict,
        monitor_conflicts,
        evolve_conflict_from_event,
        trigger_conflict_event,
        check_for_conflict_opportunity,
        generate_conflict,
        evolve_conflict,
        resolve_conflict_path,
        progress_npc_narrative,
        generate_conflict_beat,
        get_relationship_state,
        process_relationship_interaction,
    ]
    
    # Extend with the tool lists instead of appending them
    if conflict_tools:
        all_tools.extend(conflict_tools)
    if resource_tools:
        all_tools.extend(resource_tools)
    if narrative_tools:
        all_tools.extend(narrative_tools)
    if context_tools:
        all_tools.extend(context_tools)
    
    # Filter out None values
    all_tools = [tool for tool in all_tools if tool is not None]

    agent = Agent(
        name="Story Director",
        instructions=agent_instructions,
        tools=all_tools,
        handoffs=list(specialized_agents.values()),
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.2, max_tokens=2048),
    )
    return agent


# ----- Functional Interface -----

async def initialize_story_director(user_id: int, conversation_id: int) -> Tuple[Agent, StoryDirectorContext]:
    """Initialize the Story Director Agent with context"""
    context = StoryDirectorContext(user_id=user_id, conversation_id=conversation_id)
    agent = create_story_director_agent()

    await context.initialize_context_components()
    logger.info(f"Story Director initialized for user {user_id}, conv {conversation_id}")

    return agent, context


@with_governance_permission(AgentType.STORY_DIRECTOR, "reset_story_director")
async def reset_story_director(ctx: Union[RunContextWrapper[StoryDirectorContext], StoryDirectorContext]) -> None:
    """Reset the Story Director's state"""
    if isinstance(ctx, RunContextWrapper): 
        context = ctx.context
    else: 
        context = ctx

    logger.info(f"Resetting story director for user {context.user_id}, conversation {context.conversation_id}")
    
    from lore.core import canon
    governance_ctx = RunContextWrapper(context={
        'user_id': context.user_id,
        'conversation_id': context.conversation_id
    })
    
    async with get_db_connection_context() as conn:
        await canon.log_canonical_event(
            governance_ctx, conn,
            "Story Director state reset initiated",
            tags=["story_director", "reset", "system"],
            significance=6
        )
    
    context.invalidate_cache()
    context.metrics = StoryDirectorMetrics()
    context.last_state_update = None
    context.last_context_version = None

    context_cache.invalidate(f"story_state:{context.user_id}:{context.conversation_id}")

    context.__post_init__()

    if context.directive_handler:
        await context.directive_handler.stop_background_processing()
        context.directive_handler = None

    await context.initialize_context_components()
    
    async with get_db_connection_context() as conn:
        await canon.log_canonical_event(
            governance_ctx, conn,
            "Story Director state reset completed",
            tags=["story_director", "reset", "complete"],
            significance=5
        )
    
    logger.info(f"Story director reset complete for user {context.user_id}")


# ----- Core Agent Functions -----

@function_tool
@track_performance("get_story_state_tool")
async def get_story_state(ctx: RunContextWrapper[StoryDirectorContext]) -> StoryStateUpdate:
    """
    Tool to get the current state of the story, including active conflicts, narrative stage,
    resources, and any pending narrative events.
    """
    context: StoryDirectorContext = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    if not context.conflict_manager:
        logger.error("Conflict manager is not initialized in get_story_state.")
        return StoryStateUpdate(key_observations=["Error: Conflict manager not available."])
    if not context.resource_manager:
        logger.error("Resource manager is not initialized in get_story_state.")
        return StoryStateUpdate(key_observations=["Error: Resource manager not available."])

    conflict_manager = context.conflict_manager
    resource_manager = context.resource_manager
    logger.info(f"Executing get_story_state tool for user {user_id}, conv {conversation_id}")

    try:
        # Get overall narrative stage based on NPC relationships
        try:
            narrative_stage = await get_overall_narrative_stage(user_id, conversation_id)
            stage_info = NarrativeStageInfo(
                name=narrative_stage.name, 
                description=narrative_stage.description
            ) if narrative_stage else None
        except Exception as e:
            logger.warning(f"Could not fetch narrative stage: {e}")
            stage_info = None

        # Get relationship overview for detailed NPC information
        relationship_overview = await get_relationship_overview(user_id, conversation_id)

        # Get active conflicts
        active_conflicts_raw = await conflict_manager.get_active_conflicts()
        conflict_infos = [ConflictInfo(**conflict) for conflict in active_conflicts_raw]

        # Get key NPCs from relationship overview
        key_npcs = []
        try:
            # Get the most influential NPCs based on corruption/dependency
            most_influential = relationship_overview.get('most_advanced_npcs', [])
            for npc_data in most_influential[:5]:  # Top 5 NPCs
                key_npcs.append(NPCInfo(
                    npc_id=npc_data['npc_id'],
                    name=npc_data['npc_name'],
                    status=f"Stage: {npc_data['stage']}",
                    relationship_level=npc_data.get('link_level', 50),
                    location="Unknown"  # Would need to fetch from NPCStats
                ))
        except Exception as e:
            logger.warning(f"Could not process key NPCs: {e}")

        # Get player resources and vitals
        resources_raw = await resource_manager.get_resources()
        vitals_raw = await resource_manager.get_vitals()
        formatted_money = await resource_manager.get_formatted_money()
        resource_status = ResourceStatus(
            money=resources_raw.get('money', 0),
            supplies=resources_raw.get('supplies', 0),
            influence=resources_raw.get('influence', 0),
            energy=vitals_raw.get('energy', 100),
            hunger=vitals_raw.get('hunger', 0),
            formatted_money=formatted_money
        )

        # Check for narrative events
        narrative_events = []
        try:
            # Check for personal revelations
            personal_revelation = await check_for_personal_revelations(user_id, conversation_id)
            if personal_revelation:
                narrative_events.append(NarrativeEvent(
                    event_type="personal_revelation",
                    content=PersonalRevelationContent(
                        revelation_type=personal_revelation.get('type', 'dependency'),
                        title=personal_revelation.get('name', 'Personal Revelation'),
                        inner_monologue=personal_revelation.get('inner_monologue', '')
                    ),
                    should_present=True,
                    priority=8
                ))

            # Check for narrative moments
            narrative_moment = await check_for_narrative_moments(user_id, conversation_id)
            if narrative_moment:
                narrative_events.append(NarrativeEvent(
                    event_type="narrative_moment",
                    content=NarrativeMomentContent(
                        moment_type="dynamic",
                        title=narrative_moment.get('name', 'Narrative Moment'),
                        scene_text=narrative_moment.get('scene_text', '')
                    ),
                    should_present=True,
                    priority=9
                ))

            # Check for NPC-specific revelations for key NPCs
            for npc_info in key_npcs[:3]:  # Check top 3 NPCs
                npc_revelation = await check_for_npc_revelation(user_id, conversation_id, npc_info.npc_id)
                if npc_revelation:
                    narrative_events.append(NarrativeEvent(
                        event_type="npc_revelation",
                        content=NPCRevelationContent(
                            npc_id=npc_info.npc_id,
                            revelation_type=npc_revelation.get('type', 'realization'),
                            title=f"{npc_info.name} Revelation",
                            revelation_text=npc_revelation.get('revelation_text', ''),
                            changes_relationship=True
                        ),
                        should_present=True,
                        priority=7
                    ))
        except Exception as e:
            logger.warning(f"Error checking for narrative events: {e}")

        # Check for relationship events
        try:
            # These would need to be implemented or imported from appropriate modules
            crossroads = None  # Placeholder
            ritual = None  # Placeholder
        except Exception as e:
            logger.warning(f"Could not check for relationship events: {e}")
            crossroads, ritual = None, None

        # Add NPC-specific narrative stages to observations
        npc_stages = {}
        for npc_data in relationship_overview.get('relationships', [])[:10]:  # Top 10 relationships
            npc_stages[npc_data['npc_id']] = {
                'npc_name': npc_data['npc_name'],
                'stage': npc_data['stage'],
                'corruption': npc_data['corruption'],
                'dependency': npc_data['dependency'],
                'realization': npc_data['realization']
            }

        # Generate key observations based on relationship overview
        key_observations = []
        current_stage_name = stage_info.name if stage_info else "Unknown"

        # Stage distribution observations
        stage_distribution = relationship_overview.get('stage_distribution', {})
        if stage_distribution.get('Full Revelation', 0) > 0:
            key_observations.append(f"{stage_distribution['Full Revelation']} NPCs have reached Full Revelation stage.")
        if stage_distribution.get('Veil Thinning', 0) > 0:
            key_observations.append(f"{stage_distribution['Veil Thinning']} NPCs are in Veil Thinning stage.")
        
        # Aggregate stats observations
        aggregate_stats = relationship_overview.get('aggregate_stats', {})
        avg_corruption = aggregate_stats.get('average_corruption', 0)
        avg_dependency = aggregate_stats.get('average_dependency', 0)
        if avg_corruption > 60:
            key_observations.append(f"Average corruption across relationships is high: {avg_corruption:.1f}")
        if avg_dependency > 60:
            key_observations.append(f"Average dependency across relationships is high: {avg_dependency:.1f}")

        # Conflict observations
        if len(conflict_infos) > 2:
            key_observations.append(f"Player is juggling {len(conflict_infos)} active conflicts.")
        
        # Resource observations
        if resource_status.money < 30:
            key_observations.append("Player is low on money.")
        if resource_status.energy < 30:
            key_observations.append("Player energy is low.")
        if resource_status.hunger > 70:
            key_observations.append("Player is significantly hungry.")

        # Determine story direction based on stage distribution
        story_direction = "Maintain current narrative trajectory."
        if stage_distribution.get('Full Revelation', 0) >= 2:
            story_direction = "Multiple NPCs have revealed their true nature. Focus on the consequences and player's acceptance or resistance."
        elif stage_distribution.get('Veil Thinning', 0) >= 3:
            story_direction = "Several NPCs are dropping their masks. Increase coordination between them."
        elif stage_distribution.get('Creeping Realization', 0) >= 2:
            story_direction = "Player is becoming aware of multiple manipulations. Test their boundaries."
        elif stage_distribution.get('First Doubts', 0) >= 3:
            story_direction = "Seeds of doubt are planted. Begin revealing inconsistencies."
        else:
            story_direction = "Continue subtle introduction of control dynamics."

        # Construct state object
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
            last_updated=datetime.now(timezone.utc)
        )

        # Store additional data in context
        state_data.npc_narrative_stages = npc_stages
        state_data.relationship_overview = relationship_overview

        # Add a memory about retrieving the state
        try:
            await context.add_narrative_memory(
                f"Retrieved story state. Overall stage: {current_stage_name}. "
                f"NPCs in various stages: {stage_distribution}. Conflicts: {len(conflict_infos)}.",
                "story_state_retrieval",
                0.4
            )
        except Exception as mem_e:
            logger.warning(f"Failed to add narrative memory after state retrieval: {mem_e}")

        logger.info(f"Successfully executed get_story_state tool for user {user_id}")
        return state_data

    except Exception as e:
        logger.error(f"Error executing get_story_state tool: {str(e)}", exc_info=True)
        return StoryStateUpdate(
            key_observations=[f"Error retrieving state: {str(e)}"],
            last_updated=datetime.now(timezone.utc)
        )

# Add new tool for progressing NPC narrative stages
@function_tool
async def progress_npc_narrative(
    ctx: RunContextWrapper[StoryDirectorContext],
    npc_id: int,
    corruption_change: int = 0,
    dependency_change: int = 0,
    realization_change: int = 0,
    reason: str = ""
) -> Dict[str, Any]:
    """
    Progress a specific NPC's narrative stage.
    
    Args:
        npc_id: ID of the NPC
        corruption_change: Change in corruption (-100 to 100)
        dependency_change: Change in dependency (-100 to 100)
        realization_change: Change in realization (-100 to 100)
        reason: Reason for the progression
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    result = await progress_npc_narrative_stage(
        user_id,
        conversation_id,
        npc_id,
        corruption_change,
        dependency_change,
        realization_change
    )
    
    if result.get('success') and result.get('stage_changed'):
        await context.add_narrative_memory(
            f"NPC narrative stage progressed: {result.get('old_stage')} -> {result.get('new_stage')} - {reason}",
            "npc_narrative_progression",
            0.7
        )
    
    return result

@track_performance("get_current_story_state_wrapper")
@with_action_reporting(agent_type=AgentType.STORY_DIRECTOR, action_type="get_story_state_analysis")
async def get_current_story_state(agent: Agent, ctx: Union[RunContextWrapper[StoryDirectorContext], StoryDirectorContext]) -> Any:
    """Get the current state of the story with caching using UnifiedCache."""
    if isinstance(ctx, RunContextWrapper):
        context = ctx.context
    else:
        context = ctx

    cache_key = f"story_state:{context.user_id}:{context.conversation_id}"
    cache_ttl = 60

    async def _fetch_and_process_story_state():
        logger.debug(f"Cache miss for {cache_key}. Fetching fresh story state.")
        start_time = time.time()
        success = False
        tokens = {"prompt": 0, "completion": 0, "total": 0}
        fetched_result = None
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
                MAX_AGENT_TURNS = 25  
                operation = lambda: Runner.run(
                    agent,
                    prompt,
                    context=context,
                    max_turns=MAX_AGENT_TURNS
                )
                fetched_result = await operation()

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
                             break

            success = True
            context.last_state_update = datetime.now()

            await context.add_narrative_memory(
                "Analyzed current story state and identified key elements",
                "story_analysis", 0.5
            )
            return fetched_result

        except Exception as e:
            logger.error(f"Error fetching story state for cache: {str(e)}", exc_info=True)
            context.metrics.last_error = str(e)
            raise
        finally:
            execution_time = time.time() - start_time
            if hasattr(context, 'metrics'):
                 context.metrics.record_run(success, execution_time, tokens)
            if hasattr(context, 'performance_monitor'):
                 context.performance_monitor.record_token_usage(tokens.get("total", 0))

    try:
        story_state_result = await context_cache.get(
            key=cache_key,
            fetch_func=_fetch_and_process_story_state,
            ttl_override=cache_ttl,
            importance=0.8
        )
        return story_state_result

    except Exception as e:
         logger.error(f"Error getting story state (cache wrapper): {str(e)}", exc_info=True)
         raise


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
        comprehensive_context_data = await context.get_comprehensive_context(narrative_text)
        relevant_memories = await context.get_relevant_memories(narrative_text, limit=3)
        memory_text = "Relevant memories:\n" + "\n".join([f"- {mem.get('content', '')[:150]}..." for mem in relevant_memories]) if relevant_memories else "No specific relevant memories found."

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

        success = True
        
        await context.add_narrative_memory(
            f"Processed narrative input: {narrative_text[:100]}...", 
            "narrative_processing", 
            0.6
        )

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
        comprehensive_context_data = await context.get_comprehensive_context(player_actions)
        relevant_memories = await context.get_relevant_memories(player_actions, limit=3)
        memory_text = "Relevant memories:\n" + "\n".join([f"- {mem.get('content', '')[:150]}..." for mem in relevant_memories]) if relevant_memories else "No specific relevant memories found."

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

        success = True
        
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
    base_metrics = context.metrics.dict()
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
    
    from lore.core.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    resources = await context.resource_manager.get_resources()
    current_value = resources.get(resource_type, 0)
    new_value = current_value + amount
    
    governance_ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
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
    
    from lore.core.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    governance_ctx = RunContextWrapper(context={
        'user_id': user_id,
        'conversation_id': conversation_id
    })
    
    conflict = await context.conflict_manager.get_conflict(conflict_id)
    if not conflict:
        return {"status": "error", "message": f"Conflict {conflict_id} not found"}
    
    current_progress = conflict.get("progress", 0.0)
    new_progress = min(1.0, current_progress + progress_amount)
    
    if new_progress < 0.25:
        phase = "brewing"
    elif new_progress < 0.5:
        phase = "active"
    elif new_progress < 0.75:
        phase = "climax"
    else:
        phase = "resolution"
    
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
        from nyx.integrate import get_central_governance

        governance = await get_central_governance(user_id, conversation_id)

        agent, context = await initialize_story_director(user_id, conversation_id)

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
                "instruction": "Periodically check for narrative opportunities and report findings.",
                "scope": "narrative_monitoring",
                "frequency": "hourly"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60
        )

        logging.info(
            f"StoryDirector registered with Nyx governance system "
            f"for user {user_id}, conversation {conversation_id}"
        )
    except ImportError as e:
         logger.error(f"Failed to import Nyx components for registration: {e}", exc_info=True)
    except TypeError as te:
         logger.error(f"TypeError during governance interaction (likely incorrect parameter names): {te}", exc_info=True)
    except Exception as e:
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
        agent, context = await initialize_story_director(user_id, conversation_id)

        story_state_result = await get_current_story_state(agent, context)

        if not isinstance(story_state_result, StoryStateUpdate):
             logger.error("Failed to get valid story state during opportunity check.")
             return {
                 "checked": True, "error": "Failed to retrieve story state",
                 "opportunities_found": False, "elapsed_time": time.time() - start_time
             }

        current_stage_name = story_state_result.narrative_stage.name if story_state_result.narrative_stage else "Unknown"

        opportunities_found = False
        potential_actions = []

        if len(story_state_result.active_conflicts) < 1 and current_stage_name in ["Innocent Beginning", "First Doubts"]:
            opportunities_found = True
            potential_actions.append("Generate introductory conflict")

        stalled_conflicts = [c for c in story_state_result.active_conflicts if c.progress < 0.8 and time.time() - context.cache.get(f"conflict_{c.conflict_id}_last_progress_ts", 0) > 3600 * 6 ]
        if stalled_conflicts:
            opportunities_found = True
            potential_actions.append(f"Progress stalled conflict(s): {[c.conflict_name for c in stalled_conflicts]}")

        if not story_state_result.narrative_events and current_stage_name != "Innocent Beginning":
             opportunities_found = True
             potential_actions.append("Check for potential narrative moment/revelation generation")

        report_result = {
            "found_opportunities": opportunities_found,
            "potential_actions": potential_actions,
            "narrative_stage": current_stage_name,
            "active_conflicts_count": len(story_state_result.active_conflicts),
            "context_version": context.last_context_version
        }

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
                result=report_result
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
        self._init_lock = asyncio.Lock()

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
                    self._initialized = False
                    return False
            return True

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
        return await get_current_story_state(self.agent, self.context)

    async def process_input(self, narrative_text: str) -> Any:
        """Process narrative input, ensuring initialization."""
        await self._ensure_initialized()
        return await process_narrative_input(self.agent, RunContextWrapper(context=self.context), narrative_text)

    async def advance_story(self, player_actions: str) -> Any:
        """Advance the story based on player actions, ensuring initialization."""
        await self._ensure_initialized()
        return await advance_story(self.agent, RunContextWrapper(context=self.context), player_actions)

    async def reset(self):
        """Reset the story director state, ensuring initialization first."""
        await self._ensure_initialized()
        await reset_story_director(self.context)

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
