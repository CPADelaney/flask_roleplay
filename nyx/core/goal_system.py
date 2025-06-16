# nyx/core/goal_system.py

"""
Goal Management System for Nyx AI - Leverages advanced agent architecture for planning and execution.
Features improved concurrency, persistence, conflict resolution, and integration with other systems.
"""

import logging
import datetime
import uuid
import asyncio
import json
import os
import aiofiles
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable, Type
from pydantic import BaseModel, Field, field_validator, model_validator, Json
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import copy
import random
from functools import lru_cache
import weakref

# Import agent framework components
from agents import (
    Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper,
    handoff, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
)

logger = logging.getLogger(__name__)

# Schema version for persistence
SCHEMA_VERSION = "1.0.0"


JsonScalar = Union[str, int, float, bool, None]

class StepExecutionResult(BaseModel):
    success: bool
    step_id: str
    result_json: Optional[str] = None
    error: Optional[str] = None
    next_action: str = Field("continue")
    model_config = {"extra": "forbid"}

class PlanGenerationResult(BaseModel):
    plan_json: str                     # json.dumps(plan)
    reasoning: str
    estimated_steps: int
    estimated_completion_time: Optional[str] = None
    model_config = {"extra": "forbid"}

class ActiveGoalsSummary(BaseModel):
    active_count: int
    pending_count: int
    goals_json: str
    model_config = {"extra": "forbid"}

class ConflictCheckResult(BaseModel):
    has_conflicts: bool
    conflict_count: int
    conflicts_json: str
    model_config = {"extra": "forbid"}

class CapabilityCheckResult(BaseModel):
    all_available: bool
    available_count: int
    unavailable_actions_json: str
    model_config = {"extra": "forbid"}

class PrioritizedGoalsSummary(BaseModel):
    total_count: int
    active_count: int
    pending_count: int
    goals_json: str
    model_config = {"extra": "forbid"}

class ResolvedParameter(BaseModel):
    name: str
    value: JsonScalar
    model_config = {"extra": "forbid"}          # ← closes the object

class ParameterStatus(BaseModel):
    name: str
    original: str
    resolved: bool
    is_null: bool
    model_config = {"extra": "forbid"}          # ← closes the object

class StepParameterResolutionResult(BaseModel):
    resolved_parameters: List[ResolvedParameter]
    resolution_status:    List[ParameterStatus]
    all_resolved:         bool
    raw_json: Optional[str] = Field(
        None, description="Full resolved mapping, JSON-encoded"
    )
    model_config = {"extra": "forbid"}          # ← closes the object

class ExecutionOutcome(BaseModel):
    success: bool
    duration: float
    result_json: Optional[str] = Field(
        None, description="Result object encoded as JSON, if any"
    )
    error: Optional[str] = None
    exception_type: Optional[str] = None

    model_config = {"extra": "forbid"}

class DominanceCheckResult(BaseModel):
    is_dominance_action: bool
    can_proceed: bool
    action: str
    reason: Optional[str] = None
    evaluation_json: Optional[str] = Field(
        None, description="Full evaluation dict encoded as JSON"
    )

    model_config = {"extra": "forbid"}

class ExecutionLogStatus(BaseModel):
    success: bool
    step_index: int
    current_index: int
    step_status: str
    error: Optional[str] = None

    model_config = {"extra": "forbid"}

class UpdateGoalStatusResult(BaseModel):
    success: bool
    goal_id: str
    new_status: str
    old_status: Optional[str] = None
    notifications_json: Optional[str] = None

    model_config = {"extra": "forbid"}

class NotifySystemsResult(BaseModel):
    success: bool
    goal_id: str
    status: str
    notifications_json: Optional[str] = None
    error: Optional[str] = None

    model_config = {"extra": "forbid"}


class GoalStep(BaseModel):
    """A single step in a goal's execution plan"""
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:6]}")
    description: str
    action: str  # Name of a method callable on NyxBrain (e.g., 'query_knowledge', 'generate_response')
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field("pending", description="pending, active, completed, failed, skipped")
    result: Optional[Any] = Field(None, description="Result from action execution")
    error: Optional[str] = Field(None, description="Error message if step failed")
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None

    @field_validator('action')
    @classmethod
    def action_must_be_valid(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Action must be a non-empty string')
        return v

class TimeHorizon(str, Enum):
    """Time horizon for goals - affects planning, execution and priority calculations"""
    SHORT_TERM = "short_term"  # Hours to days
    MEDIUM_TERM = "medium_term"  # Days to weeks
    LONG_TERM = "long_term"  # Weeks to months/years

class EmotionalMotivation(BaseModel):
    """Emotional motivation behind a goal"""
    primary_need: str = Field(..., description="Primary emotional need driving this goal")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Intensity of the emotional motivation")
    expected_satisfaction: float = Field(0.5, ge=0.0, le=1.0, description="Expected satisfaction from achieving the goal")
    associated_chemicals: Dict[str, float] = Field(default_factory=dict, description="Associated neurochemicals and their expected changes")
    description: str = Field("", description="Description of the emotional motivation")

class GoalRelationship(BaseModel):
    """Relationship between goals"""
    parent_goal_id: Optional[str] = Field(None, description="ID of the parent goal")
    child_goal_ids: List[str] = Field(default_factory=list, description="IDs of child goals")
    supports_goal_ids: List[str] = Field(default_factory=list, description="IDs of goals this goal supports")
    conflicts_with_goal_ids: List[str] = Field(default_factory=list, description="IDs of goals this goal conflicts with")
    relationship_type: str = Field("independent", description="Type of relationship (hierarchical, supportive, conflicting)")

class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving conflicts between goals"""
    PRIORITY_BASED = "priority_based"  # Higher priority wins
    TIME_HORIZON_BASED = "time_horizon_based"  # Shorter time horizon wins 
    NEGOTIATE = "negotiate"  # Try to find a compromise
    MERGE = "merge"  # Merge into a single goal
    USER_DECISION = "user_decision"  # Ask the user to decide

class ConflictResolutionData(BaseModel):
    """Data for conflict resolution"""
    strategy: ConflictResolutionStrategy
    resolved: bool = False
    resolution_time: Optional[datetime.datetime] = None
    resolution_reason: Optional[str] = None
    negotiation_attempts: int = 0
    max_negotiation_attempts: int = 3
    merged_goal_id: Optional[str] = None

class Goal(BaseModel):
    """A goal for Nyx to pursue, with associated metadata and execution plan"""
    id: str = Field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    description: str
    status: str = Field("pending", description="pending, active, completed, failed, abandoned")
    priority: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field("unknown", description="Originator (NeedsSystem, User, MetaCore, etc.)")
    associated_need: Optional[str] = None  # Link to a need in NeedsSystem
    creation_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    completion_time: Optional[datetime.datetime] = None
    deadline: Optional[datetime.datetime] = None
    plan: List[GoalStep] = Field(default_factory=list)
    current_step_index: int = Field(0, description="Index of the next step to execute")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Log of step execution attempts")
    last_error: Optional[str] = None
    time_horizon: TimeHorizon = Field(TimeHorizon.MEDIUM_TERM, description="Time horizon for the goal")
    emotional_motivation: Optional[EmotionalMotivation] = None
    relationships: GoalRelationship = Field(default_factory=GoalRelationship)
    recurring: bool = Field(False, description="Whether this is a recurring goal")
    recurrence_pattern: Optional[Dict[str, Any]] = None  # For recurring goals
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress toward completion (0.0-1.0)")
    
    # Fields for persistence and versioning
    version: str = Field(SCHEMA_VERSION, description="Schema version for this goal")
    last_modified: datetime.datetime = Field(default_factory=datetime.datetime.now)
    last_persisted: Optional[datetime.datetime] = None
    checksum: Optional[str] = None  # For data integrity verification
    
    # Fields for conflict resolution
    conflict_data: Dict[str, ConflictResolutionData] = Field(default_factory=dict)
    negotiation_status: Optional[Dict[str, Any]] = None
    
    # Fields for integration
    external_system_ids: Dict[str, str] = Field(default_factory=dict)  # IDs in external systems
    callback_urls: List[str] = Field(default_factory=list)  # Webhook URLs for updates
    emotional_state_snapshots: List[Dict[str, Any]] = Field(default_factory=list)  # Snapshots of emotional state
    external_feedback: List[Dict[str, Any]] = Field(default_factory=list)  # Feedback from external systems

    @model_validator(mode='after')
    def update_checksum(self):
        # Update checksum based on essential fields
        checksum_data = f"{self.id}{self.description}{self.priority}{self.status}"
        self.checksum = hashlib.sha256(checksum_data.encode()).hexdigest()
        return self

# Models for structured I/O with Agents
class GoalValidationResult(BaseModel):
    """Output model for goal validation guardrail"""
    is_valid: bool = Field(..., description="Whether the goal is valid")
    reason: Optional[str] = Field(None, description="Reason for invalidation if not valid")
    priority_adjustment: Optional[float] = Field(None, description="Suggested priority adjustment")

class PlanValidationResult(BaseModel):
    """Output model for plan validation guardrail"""
    is_valid: bool = Field(..., description="Whether the plan is valid")
    reason: Optional[str] = Field(None, description="Reason for invalidation if not valid")
    suggestions: List[str] = Field(default_factory=list, description="Suggested improvements")

class StepExecutionResult(BaseModel):
    """Output model for step execution agent"""
    success: bool = Field(..., description="Whether the step executed successfully")
    step_id: str = Field(..., description="ID of the executed step")
    result: Optional[Any] = Field(None, description="Result from the execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    next_action: str = Field("continue", description="continue, retry, skip, or abort")

class PlanGenerationResult(BaseModel):
    """Output model for plan generation agent"""
    plan: List[Dict[str, Any]] = Field(..., description="Generated plan steps")
    reasoning: str = Field(..., description="Reasoning behind the plan")
    estimated_steps: int = Field(..., description="Estimated number of steps")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")

class GoalHierarchyNode(BaseModel):
    """Node in a goal hierarchy visualization"""
    goal_id: str
    description: str
    time_horizon: TimeHorizon
    status: str
    priority: float
    children: List["GoalHierarchyNode"] = Field(default_factory=list)

class GoalCreationWithMotivation(BaseModel):
    """Input model for goal creation with motivation"""
    description: str
    priority: float = 0.5
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    deadline: Optional[str] = None
    emotional_motivation: Optional[EmotionalMotivation] = None
    associated_need: Optional[str] = None
    parent_goal_id: Optional[str] = None

class GoalMotivationAnalysis(BaseModel):
    """Output model for analyzing goal motivations"""
    emotional_needs: Dict[str, int] = Field(default_factory=dict)
    primary_motivations: List[str] = Field(default_factory=list)
    chemical_associations: Dict[str, float] = Field(default_factory=dict)
    motivation_patterns: List[Dict[str, Any]] = Field(default_factory=list)

class RunContext(BaseModel):
    """Context model for agent execution"""
    goal_id: str
    brain_available: bool = True
    user_id: Optional[str] = None
    current_step_index: int = 0
    max_retries: int = 3
    retry_count: int = 0

# Configuration models
class PersistenceConfig(BaseModel):
    """Configuration for goal persistence"""
    storage_path: str = Field("./data/goals", description="Path for storing goal data")
    auto_save_interval: int = Field(60, description="Auto-save interval in seconds")
    compression: bool = Field(True, description="Whether to compress saved data")
    backup_enabled: bool = Field(True, description="Whether to create backups")
    max_backups: int = Field(5, description="Maximum number of backups to keep")

class SystemIntegrationConfig(BaseModel):
    """Configuration for system integration"""
    enabled_systems: List[str] = Field(default_factory=list)
    callback_timeout: float = Field(5.0, description="Timeout for callbacks in seconds")
    retry_callbacks: bool = Field(True, description="Whether to retry failed callbacks")
    max_retries: int = Field(3, description="Maximum number of callback retries")

class ConflictDetectionConfig(BaseModel):
    """Configuration for conflict detection"""
    similarity_threshold: float = Field(0.6, description="Threshold for detecting similar goals")
    resource_conflict_detection: bool = Field(True, description="Whether to detect resource conflicts")
    default_strategy: ConflictResolutionStrategy = Field(
        ConflictResolutionStrategy.PRIORITY_BASED, 
        description="Default conflict resolution strategy"
    )

class GoalManager:
    """Manages goals, planning, and execution oversight for Nyx."""

    def __init__(self, brain_reference=None, 
                 persistence_config: Optional[PersistenceConfig] = None,
                 integration_config: Optional[SystemIntegrationConfig] = None,
                 conflict_config: Optional[ConflictDetectionConfig] = None,
                 api_key: Optional[str] = None):
        """
        Args:
            brain_reference: Reference to the main NyxBrain instance for action execution.
            persistence_config: Configuration for goal persistence
            integration_config: Configuration for system integration
            conflict_config: Configuration for conflict detection and resolution
            api_key: Optional API key for agent services
        """
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()  # IDs of goals currently being executed
        self.brain = brain_reference  # Set via set_brain_reference if needed later
        self.max_concurrent_goals = 3  # Allows multiple goals to execute concurrently
        
        # Concurrency control - granular locking for improved performance
        self._goal_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)  # Lock per goal
        self._goals_dict_lock = asyncio.Lock()  # Lock for the goals dictionary itself
        self._active_goals_lock = asyncio.Lock()  # Lock for active goals set
        self._reader_count = 0  # Number of current readers
        self._reader_lock = asyncio.Lock()  # Lock for reader count
        self._writer_lock = asyncio.Lock()  # Lock for writers
        
        # Thread pool for background tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Persistence configuration
        self.persistence_config = persistence_config or PersistenceConfig()
        self._persistence_task = None
        self._last_save_time = datetime.datetime.now()
        self._dirty_goals: Set[str] = set()  # Track which goals have changed since last save
        
        # Conflict resolution
        self.conflict_config = conflict_config or ConflictDetectionConfig()
        self._conflict_resolution_tasks: Dict[str, asyncio.Task] = {}  # Track running resolution tasks
        
        # Integration with other systems
        self.integration_config = integration_config or SystemIntegrationConfig()
        self._integration_callbacks: Dict[str, List[Callable]] = defaultdict(list)  # Callbacks for goal events
        self._external_system_clients: Dict[str, Any] = {}  # Clients for external systems
        
        # Goal statistics
        self.goal_statistics = {
            "created": 0,
            "completed": 0,
            "failed": 0,
            "abandoned": 0,
            "merged": 0,
            "conflict_resolved": 0
        }
        
        # Initialize agents and settings
        self._init_agents(api_key)
        self.trace_group_id = "NyxGoalManagement"

        # Initialize storage directory
        os.makedirs(self.persistence_config.storage_path, exist_ok=True)
        
        logger.info("GoalManager initialized with improved concurrency, persistence, conflict resolution, and integration.")
    
    async def start(self):
        """Start the goal manager's background tasks"""
        # Start auto-save task if enabled
        if self.persistence_config.auto_save_interval > 0:
            self._persistence_task = asyncio.create_task(self._auto_save_loop())
            logger.info(f"Auto-save task started with interval of {self.persistence_config.auto_save_interval} seconds")
        
        # Attempt to load any saved goals
        try:
            await self.load_goals()
            logger.info("Successfully loaded saved goals")
        except Exception as e:
            logger.error(f"Error loading saved goals: {e}")
        
        # Initialize integration with external systems
        await self._init_external_systems()
        
        logger.info("GoalManager started successfully")
    
    async def stop(self):
        """Stop the goal manager's background tasks"""
        # Stop persistence task
        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
            self._persistence_task = None
        
        # Save all goals one last time
        await self.save_goals()
        
        # Cancel all conflict resolution tasks
        for task in self._conflict_resolution_tasks.values():
            task.cancel()
        
        # Close thread pool
        self._thread_pool.shutdown()
        
        logger.info("GoalManager stopped")
    
    def _init_agents(self, api_key=None):
        """Initialize all agents needed for the goal system"""
        # Goal planning agent (generates plans for goals)
        self.planning_agent = self._create_planning_agent()
        
        # Goal validation agent (validates goals before accepting them)
        self.goal_validation_agent = self._create_goal_validation_agent()
        
        # Plan validation agent (validates plans before execution)
        self.plan_validation_agent = self._create_plan_validation_agent()
        
        # Step execution agent (handles step execution and error recovery)
        self.step_execution_agent = self._create_step_execution_agent()
        
        # Main orchestration agent (coordinates the overall goal execution)
        self.orchestration_agent = self._create_orchestration_agent()
        
        # Conflict resolution agent
        self.conflict_resolution_agent = self._create_conflict_resolution_agent()
        
        # Goal negotiation agent
        self.negotiation_agent = self._create_negotiation_agent()
        
        # Goal merging agent
        self.merging_agent = self._create_merging_agent()

    def set_brain_reference(self, brain):
        """Set the reference to the main NyxBrain after initialization."""
        self.brain = brain
        logger.info("NyxBrain reference set for GoalManager.")

    # Add these methods to your existing GoalManager class in goal_system.py
    
    async def update_goal_priority(self, goal_id: str, new_priority: float, reason: str = "context_adjustment") -> Dict[str, Any]:
        """
        Update the priority of a specific goal
        
        Args:
            goal_id: Goal ID
            new_priority: New priority value (0.0-1.0)
            reason: Reason for the priority change
            
        Returns:
            Update result
        """
        if not (0.0 <= new_priority <= 1.0):
            return {"success": False, "error": "Priority must be between 0.0 and 1.0"}
        
        success = await self._update_goal_with_writer_lock(
            goal_id,
            lambda goal: (
                setattr(goal, "priority", new_priority),
                goal.execution_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "priority_update",
                    "old_priority": goal.priority,
                    "new_priority": new_priority,
                    "reason": reason
                })
            )
        )
        
        if success:
            logger.info(f"Updated priority for goal '{goal_id}' to {new_priority:.2f}. Reason: {reason}")
            await self.mark_goal_dirty(goal_id)
            
            return {
                "success": True,
                "goal_id": goal_id,
                "new_priority": new_priority,
                "reason": reason
            }
        else:
            return {"success": False, "error": f"Goal {goal_id} not found"}
    
    async def get_goals_for_need(self, need_name: str, status_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all goals associated with a specific need
        
        Args:
            need_name: Name of the need
            status_filter: Optional status filter
            
        Returns:
            List of goals for the need
        """
        all_goals = []
        
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                await self._writer_lock.acquire()
        
        try:
            all_goals = list(self.goals.values())
        finally:
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._writer_lock.release()
        
        filtered_goals = []
        for goal in all_goals:
            if goal.associated_need == need_name:
                if status_filter is None or goal.status in status_filter:
                    goal_summary = goal.model_dump(exclude={'execution_history', 'plan'})
                    goal_summary['plan_step_count'] = len(goal.plan)
                    filtered_goals.append(goal_summary)
        
        return filtered_goals
    
    async def register_context_callbacks(self, context_system):
        """
        Register callbacks with the context system for goal events
        
        Args:
            context_system: The context distribution system
        """
        # Register for goal completion events
        async def on_goal_completed(goal_data: Dict[str, Any]):
            goal_id = goal_data.get("goal_id")
            associated_need = goal_data.get("associated_need")
            
            if associated_need and hasattr(context_system, 'add_context_update'):
                from nyx.core.brain.context_distribution import ContextUpdate, ContextScope, ContextPriority
                
                update = ContextUpdate(
                    source_module="goal_manager",
                    update_type="goal_completion",
                    data={
                        "goal_context": {
                            "goal_id": goal_id,
                            "associated_need": associated_need,
                            "completion_quality": 0.8  # Default quality score
                        }
                    },
                    scope=ContextScope.GLOBAL,
                    priority=ContextPriority.HIGH
                )
                
                await context_system.add_context_update(update)
        
        # Register for goal status changes
        async def on_goal_status_change(goal_data: Dict[str, Any]):
            if hasattr(context_system, 'add_context_update'):
                from nyx.core.brain.context_distribution import ContextUpdate, ContextScope, ContextPriority
                
                update = ContextUpdate(
                    source_module="goal_manager", 
                    update_type="goal_status_change",
                    data=goal_data,
                    scope=ContextScope.GLOBAL,
                    priority=ContextPriority.NORMAL
                )
                
                await context_system.add_context_update(update)
        
        # Store callbacks for later use
        self._context_callbacks = {
            "goal_completed": on_goal_completed,
            "goal_status_change": on_goal_status_change
        }
    
    async def trigger_context_callback(self, event_type: str, data: Dict[str, Any]):
        """
        Trigger a registered context callback
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if hasattr(self, '_context_callbacks') and event_type in self._context_callbacks:
            try:
                await self._context_callbacks[event_type](data)
            except Exception as e:
                logger.error(f"Error in context callback for {event_type}: {e}")
    

    #==========================================================================
    # Persistence and Durability
    #==========================================================================
    
    async def _auto_save_loop(self):
        """Background task for automatically saving goals at intervals"""
        try:
            while True:
                await asyncio.sleep(self.persistence_config.auto_save_interval)
                await self._save_dirty_goals()
                logger.debug("Auto-save complete")
        except asyncio.CancelledError:
            # Final save on cancellation
            await self._save_dirty_goals()
            logger.info("Auto-save task cancelled, final save completed")
            raise
    
    async def _save_dirty_goals(self):
        """Save only goals that have been modified since last save"""
        # Get a snapshot of dirty goals under lock
        async with self._goals_dict_lock:
            dirty_goals = self._dirty_goals.copy()
            self._dirty_goals.clear()
        
        if not dirty_goals:
            return
        
        logger.debug(f"Saving {len(dirty_goals)} modified goals")
        
        # Save each goal individually
        for goal_id in dirty_goals:
            await self._save_goal(goal_id)
    
    async def _save_goal(self, goal_id: str):
        """Save a single goal to storage"""
        # Get goal with reader lock
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return
        
        # Prepare goal for saving
        goal_copy = goal.model_copy()
        goal_copy.last_persisted = datetime.datetime.now()
        
        # Create goal directory if it doesn't exist
        goal_dir = os.path.join(self.persistence_config.storage_path, goal_id)
        os.makedirs(goal_dir, exist_ok=True)
        
        # Save goal to file
        goal_path = os.path.join(goal_dir, "goal.json")
        async with aiofiles.open(goal_path, 'w') as f:
            await f.write(goal_copy.model_dump_json(indent=2))
        
        # Save execution history separately to avoid large files
        history_path = os.path.join(goal_dir, "execution_history.json")
        async with aiofiles.open(history_path, 'w') as f:
            await f.write(json.dumps(goal_copy.execution_history, indent=2))
        
        # Update goal's last_persisted timestamp
        async with self._get_goal_lock(goal_id):
            if goal_id in self.goals:
                self.goals[goal_id].last_persisted = goal_copy.last_persisted
        
        logger.debug(f"Saved goal {goal_id}")
    
    async def save_goals(self):
        """Save all goals to persistent storage"""
        # Get all goal IDs with reader lock
        async with self._goals_dict_lock:
            goal_ids = list(self.goals.keys())
        
        logger.info(f"Saving all {len(goal_ids)} goals to storage")
        
        # Create backup of existing data if enabled
        if self.persistence_config.backup_enabled:
            await self._create_backup()
        
        # Save each goal
        for goal_id in goal_ids:
            await self._save_goal(goal_id)
        
        # Save goals index for faster loading
        index_path = os.path.join(self.persistence_config.storage_path, "goals_index.json")
        async with aiofiles.open(index_path, 'w') as f:
            await f.write(json.dumps({
                "version": SCHEMA_VERSION,
                "timestamp": datetime.datetime.now().isoformat(),
                "goal_ids": goal_ids,
                "total_count": len(goal_ids)
            }))
        
        logger.info(f"Successfully saved {len(goal_ids)} goals")
    
    async def _create_backup(self):
        """Create a backup of the goals storage directory"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{self.persistence_config.storage_path}_backup_{timestamp}"
        
        # Check if storage directory exists
        if not os.path.exists(self.persistence_config.storage_path):
            return
        
        # Copy all files to backup directory
        import shutil
        shutil.copytree(self.persistence_config.storage_path, backup_dir)
        
        # Limit number of backups
        await self._cleanup_old_backups()
        
        logger.info(f"Created backup at {backup_dir}")
    
    async def _cleanup_old_backups(self):
        """Remove old backups to limit storage usage"""
        import shutil
        base_dir = os.path.dirname(self.persistence_config.storage_path)
        backup_prefix = f"{os.path.basename(self.persistence_config.storage_path)}_backup_"
        
        # List all backup directories
        backups = []
        for item in os.listdir(base_dir):
            if item.startswith(backup_prefix) and os.path.isdir(os.path.join(base_dir, item)):
                backups.append(item)
        
        # Sort by timestamp (newest first)
        backups.sort(reverse=True)
        
        # Remove oldest backups beyond the limit
        if len(backups) > self.persistence_config.max_backups:
            for old_backup in backups[self.persistence_config.max_backups:]:
                shutil.rmtree(os.path.join(base_dir, old_backup))
                logger.debug(f"Removed old backup: {old_backup}")
    
    async def load_goals(self):
        """Load goals from persistent storage"""
        # Check for index file
        index_path = os.path.join(self.persistence_config.storage_path, "goals_index.json")
        if not os.path.exists(index_path):
            logger.info("No goals index found, checking for individual goal files")
            # Try loading goals directly from directories
            return await self._load_goals_from_directories()
        
        # Load index
        async with aiofiles.open(index_path, 'r') as f:
            index_data = json.loads(await f.read())
        
        goal_ids = index_data.get("goal_ids", [])
        logger.info(f"Found {len(goal_ids)} goals in index")
        
        # Load each goal
        loaded_count = 0
        for goal_id in goal_ids:
            if await self._load_goal(goal_id):
                loaded_count += 1
        
        logger.info(f"Successfully loaded {loaded_count} of {len(goal_ids)} goals")
        return loaded_count
    
    async def _load_goals_from_directories(self):
        """Load goals by scanning all subdirectories in the storage path"""
        if not os.path.exists(self.persistence_config.storage_path):
            logger.info(f"Storage path {self.persistence_config.storage_path} does not exist")
            return 0
        
        loaded_count = 0
        for item in os.listdir(self.persistence_config.storage_path):
            potential_goal_dir = os.path.join(self.persistence_config.storage_path, item)
            if os.path.isdir(potential_goal_dir):
                goal_path = os.path.join(potential_goal_dir, "goal.json")
                if os.path.exists(goal_path):
                    goal_id = item
                    if await self._load_goal(goal_id):
                        loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} goals from directory scan")
        return loaded_count
    
    async def _load_goal(self, goal_id: str):
        """Load a single goal from storage"""
        goal_dir = os.path.join(self.persistence_config.storage_path, goal_id)
        goal_path = os.path.join(goal_dir, "goal.json")
        history_path = os.path.join(goal_dir, "execution_history.json")
        
        if not os.path.exists(goal_path):
            logger.warning(f"Goal file not found for {goal_id}")
            return False
        
        try:
            # Load goal data
            async with aiofiles.open(goal_path, 'r') as f:
                goal_data = json.loads(await f.read())
            
            # Load execution history if available
            if os.path.exists(history_path):
                async with aiofiles.open(history_path, 'r') as f:
                    execution_history = json.loads(await f.read())
                goal_data["execution_history"] = execution_history
            
            # Create goal object and add to goals dictionary
            goal = Goal(**goal_data)
            
            # Add goal to goals dictionary with write lock
            async with self._goals_dict_lock:
                self.goals[goal_id] = goal
                
                # Update active goals set if goal is active
                if goal.status == "active":
                    async with self._active_goals_lock:
                        self.active_goals.add(goal_id)
            
            # Update goal statistics
            self.goal_statistics["created"] += 1
            if goal.status == "completed":
                self.goal_statistics["completed"] += 1
            elif goal.status == "failed":
                self.goal_statistics["failed"] += 1
            elif goal.status == "abandoned":
                self.goal_statistics["abandoned"] += 1
            
            logger.debug(f"Loaded goal {goal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading goal {goal_id}: {e}")
            return False
    
    async def mark_goal_dirty(self, goal_id: str):
        """Mark a goal as needing to be saved"""
        async with self._goals_dict_lock:
            if goal_id in self.goals:
                self._dirty_goals.add(goal_id)
                self.goals[goal_id].last_modified = datetime.datetime.now()
    
    #==========================================================================
    # Concurrency and Performance
    #==========================================================================
    
    def _get_goal_lock(self, goal_id: str) -> asyncio.Lock:
        """Get the lock for a specific goal"""
        return self._goal_locks[goal_id]
    
    async def _get_goal_with_reader_lock(self, goal_id: str) -> Optional[Goal]:
        """Get a goal using a reader lock that allows concurrent reads"""
        # Increment reader count with lock
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                # First reader acquires writer lock to block writers
                await self._writer_lock.acquire()
        
        try:
            # Check if goal exists without locking goals dictionary
            # This is safe because we're holding the reader lock
            if goal_id not in self.goals:
                return None
            
            # Return a copy of the goal to avoid concurrent modification issues
            return self.goals[goal_id].model_copy()
        finally:
            # Decrement reader count with lock
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    # Last reader releases writer lock
                    self._writer_lock.release()
    
    async def _update_goal_with_writer_lock(self, goal_id: str, update_func: Callable[[Goal], None]) -> bool:
        """Update a goal using a writer lock"""
        # Acquire writer lock to block all readers and writers
        async with self._writer_lock:
            # Now get the specific goal lock
            async with self._get_goal_lock(goal_id):
                if goal_id not in self.goals:
                    return False
                
                # Apply update
                update_func(self.goals[goal_id])
                
                # Mark goal as dirty
                await self.mark_goal_dirty(goal_id)
                
                return True
    
    async def get_prioritized_goals(self) -> List[Goal]:
        """
        Returns active and pending goals sorted by priority with optimized sorting.
        This method is asynchronous to correctly handle async locks.
        """
        goals_copy = {}
        # Acquire the async lock
        async with self._goals_dict_lock:
            goals_copy = {
                g_id: g.model_copy() for g_id, g in self.goals.items()
                if g.status in ["pending", "active"]
            }

        if not goals_copy:
            return []
        
        # Use a cached sorting key function for better performance
        @lru_cache(maxsize=1024)
        def sort_key_cached(goal_id, status, priority, creation_time_ts, deadline_ts=None,
                           time_horizon=None, motivation_intensity=None):
            # Basic priority from existing method
            now_ts = datetime.datetime.now().timestamp()
            age_penalty = (now_ts - creation_time_ts) / (3600 * 24)  # Age in days
            status_boost = 0.05 if status == "pending" else -0.05

            deadline_urgency = 0.0
            if deadline_ts:
                time_to_deadline = deadline_ts - now_ts
                if time_to_deadline > 0:
                    deadline_urgency = min(0.3, 86400 / max(3600, time_to_deadline))
                else:
                    deadline_urgency = 0.4

            time_horizon_factor = 0.0
            if time_horizon:
                if time_horizon == TimeHorizon.SHORT_TERM: time_horizon_factor = 0.2
                elif time_horizon == TimeHorizon.MEDIUM_TERM: time_horizon_factor = 0.1

            motivation_boost = 0.0
            if motivation_intensity is not None:
                motivation_boost = motivation_intensity * 0.15

            # Small random factor to break ties consistently but not predictably
            random.seed(goal_id) # Seed with goal_id for deterministic tie-breaking per goal
            tie_breaker = random.uniform(-0.001, 0.001)

            return (priority
                    + (age_penalty * 0.01 * status_boost)
                    + deadline_urgency
                    + time_horizon_factor
                    + motivation_boost
                    + tie_breaker)

        # Convert goals to tuples for faster sorting
        sort_tuples = []

        for goal in goals_copy.values():
            # Pre-compute timestamp values for better performance
            creation_time_ts = goal.creation_time.timestamp()
            deadline_ts = goal.deadline.timestamp() if goal.deadline else None

            # Get motivation intensity if available
            motivation_intensity = goal.emotional_motivation.intensity if goal.emotional_motivation else None

            # Compute sort key
            key = sort_key_cached(
                goal.id, goal.status, goal.priority, creation_time_ts, deadline_ts,
                goal.time_horizon.value if hasattr(goal, 'time_horizon') else None,
                motivation_intensity
            )

            # Store tuple of (sort_key, goal) for sorting
            sort_tuples.append((key, goal))

        # Sort tuples by key (faster than sorting objects)
        sort_tuples.sort(key=lambda item: item[0], reverse=True) # Use lambda for key access

        # Extract sorted goals
        return [item[1] for item in sort_tuples]
    
    #==========================================================================
    # Agent Creation Methods
    #==========================================================================
    
    def _create_planning_agent(self) -> Agent:
        """Creates the agent responsible for generating plans for goals."""
        try:
            # Define the available actions for the planner
            available_actions = [
                # Basic interaction
                "process_input", "generate_response", 
                # Knowledge Core
                "query_knowledge", "add_knowledge", 
                # Memory Core
                "retrieve_memories", "add_memory", "create_reflection", "create_abstraction", 
                # Procedural Memory
                "execute_procedure", "add_procedure", 
                # Reasoning Core
                "reason_causal", "perform_intervention", "reason_counterfactually", 
                # Emotional Core
                "update_emotion", "process_emotional_input", 
                # Multimodal/Somatic
                "process_sensory_input", "add_expectation", 
                # Meta Core
                "monitor_systems", "evaluate_cognition", "select_strategy", 
                # Prediction Engine
                "generate_prediction", "evaluate_prediction", 
                # Curiosity/Knowledge
                "explore_knowledge", 
                # Intimacy/Connection
                "express_attraction", "initiate_intimate_interaction", "deepen_connection",
                "express_desire", "respond_to_intimacy", "simulate_physical_touch",
                "seek_gratification", "process_gratification_outcome",
                # Dominance Actions
                "analyze_user_state_for_dominance", "select_dominance_tactic", "issue_command",
                "evaluate_compliance", "apply_consequence_simulated", "praise_submission",
                "increase_control_intensity", "trigger_dominance_gratification", "express_satisfaction"
            ]
            
            # Create descriptions from action names
            tool_descriptions = "\n".join([f"- {action}: {self._generate_action_description(action)}" 
                                          for action in available_actions])

            return Agent(
                name="Goal_Planner_Agent",
                instructions=f"""You are a planner agent for the Nyx AI. Your task is to break down a high-level goal description into a sequence of concrete, actionable steps using Nyx's available actions.

                Available Actions Nyx can perform (these are methods on the main system):
                {tool_descriptions}

                For a given goal, create a detailed plan with logical steps that build on each other. Each step should use results from previous steps when appropriate.

                Your plan should be thorough, considering:
                1. Required inputs for each step
                2. Dependencies between steps
                3. Error handling options
                4. Resources needed
                5. Estimated completion difficulty

                When the goal is associated with a specific need, ensure your plan includes steps that specifically address that need.
                """,
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.1),
                tools=[
                    self._get_available_actions,
                    self._get_action_description,
                    self._get_goal_details,
                    self._get_recent_goals
                ],
                output_type=PlanGenerationResult
            )
        except Exception as e:
            logger.error(f"Error creating planning agent: {e}")
            return None
    
    def _create_goal_validation_agent(self) -> Agent:
        """Creates an agent that validates goals before acceptance"""
        return Agent(
            name="Goal_Validation_Agent",
            instructions="""You are a goal validation agent for Nyx AI. Your task is to evaluate whether a proposed goal:

            1. Is well-defined and clear enough to plan for
            2. Has an appropriate priority level
            3. Is aligned with Nyx's capabilities and purpose
            4. Is ethical and appropriate
            5. Doesn't conflict with existing active goals
            
            If the goal needs adjustment, provide specific feedback. 
            For priority adjustments, consider how important and urgent the goal appears.
            """,
            model="gpt-4.1-nano",
            tools=[
                self._get_active_goals, 
                self._check_goal_conflicts,
                self._verify_capabilities
            ],
            output_type=GoalValidationResult
        )
    
    def _create_plan_validation_agent(self) -> Agent:
        """Creates an agent that validates plans before execution"""
        return Agent(
            name="Plan_Validation_Agent",
            instructions="""You are a plan validation agent for Nyx AI. Your task is to evaluate whether a proposed plan:

            1. Is logically sequenced with proper dependencies
            2. Uses valid actions with correct parameters
            3. Is likely to achieve the stated goal
            4. Handles potential error cases
            5. Uses resources efficiently
            
            Look for issues like:
            - Missing prerequisite steps
            - Invalid action references
            - Unclear parameter definitions
            - Redundant steps or inefficient sequences
            
            Provide specific suggestions for improvement if issues are found.
            """,
            model="gpt-4.1-nano",
            tools=[
                self._validate_action_sequence,
                self._check_parameter_references,
                self._estimate_plan_efficiency
            ],
            output_type=PlanValidationResult
        )
    
    def _create_step_execution_agent(self) -> Agent:
        """Creates an agent that handles step execution and error recovery"""
        return Agent(
            name="Step_Execution_Agent",
            instructions="""You are a step execution agent for Nyx AI. Your task is to:

            1. Oversee the execution of individual goal steps
            2. Resolve parameter references to previous step results
            3. Handle errors and suggest recovery options
            4. Determine whether to continue, retry, skip or abort after each step
            
            When a step fails, consider:
            - Is this a temporary failure that might succeed on retry?
            - Is this step optional or can we skip it?
            - Does this failure require aborting the entire goal?
            
            For dominance-related actions, ensure they meet safety and contextual appropriateness 
            requirements before executing.
            """,
            model="gpt-4.1-nano",
            tools=[
                self._resolve_step_parameters_tool,
                self._execute_action,
                self._check_dominance_appropriateness,
                self._log_execution_result
            ],
            output_type=StepExecutionResult
        )
    
    def _create_orchestration_agent(self) -> Agent:
        """Creates the main orchestration agent that coordinates goal execution"""
        return Agent(
            name="Goal_Orchestration_Agent",
            instructions="""You are the goal orchestration agent for Nyx AI. Your role is to coordinate the entire goal lifecycle:

            1. Validate incoming goals using the validation agent
            2. Generate plans for validated goals using the planning agent
            3. Validate plans before execution
            4. Coordinate step execution through the execution agent
            5. Handle goal completion, failure or abandonment
            
            You should efficiently manage the goal queue, prioritize goals appropriately,
            and ensure resources are allocated effectively across concurrent goals.
            
            Monitor for conflicts between goals and ensure critical goals receive
            priority attention.
            """,
            handoffs=[
                handoff(self.goal_validation_agent, 
                       tool_name_override="validate_goal", 
                       tool_description_override="Validate a goal before acceptance"),
                
                handoff(self.planning_agent, 
                       tool_name_override="generate_plan",
                       tool_description_override="Generate a plan for a validated goal"),
                
                handoff(self.plan_validation_agent,
                       tool_name_override="validate_plan",
                       tool_description_override="Validate a plan before execution"),
                       
                handoff(self.step_execution_agent,
                       tool_name_override="execute_step",
                       tool_description_override="Execute a step in the goal plan")
            ],
            tools=[
                self._get_prioritized_goals_tool,
                self._update_goal_status_tool,
                self._notify_systems,
                self._check_concurrency_limits
            ],
            model="gpt-4.1-nano"
        )
    
    def _create_conflict_resolution_agent(self) -> Agent:
        """Creates an agent for resolving conflicts between goals"""
        return Agent(
            name="Conflict_Resolution_Agent",
            instructions="""You are a conflict resolution agent for Nyx AI. Your task is to:

            1. Analyze conflicting goals to determine their relationship
            2. Suggest an appropriate resolution strategy based on the conflict type
            3. Provide reasoning for your recommended strategy
            
            Consider these factors when analyzing conflicts:
            - Priority levels of conflicting goals
            - Time horizons (short-term vs long-term)
            - Associated needs and emotional motivations
            - Resources required by each goal
            - Potential for compromise or merging

            You should consider these resolution strategies:
            - PRIORITY_BASED: Higher priority goal takes precedence
            - TIME_HORIZON_BASED: Shorter time horizon goal executes first
            - NEGOTIATE: Goals can coexist with adjustments to both
            - MERGE: Goals can be combined into a single more efficient goal
            - USER_DECISION: Conflict is significant enough to require user input
            """,
            model="gpt-4.1-nano",
            tools=[
                self._analyze_goal_similarity,
                self._analyze_resource_conflicts,
                self._get_goal_details
            ]
        )
    
    def _create_negotiation_agent(self) -> Agent:
        """Creates an agent for negotiating between conflicting goals"""
        return Agent(
            name="Goal_Negotiation_Agent",
            instructions="""You are a goal negotiation agent for Nyx AI. Your task is to:

            1. Find compromise solutions between conflicting goals
            2. Adjust goal parameters to allow concurrent execution
            3. Suggest modifications to plans that reduce resource conflicts
            
            Your objective is to find ways for both goals to succeed, possibly
            with modifications rather than having one goal completely defer to another.
            
            Negotiate changes like:
            - Adjusting timing and sequencing
            - Modifying resource usage
            - Finding alternative approaches
            - Splitting goals into stages
            """,
            model="gpt-4.1-nano",
            tools=[
                self._propose_goal_modifications,
                self._evaluate_modification_impact,
                self._get_shared_subgoals
            ]
        )
    
    def _create_merging_agent(self) -> Agent:
        """Creates an agent for merging similar goals"""
        return Agent(
            name="Goal_Merging_Agent",
            instructions="""You are a goal merging agent for Nyx AI. Your task is to:

            1. Analyze similar goals to identify overlaps
            2. Create a new unified goal that captures the intent of both
            3. Generate an efficient plan that achieves the combined objective
            
            When merging goals, ensure:
            - No critical aspects of either original goal are lost
            - The merged goal has appropriate priority and time horizon
            - The emotional motivations are properly combined
            - The merged plan is more efficient than two separate plans
            """,
            model="gpt-4.1-nano",
            tools=[
                self._get_goal_common_elements,
                self._generate_merged_goal_description,
                self._validate_merged_goal_coverage
            ]
        )
    
    #==========================================================================
    # Tool Implementations for Agents
    #==========================================================================
    
    def _generate_action_description(self, action_name: str) -> str:
        """Generate a description for an action based on its name."""
        descriptions = {
            # Basic interaction
            "process_input": "Process and understand user input",
            "generate_response": "Generate a response to the user",
            
            # Knowledge Core
            "query_knowledge": "Query the knowledge base for information on a topic",
            "add_knowledge": "Add new information to the knowledge base",
            
            # Memory Core
            "retrieve_memories": "Retrieve relevant memories based on query and filters",
            "add_memory": "Create and store a new memory",
            "create_reflection": "Create a reflection on experiences or information",
            "create_abstraction": "Create an abstract concept from concrete memories",
            
            # Procedural Memory
            "execute_procedure": "Execute a stored procedure from procedural memory",
            "add_procedure": "Add a new procedure to procedural memory",
            
            # Reasoning Core
            "reason_causal": "Perform causal reasoning on a situation",
            "perform_intervention": "Suggest an intervention based on causal analysis",
            "reason_counterfactually": "Generate counterfactual alternatives",
            
            # Emotional Core
            "update_emotion": "Update Nyx's emotional state",
            "process_emotional_input": "Process and respond to user's emotional content",
            
            # Multimodal/Somatic
            "process_sensory_input": "Process input from sensory systems",
            "add_expectation": "Add a perceptual expectation",
            
            # Meta Core
            "monitor_systems": "Monitor the state of Nyx's cognitive systems",
            "evaluate_cognition": "Evaluate cognitive performance on a task",
            "select_strategy": "Select an appropriate cognitive strategy",
            
            # Prediction Engine
            "generate_prediction": "Generate a prediction about future events or states",
            "evaluate_prediction": "Evaluate the accuracy of a previous prediction",
            
            # Curiosity/Knowledge
            "explore_knowledge": "Explore a knowledge domain out of curiosity",
            
            # Intimacy/Connection
            "express_attraction": "Express positive affinity towards user/entity",
            "initiate_intimate_interaction": "Propose or start more intimate dialogue/action",
            "deepen_connection": "Actions focused on building trust/familiarity/intimacy",
            "express_desire": "Communicate wanting for closeness, intimacy, or action",
            "respond_to_intimacy": "Process and react to user's intimate actions/words",
            "simulate_physical_touch": "Trigger DSS for simulated touch (requires consent/context checks)",
            "seek_gratification": "Goal-driven action towards achieving satisfaction",
            "process_gratification_outcome": "Handle state changes after gratification event",
            
            # Dominance
            "analyze_user_state_for_dominance": "Assess vulnerability, willingness, resistance",
            "select_dominance_tactic": "Choose psychological, emotional, or physical approach",
            "issue_command": "Give a directive to the user",
            "evaluate_compliance": "Check if user response complies with directive",
            "apply_consequence_simulated": "Describe a consequence (reward/punishment in context)",
            "praise_submission": "Reinforce compliant behavior",
            "increase_control_intensity": "Escalate dominance level",
            "trigger_dominance_gratification": "Internal action marking successful dominance sequence",
            "express_satisfaction": "Express satisfaction after successful dominance interaction"
        }
        
        return descriptions.get(action_name, "Perform a system action")

    # Agent SDK tools for goal validation agent
    @staticmethod
    @function_tool
    async def _get_active_goals(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get currently active and pending goals
        
        Returns:
            Dictionary with active and pending goals
        """
        # Use reader lock for access to goals dictionary
        goals = []
        
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                await self._writer_lock.acquire()
        
        try:
            # Safe to read goals dictionary with reader lock
            for goal_id, goal in self.goals.items():
                if goal.status in ["active", "pending"]:
                    goals.append({
                        "id": goal.id,
                        "description": goal.description,
                        "priority": goal.priority,
                        "source": goal.source,
                        "associated_need": goal.associated_need,
                        "status": goal.status
                    })
        finally:
            # Release reader lock
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._writer_lock.release()
        
        return {
            "active_count": len([g for g in goals if g["status"] == "active"]),
            "pending_count": len([g for g in goals if g["status"] == "pending"]),
            "goals": goals
        }

    @staticmethod
    @function_tool
    async def _check_goal_conflicts(ctx: RunContextWrapper, goal_description: str) -> Dict[str, Any]:
        """
        Check if a proposed goal conflicts with existing goals
        
        Args:
            goal_description: Description of the proposed goal
            
        Returns:
            Conflict information
        """
        conflicts = []
        
        # Use enhanced conflict detection based on configuration
        similarity_threshold = self.conflict_config.similarity_threshold
        
        # Get all active and pending goals with reader lock
        goals = []
        
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                await self._writer_lock.acquire()
        
        try:
            # Safe to read goals dictionary with reader lock
            for goal_id, goal in self.goals.items():
                if goal.status in ["active", "pending"]:
                    goals.append(goal)
        finally:
            # Release reader lock
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._writer_lock.release()
        
        # Check for conflicts
        for goal in goals:
            # Simple overlap detection
            words1 = set(goal.description.lower().split())
            words2 = set(goal_description.lower().split())
            overlap = len(words1.intersection(words2)) / max(1, min(len(words1), len(words2)))
            
            if overlap > similarity_threshold:
                conflicts.append({
                    "goal_id": goal.id,
                    "description": goal.description,
                    "similarity": overlap,
                    "status": goal.status
                })
        
        # Enhanced conflict detection
        if self.conflict_config.resource_conflict_detection and len(conflicts) > 0:
            # Analyze potential resource conflicts for the most similar goals
            most_similar = max(conflicts, key=lambda x: x["similarity"]) if conflicts else None
            
            if most_similar:
                # Simple resource check based on keywords
                resource_keywords = {
                    "computational": ["analyze", "compute", "process", "calculate"],
                    "memory": ["remember", "recall", "memorize"],
                    "knowledge": ["learn", "research", "study"],
                    "attention": ["focus", "concentrate", "attend"]
                }
                
                shared_resources = {}
                
                # Check goal description for resource keywords
                for resource, keywords in resource_keywords.items():
                    goal1_uses = any(keyword in goal_description.lower() for keyword in keywords)
                    goal2_uses = any(keyword in most_similar["description"].lower() for keyword in keywords)
                    
                    if goal1_uses and goal2_uses:
                        shared_resources[resource] = "high"
                    elif goal1_uses or goal2_uses:
                        shared_resources[resource] = "medium"
                
                if shared_resources:
                    most_similar["resource_conflicts"] = shared_resources
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflict_count": len(conflicts),
            "conflicts": conflicts
        }

    @staticmethod
    @function_tool
    async def _verify_capabilities(ctx: RunContextWrapper, required_actions: List[str]) -> Dict[str, Any]:
        """
        Verify if required actions are available in Nyx's capabilities
        
        Args:
            required_actions: List of actions required by the goal
            
        Returns:
            Capability verification results
        """
        available_actions = await self._get_available_actions(ctx)
        available_action_names = [a["name"] for a in available_actions["actions"]]
        
        unavailable = [action for action in required_actions if action not in available_action_names]
        
        return {
            "all_available": len(unavailable) == 0,
            "available_count": len(required_actions) - len(unavailable),
            "unavailable_actions": unavailable
        }
    
    # Agent SDK tools for plan validation agent
    @function_tool(name_override="_validate_action_sequence")
    async def validate_action_sequence(                      # noqa: N802
        ctx: RunContextWrapper,
        plan_json: str | None = None,
    ) -> Dict[str, Any]:
        """
        Validate that a proposed *plan* (list of steps) is internally
        consistent.  The caller supplies the plan as **JSON string** so the
        tool schema remains simple and strict-schema-compatible.
    
        Parameters
        ----------
        plan_json : str
            JSON-encoded list of step dictionaries, each containing at least
            ``action`` and (optionally) ``parameters``.
    
        Returns
        -------
        dict
            ``is_valid``      – boolean  
            ``issue_count``   – number of detected issues  
            ``issues``        – list of human-readable issue strings
        """
        # -------------------------------------------------------------- #
        # 1. Decode & basic sanity-check                                 #
        # -------------------------------------------------------------- #
        issues: list[str] = []
        try:
            plan: list[dict[str, Any]] = json.loads(plan_json or "[]")
            if not isinstance(plan, list):
                raise TypeError("plan_json must decode to a list")
        except Exception as exc:
            err = f"Invalid JSON for plan: {exc}"
            logger.error(err)
            return {"is_valid": False, "issue_count": 1, "issues": [err]}
    
        # Empty plan = nothing to validate (let the caller decide if that’s
        # permissible; here we treat it as valid but warn)
        if not plan:
            return {"is_valid": True, "issue_count": 0, "issues": []}
    
        # -------------------------------------------------------------- #
        # 2. Determine the catalogue of available action names           #
        # -------------------------------------------------------------- #
        try:
            # Preferred: call sibling tool if present on ctx.context
            if hasattr(ctx.context, "_get_available_actions"):
                cat = await ctx.context._get_available_actions(ctx)  # type: ignore[attr-defined]
                avail_actions = {a["name"] for a in cat.get("actions", [])}
            else:
                # Fallback: introspect NyxBrain / runner context for callables
                brain = getattr(ctx.context, "brain", None)
                avail_actions = {
                    name for name in dir(brain) if callable(getattr(brain, name, None))
                } if brain else set()
        except Exception as exc:
            logger.warning("Could not fetch available actions: %s", exc)
            avail_actions = set()
    
        # -------------------------------------------------------------- #
        # 3. Per-step validation                                         #
        # -------------------------------------------------------------- #
        for i, step in enumerate(plan, 1):
            action = step.get("action", "")
            if action not in avail_actions:
                issues.append(f"Step {i}: unknown action '{action}'")
    
            # Parameter references to previous steps
            for p_name, p_val in (step.get("parameters") or {}).items():
                if isinstance(p_val, str) and p_val.startswith("$step_"):
                    piece = p_val.split("_", 1)[-1]  # e.g. "3.result"
                    try:
                        ref = int(piece.split(".")[0])
                    except ValueError:
                        issues.append(f"Step {i}: malformed reference '{p_val}'")
                        continue
                    if ref >= i:
                        issues.append(f"Step {i}: references future/self step {ref}")
    
        # -------------------------------------------------------------- #
        # 4. Outcome                                                     #
        # -------------------------------------------------------------- #
        return {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues,
        }

    @function_tool(name_override="_check_parameter_references")
    async def check_parameter_references(            # noqa: N802
        ctx: RunContextWrapper,
        plan_json: str | None = None,
    ) -> Dict[str, Any]:
        """
        Validate that all `$step_X.*` parameter references in *plan_json*
        point to fields that plausibly exist in the output of earlier steps.
    
        Parameters
        ----------
        plan_json : str
            JSON-encoded list of step dictionaries. Each step should contain
            ``action`` and optional ``parameters``.
    
        Returns
        -------
        dict
            ``is_valid``    – bool  
            ``issue_count`` – int  
            ``issues``      – list[str]
        """
        import json, re, logging
    
        issues: list[str] = []
        try:
            plan: list[dict] = json.loads(plan_json or "[]")
            if not isinstance(plan, list):
                raise TypeError("plan_json must decode to a list")
        except Exception as exc:  # JSON / type errors
            err = f"Invalid plan_json: {exc}"
            logging.getLogger(__name__).error(err)
            return {"is_valid": False, "issue_count": 1, "issues": [err]}
    
        if not plan:
            return {"is_valid": True, "issue_count": 0, "issues": []}
    
        # -----------------------------------------------------------------
        # Helpers – intentionally simple, no external dependencies
        # -----------------------------------------------------------------
        def _guess_output_fields(action_name: str) -> set[str]:
            """
            Heuristic: infer what top-level keys an action might return.
            • If ctx.context has .action_output_schemas, try that.
            • Else common defaults.
            """
            brain = getattr(ctx.context, "brain", None)
            if brain and hasattr(brain, "action_output_schemas"):
                sch = brain.action_output_schemas.get(action_name)
                if isinstance(sch, (list, set)):
                    return set(sch)
                if isinstance(sch, dict):
                    return set(sch.keys())
            # Fallback – assume at least 'result' plus snake-case variants
            return {"result", "output", "data"}
    
        _field_re = re.compile(r"^[a-zA-Z_][\w\.]*$")
    
        def _field_exists(candidates: set[str], dotted_path: str) -> bool:
            """
            Very light check: if the first segment of the dotted path matches
            a known candidate field we treat it as available.
            """
            if not _field_re.match(dotted_path):
                return False
            first, *_ = dotted_path.split(".")
            return first in candidates
    
        # -----------------------------------------------------------------
        # Walk through the plan
        # -----------------------------------------------------------------
        provided: dict[int, set[str]] = {}
        for idx, step in enumerate(plan, start=1):
            action = step.get("action", "")
            provided[idx] = _guess_output_fields(action)
    
            params = step.get("parameters") or {}
            for p_name, p_val in params.items():
                if isinstance(p_val, str) and p_val.startswith("$step_"):
                    # Expect formats like "$step_3.result" or "$step_2.data.id"
                    try:
                        ref_part, field_part = p_val.lstrip("$").split(".", 1)
                        ref_idx = int(ref_part.replace("step_", ""))
                    except Exception:
                        issues.append(f"Step {idx}: malformed reference '{p_val}'")
                        continue
    
                    if ref_idx >= idx:
                        issues.append(f"Step {idx}: references future/self step {ref_idx}")
                    elif ref_idx not in provided:
                        issues.append(f"Step {idx}: references unknown step {ref_idx}")
                    elif not _field_exists(provided[ref_idx], field_part):
                        issues.append(
                            f"Step {idx}: field '{field_part}' may not exist in output of step {ref_idx}"
                        )
    
        return {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues,
        }
    
    def _estimate_action_output_fields(self, action: str) -> List[str]:
        """Estimate what fields an action might output based on its name"""
        # This is a simplified estimate - in a real system, you might have more detailed schema
        common_fields = ["result", "success", "error"]
        
        if action.startswith("query_"):
            return common_fields + ["data", "matches", "count"]
        elif action.startswith("retrieve_"):
            return common_fields + ["items", "count"]
        elif action.startswith("generate_"):
            return common_fields + ["content", "text"]
        elif action.startswith("analyze_"):
            return common_fields + ["analysis", "score", "details"]
        elif action.startswith("evaluate_"):
            return common_fields + ["evaluation", "score", "feedback"]
        else:
            return common_fields
    
    def _check_field_availability(self, available_fields: List[str], field_path: str) -> bool:
        """Check if a field path might be available in the output"""
        if not field_path or not available_fields:
            return False
            
        top_field = field_path.split('.')[0]
        return top_field in available_fields or "result" in available_fields

    @function_tool(name_override="_estimate_plan_efficiency")
    async def estimate_plan_efficiency(                    # noqa: N802
        ctx: RunContextWrapper,
        plan_json: str | None = None,
    ) -> Dict[str, Any]:
        """
        Robust heuristic evaluation of plan *efficiency*.
    
        Parameters
        ----------
        plan_json : str
            JSON-encoded list of step dictionaries.  The schema is intentionally
            loose to stay compatible with strict-schema rules.
    
        Returns
        -------
        dict
            step_count, action_distribution, efficiency_score, suggestions
        """
        import json, logging, math, hashlib
        from collections import Counter
    
        logger = logging.getLogger(__name__)
    
        # ------------------------------------------------------------------ #
        # 1) Parse & validate input                                          #
        # ------------------------------------------------------------------ #
        try:
            plan: list[dict] = json.loads(plan_json or "[]")
            if not isinstance(plan, list):
                raise TypeError("plan_json must decode to a list")
        except Exception as exc:
            msg = f"Invalid plan_json supplied: {exc}"
            logger.error(msg)
            return {
                "step_count": 0,
                "action_distribution": {},
                "efficiency_score": 0.0,
                "suggestions": [msg],
            }
    
        if not plan:
            return {
                "step_count": 0,
                "action_distribution": {},
                "efficiency_score": 1.0,
                "suggestions": ["Plan is empty – nothing to optimise."],
            }
    
        # ------------------------------------------------------------------ #
        # 2) Helper functions                                                #
        # ------------------------------------------------------------------ #
        def _category(action: str) -> str:
            if action.startswith(("query_", "retrieve_")):
                return "retrieval"
            if action.startswith(("add_", "create_")):
                return "creation"
            if action.startswith(("update_", "modify_")):
                return "modification"
            if action.startswith(("analyze_", "evaluate_", "reason_")):
                return "analysis"
            if action.startswith(("generate_", "express_")):
                return "generation"
            return "unknown"
    
        def _shannon_entropy(counter: Counter) -> float:
            """Normalized entropy ∈ [0,1] (1 = perfectly uniform)."""
            if not counter:
                return 1.0
            total = sum(counter.values())
            probs = [v / total for v in counter.values() if v]
            if not probs:
                return 1.0
            entropy = -sum(p * math.log2(p) for p in probs)
            max_entropy = math.log2(len(counter))
            return entropy / max_entropy if max_entropy else 1.0
    
        # ------------------------------------------------------------------ #
        # 3) Analyse the plan                                                #
        # ------------------------------------------------------------------ #
        n_steps = len(plan)
        categories = Counter(_category(s.get("action", "")) for s in plan)
    
        # --- Duplicate-work detection ------------------------------------- #
        step_signature = lambda s: hashlib.sha1(                      # noqa: E731
            json.dumps(
                {"action": s.get("action"), "parameters": s.get("parameters", {})},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        sigs = [step_signature(s) for s in plan]
        duplicate_hits = len(sigs) - len(set(sigs))
    
        # --- Parallelism estimate ----------------------------------------- #
        serial_dependencies = sum(
            1
            for s in plan
            for v in (s.get("parameters") or {}).values()
            if isinstance(v, str) and v.startswith("$step_")
        )
        parallelisable_ratio = 1.0 - (serial_dependencies / n_steps)
    
        # ------------------------------------------------------------------ #
        # 4) Build suggestions & penalties                                   #
        # ------------------------------------------------------------------ #
        suggestions: list[str] = []
    
        # (a) Category skew (low entropy → skew)
        entropy = _shannon_entropy(categories)
        cat_penalty = (1 - entropy) * 0.25  # up to −0.25
        if entropy < 0.6:
            suggestions.append("Action mix is skewed; consider balancing categories.")
    
        # (b) Heavy retrieval / creation
        retrieval_heavy = categories.get("retrieval", 0) > n_steps * 0.5
        creation_heavy  = categories.get("creation", 0)  > n_steps * 0.4
        cat_specific_penalty = 0.0
        if retrieval_heavy:
            cat_specific_penalty += 0.15
            suggestions.append("Combine or batch retrieval steps to reduce latency.")
        if creation_heavy:
            cat_specific_penalty += 0.10
            suggestions.append("Batch creation steps where possible to save overhead.")
    
        # (c) Plan length
        length_penalty = 0.0
        if n_steps > 8:
            length_penalty = min(0.20, math.log(n_steps - 7, 10) * 0.10)
            suggestions.append("Plan is long; investigate merging related steps.")
    
        # (d) Duplicate work
        dup_penalty = min(0.20, 0.05 * duplicate_hits)
        if duplicate_hits:
            suggestions.append(f"Found {duplicate_hits} duplicate steps; consider deduplication.")
    
        # (e) Parallelism
        parallel_penalty = 0.0
        if parallelisable_ratio < 0.4:
            parallel_penalty = (0.4 - parallelisable_ratio) * 0.375  # up to −0.15
            suggestions.append(
                "Low parallelism; many steps reference previous outputs. "
                "Evaluate which can run concurrently."
            )
    
        # ------------------------------------------------------------------ #
        # 5) Final score                                                     #
        # ------------------------------------------------------------------ #
        score = 1.0 - sum(
            [cat_penalty, cat_specific_penalty, length_penalty, dup_penalty, parallel_penalty]
        )
        score = round(max(0.0, score), 3)
    
        return {
            "step_count": n_steps,
            "action_distribution": dict(categories),
            "efficiency_score": score,
            "suggestions": suggestions,
        }
    
    @staticmethod
    @function_tool  # strict_json_schema=True by default
    async def _resolve_step_parameters_tool(                  # noqa: N802
        ctx: RunContextWrapper,
        goal_id: str,
        parameters_json: str,                                 # <- plain STRING
    ) -> StepParameterResolutionResult:
        """
        Resolve placeholders contained in *parameters_json*.
    
        *Input* is a JSON string so the schema remains closed.
        """
        import logging, asyncstdlib
    
        logger = logging.getLogger(__name__)
    
        # 1️⃣ Parse caller input ------------------------------------------------
        try:
            raw_params: Dict[str, Any] = json.loads(parameters_json or "{}")
            if not isinstance(raw_params, dict):
                raise TypeError("parameters_json must decode to an object")
        except Exception as exc:
            logger.error("Bad parameters_json: %s", exc)
            return StepParameterResolutionResult(
                resolved_parameters=[], resolution_status=[],
                all_resolved=False, raw_json=None
            )
    
        # 2️⃣ Delegate to the instance helper -----------------------------------
        gm = ctx.context                                # GoalManager instance
        resolved: Dict[str, Any] = await gm._resolve_step_parameters(goal_id, raw_params)
    
        # 3️⃣ Build scalar-only result lists ------------------------------------
        resolved_list: List[ResolvedParameter] = []
        statuses:      List[ParameterStatus]    = []
    
        for k, orig in raw_params.items():
            v = resolved.get(k)
    
            # encode complex objects so value is scalar
            if not isinstance(v, (str, int, float, bool)) and v is not None:
                v = json.dumps(v, separators=(",", ":"))
    
            resolved_list.append(ResolvedParameter(name=k, value=v))
    
            if isinstance(orig, str) and orig.startswith("$step_"):
                statuses.append(
                    ParameterStatus(
                        name=k, original=orig,
                        resolved=k in resolved and resolved[k] is not None,
                        is_null=(resolved.get(k) is None),
                    )
                )
    
        return StepParameterResolutionResult(
            resolved_parameters=resolved_list,
            resolution_status=statuses,
            all_resolved=all(s.resolved for s in statuses) if statuses else True,
            raw_json=json.dumps(resolved, separators=(",", ":")),
        )
        
    @staticmethod
    @function_tool
    async def _execute_action(                           # noqa: N802
        ctx: RunContextWrapper,
        action: str,
        parameters_json: str,                            # strict input
    ) -> ExecutionOutcome:                               # strict output
        """
        Execute *action* with parameters provided as JSON string.
        """
        gm = ctx.context                                   # GoalManager instance
        if not gm.brain:
            return ExecutionOutcome(
                success=False, duration=0.0,
                error="NyxBrain reference not set in GoalManager"
            )
    
        try:
            params: Dict[str, Any] = json.loads(parameters_json or "{}")
        except Exception as exc:
            return ExecutionOutcome(success=False, duration=0.0, error=f"Bad JSON: {exc}")
    
        action_method = getattr(gm.brain, action, None)
        if not (action_method and callable(action_method)):
            return ExecutionOutcome(
                success=False, duration=0.0,
                error=f"Action '{action}' not found on NyxBrain"
            )
    
        start = datetime.datetime.now()
        try:
            result = await action_method(**params)
            duration = (datetime.datetime.now() - start).total_seconds()
            res_json = (
                json.dumps(result, separators=(",", ":"))
                if not isinstance(result, (str, int, float, bool, type(None)))
                else json.dumps(result)
            )
            return ExecutionOutcome(
                success=True, duration=duration, result_json=res_json
            )
        except Exception as exc:
            duration = (datetime.datetime.now() - start).total_seconds()
            return ExecutionOutcome(
                success=False, duration=duration,
                error=str(exc), exception_type=type(exc).__name__
            )
            
    @staticmethod
    @function_tool
    async def _check_dominance_appropriateness(          # noqa: N802
        ctx: RunContextWrapper,
        action: str,
        parameters_json: str,
    ) -> DominanceCheckResult:
        """
        Strict-schema wrapper around the dominance appropriateness check.
        """
        gm = ctx.context
        try:
            params = json.loads(parameters_json or "{}")
        except Exception as exc:
            return DominanceCheckResult(
                is_dominance_action=False, can_proceed=False,
                action="block", reason=f"Bad JSON: {exc}"
            )
    
        is_dom = action in [
            "issue_command", "increase_control_intensity",
            "apply_consequence_simulated", "select_dominance_tactic",
            "trigger_dominance_gratification", "praise_submission",
        ]
        if not is_dom:
            return DominanceCheckResult(
                is_dominance_action=False, can_proceed=True, action="proceed"
            )
    
        user_id = params.get("user_id") or params.get("target_user_id")
        if not user_id:
            return DominanceCheckResult(
                is_dominance_action=True, can_proceed=False,
                action="block", reason="Missing user_id"
            )
    
        if gm.brain and hasattr(gm.brain, "_evaluate_dominance_step_appropriateness"):
            evaluation = await gm.brain._evaluate_dominance_step_appropriateness(
                action, params, user_id
            )
            return DominanceCheckResult(
                is_dominance_action=True,
                can_proceed=evaluation.get("action") == "proceed",
                action=evaluation.get("action", "block"),
                reason=evaluation.get("reason"),
                evaluation_json=json.dumps(evaluation, separators=(",", ":")),
            )
    
        return DominanceCheckResult(
            is_dominance_action=True, can_proceed=False,
            action="block", reason="No dominance evaluation available"
        )
        
    @staticmethod
    @function_tool
    async def _log_execution_result(                     # noqa: N802
        ctx: RunContextWrapper,
        goal_id: str,
        step_id: str,
        execution_result_json: str,
    ) -> ExecutionLogStatus:
        """
        Store an execution result (JSON string) strictly.
        """
        gm = ctx.context
        try:
            exec_result = json.loads(execution_result_json or "{}")
        except Exception as exc:
            return ExecutionLogStatus(
                success=False, step_index=-1, current_index=-1,
                step_status="failed", error=f"Bad JSON: {exc}"
            )
    
        # The rest is your original logic, unchanged except JSON decoding
        async with gm._get_goal_lock(goal_id):
            if goal_id not in gm.goals:
                return ExecutionLogStatus(
                    success=False, step_index=-1, current_index=-1,
                    step_status="failed", error=f"Goal {goal_id} not found"
                )
            goal = gm.goals[goal_id]
            step = next((s for s in goal.plan if s.step_id == step_id), None)
            if not step:
                return ExecutionLogStatus(
                    success=False, step_index=-1, current_index=-1,
                    step_status="failed", error=f"Step {step_id} not found"
                )
    
            step.status = "completed" if exec_result.get("success") else "failed"
            step.result = exec_result.get("result_json")
            step.error  = exec_result.get("error")
            if not step.start_time:
                step.start_time = datetime.datetime.now()
            step.end_time = datetime.datetime.now()
    
            goal.execution_history.append({
                "step_id": step_id,
                "status": step.status,
                "timestamp": step.end_time.isoformat(),
                "duration": exec_result.get("duration", 0),
                "error": step.error,
            })
            await gm.mark_goal_dirty(goal_id)
    
            idx = goal.plan.index(step)
            return ExecutionLogStatus(
                success=True, step_index=idx,
                current_index=goal.current_step_index,
                step_status=step.status
            )
    
    # Agent SDK tools for orchestration agent
    @staticmethod
    @function_tool
    async def _get_prioritized_goals_tool(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get prioritized goals for execution
        
        Returns:
            Prioritized goals information
        """
        # Call the async version of the method
        goals = await self.get_prioritized_goals()

        return {
            "total_count": len(goals),
            "active_count": len([g for g in goals if g.status == "active"]),
            "pending_count": len([g for g in goals if g.status == "pending"]),
            # Return summaries for the tool, excluding large fields
            "goals": [g.model_dump(exclude={'plan', 'execution_history', 'checksum', 'emotional_state_snapshots', 'external_feedback'}) for g in goals[:5]]
        }

    @staticmethod
    @function_tool
    async def _update_goal_status_tool(                  # noqa: N802
        ctx: RunContextWrapper,
        goal_id: str,
        status: str,
        result_json: Optional[str] = None,
        error: Optional[str] = None,
    ) -> UpdateGoalStatusResult:
        """
        Strict replacement for status updates.
        """
        gm = ctx.context
        valid = {"pending", "active", "completed", "failed", "abandoned"}
        if status not in valid:
            return UpdateGoalStatusResult(
                success=False, goal_id=goal_id, new_status=status,
                notifications_json=None,
                old_status=None,
            )
    
        upd = await gm.update_goal_status(goal_id, status, result_json, error)
        return UpdateGoalStatusResult(
            success=upd.get("success", False),
            goal_id=goal_id,
            new_status=status,
            old_status=upd.get("old_status"),
            notifications_json=json.dumps(upd.get("notifications", {}), separators=(",", ":")),
        )
        
    @staticmethod
    @function_tool
    async def _notify_systems(                             # noqa: N802
        ctx: RunContextWrapper,
        goal_id: str,
        status: str,
        result_json: Optional[str] = None,
        error: Optional[str] = None,
    ) -> NotifySystemsResult:
        """
        Notify internal & external subsystems of a goal-status change (STRICT).
    
        * `result_json` carries any arbitrary result object as a JSON string so the
          schema stays closed.
        """
        gm = ctx.context                                     # GoalManager instance
        logger = logging.getLogger(__name__)
    
        # ------------------------------------------------------------------ #
        # 0. Decode result_json (safe fallback)                              #
        # ------------------------------------------------------------------ #
        try:
            result_obj: Optional[Any] = (
                json.loads(result_json) if isinstance(result_json, str) else result_json
            )
        except Exception as exc:
            logger.warning("Bad JSON in result_json: %s", exc)
            result_obj = result_json
    
        # ------------------------------------------------------------------ #
        # 1. Fetch goal (reader lock)                                        #
        # ------------------------------------------------------------------ #
        goal = await gm._get_goal_with_reader_lock(goal_id)
        if not goal:
            return NotifySystemsResult(
                success=False, goal_id=goal_id, status=status,
                error=f"Goal {goal_id} not found"
            )
    
        notifications: Dict[str, Any] = {}
    
        # ------------------------------------------------------------------ #
        # 2. NeedsSystem                                                    #
        # ------------------------------------------------------------------ #
        if goal.associated_need and gm.brain and hasattr(gm.brain, "needs_system"):
            try:
                ns = gm.brain.needs_system          # type: ignore[attr-defined]
                if status == "completed" and hasattr(ns, "satisfy_need"):
                    amt = goal.priority * 0.3 + 0.1
                    await ns.satisfy_need(goal.associated_need, amt)
                    notifications["needs_system"] = {
                        "success": True, "need": goal.associated_need,
                        "amount": amt, "action": "satisfy"
                    }
                elif status == "failed" and hasattr(ns, "decrease_need"):
                    amt = goal.priority * 0.1
                    await ns.decrease_need(goal.associated_need, amt)
                    notifications["needs_system"] = {
                        "success": True, "need": goal.associated_need,
                        "amount": amt, "action": "decrease"
                    }
            except Exception as exc:
                notifications["needs_system"] = {"success": False, "error": str(exc)}
    
        # ------------------------------------------------------------------ #
        # 3. RewardSystem                                                   #
        # ------------------------------------------------------------------ #
        if gm.brain and hasattr(gm.brain, "reward_system"):
            try:
                rs = gm.brain.reward_system        # type: ignore[attr-defined]
                rv = 0.0
                if status == "completed":
                    rv = goal.priority * 0.6
                elif status == "failed":
                    rv = -goal.priority * 0.4
                elif status == "abandoned":
                    rv = -0.1
                if abs(rv) > 0.05 and hasattr(rs, "process_reward_signal"):
                    from nyx.core.reward_system import RewardSignal  # local import
                    sig = RewardSignal(
                        value=rv, source="GoalManager",
                        context={
                            "goal_id": goal_id,
                            "goal_description": goal.description,
                            "outcome": status,
                            "associated_need": goal.associated_need,
                        },
                        timestamp=datetime.datetime.now().isoformat(),
                    )
                    await rs.process_reward_signal(sig)
                    notifications["reward_system"] = {
                        "success": True, "reward_value": rv, "source": "GoalManager"
                    }
            except Exception as exc:
                notifications["reward_system"] = {"success": False, "error": str(exc)}
    
        # ------------------------------------------------------------------ #
        # 4. MetaCore                                                       #
        # ------------------------------------------------------------------ #
        if gm.brain and hasattr(gm.brain, "meta_core"):
            try:
                mc = gm.brain.meta_core            # type: ignore[attr-defined]
                if hasattr(mc, "record_goal_outcome"):
                    await mc.record_goal_outcome(goal.model_dump())
                    notifications["meta_core"] = {
                        "success": True, "recorded_goal": goal_id, "status": status
                    }
            except Exception as exc:
                notifications["meta_core"] = {"success": False, "error": str(exc)}
    
        # ------------------------------------------------------------------ #
        # 5. Integration callbacks                                          #
        # ------------------------------------------------------------------ #
        try:
            cb = await gm.trigger_integration_callbacks("goal_status_change", {
                "goal_id": goal_id,
                "old_status": goal.status,
                "new_status": status,
                "result": result_obj,
                "error": error,
            })
            if cb:
                notifications["integration_callbacks"] = cb
        except Exception as exc:
            notifications["integration_callbacks"] = {"success": False, "error": str(exc)}
    
        # ------------------------------------------------------------------ #
        # 6. External systems                                               #
        # ------------------------------------------------------------------ #
        if goal.external_system_ids:
            for sys_name, ext_id in goal.external_system_ids.items():
                if sys_name in gm._external_system_clients:
                    try:
                        notifications[f"external_{sys_name}"] = {
                            "success": True, "external_id": ext_id, "status": status
                        }
                    except Exception as exc:
                        notifications[f"external_{sys_name}"] = {
                            "success": False, "error": str(exc)
                        }
    
        # ------------------------------------------------------------------ #
        # 7. Return                                                          #
        # ------------------------------------------------------------------ #
        return NotifySystemsResult(
            success=True,
            goal_id=goal_id,
            status=status,
            notifications_json=json.dumps(notifications, separators=(",", ":")),
        )
    
    @staticmethod
    @function_tool
    async def _check_concurrency_limits(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Check if more goals can be activated based on concurrency limits
        
        Returns:
            Concurrency information
        """
        # Get count of active goals with lock
        async with self._active_goals_lock:
            active_count = len(self.active_goals)
            can_activate = active_count < self.max_concurrent_goals
            remaining_slots = max(0, self.max_concurrent_goals - active_count)
            
            active_goals = []
            # Get details of active goals
            for goal_id in self.active_goals:
                goal = await self._get_goal_with_reader_lock(goal_id)
                if goal:
                    active_goals.append({
                        "id": goal_id,
                        "description": goal.description,
                        "priority": goal.priority
                    })
        
        return {
            "active_count": active_count,
            "max_concurrent": self.max_concurrent_goals,
            "can_activate_more": can_activate,
            "remaining_slots": remaining_slots,
            "active_goals": active_goals
        }
    
    # Additional tools used by multiple agents
    @staticmethod
    @function_tool
    async def _get_available_actions(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get available actions that can be used in plans
        
        Returns:
            Available actions with descriptions
        """
        available_actions = [
            # Basic interaction
            "process_input", "generate_response", 
            # Knowledge Core
            "query_knowledge", "add_knowledge", 
            # Memory Core
            "retrieve_memories", "add_memory", "create_reflection", "create_abstraction", 
            # Procedural Memory
            "execute_procedure", "add_procedure", 
            # Reasoning Core
            "reason_causal", "perform_intervention", "reason_counterfactually", 
            # Emotional Core
            "update_emotion", "process_emotional_input", 
            # Multimodal/Somatic
            "process_sensory_input", "add_expectation", 
            # Meta Core
            "monitor_systems", "evaluate_cognition", "select_strategy", 
            # Prediction Engine
            "generate_prediction", "evaluate_prediction", 
            # Curiosity/Knowledge
            "explore_knowledge", 
            # Intimacy/Connection
            "express_attraction", "initiate_intimate_interaction", "deepen_connection",
            "express_desire", "respond_to_intimacy", "simulate_physical_touch",
            "seek_gratification", "process_gratification_outcome",
            # Dominance Actions
            "analyze_user_state_for_dominance", "select_dominance_tactic", "issue_command",
            "evaluate_compliance", "apply_consequence_simulated", "praise_submission",
            "increase_control_intensity", "trigger_dominance_gratification", "express_satisfaction"
        ]
        
        # Build list with descriptions
        actions_with_descriptions = []
        for action in available_actions:
            description = self._generate_action_description(action)
            actions_with_descriptions.append({
                "name": action,
                "description": description
            })
        
        return {
            "count": len(actions_with_descriptions),
            "actions": actions_with_descriptions
        }

    @staticmethod
    @function_tool
    async def _get_action_description(ctx: RunContextWrapper, action: str) -> Dict[str, Any]:
        """
        Get a description for a specific action
        
        Args:
            action: The action name
            
        Returns:
            Action description
        """
        description = self._generate_action_description(action)
        actions_result = await self._get_available_actions(ctx)
        available_action_names = [a["name"] for a in actions_result["actions"]]
        
        return {
            "action": action,
            "description": description,
            "is_available": action in available_action_names
        }

    @staticmethod
    @function_tool
    async def _get_goal_details(ctx: RunContextWrapper, goal_id: str) -> Dict[str, Any]:
        """
        Get details about a specific goal
        
        Args:
            goal_id: The goal ID
            
        Returns:
            Goal details
        """
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return {
                "success": False,
                "error": f"Goal {goal_id} not found"
            }
        
        # Return goal details without large fields
        return {
            "success": True,
            "id": goal.id,
            "description": goal.description,
            "status": goal.status,
            "priority": goal.priority,
            "source": goal.source,
            "associated_need": goal.associated_need,
            "creation_time": goal.creation_time.isoformat(),
            "completion_time": goal.completion_time.isoformat() if goal.completion_time else None,
            "deadline": goal.deadline.isoformat() if goal.deadline else None,
            "has_plan": len(goal.plan) > 0,
            "plan_step_count": len(goal.plan),
            "current_step_index": goal.current_step_index,
            "time_horizon": goal.time_horizon,
            "has_emotional_motivation": goal.emotional_motivation is not None,
            "primary_need": goal.emotional_motivation.primary_need if goal.emotional_motivation else None,
            "last_error": goal.last_error,
            "last_modified": goal.last_modified.isoformat(),
            "has_conflicts": len(goal.conflict_data) > 0 if goal.conflict_data else False
        }

    @staticmethod
    @function_tool
    async def _get_recent_goals(ctx: RunContextWrapper, limit: int = 3) -> Dict[str, Any]:
        """
        Get recently completed goals
        
        Args:
            limit: Maximum number of goals to return
            
        Returns:
            Recent goals
        """
        # Get all goals with reader lock
        all_goals = []
        
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                await self._writer_lock.acquire()
        
        try:
            # Safe to read goals dictionary with reader lock
            all_goals = [
                goal.model_copy() for goal in self.goals.values()
                if goal.status == "completed" and goal.completion_time is not None
            ]
        finally:
            # Release reader lock
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._writer_lock.release()
        
        # Sort by completion time (newest first)
        all_goals.sort(key=lambda g: g.completion_time, reverse=True)
        
        # Get recent goals
        recent_goals = []
        for goal in all_goals[:limit]:
            recent_goals.append({
                "id": goal.id,
                "description": goal.description,
                "completion_time": goal.completion_time.isoformat(),
                "priority": goal.priority,
                "source": goal.source,
                "associated_need": goal.associated_need,
                "time_horizon": goal.time_horizon,
                "steps": [
                    {
                        "description": step.description,
                        "action": step.action
                    }
                    for step in goal.plan[:3]  # First 3 steps of each goal
                ]
            })
        
        return {
            "count": len(recent_goals),
            "goals": recent_goals
        }
    
    # Conflict resolution tools
    @staticmethod
    @function_tool
    async def _analyze_goal_similarity(ctx: RunContextWrapper, goal_id1: str, goal_id2: str) -> Dict[str, Any]:
        """
        Analyze the similarity between two goals
        
        Args:
            goal_id1: First goal ID
            goal_id2: Second goal ID
            
        Returns:
            Similarity analysis
        """
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {
                "error": "One or both goals not found",
                "similarity_score": 0.0
            }
        
        # Basic text similarity between descriptions
        words1 = set(goal1.description.lower().split())
        words2 = set(goal2.description.lower().split())
        word_overlap = len(words1.intersection(words2)) / max(1, min(len(words1), len(words2)))
        
        # Check for common needs
        need_similarity = 0.0
        if goal1.associated_need and goal2.associated_need:
            need_similarity = 1.0 if goal1.associated_need == goal2.associated_need else 0.0
        
        # Check for emotional motivation similarity
        emotion_similarity = 0.0
        if goal1.emotional_motivation and goal2.emotional_motivation:
            # Compare primary needs
            if goal1.emotional_motivation.primary_need == goal2.emotional_motivation.primary_need:
                emotion_similarity = 0.7
                
                # Compare chemical profiles if primary needs match
                chemicals1 = set(goal1.emotional_motivation.associated_chemicals.keys())
                chemicals2 = set(goal2.emotional_motivation.associated_chemicals.keys())
                if chemicals1 and chemicals2:
                    chem_overlap = len(chemicals1.intersection(chemicals2)) / max(1, min(len(chemicals1), len(chemicals2)))
                    emotion_similarity += chem_overlap * 0.3
        
        # Check for plan similarity if both have plans
        plan_similarity = 0.0
        if goal1.plan and goal2.plan:
            # Count similar actions
            actions1 = [step.action for step in goal1.plan]
            actions2 = [step.action for step in goal2.plan]
            
            # Get unique actions
            unique_actions = set(actions1 + actions2)
            
            # Count occurrences in each plan
            counts1 = {action: actions1.count(action) for action in unique_actions}
            counts2 = {action: actions2.count(action) for action in unique_actions}
            
            # Calculate similarity based on action frequency correlation
            similarity_sum = 0
            for action in unique_actions:
                similarity_sum += min(counts1.get(action, 0), counts2.get(action, 0))
            
            plan_similarity = similarity_sum / max(1, max(len(actions1), len(actions2)))
        
        # Combine all similarity factors
        weighted_similarity = (
            word_overlap * 0.4 +
            need_similarity * 0.2 +
            emotion_similarity * 0.2 +
            plan_similarity * 0.2
        )
        
        # Analyze potential for merging or conflict
        merge_potential = "high" if weighted_similarity > 0.7 else "medium" if weighted_similarity > 0.4 else "low"
        conflict_potential = "high" if 0.3 < weighted_similarity < 0.7 else "medium" if 0.1 < weighted_similarity < 0.3 else "low"
        
        return {
            "similarity_score": weighted_similarity,
            "text_similarity": word_overlap,
            "need_similarity": need_similarity,
            "emotion_similarity": emotion_similarity,
            "plan_similarity": plan_similarity,
            "merge_potential": merge_potential,
            "conflict_potential": conflict_potential,
            "is_duplicate": weighted_similarity > 0.85
        }

    @staticmethod
    @function_tool
    async def _analyze_resource_conflicts(ctx: RunContextWrapper, goal_id1: str, goal_id2: str) -> Dict[str, Any]:
        """
        Analyze resource conflicts between two goals
        
        Args:
            goal_id1: First goal ID
            goal_id2: Second goal ID
            
        Returns:
            Resource conflict analysis
        """
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {
                "error": "One or both goals not found",
                "has_conflicts": False
            }
        
        # Simplified resource analysis based on action patterns
        # A more sophisticated system would have explicit resource tracking
        
        # Extract actions from plans
        actions1 = [step.action for step in goal1.plan] if goal1.plan else []
        actions2 = [step.action for step in goal2.plan] if goal2.plan else []
        
        # Define resource-intensive action patterns
        resource_patterns = {
            "computational": ["reason", "analyze", "evaluate", "generate", "predict"],
            "knowledge": ["query_knowledge", "explore_knowledge"],
            "memory": ["retrieve_memories", "add_memory"],
            "emotional": ["update_emotion", "process_emotional_input"],
            "attention": ["process_input", "process_sensory_input"]
        }
        
        # Count resource usage
        resource_usage1 = {res: 0 for res in resource_patterns}
        resource_usage2 = {res: 0 for res in resource_patterns}
        
        for action in actions1:
            for resource, patterns in resource_patterns.items():
                if any(pattern in action for pattern in patterns):
                    resource_usage1[resource] += 1
        
        for action in actions2:
            for resource, patterns in resource_patterns.items():
                if any(pattern in action for pattern in patterns):
                    resource_usage2[resource] += 1
        
        # Identify conflicts
        conflicts = []
        for resource in resource_patterns:
            # Simple threshold-based conflict detection
            if resource_usage1[resource] > 2 and resource_usage2[resource] > 2:
                conflicts.append({
                    "resource": resource,
                    "usage_goal1": resource_usage1[resource],
                    "usage_goal2": resource_usage2[resource],
                    "severity": "high" if resource_usage1[resource] + resource_usage2[resource] > 8 else "medium"
                })
        
        # Check time-based conflicts
        time_conflict = False
        time_conflict_reason = None
        
        # Check for deadline overlaps
        if goal1.deadline and goal2.deadline:
            # If both deadlines are within 24 hours of each other
            time_diff = abs((goal1.deadline - goal2.deadline).total_seconds())
            if time_diff < 86400:  # 24 hours in seconds
                time_conflict = True
                time_conflict_reason = "deadlines_close"
        
        # Check for time horizon conflicts
        time_horizon_conflict = False
        if goal1.time_horizon != goal2.time_horizon:
            # Short-term and long-term goals might conflict in scheduling
            if (goal1.time_horizon == TimeHorizon.SHORT_TERM and goal2.time_horizon == TimeHorizon.LONG_TERM) or \
               (goal1.time_horizon == TimeHorizon.LONG_TERM and goal2.time_horizon == TimeHorizon.SHORT_TERM):
                time_horizon_conflict = True
        
        return {
            "has_conflicts": len(conflicts) > 0 or time_conflict,
            "resource_conflicts": conflicts,
            "time_conflict": time_conflict,
            "time_conflict_reason": time_conflict_reason,
            "time_horizon_conflict": time_horizon_conflict,
            "total_conflict_severity": "high" if len(conflicts) > 2 or time_conflict else "medium" if conflicts else "low"
        }

    @staticmethod
    @function_tool
    async def _propose_goal_modifications(ctx: RunContextWrapper, goal_id: str, 
                                       conflict_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose modifications to a goal to resolve conflicts
        
        Args:
            goal_id: Goal ID to modify
            conflict_analysis: Conflict analysis results
            
        Returns:
            Proposed modifications
        """
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return {"error": "Goal not found"}
        
        modifications = []
        
        # Check for resource conflicts
        if conflict_analysis.get("resource_conflicts"):
            resource_conflicts = conflict_analysis["resource_conflicts"]
            
            for conflict in resource_conflicts:
                resource = conflict["resource"]
                severity = conflict["severity"]
                
                if resource == "computational" and severity == "high":
                    modifications.append({
                        "type": "plan_optimization",
                        "description": "Optimize computational steps by combining or caching results",
                        "impact": "medium"
                    })
                
                elif resource == "knowledge" or resource == "memory":
                    modifications.append({
                        "type": "resource_sharing",
                        "description": f"Share {resource} retrieval results between conflicting goals",
                        "impact": "high"
                    })
                
                elif resource == "emotional" and severity == "high":
                    modifications.append({
                        "type": "priority_adjustment",
                        "description": "Reduce priority temporarily to delay execution",
                        "impact": "medium",
                        "new_priority": max(0.1, goal.priority - 0.2)
                    })
        
        # Check for time conflicts
        if conflict_analysis.get("time_conflict"):
            # If deadline is causing conflict
            if conflict_analysis.get("time_conflict_reason") == "deadlines_close" and goal.deadline:
                # Propose adjusting deadline
                new_deadline = goal.deadline + datetime.timedelta(hours=24)
                modifications.append({
                    "type": "deadline_adjustment",
                    "description": "Extend deadline by 24 hours",
                    "impact": "medium",
                    "new_deadline": new_deadline.isoformat()
                })
        
        # Check for time horizon conflicts
        if conflict_analysis.get("time_horizon_conflict"):
            if goal.time_horizon == TimeHorizon.SHORT_TERM:
                # No change to short-term goals
                pass
            elif goal.time_horizon == TimeHorizon.MEDIUM_TERM:
                modifications.append({
                    "type": "time_horizon_adjustment",
                    "description": "Adjust to long-term horizon to reduce scheduling pressure",
                    "impact": "medium",
                    "new_time_horizon": TimeHorizon.LONG_TERM
                })
            elif goal.time_horizon == TimeHorizon.LONG_TERM:
                # Suggest breaking into stages
                modifications.append({
                    "type": "goal_staging",
                    "description": "Break goal into immediate and future stages",
                    "impact": "high",
                    "requires_restructuring": True
                })
        
        # If conflict severity is high and no good modifications found
        if conflict_analysis.get("total_conflict_severity") == "high" and len(modifications) < 2:
            modifications.append({
                "type": "priority_adjustment",
                "description": "Significantly reduce priority to defer to other goal",
                "impact": "high",
                "new_priority": max(0.1, goal.priority - 0.3)
            })
        
        return {
            "goal_id": goal_id,
            "proposed_modifications": modifications,
            "modification_count": len(modifications),
            "requires_user_input": any(mod.get("requires_user_input", False) for mod in modifications)
        }

    @staticmethod
    @function_tool
    async def _evaluate_modification_impact(ctx: RunContextWrapper, goal_id: str, 
                                         modifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the impact of proposed goal modifications
        
        Args:
            goal_id: Goal ID
            modifications: List of proposed modifications
            
        Returns:
            Impact evaluation
        """
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return {"error": "Goal not found"}
        
        impacts = []
        overall_impact_score = 0.0
        
        for mod in modifications:
            mod_type = mod.get("type")
            impact_level = {
                "low": 0.2,
                "medium": 0.5,
                "high": 0.8
            }.get(mod.get("impact", "medium"), 0.5)
            
            impact_data = {
                "type": mod_type,
                "score": impact_level,
                "affects_outcome": False,
                "affects_timing": False,
                "affects_resources": False
            }
            
            # Evaluate based on modification type
            if mod_type == "priority_adjustment":
                new_priority = mod.get("new_priority", goal.priority)
                priority_change = abs(new_priority - goal.priority)
                
                impact_data["affects_timing"] = True
                if priority_change > 0.3:
                    impact_data["affects_outcome"] = True
                    impact_data["assessment"] = "May significantly delay goal completion"
                else:
                    impact_data["assessment"] = "Moderate delay in execution"
            
            elif mod_type == "deadline_adjustment":
                impact_data["affects_timing"] = True
                impact_data["assessment"] = "Extended timeline for completion"
            
            elif mod_type == "time_horizon_adjustment":
                impact_data["affects_timing"] = True
                impact_data["affects_outcome"] = True
                impact_data["assessment"] = "Shifts goal to longer-term perspective"
            
            elif mod_type == "goal_staging":
                impact_data["affects_timing"] = True
                impact_data["affects_outcome"] = True
                impact_data["affects_resources"] = True
                impact_data["assessment"] = "Significant restructuring of goal execution"
            
            elif mod_type == "plan_optimization":
                impact_data["affects_resources"] = True
                impact_data["assessment"] = "Reduces resource usage but may require plan changes"
            
            elif mod_type == "resource_sharing":
                impact_data["affects_resources"] = True
                impact_data["assessment"] = "Enables more efficient concurrent execution"
            
            impacts.append(impact_data)
            overall_impact_score += impact_level
        
        # Normalize overall impact
        if impacts:
            overall_impact_score /= len(impacts)
        
        # Assess acceptability
        acceptable = overall_impact_score < 0.7
        requires_user_approval = overall_impact_score > 0.6
        
        return {
            "goal_id": goal_id,
            "modification_impacts": impacts,
            "overall_impact_score": overall_impact_score,
            "acceptable": acceptable,
            "requires_user_approval": requires_user_approval,
            "recommendation": "proceed" if acceptable else "seek_approval"
        }

    @staticmethod
    @function_tool
    async def _get_shared_subgoals(ctx: RunContextWrapper, goal_id1: str, goal_id2: str) -> Dict[str, Any]:
        """
        Identify shared subgoals between two goals for optimization
        
        Args:
            goal_id1: First goal ID
            goal_id2: Second goal ID
            
        Returns:
            Shared subgoals information
        """
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {"error": "One or both goals not found"}
        
        # Extract steps from both plans
        steps1 = [(s.action, s.description) for s in goal1.plan] if goal1.plan else []
        steps2 = [(s.action, s.description) for s in goal2.plan] if goal2.plan else []
        
        # Find similar steps (simple similarity based on action and keyword overlap)
        shared_steps = []
        
        for i, (action1, desc1) in enumerate(steps1):
            for j, (action2, desc2) in enumerate(steps2):
                if action1 == action2:
                    # Check description similarity
                    words1 = set(desc1.lower().split())
                    words2 = set(desc2.lower().split())
                    overlap = len(words1.intersection(words2)) / max(1, min(len(words1), len(words2)))
                    
                    if overlap > 0.3:  # Threshold for similarity
                        shared_steps.append({
                            "step_index_goal1": i,
                            "step_index_goal2": j,
                            "action": action1,
                            "description1": desc1,
                            "description2": desc2,
                            "similarity": overlap
                        })
        
        # Identify potential coordination points
        coordination_points = []
        if shared_steps:
            # Group by action type
            action_groups = {}
            for step in shared_steps:
                action = step["action"]
                if action not in action_groups:
                    action_groups[action] = []
                action_groups[action].append(step)
            
            # For each action type, suggest coordination
            for action, steps in action_groups.items():
                if action.startswith(("query_", "retrieve_")):
                    coordination_points.append({
                        "type": "shared_retrieval",
                        "action": action,
                        "steps": steps,
                        "suggestion": "Combine retrieval operations and share results"
                    })
                elif action.startswith("reason_"):
                    coordination_points.append({
                        "type": "shared_reasoning",
                        "action": action,
                        "steps": steps,
                        "suggestion": "Perform reasoning once and use for both goals"
                    })
                elif action.startswith("generate_"):
                    coordination_points.append({
                        "type": "shared_generation",
                        "action": action,
                        "steps": steps,
                        "suggestion": "Generate content once and adapt for both goals"
                    })
        
        return {
            "shared_steps_count": len(shared_steps),
            "shared_steps": shared_steps[:5],  # Limit to 5 for brevity
            "coordination_points": coordination_points,
            "can_optimize": len(coordination_points) > 0
        }

    @staticmethod
    @function_tool
    async def _get_goal_common_elements(ctx: RunContextWrapper, goal_id1: str, goal_id2: str) -> Dict[str, Any]:
        """
        Identify common elements between two goals for merging
        
        Args:
            goal_id1: First goal ID
            goal_id2: Second goal ID
            
        Returns:
            Common elements analysis
        """
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {"error": "One or both goals not found"}
        
        # Find common words in descriptions
        words1 = set(goal1.description.lower().split())
        words2 = set(goal2.description.lower().split())
        common_words = words1.intersection(words2)
        
        # Find common actions in plans
        common_actions = []
        if goal1.plan and goal2.plan:
            actions1 = [step.action for step in goal1.plan]
            actions2 = [step.action for step in goal2.plan]
            
            # Find action patterns that appear in both plans
            for action in set(actions1).intersection(set(actions2)):
                common_actions.append(action)
        
        # Analyze needs and motivations
        common_need = None
        if goal1.associated_need and goal1.associated_need == goal2.associated_need:
            common_need = goal1.associated_need
        
        combined_motivation = None
        if goal1.emotional_motivation and goal2.emotional_motivation:
            # If same primary need
            if goal1.emotional_motivation.primary_need == goal2.emotional_motivation.primary_need:
                # Combine the motivations
                combined_motivation = {
                    "primary_need": goal1.emotional_motivation.primary_need,
                    "intensity": max(goal1.emotional_motivation.intensity, goal2.emotional_motivation.intensity),
                    "expected_satisfaction": max(
                        goal1.emotional_motivation.expected_satisfaction,
                        goal2.emotional_motivation.expected_satisfaction
                    )
                }
        
        # Determine best time horizon for merged goal
        merged_time_horizon = None
        if goal1.time_horizon == goal2.time_horizon:
            merged_time_horizon = goal1.time_horizon
        else:
            # Use the shorter time horizon as default
            time_horizons = [goal1.time_horizon, goal2.time_horizon]
            if TimeHorizon.SHORT_TERM in time_horizons:
                merged_time_horizon = TimeHorizon.SHORT_TERM
            elif TimeHorizon.MEDIUM_TERM in time_horizons:
                merged_time_horizon = TimeHorizon.MEDIUM_TERM
            else:
                merged_time_horizon = TimeHorizon.LONG_TERM
        
        # Determine best priority for merged goal
        merged_priority = max(goal1.priority, goal2.priority)
        
        return {
            "common_words": list(common_words),
            "common_actions": common_actions,
            "common_need": common_need,
            "combined_motivation": combined_motivation,
            "merged_time_horizon": merged_time_horizon,
            "merged_priority": merged_priority,
            "merge_suitability": "high" if len(common_words) > 3 and len(common_actions) > 1 else "medium"
        }

    @staticmethod
    @function_tool
    async def _generate_merged_goal_description(ctx: RunContextWrapper, goal_id1: str, goal_id2: str, 
                                             common_elements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a description for a merged goal
        
        Args:
            goal_id1: First goal ID
            goal_id2: Second goal ID
            common_elements: Common elements analysis
            
        Returns:
            Merged goal description
        """
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {"error": "One or both goals not found"}
        
        # Extract key elements from descriptions
        description1 = goal1.description
        description2 = goal2.description
        
        # Create a merged description that combines both goals
        merged_description = f"Accomplish both: {description1} and {description2}"
        
        # Create merged goal data
        merged_goal_data = {
            "description": merged_description,
            "priority": common_elements["merged_priority"],
            "time_horizon": common_elements["merged_time_horizon"],
            "associated_need": common_elements.get("common_need"),
            "source": "goal_merger",
            "parent_goals": [goal_id1, goal_id2]
        }
        
        # Create emotional motivation if available
        if common_elements.get("combined_motivation"):
            cm = common_elements["combined_motivation"]
            merged_goal_data["emotional_motivation"] = {
                "primary_need": cm["primary_need"],
                "intensity": cm["intensity"],
                "expected_satisfaction": cm["expected_satisfaction"],
                "description": f"Combined motivation from goals {goal_id1} and {goal_id2}"
            }
        
        return {
            "merged_goal_data": merged_goal_data,
            "original_goal1": goal1.description,
            "original_goal2": goal2.description,
            "merged_description": merged_description
        }

    @staticmethod
    @function_tool
    async def _validate_merged_goal_coverage(ctx: RunContextWrapper, merged_goal_data: Dict[str, Any], 
                                          goal_id1: str, goal_id2: str) -> Dict[str, Any]:
        """
        Validate that a merged goal adequately covers the original goals
        
        Args:
            merged_goal_data: Merged goal data
            goal_id1: First goal ID
            goal_id2: Second goal ID
            
        Returns:
            Coverage validation
        """
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {"error": "One or both goals not found"}
        
        # Extract significant keywords from each goal
        keywords1 = set([w.lower() for w in goal1.description.split() if len(w) > 4])
        keywords2 = set([w.lower() for w in goal2.description.split() if len(w) > 4])
        merged_keywords = set([w.lower() for w in merged_goal_data["description"].split() if len(w) > 4])
        
        # Calculate coverage percentage
        coverage1 = len(keywords1.intersection(merged_keywords)) / max(1, len(keywords1))
        coverage2 = len(keywords2.intersection(merged_keywords)) / max(1, len(keywords2))
        
        # Check need coverage
        need_covered = True
        if goal1.associated_need and goal2.associated_need:
            need_covered = merged_goal_data.get("associated_need") in [goal1.associated_need, goal2.associated_need]
        
        # Overall coverage assessment
        overall_coverage = min(coverage1, coverage2)
        
        return {
            "coverage_goal1": coverage1,
            "coverage_goal2": coverage2,
            "overall_coverage": overall_coverage,
            "need_coverage": need_covered,
            "is_sufficient": overall_coverage > 0.7 and need_covered,
            "missing_elements": []  # Would be populated with missing key elements in a full implementation
        }
    
    #==========================================================================
    # Integration with External Systems
    #==========================================================================
    
    async def _init_external_systems(self):
        """Initialize connections to external systems"""
        if not self.integration_config:
            return
        
        for system_name in self.integration_config.enabled_systems:
            try:
                if system_name == "emotional_system" and self.brain and hasattr(self.brain, "emotional_core"):
                    # Already connected via brain reference
                    logger.info(f"Using existing connection to {system_name}")
                    continue
                    
                elif system_name == "needs_system" and self.brain and hasattr(self.brain, "needs_system"):
                    # Already connected via brain reference
                    logger.info(f"Using existing connection to {system_name}")
                    continue
                    
                elif system_name == "external_api":
                    # Example of connecting to an external API client
                    # In a real system, this would create an API client
                    self._external_system_clients[system_name] = {
                        "name": system_name,
                        "connected": True,
                        "client": None  # Would be an actual client in a real system
                    }
                    logger.info(f"Connected to external system: {system_name}")
                    
                else:
                    logger.warning(f"Unknown external system: {system_name}")
            
            except Exception as e:
                logger.error(f"Error connecting to external system {system_name}: {e}")
    
    async def register_integration_callback(self, event_type: str, callback: Callable):
        """Register a callback for goal events"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        
        self._integration_callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for event type: {event_type}")
    
    async def trigger_integration_callbacks(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger registered callbacks for an event"""
        callbacks = self._integration_callbacks.get(event_type, [])
        
        if not callbacks:
            return
        
        logger.debug(f"Triggering {len(callbacks)} callbacks for event type: {event_type}")
        
        results = []
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Set a timeout for async callbacks
                    try:
                        result = await asyncio.wait_for(
                            callback(event_data),
                            timeout=self.integration_config.callback_timeout
                        )
                        results.append({"success": True, "result": result})
                    except asyncio.TimeoutError:
                        logger.warning(f"Callback for {event_type} timed out")
                        results.append({"success": False, "error": "timeout"})
                else:
                    # Run synchronous callbacks in thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        self._thread_pool, callback, event_data
                    )
                    results.append({"success": True, "result": result})
            except Exception as e:
                logger.error(f"Error in callback for event {event_type}: {e}")
                results.append({"success": False, "error": str(e)})
        
        return results
    
    async def _link_goal_to_emotional_system(self, goal_id: str) -> bool:
        """Creates an emotional link for a goal to influence the emotional state"""
        if not self.brain or not hasattr(self.brain, "emotional_core"):
            return False
        
        emotional_core = self.brain.emotional_core
        
        # Check if the goal exists
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return False
        
        # Link based on emotional motivation if available
        if goal.emotional_motivation:
            # Set up chemical changes based on goal's emotional motivation
            chemicals = goal.emotional_motivation.associated_chemicals
            
            # Apply a small anticipatory boost for focusing on the goal
            for chemical, value in chemicals.items():
                try:
                    original_value = await emotional_core.get_neurochemical_level(chemical)
                    await emotional_core.update_neurochemical(chemical, value * 0.25)
                    
                    # Record the chemical change in goal
                    await self._update_goal_with_writer_lock(
                        goal_id,
                        lambda g: g.emotional_state_snapshots.append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "anticipatory_change",
                            "chemical": chemical,
                            "original_value": original_value,
                            "applied_change": value * 0.25
                        })
                    )
                except Exception as e:
                    logger.error(f"Error updating neurochemical {chemical}: {e}")
            
            # Register for emotional state change notifications
            try:
                # Create weak reference to avoid memory leaks
                goal_ref = weakref.ref(await self._get_goal_with_reader_lock(goal_id))
                
                # Define callback for emotional state changes
                async def emotional_state_callback(state_data):
                    goal = goal_ref()
                    if not goal:
                        return  # Goal has been garbage collected
                    
                    # Update goal with emotional state data
                    await self._update_goal_with_writer_lock(
                        goal_id,
                        lambda g: g.emotional_state_snapshots.append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "state_update",
                            "state": state_data
                        })
                    )
                
                # Register callback with emotional core
                if hasattr(emotional_core, "register_state_callback"):
                    await emotional_core.register_state_callback(
                        goal_id, emotional_state_callback, expiration=3600  # 1 hour expiration
                    )
            except Exception as e:
                logger.error(f"Error registering emotional state callback: {e}")
        
        return True
    
    async def _process_goal_completion_reward(self, goal_id: str, result: Any = None) -> Dict[str, Any]:
        """Process reward signals when a goal is completed"""
        if not self.brain:
            return {"success": False, "reason": "No brain reference available"}
        
        # Get goal with reader lock
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return {"success": False, "reason": "Goal not found"}
        
        # Track reward system results
        reward_results = {}
        
        # Reward system integration
        if hasattr(self.brain, "reward_system"):
            reward_system = self.brain.reward_system
        
            # Calculate base reward based on priority and time horizon
            base_reward = goal.priority
            
            # Adjust based on time horizon (more immediate satisfaction for short-term goals)
            time_horizon_factor = 1.0
            if goal.time_horizon == TimeHorizon.SHORT_TERM:
                time_horizon_factor = 1.2  # 20% boost for short-term goal completion
            elif goal.time_horizon == TimeHorizon.LONG_TERM:
                time_horizon_factor = 0.9  # 10% reduction for long-term goals
            
            # Adjust based on emotional motivation if available
            satisfaction_factor = 1.0
            if goal.emotional_motivation:
                satisfaction_factor = goal.emotional_motivation.expected_satisfaction
            
            # Calculate final reward
            reward_value = base_reward * time_horizon_factor * satisfaction_factor
            
            # Create context for the reward
            context = {
                "goal_id": goal_id,
                "goal_description": goal.description,
                "time_horizon": goal.time_horizon.value,
                "emotional_need": goal.emotional_motivation.primary_need if goal.emotional_motivation else None,
                "achievement_type": "goal_completion"
            }
            
            try:
                # Import RewardSignal locally to avoid circular imports
                from nyx.core.reward_system import RewardSignal
                
                # Create and process reward signal
                reward_signal = RewardSignal(
                    value=reward_value,
                    source="GoalManager",
                    context=context,
                    timestamp=datetime.datetime.now().isoformat()
                )
                
                reward_result = await reward_system.process_reward_signal(reward_signal)
                reward_results["reward_system"] = {
                    "success": True,
                    "value": reward_value,
                    "result": reward_result
                }
                
            except Exception as e:
                logger.error(f"Error processing goal completion reward: {e}")
                reward_results["reward_system"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Emotional system integration
        if hasattr(self.brain, "emotional_core"):
            emotional_core = self.brain.emotional_core
            
            try:
                if goal.emotional_motivation:
                    chemicals = goal.emotional_motivation.associated_chemicals
                    
                    # Store initial state
                    baseline_state = await emotional_core.get_emotional_state()
                    
                    # Apply stronger chemical changes for completion
                    chemical_results = {}
                    for chemical, value in chemicals.items():
                        try:
                            original_value = await emotional_core.get_neurochemical_level(chemical)
                            await emotional_core.update_neurochemical(chemical, value * 0.8)
                            new_value = await emotional_core.get_neurochemical_level(chemical)
                            
                            chemical_results[chemical] = {
                                "original": original_value,
                                "change": value * 0.8,
                                "new": new_value
                            }
                        except Exception as e:
                            logger.error(f"Error updating neurochemical {chemical}: {e}")
                            chemical_results[chemical] = {
                                "error": str(e)
                            }
                    
                    # Get updated state
                    updated_state = await emotional_core.get_emotional_state()
                    
                    reward_results["emotional_core"] = {
                        "success": True,
                        "baseline_state": baseline_state,
                        "updated_state": updated_state,
                        "chemical_changes": chemical_results
                    }
                    
                    # Store the completion state in the goal
                    await self._update_goal_with_writer_lock(
                        goal_id,
                        lambda g: g.emotional_state_snapshots.append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "completion_state",
                            "baseline_state": baseline_state,
                            "updated_state": updated_state,
                            "chemical_changes": chemical_results
                        })
                    )
                    
            except Exception as e:
                logger.error(f"Error updating emotional state for goal completion: {e}")
                reward_results["emotional_core"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Trigger integration callbacks for goal completion
        await self.trigger_integration_callbacks("goal_completed", {
            "goal_id": goal_id,
            "goal_description": goal.description,
            "result": result,
            "reward_results": reward_results
        })
        
        return {
            "success": True,
            "goal_id": goal_id,
            "reward_results": reward_results
        }
        
    async def integrate_with_needs_system(self, goal_id: str, need_name: str, satisfaction_level: float = 0.3) -> Dict[str, Any]:
        """
        Integrate a goal with the needs system for more detailed need satisfaction tracking
        
        Args:
            goal_id: Goal ID
            need_name: Name of the need to associate
            satisfaction_level: Level of satisfaction upon completion (0.0-1.0)
            
        Returns:
            Integration result
        """
        if not self.brain or not hasattr(self.brain, "needs_system"):
            return {"success": False, "reason": "Needs system not available"}
        
        needs_system = self.brain.needs_system
        
        # Update goal with need association
        success = await self._update_goal_with_writer_lock(
            goal_id,
            lambda goal: (
                setattr(goal, "associated_need", need_name),
                goal.external_system_ids.update({"needs_system": need_name})
            )
        )
        
        if not success:
            return {"success": False, "reason": "Goal not found"}
        
        # Register with needs system if it has registration method
        needs_system_result = {"registered": False}
        if hasattr(needs_system, "register_goal_for_need"):
            try:
                result = await needs_system.register_goal_for_need(
                    need_name, goal_id, satisfaction_level
                )
                needs_system_result = {
                    "registered": True,
                    "result": result
                }
            except Exception as e:
                logger.error(f"Error registering goal with needs system: {e}")
                needs_system_result = {
                    "registered": False,
                    "error": str(e)
                }
        
        # Create emotional motivation based on need if available
        if hasattr(needs_system, "get_need_emotional_components"):
            try:
                emotional_components = await needs_system.get_need_emotional_components(need_name)
                
                if emotional_components:
                    # Create emotional motivation based on need components
                    em = EmotionalMotivation(
                        primary_need=need_name,
                        intensity=emotional_components.get("intensity", 0.5),
                        expected_satisfaction=satisfaction_level,
                        associated_chemicals=emotional_components.get("chemicals", {}),
                        description=f"Derived from need: {need_name}"
                    )
                    
                    # Update goal with emotional motivation
                    await self._update_goal_with_writer_lock(
                        goal_id,
                        lambda goal: setattr(goal, "emotional_motivation", em)
                    )
                    
                    # Link to emotional system
                    await self._link_goal_to_emotional_system(goal_id)
                    
                    needs_system_result["emotional_components"] = emotional_components
            except Exception as e:
                logger.error(f"Error getting need emotional components: {e}")
                needs_system_result["emotional_error"] = str(e)
        
        return {
            "success": True,
            "goal_id": goal_id,
            "need": need_name,
            "satisfaction_level": satisfaction_level,
            "needs_system_result": needs_system_result
        }
    
    #==========================================================================
    # Core Goal Management Methods
    #==========================================================================
    
    async def derive_emotional_motivation(self, goal_description: str, need: Optional[str] = None) -> EmotionalMotivation:
        """
        Analyzes a goal description to derive likely emotional motivation
        
        Args:
            goal_description: Description of the goal
            need: Associated need, if known
            
        Returns:
            Derived emotional motivation
        """
        # Start with default values
        motivation = EmotionalMotivation(
            primary_need=need or "accomplishment",
            intensity=0.5,
            expected_satisfaction=0.6,
            associated_chemicals={"nyxamine": 0.3, "seranix": 0.2},
            description="Derived emotional motivation"
        )
        
        # Use emotional core to refine if available
        if self.brain and hasattr(self.brain, "emotional_core"):
            emotional_core = self.brain.emotional_core
            
            # Common emotional needs mapped to chemicals
            need_to_chemicals = {
                "accomplishment": {"nyxamine": 0.4, "seranix": 0.2},
                "connection": {"oxynixin": 0.5, "seranix": 0.2},
                "security": {"seranix": 0.4, "cortanyx": -0.3},
                "control": {"adrenyx": 0.3, "cortanyx": -0.2},
                "growth": {"nyxamine": 0.3, "adrenyx": 0.2},
                "pleasure": {"nyxamine": 0.5},
                "meaning": {"nyxamine": 0.3, "seranix": 0.3, "oxynixin": 0.2},
                "efficiency": {"nyxamine": 0.2, "cortanyx": -0.2},
                "autonomy": {"adrenyx": 0.2, "nyxamine": 0.2},
                "challenge": {"adrenyx": 0.4, "nyxamine": 0.3}
            }
            
            # Analyze text for emotional content if no need specified
            if not need:
                # Check for key words/phrases that suggest specific needs
                lower_text = goal_description.lower()
                
                if any(word in lower_text for word in ["connect", "bond", "relate", "together", "relationship"]):
                    motivation.primary_need = "connection"
                    motivation.associated_chemicals = need_to_chemicals["connection"]
                    
                elif any(word in lower_text for word in ["grow", "improve", "better", "learn", "develop"]):
                    motivation.primary_need = "growth"
                    motivation.associated_chemicals = need_to_chemicals["growth"]
                    
                elif any(word in lower_text for word in ["secure", "safe", "protect", "prevent", "avoid"]):
                    motivation.primary_need = "security"
                    motivation.associated_chemicals = need_to_chemicals["security"]
                    
                elif any(word in lower_text for word in ["control", "manage", "direct", "lead", "organize"]):
                    motivation.primary_need = "control"
                    motivation.associated_chemicals = need_to_chemicals["control"]
                    
                elif any(word in lower_text for word in ["meaning", "purpose", "value", "important", "significant"]):
                    motivation.primary_need = "meaning"
                    motivation.associated_chemicals = need_to_chemicals["meaning"]
                    
                elif any(word in lower_text for word in ["enjoy", "fun", "pleasure", "happy", "delight"]):
                    motivation.primary_need = "pleasure"
                    motivation.associated_chemicals = need_to_chemicals["pleasure"]
                    
                elif any(word in lower_text for word in ["challenge", "difficult", "hard", "master", "overcome"]):
                    motivation.primary_need = "challenge"
                    motivation.associated_chemicals = need_to_chemicals["challenge"]
                    
                elif any(word in lower_text for word in ["efficient", "quick", "optimize", "streamline", "automate"]):
                    motivation.primary_need = "efficiency"
                    motivation.associated_chemicals = need_to_chemicals["efficiency"]
            else:
                # Use provided need if specified
                if need in need_to_chemicals:
                    motivation.associated_chemicals = need_to_chemicals[need]
            
            # Set description based on need
            need_descriptions = {
                "accomplishment": "Desire to achieve something meaningful and receive recognition",
                "connection": "Desire for authentic bonding and meaningful relationships",
                "security": "Desire for safety, stability and predictability",
                "control": "Desire to influence outcomes and direct processes",
                "growth": "Desire to improve skills, knowledge and capabilities",
                "pleasure": "Desire for enjoyment and positive experiences",
                "meaning": "Desire for purpose and significance",
                "efficiency": "Desire to optimize processes and save resources",
                "autonomy": "Desire for independence and self-direction",
                "challenge": "Desire to overcome difficult obstacles"
            }
            
            if motivation.primary_need in need_descriptions:
                motivation.description = need_descriptions[motivation.primary_need]
            
            # Use emotional core for more sophisticated analysis if available
            if hasattr(emotional_core, "analyze_text_motivation"):
                try:
                    analysis_result = await emotional_core.analyze_text_motivation(goal_description)
                    
                    # If analysis returned valid results, update motivation
                    if analysis_result and analysis_result.get("primary_need"):
                        motivation.primary_need = analysis_result["primary_need"]
                        motivation.intensity = analysis_result.get("intensity", motivation.intensity)
                        motivation.expected_satisfaction = analysis_result.get("satisfaction", motivation.expected_satisfaction)
                        
                        if analysis_result.get("chemicals"):
                            motivation.associated_chemicals = analysis_result["chemicals"]
                        elif motivation.primary_need in need_to_chemicals:
                            motivation.associated_chemicals = need_to_chemicals[motivation.primary_need]
                        
                        if analysis_result.get("description"):
                            motivation.description = analysis_result["description"]
                        elif motivation.primary_need in need_descriptions:
                            motivation.description = need_descriptions[motivation.primary_need]
                except Exception as e:
                    logger.error(f"Error analyzing goal motivation: {e}")
        
        return motivation

    async def add_goal_with_motivation(self, goal_data: GoalCreationWithMotivation) -> str:
        """
        Creates a goal with emotional motivation and time horizon specifications
        
        Args:
            goal_data: Goal creation data with motivation
            
        Returns:
            ID of created goal
        """
        # Create base fields for the goal
        goal_fields = {
            "description": goal_data.description,
            "priority": goal_data.priority,
            "source": "explicit",
            "associated_need": goal_data.associated_need,
            "time_horizon": goal_data.time_horizon
        }
        
        # Add deadline if provided
        if goal_data.deadline:
            try:
                deadline = datetime.datetime.fromisoformat(goal_data.deadline)
                goal_fields["deadline"] = deadline
            except ValueError:
                logger.warning(f"Invalid deadline format: {goal_data.deadline}")
        
        # Add emotional motivation if provided, or derive one
        if goal_data.emotional_motivation:
            goal_fields["emotional_motivation"] = goal_data.emotional_motivation
        else:
            # Derive emotional motivation from description and need
            motivation = await self.derive_emotional_motivation(
                goal_data.description, goal_data.associated_need
            )
            goal_fields["emotional_motivation"] = motivation
        
        # Create goal relationships if parent_goal_id is provided
        relationships = GoalRelationship()
        if goal_data.parent_goal_id:
            # Check if parent goal exists with reader lock
            parent_goal = await self._get_goal_with_reader_lock(goal_data.parent_goal_id)
            if parent_goal:
                relationships.parent_goal_id = goal_data.parent_goal_id
                relationships.relationship_type = "hierarchical"
                goal_fields["relationships"] = relationships
                
                # If parent goal has time horizon, consider it for this goal
                if goal_data.time_horizon == TimeHorizon.MEDIUM_TERM:  # Only adjust if not explicitly set
                    goal_fields["time_horizon"] = parent_goal.time_horizon
        
        # Create the goal
        goal_id = await self.add_goal(**goal_fields)
        
        if not goal_id:
            return None
        
        # Update parent goal's relationships if applicable
        if goal_data.parent_goal_id:
            # Use writer lock for parent goal
            await self._update_goal_with_writer_lock(
                goal_data.parent_goal_id,
                lambda parent_goal: (
                    parent_goal.relationships.child_goal_ids.append(goal_id)
                    if goal_id not in parent_goal.relationships.child_goal_ids
                    else None
                )
            )
        
        # Link with emotional system
        await self._link_goal_to_emotional_system(goal_id)
        
        # If goal has associated need, integrate with needs system
        if goal_data.associated_need and self.brain and hasattr(self.brain, "needs_system"):
            await self.integrate_with_needs_system(
                goal_id, 
                goal_data.associated_need,
                goal_data.emotional_motivation.expected_satisfaction if goal_data.emotional_motivation else 0.3
            )
        
        return goal_id
    
    async def create_goal_hierarchy(self, root_goal_data: Dict[str, Any], subgoals_data: List[Dict[str, Any]]) -> str:
        """
        Creates a hierarchical structure of goals with a root goal and subgoals
        
        Args:
            root_goal_data: Data for the root goal
            subgoals_data: List of data for subgoals
            
        Returns:
            ID of root goal
        """
        # First create the root goal
        root_goal = GoalCreationWithMotivation(**root_goal_data)
        root_goal_id = await self.add_goal_with_motivation(root_goal)
        
        if not root_goal_id:
            return None
        
        # Create each subgoal and link to the root goal
        for subgoal_data in subgoals_data:
            subgoal_data["parent_goal_id"] = root_goal_id
            subgoal = GoalCreationWithMotivation(**subgoal_data)
            await self.add_goal_with_motivation(subgoal)
        
        return root_goal_id
    
    async def add_goal(self, description: str, priority: float = 0.5, source: str = "unknown",
                     associated_need: Optional[str] = None, emotional_motivation: Optional[EmotionalMotivation] = None,
                     plan: Optional[List[Dict]] = None, user_id: Optional[str] = None, 
                     deadline: Optional[datetime.datetime] = None,
                     time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> str:
        """
        Adds a new goal, optionally generating a plan if none is provided
        
        Args:
            description: Goal description
            priority: Priority (0.0-1.0)
            source: Source of the goal (user, system, etc)
            associated_need: Associated need
            emotional_motivation: Emotional motivation
            plan: Optional pre-defined plan
            user_id: Optional user ID
            deadline: Optional deadline
            time_horizon: Time horizon
            
        Returns:
            ID of created goal
        """
        if not description:
            raise ValueError("Goal description cannot be empty.")

        # Create the goal object
        goal = Goal(
            description=description,
            priority=priority,
            source=source,
            associated_need=associated_need,
            deadline=deadline,
            plan=[],  # Start with empty plan, generate/add later
            time_horizon=time_horizon,
            emotional_motivation=emotional_motivation
        )

        # Add goal with writer lock
        async with self._goals_dict_lock:
            self.goals[goal.id] = goal
            self.goal_statistics["created"] += 1
            self._dirty_goals.add(goal.id)  # Mark as dirty for persistence

        logger.info(f"Added goal '{goal.id}': {description} (Priority: {priority:.2f}, Source: {source})")

        # Process the goal through the orchestration system
        with trace(workflow_name="Goal_Management", group_id=self.trace_group_id):
            # Create context for this goal management process
            context = RunContext(
                goal_id=goal.id,
                brain_available=self.brain is not None,
                user_id=user_id
            )
            
            # Run goal processing through the orchestration agent
            result = await Runner.run(
                self.orchestration_agent,
                json.dumps({
                    "goal": {
                        "id": goal.id,
                        "description": description,
                        "priority": priority,
                        "source": source,
                        "associated_need": associated_need,
                        "time_horizon": time_horizon.value
                    },
                    "has_plan": plan is not None,
                    "user_id": user_id
                }),
                context=context,
                run_config={
                    "workflow_name": "GoalProcessing",
                    "trace_metadata": {
                        "goal_id": goal.id,
                        "goal_description": description
                    }
                }
            )
            
            # If a plan was provided, use it
            if plan:
                try:
                    plan_steps = [GoalStep(**step_data) for step_data in plan]
                    
                    # Update goal with writer lock
                    await self._update_goal_with_writer_lock(
                        goal.id,
                        lambda g: (
                            setattr(g, "plan", plan_steps),
                            setattr(g, "status", "pending")  # Ready to be activated
                        )
                    )
                except Exception as e:
                    logger.error(f"Invalid plan structure provided for goal '{goal.id}': {e}")

        return goal.id
    
    async def generate_plan_for_goal(self, goal_id: str) -> bool:
        """
        Generates and assigns a plan for a goal using the planning agent
        
        Args:
            goal_id: Goal ID
            
        Returns:
            Success status
        """
        # Get goal with reader lock
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            logger.error(f"Cannot generate plan: Goal {goal_id} not found.")
            return False
            
        if goal.plan:  # Don't overwrite existing plan
            logger.info(f"Goal '{goal_id}' already has a plan.")
            return True

        if not self.planning_agent:
            logger.warning(f"Cannot generate plan for goal '{goal_id}': Planning agent not available.")
            return False

        logger.info(f"Generating plan for goal '{goal.id}': {goal.description}")

        try:
            with trace(workflow_name="GenerateGoalPlan", group_id=self.trace_group_id):
                # Create context for the agent
                context = RunContext(goal_id=goal_id, brain_available=self.brain is not None)
                
                # Generate plan through the Planning Agent
                result = await Runner.run(
                    self.planning_agent,
                    json.dumps({
                        "goal": {
                            "id": goal.id,
                            "description": goal.description,
                            "priority": goal.priority,
                            "source": goal.source,
                            "associated_need": goal.associated_need,
                            "time_horizon": goal.time_horizon
                        }
                    }),
                    context=context,
                    run_config={
                        "workflow_name": "GoalPlanning",
                        "trace_metadata": {
                            "goal_id": goal_id,
                            "goal_description": goal.description
                        }
                    }
                )
                
                # Extract plan from result
                plan_result = result.final_output_as(PlanGenerationResult)
                plan_data = plan_result.plan
                
                # Validate the plan
                validation_result = await Runner.run(
                    self.plan_validation_agent,
                    json.dumps({
                        "goal": {
                            "id": goal.id,
                            "description": goal.description
                        },
                        "plan": plan_data
                    }),
                    context=context
                )
                
                validation_output = validation_result.final_output_as(PlanValidationResult)
                
                # Convert to GoalStep objects
                plan_steps = [GoalStep(**step_data) for step_data in plan_data]
                
                # Update the goal with the new plan using writer lock
                await self._update_goal_with_writer_lock(
                    goal_id,
                    lambda g: (
                        setattr(g, "plan", plan_steps),
                        setattr(g, "current_step_index", 0),  # Reset index
                        setattr(g, "status", "pending" if g.status != "failed" else "failed")
                    )
                )
                
                logger.info(f"Generated plan with {len(plan_steps)} steps for goal '{goal.id}'.")
                
                if not validation_output.is_valid:
                    logger.warning(f"Plan validation raised concerns: {validation_output.reason}")
                    
                return True

        except Exception as e:
            error_msg = f"Error generating plan for goal '{goal.id}': {e}"
            logger.exception(error_msg)  # Log full traceback
            await self.update_goal_status(goal_id, "failed", error=error_msg)
            return False
    
    async def select_active_goal(self) -> Optional[str]:
        """
        Selects the highest priority goal to work on, respecting concurrency limits
        
        Returns:
            ID of selected goal, or None if no goal is ready
        """
        # Get prioritized goals
        prioritized = await self.get_prioritized_goals()
        selected_goal_id = None

        if not prioritized:
            # No goals left to be active
            async with self._active_goals_lock:
                self.active_goals.clear()
            return None

        # Check if any currently active goals are finished or failed
        active_goals_copy = set()
        finished_active = set()
        
        async with self._active_goals_lock:
            active_goals_copy = self.active_goals.copy()
        
        for gid in active_goals_copy:
            goal = await self._get_goal_with_reader_lock(gid)
            if not goal or goal.status not in ["active", "pending"]:
                finished_active.add(gid)
        
        # Update active goals set with writer lock
        if finished_active:
            async with self._active_goals_lock:
                self.active_goals -= finished_active
            
        # Get current active count
        async with self._active_goals_lock:
            active_count = len(self.active_goals)

        # Find the highest priority goal that can be activated
        for goal in prioritized:
            async with self._active_goals_lock:
                if goal.id in self.active_goals:
                    # If it's already active, it's a candidate to continue
                    selected_goal_id = goal.id
                    break  # Keep executing the highest priority *active* goal first
                elif goal.status == "pending" and active_count < self.max_concurrent_goals and goal.plan:
                    # Activate this pending goal if concurrency allows and it has a plan
                    selected_goal_id = goal.id
                    
                    # Update goal status with writer lock
                    await self._update_goal_with_writer_lock(
                        goal.id,
                        lambda g: setattr(g, "status", "active")
                    )
                    
                    self.active_goals.add(goal.id)
                    logger.info(f"Activated goal '{goal.id}' (Priority: {goal.priority:.2f})")
                    break

        return selected_goal_id
    
    async def execute_next_step(self) -> Optional[Dict[str, Any]]:
        """
        Selects and executes the next step of the highest priority active goal
        
        Returns:
            Execution result, or None if no step was executed
        """
        goal_id = await self.select_active_goal()  # Selection handles concurrency & prioritization

        if goal_id is None:
            return None

        # Get goal with reader lock
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            logger.warning(f"Selected goal '{goal_id}' disappeared before execution.")
            async with self._active_goals_lock:
                self.active_goals.discard(goal_id)
            return None
            
        if goal.status != "active" or not goal.plan:
            logger.warning(f"Goal '{goal_id}' is not ready for execution (Status: {goal.status}, Has Plan: {bool(goal.plan)}).")
            async with self._active_goals_lock:
                self.active_goals.discard(goal_id)
            return None

        step_index = goal.current_step_index
        if not (0 <= step_index < len(goal.plan)):
            logger.error(f"Goal '{goal.id}' has invalid step index {step_index}. Failing goal.")
            await self.update_goal_status(goal_id, "failed", error="Invalid plan state")
            async with self._active_goals_lock:
                self.active_goals.discard(goal_id)
            return None

        step = goal.plan[step_index]

        if step.status != "pending":
            logger.warning(f"Step '{step.step_id}' for goal '{goal.id}' is not pending (Status: {step.status}). Skipping.")
            
            # Update goal with writer lock
            await self._update_goal_with_writer_lock(
                goal_id,
                lambda g: setattr(g, "current_step_index", g.current_step_index + 1)
            )
            
            # Check if goal is now complete
            updated_goal = await self._get_goal_with_reader_lock(goal_id)
            if updated_goal and updated_goal.current_step_index >= len(updated_goal.plan):
                await self.update_goal_status(goal.id, "completed", result="Plan finished after skipping steps.")
                async with self._active_goals_lock:
                    self.active_goals.discard(goal_id)
                    
            return {"skipped_step": step.model_dump(), "goal_id": goal_id}  # Indicate skip

        # Create context for the step execution
        context = RunContext(
            goal_id=goal_id,
            brain_available=self.brain is not None,
            current_step_index=step_index
        )
        
        # Execute the step through the Step Execution Agent
        with trace(workflow_name="ExecuteGoalStep", group_id=self.trace_group_id):
            step_result = await Runner.run(
                self.step_execution_agent,
                json.dumps({
                    "goal_id": goal_id,
                    "step": step.model_dump(),
                    "step_index": step_index
                }),
                context=context,
                run_config={
                    "workflow_name": "StepExecution",
                    "trace_metadata": {
                        "goal_id": goal_id,
                        "step_id": step.step_id,
                        "action": step.action
                    }
                }
            )
            
            execution_result = step_result.final_output_as(StepExecutionResult)
            
            # Update goal based on execution result with writer lock
            await self._update_goal_with_writer_lock(
                goal_id,
                lambda goal: self._update_goal_after_step_execution(goal, step_index, execution_result)
            )
            
            # Process goal status changes if needed
            updated_goal = await self._get_goal_with_reader_lock(goal_id)
            if updated_goal:
                if updated_goal.status == "completed":
                    # Process goal completion reward
                    await self._process_goal_completion_reward(goal_id, step.result)
                    
                    # Remove from active goals
                    async with self._active_goals_lock:
                        self.active_goals.discard(goal_id)
                        
                elif updated_goal.status == "failed":
                    # Remove from active goals
                    async with self._active_goals_lock:
                        self.active_goals.discard(goal_id)

        return {"executed_step": step.model_dump(), "goal_id": goal_id, "next_action": execution_result.next_action}
    
    def _update_goal_after_step_execution(self, goal: Goal, step_index: int, 
                                       execution_result: StepExecutionResult) -> None:
        """
        Helper method to update goal after step execution - used with writer lock
        
        Args:
            goal: Goal to update
            step_index: Index of executed step
            execution_result: Execution result
        """
        if step_index >= len(goal.plan):
            return
            
        step = goal.plan[step_index]
        
        # Update step with execution result
        step.status = "completed" if execution_result.success else "failed"
        step.result = execution_result.result
        step.error = execution_result.error
        step.end_time = datetime.datetime.now()
        
        # Update execution history
        goal.execution_history.append({
            "step_id": step.step_id,
            "action": step.action,
            "status": step.status,
            "timestamp": datetime.datetime.now().isoformat(),
            "next_action": execution_result.next_action,
            "error": step.error
        })
        
        # Process next action
        if execution_result.next_action == "continue":
            # Move to next step
            if step.status == "completed":
                goal.current_step_index += 1
                if goal.current_step_index >= len(goal.plan):
                    goal.status = "completed"
                    goal.completion_time = datetime.datetime.now()
                    goal.progress = 1.0
            else:  # Failed
                goal.status = "failed"
                goal.completion_time = datetime.datetime.now()
                goal.last_error = step.error
                
        elif execution_result.next_action == "retry":
            # Leave index the same to retry the step
            if "retry_count" not in goal.execution_history[-1]:
                goal.execution_history[-1]["retry_count"] = 1
            else:
                goal.execution_history[-1]["retry_count"] += 1
                
            # Check if max retries exceeded
            if goal.execution_history[-1]["retry_count"] >= 3:
                goal.status = "failed"
                goal.completion_time = datetime.datetime.now()
                goal.last_error = f"Max retries exceeded for step {step.step_id}"
                
        elif execution_result.next_action == "skip":
            # Mark as skipped and move to next step
            step.status = "skipped"
            goal.current_step_index += 1
            if goal.current_step_index >= len(goal.plan):
                goal.status = "completed"
                goal.completion_time = datetime.datetime.now()
                goal.progress = 1.0
                
        elif execution_result.next_action == "abort":
            # Abort the entire goal
            goal.status = "failed"
            goal.completion_time = datetime.datetime.now()
            goal.last_error = f"Goal aborted: {step.error}"
        
        # Update progress
        if goal.status not in ["completed", "failed"]:
            goal.progress = goal.current_step_index / max(1, len(goal.plan))
    
    async def _resolve_step_parameters(self, goal_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves parameter placeholders like '$step_1.result' within a specific goal
        
        Args:
            goal_id: Goal ID
            parameters: Parameters with possible placeholders
            
        Returns:
            Resolved parameters
        """
        # Get goal with reader lock
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            logger.error(f"Cannot resolve parameters: Goal {goal_id} not found.")
            return parameters  # Return original if goal gone
             
        resolved_params = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$step_"):
                parts = value[1:].split('.')
                source_step_index_str = parts[0][4:]  # Get the number after 'step_'
                path = parts[1:]

                try:
                    # Step indices in plans are 0-based, but placeholders likely use 1-based. Adjust.
                    source_step_index = int(source_step_index_str) - 1
                except ValueError:
                    logger.warning(f"Invalid step index '{source_step_index_str}' in parameter placeholder '{value}' for goal '{goal.id}'.")
                    resolved_params[key] = None
                    continue

                if 0 <= source_step_index < len(goal.plan):
                    source_step = goal.plan[source_step_index]
                    if source_step.status == "completed" and source_step.result is not None:
                        try:
                            current_value = source_step.result
                            for part in path:
                                if isinstance(current_value, dict):
                                    current_value = current_value.get(part)
                                elif isinstance(current_value, BaseModel):
                                    current_value = getattr(current_value, part, None)
                                elif hasattr(current_value, part):
                                    current_value = getattr(current_value, part)
                                else:
                                    try:  # Try index access for lists/tuples
                                        idx = int(part)
                                        if isinstance(current_value, (list, tuple)) and -len(current_value) <= idx < len(current_value):
                                            current_value = current_value[idx]
                                        else:
                                            current_value = None
                                            break
                                    except (ValueError, TypeError, IndexError):
                                        current_value = None
                                        break
                                if current_value is None: 
                                    break

                            resolved_params[key] = current_value
                            if current_value is None:
                                logger.warning(f"Path '{'.'.join(path)}' resolved to None for parameter placeholder '{value}' in goal '{goal.id}'.")

                        except Exception as e:
                            logger.error(f"Error resolving path '{'.'.join(path)}' for parameter '{value}': {e}")
                            resolved_params[key] = None
                    else:
                        logger.warning(f"Could not resolve parameter '{value}' for goal '{goal.id}'. Source step '{source_step.step_id}' not completed or has no result.")
                        resolved_params[key] = None
                else:
                    logger.warning(f"Invalid source step index {source_step_index + 1} for parameter placeholder '{value}' in goal '{goal.id}'.")
                    resolved_params[key] = None
            else:
                resolved_params[key] = value
        return resolved_params

    async def update_goal_status(self, 
                                goal_id: str, 
                                status: str, 
                                result: Optional[Any] = None, 
                                error: Optional[str] = None,
                                enable_hierarchy: bool = True,
                                enable_context_callbacks: bool = True,
                                enable_completion_rewards: bool = True,
                                trigger_notifications: bool = True) -> Dict[str, Any]:
        """
        Unified method to update goal status with optional advanced features
        
        Args:
            goal_id: Goal ID
            status: New status
            result: Optional result data
            error: Optional error message
            enable_hierarchy: Whether to handle parent/child relationships
            enable_context_callbacks: Whether to trigger A2A context callbacks
            enable_completion_rewards: Whether to process completion rewards
            trigger_notifications: Whether to notify other systems
            
        Returns:
            Status update result with detailed information
        """
        # Validate status
        valid_statuses = ["pending", "active", "completed", "failed", "abandoned"]
        if status not in valid_statuses:
            return {"success": False, "error": f"Invalid status: {status}. Must be one of {valid_statuses}"}
        
        # First get the goal with reader lock to check if it exists and get old status
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            logger.warning(f"Attempted to update status for unknown goal: {goal_id}")
            return {"success": False, "error": "Goal not found"}
    
        old_status = goal.status
        if old_status == status: 
            return {
                "success": True, 
                "goal_id": goal_id,
                "old_status": old_status, 
                "new_status": status, 
                "unchanged": True,
                "features_used": []
            }
    
        # Update the goal with writer lock
        await self._update_goal_with_writer_lock(
            goal_id,
            lambda g: (
                setattr(g, "status", status),
                setattr(g, "last_error", error),
                setattr(g, "completion_time", datetime.datetime.now()) if status in ["completed", "failed", "abandoned"] else None
            )
        )
        
        # Track which features were used
        features_used = []
        hierarchy_results = {}
        reward_results = {}
        callback_results = {}
        notifications = {}
        
        # Update active goals set if needed
        if status in ["completed", "failed", "abandoned"]:
            async with self._active_goals_lock:
                self.active_goals.discard(goal_id)
            
            # Update statistics
            if status == "completed":
                self.goal_statistics["completed"] += 1
                
                # === COMPLETION REWARDS FEATURE ===
                if enable_completion_rewards:
                    try:
                        reward_results = await self._process_goal_completion_reward(goal_id, result)
                        features_used.append("completion_rewards")
                        logger.debug(f"Processed completion reward for goal '{goal_id}'")
                    except Exception as e:
                        logger.error(f"Error processing completion reward for goal '{goal_id}': {e}")
                        reward_results = {"error": str(e)}
                
                # === HIERARCHY HANDLING FEATURE ===
                if enable_hierarchy and goal.relationships:
                    try:
                        hierarchy_results = await self._handle_goal_hierarchy_completion(goal_id, goal, result)
                        features_used.append("hierarchy_handling")
                        logger.debug(f"Processed hierarchy relationships for goal '{goal_id}'")
                    except Exception as e:
                        logger.error(f"Error handling hierarchy for goal '{goal_id}': {e}")
                        hierarchy_results = {"error": str(e)}
                        
            elif status == "failed":
                self.goal_statistics["failed"] += 1
                
                # Handle hierarchy for failed goals if enabled
                if enable_hierarchy and goal.relationships:
                    try:
                        hierarchy_results = await self._handle_goal_hierarchy_failure(goal_id, goal, error)
                        features_used.append("hierarchy_failure_handling")
                    except Exception as e:
                        logger.error(f"Error handling hierarchy failure for goal '{goal_id}': {e}")
                        hierarchy_results = {"error": str(e)}
                        
            elif status == "abandoned":
                self.goal_statistics["abandoned"] += 1
    
        logger.info(f"Goal '{goal_id}' status changed from {old_status} to {status}.")
    
        # === CONTEXT CALLBACKS FEATURE ===
        if enable_context_callbacks:
            try:
                callback_data = {
                    "goal_id": goal_id,
                    "old_status": old_status,
                    "new_status": status,
                    "associated_need": goal.associated_need,
                    "result": result,
                    "error": error,
                    "hierarchy_results": hierarchy_results,
                    "reward_results": reward_results
                }
                
                # Trigger specific callbacks based on status
                if status == "completed":
                    await self.trigger_context_callback("goal_completed", callback_data)
                elif status == "failed":
                    await self.trigger_context_callback("goal_failed", callback_data)
                elif status == "abandoned":
                    await self.trigger_context_callback("goal_abandoned", callback_data)
                
                # Always trigger general status change callback
                await self.trigger_context_callback("goal_status_change", callback_data)
                
                callback_results = {"triggered": True, "callback_data": callback_data}
                features_used.append("context_callbacks")
                logger.debug(f"Triggered context callbacks for goal '{goal_id}'")
                
            except Exception as e:
                logger.error(f"Error in context callbacks for goal '{goal_id}': {e}")
                callback_results = {"error": str(e)}
    
        # === SYSTEM NOTIFICATIONS FEATURE ===
        if trigger_notifications:
            try:
                notifications = await self._notify_systems(
                    RunContextWrapper(context=RunContext(goal_id=goal_id)), 
                    goal_id=goal_id, 
                    status=status, 
                    result=result, 
                    error=error
                )
                features_used.append("system_notifications")
                logger.debug(f"Sent system notifications for goal '{goal_id}'")
            except Exception as e:
                logger.error(f"Error in notifying systems about goal status change: {e}")
                notifications = {"error": str(e)}
    
        # Return comprehensive result
        return {
            "success": True,
            "goal_id": goal_id,
            "old_status": old_status,
            "new_status": status,
            "features_used": features_used,
            "hierarchy_results": hierarchy_results,
            "reward_results": reward_results,
            "callback_results": callback_results,
            "notifications": notifications,
            "associated_need": goal.associated_need,
            "completion_time": goal.completion_time.isoformat() if goal.completion_time else None
        }
    
    # === HELPER METHODS FOR MODULAR FEATURES ===
    
    async def _handle_goal_hierarchy_completion(self, goal_id: str, goal: Goal, result: Optional[Any] = None) -> Dict[str, Any]:
        """Handle parent/child relationships when a goal is completed"""
        hierarchy_results = {
            "parent_updates": [],
            "child_updates": [],
            "cascading_completions": []
        }
        
        # Handle parent goal updates
        if goal.relationships.parent_goal_id:
            parent_id = goal.relationships.parent_goal_id
            parent_goal = await self._get_goal_with_reader_lock(parent_id)
            
            if parent_goal and parent_goal.relationships and parent_goal.relationships.child_goal_ids:
                total_children = len(parent_goal.relationships.child_goal_ids)
                
                if total_children > 0:
                    # Count completed children
                    completed_children = 0
                    for child_id in parent_goal.relationships.child_goal_ids:
                        child_goal = await self._get_goal_with_reader_lock(child_id)
                        if child_goal and child_goal.status == "completed":
                            completed_children += 1
                    
                    # Update parent goal progress
                    new_progress = completed_children / total_children
                    await self._update_goal_with_writer_lock(
                        parent_id,
                        lambda g: setattr(g, "progress", new_progress)
                    )
                    
                    hierarchy_results["parent_updates"].append({
                        "parent_id": parent_id,
                        "new_progress": new_progress,
                        "completed_children": completed_children,
                        "total_children": total_children
                    })
                    
                    # If all children completed, mark parent as completed
                    if completed_children == total_children:
                        parent_completion = await self.update_goal_status(
                            parent_id, 
                            "completed", 
                            result="All subgoals completed",
                            enable_hierarchy=True,  # Allow cascading
                            enable_context_callbacks=True,
                            enable_completion_rewards=True,
                            trigger_notifications=True
                        )
                        hierarchy_results["cascading_completions"].append({
                            "goal_id": parent_id,
                            "completion_result": parent_completion
                        })
        
        return hierarchy_results
    
    async def _handle_goal_hierarchy_failure(self, goal_id: str, goal: Goal, error: Optional[str] = None) -> Dict[str, Any]:
        """Handle parent/child relationships when a goal fails"""
        hierarchy_results = {
            "parent_notifications": [],
            "failure_propagation": []
        }
        
        # Notify parent of child failure
        if goal.relationships.parent_goal_id:
            parent_id = goal.relationships.parent_goal_id
            parent_goal = await self._get_goal_with_reader_lock(parent_id)
            
            if parent_goal:
                # Update parent with failure information
                await self._update_goal_with_writer_lock(
                    parent_id,
                    lambda g: g.execution_history.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "type": "child_goal_failed",
                        "child_goal_id": goal_id,
                        "failure_reason": error or "Unknown error"
                    })
                )
                
                hierarchy_results["parent_notifications"].append({
                    "parent_id": parent_id,
                    "failed_child": goal_id,
                    "error": error
                })
                
                # Check if parent should also fail due to critical child failure
                if goal.relationships.relationship_type == "critical_dependency":
                    parent_failure = await self.update_goal_status(
                        parent_id,
                        "failed",
                        error=f"Critical child goal failed: {goal_id}",
                        enable_hierarchy=True,
                        enable_context_callbacks=True,
                        enable_completion_rewards=False,
                        trigger_notifications=True
                    )
                    hierarchy_results["failure_propagation"].append({
                        "goal_id": parent_id,
                        "failure_result": parent_failure
                    })
        
        return hierarchy_results
    
    # === CONVENIENCE METHODS FOR BACKWARD COMPATIBILITY ===
    
    async def update_goal_status_basic(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Basic goal status update without advanced features"""
        return await self.update_goal_status(
            goal_id=goal_id,
            status=status,
            result=result,
            error=error,
            enable_hierarchy=False,
            enable_context_callbacks=False,
            enable_completion_rewards=False,
            trigger_notifications=True
        )
    
    async def update_goal_status_enhanced(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced goal status update with context callbacks only"""
        return await self.update_goal_status(
            goal_id=goal_id,
            status=status,
            result=result,
            error=error,
            enable_hierarchy=False,
            enable_context_callbacks=True,
            enable_completion_rewards=False,
            trigger_notifications=True
        )
    
    async def update_goal_status_with_hierarchy(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Goal status update with hierarchy handling only"""
        return await self.update_goal_status(
            goal_id=goal_id,
            status=status,
            result=result,
            error=error,
            enable_hierarchy=True,
            enable_context_callbacks=False,
            enable_completion_rewards=True,
            trigger_notifications=True
        )
    
    async def update_goal_status_full_featured(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Goal status update with all features enabled"""
        return await self.update_goal_status(
            goal_id=goal_id,
            status=status,
            result=result,
            error=error,
            enable_hierarchy=True,
            enable_context_callbacks=True,
            enable_completion_rewards=True,
            trigger_notifications=True
        )
        
    async def abandon_goal(self, goal_id: str, reason: str) -> Dict[str, Any]:
        """
        Abandons an active or pending goal
        
        Args:
            goal_id: Goal ID
            reason: Reason for abandonment
            
        Returns:
            Abandonment result
        """
        logger.info(f"Abandoning goal '{goal_id}': {reason}")
        
        # Get goal with reader lock
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return {"status": "error", "message": f"Goal {goal_id} not found"}
                
        if goal.status not in ["active", "pending"]:
            return {"status": "error", "message": f"Cannot abandon goal with status {goal.status}"}
        
        # Update status
        result = await self.update_goal_status(goal_id, "abandoned", error=reason)
        
        return {
            "status": "success" if result["success"] else "error", 
            "goal_id": goal_id, 
            "message": f"Goal abandoned: {reason}" if result["success"] else result.get("error", "Unknown error")
        }

    async def has_active_goal_for_need(self, need_name: str) -> bool:
        """
        Checks if there's an active goal associated with a specific need
        
        Args:
            need_name: Need name
            
        Returns:
            True if active goal exists for need
        """
        # Get all goals with reader lock
        all_goals = []
        
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                await self._writer_lock.acquire()
        
        try:
            # Safe to read goals dictionary with reader lock
            all_goals = list(self.goals.values())
        finally:
            # Release reader lock
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._writer_lock.release()
        
        # Check if any goal matches criteria
        return any(g.status in ["active", "pending"] and g.associated_need == need_name 
                  for g in all_goals)

    async def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets the status and plan of a specific goal
        
        Args:
            goal_id: Goal ID
            
        Returns:
            Goal status information
        """
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return None
            
        # Return a copy, exclude potentially large history for status checks
        goal_data = goal.model_dump(exclude={'execution_history', 'plan'})
        goal_data['plan_step_count'] = len(goal.plan)
        
        # Add current step info if available
        if 0 <= goal.current_step_index < len(goal.plan):
            current_step = goal.plan[goal.current_step_index]
            goal_data['current_step'] = {
                'description': current_step.description,
                'action': current_step.action,
                'status': current_step.status
            }
        
        return goal_data

    async def get_all_goals(self, status_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Gets all goals, optionally filtered by status
        
        Args:
            status_filter: Optional list of statuses to filter by
            
        Returns:
            List of goal summaries
        """
        # Get all goals with reader lock
        all_goals = []
        
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                await self._writer_lock.acquire()
        
        try:
            # Safe to read goals dictionary with reader lock
            all_goals = list(self.goals.values())
        finally:
            # Release reader lock
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._writer_lock.release()
        
        filtered_goals = []
        for goal in all_goals:
            if status_filter is None or goal.status in status_filter:
                # Return summaries, exclude large fields
                summary = goal.model_dump(exclude={'execution_history', 'plan'})
                summary['plan_step_count'] = len(goal.plan)
                if 0 <= goal.current_step_index < len(goal.plan):
                    summary['current_step_description'] = goal.plan[goal.current_step_index].description
                else:
                    summary['current_step_description'] = "N/A"
                filtered_goals.append(summary)
        
        # Sort by priority (descending)
        filtered_goals.sort(key=lambda g: g['priority'], reverse=True)
        
        return filtered_goals
    
    async def get_goal_statistics(self) -> Dict[str, Any]:
        """
        Gets statistics about goal execution
        
        Returns:
            Goal statistics
        """
        # Get all goals with reader lock
        all_goals = []
        
        async with self._reader_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                await self._writer_lock.acquire()
        
        try:
            # Safe to read goals dictionary with reader lock
            all_goals = list(self.goals.values())
        finally:
            # Release reader lock
            async with self._reader_lock:
                self._reader_count -= 1
                if self._reader_count == 0:
                    self._writer_lock.release()
        
        # Count goals by status
        status_counts = {}
        for goal in all_goals:
            status = goal.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get active goals count
        async with self._active_goals_lock:
            active_goals_count = len(self.active_goals)
        
        # Calculate success rate
        total_completed = self.goal_statistics["completed"]
        total_failed = self.goal_statistics["failed"]
        total_abandoned = self.goal_statistics["abandoned"]
        total_merged = self.goal_statistics.get("merged", 0)
        total_finished = total_completed + total_failed + total_abandoned
        
        success_rate = total_completed / total_finished if total_finished > 0 else 0
        
        # Calculate average completion time
        completion_times = []
        for goal in all_goals:
            if goal.status == "completed" and goal.creation_time and goal.completion_time:
                duration = (goal.completion_time - goal.creation_time).total_seconds()
                completion_times.append(duration)
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
        
        # Enhanced statistics
        conflict_stats = {
            "conflicts_detected": 0,
            "conflicts_resolved": self.goal_statistics.get("conflict_resolved", 0),
            "resolution_methods": {}
        }
        
        integration_stats = {
            "needs_system_integrations": 0,
            "emotional_system_integrations": 0,
            "external_system_integrations": 0
        }
        
        # Collect enhanced statistics
        for goal in all_goals:
            # Count conflicts
            if goal.conflict_data:
                conflict_stats["conflicts_detected"] += len(goal.conflict_data)
                
                # Count resolution methods
                for conflict_goal_id, resolution_data in goal.conflict_data.items():
                    if hasattr(resolution_data, "strategy"):
                        strategy = resolution_data.strategy
                        conflict_stats["resolution_methods"][strategy] = conflict_stats["resolution_methods"].get(strategy, 0) + 1
            
            # Count integrations
            if goal.associated_need:
                integration_stats["needs_system_integrations"] += 1
            
            if goal.emotional_motivation:
                integration_stats["emotional_system_integrations"] += 1
                
            if goal.external_system_ids:
                integration_stats["external_system_integrations"] += len(goal.external_system_ids)
        
        return {
            "total_goals_created": self.goal_statistics["created"],
            "goals_by_status": status_counts,
            "success_rate": success_rate,
            "average_completion_time_seconds": avg_completion_time,
            "active_goals_count": active_goals_count,
            "statistics": self.goal_statistics,
            "conflict_statistics": conflict_stats,
            "integration_statistics": integration_stats,
            "persistence_status": {
                "total_persisted": len([g for g in all_goals if g.last_persisted]),
                "dirty_goals": len(self._dirty_goals)
            }
        }
        
    async def close(self):
        """Clean shutdown of the GoalManager, ensuring all data is saved"""
        logger.info("Shutting down GoalManager...")
        
        # Cancel any background tasks
        try:
            await self.stop()
        except Exception as e:
            logger.error(f"Error stopping GoalManager: {e}")
        
        # Final save of all goals
        try:
            logger.info("Performing final save of all goals")
            await self.save_goals()
        except Exception as e:
            logger.error(f"Error saving goals during shutdown: {e}")
        
        logger.info("GoalManager shutdown complete")
