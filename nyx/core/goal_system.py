# nyx/core/goal_system.py

import logging
import datetime
import uuid
import asyncio
import json
import os
import aiofiles
import pickle
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable, Type
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import copy
import random
from functools import lru_cache
import asyncio
import weakref

from agents import (
    Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper,
    handoff, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
)

logger = logging.getLogger(__name__)

# Schema version for persistence
SCHEMA_VERSION = "1.0.0"

class GoalStep(BaseModel):
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
        # Basic validation - could be expanded to check against known brain methods
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

# New enum for conflict resolution strategies
class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving conflicts between goals"""
    PRIORITY_BASED = "priority_based"  # Higher priority wins
    TIME_HORIZON_BASED = "time_horizon_based"  # Shorter time horizon wins 
    NEGOTIATE = "negotiate"  # Try to find a compromise
    MERGE = "merge"  # Merge into a single goal
    USER_DECISION = "user_decision"  # Ask the user to decide

# New model for conflict resolution data
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
    
    # New fields for persistence and versioning
    version: str = Field(SCHEMA_VERSION, description="Schema version for this goal")
    last_modified: datetime.datetime = Field(default_factory=datetime.datetime.now)
    last_persisted: Optional[datetime.datetime] = None
    checksum: Optional[str] = None  # For data integrity verification
    
    # New fields for conflict resolution
    conflict_data: Optional[Dict[str, ConflictResolutionData]] = Field(default_factory=dict)
    negotiation_status: Optional[Dict[str, Any]] = None
    
    # New fields for integration
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

# New Pydantic models for structured I/O with Agents
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

# New models for persistence
class PersistenceConfig(BaseModel):
    """Configuration for goal persistence"""
    storage_path: str = Field("./data/goals", description="Path for storing goal data")
    auto_save_interval: int = Field(60, description="Auto-save interval in seconds")
    compression: bool = Field(True, description="Whether to compress saved data")
    backup_enabled: bool = Field(True, description="Whether to create backups")
    max_backups: int = Field(5, description="Maximum number of backups to keep")

# New models for integration
class SystemIntegrationConfig(BaseModel):
    """Configuration for system integration"""
    enabled_systems: List[str] = Field(default_factory=list)
    callback_timeout: float = Field(5.0, description="Timeout for callbacks in seconds")
    retry_callbacks: bool = Field(True, description="Whether to retry failed callbacks")
    max_retries: int = Field(3, description="Maximum number of callback retries")

# New model for conflict detection
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
                 conflict_config: Optional[ConflictDetectionConfig] = None):
        """
        Args:
            brain_reference: Reference to the main NyxBrain instance for action execution.
            persistence_config: Configuration for goal persistence
            integration_config: Configuration for system integration
            conflict_config: Configuration for conflict detection and resolution
        """
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()  # IDs of goals currently being executed
        self.brain = brain_reference  # Set via set_brain_reference if needed later
        self.max_concurrent_goals = 3  # Increased from 1 for better concurrency
        
        # IMPROVEMENT 1: Concurrency and Performance Improvements
        # Replace single lock with more granular locking mechanism
        self._goal_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)  # Lock per goal
        self._goals_dict_lock = asyncio.Lock()  # Separate lock for the goals dictionary itself
        self._active_goals_lock = asyncio.Lock()  # Separate lock for active goals set
        self._reader_count = 0  # Number of current readers
        self._reader_lock = asyncio.Lock()  # Lock for reader count
        self._writer_lock = asyncio.Lock()  # Lock for writers
        
        # Thread pool for background tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # IMPROVEMENT 2: Persistence and Durability
        self.persistence_config = persistence_config or PersistenceConfig()
        self._persistence_task = None
        self._last_save_time = datetime.datetime.now()
        self._dirty_goals: Set[str] = set()  # Track which goals have changed since last save
        
        # IMPROVEMENT 3: Goal Conflict Resolution
        self.conflict_config = conflict_config or ConflictDetectionConfig()
        self._conflict_resolution_tasks: Dict[str, asyncio.Task] = {}  # Track running resolution tasks
        
        # IMPROVEMENT 7: Integration Enhancements
        self.integration_config = integration_config or SystemIntegrationConfig()
        self._integration_callbacks: Dict[str, List[Callable]] = defaultdict(list)  # Callbacks for goal events
        self._external_system_clients: Dict[str, Any] = {}  # Clients for external systems
        
        # Goal outcomes for analytics
        self.goal_statistics = {
            "created": 0,
            "completed": 0,
            "failed": 0,
            "abandoned": 0,
            "merged": 0,  # New statistic for merged goals
            "conflict_resolved": 0  # New statistic for resolved conflicts
        }
        
        # Initialize agents
        self._init_agents()
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
        
    def _init_agents(self):
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
        
        # NEW: Conflict resolution agent
        self.conflict_resolution_agent = self._create_conflict_resolution_agent()
        
        # NEW: Goal negotiation agent
        self.negotiation_agent = self._create_negotiation_agent()
        
        # NEW: Goal merging agent
        self.merging_agent = self._create_merging_agent()

    def set_brain_reference(self, brain):
        """Set the reference to the main NyxBrain after initialization."""
        self.brain = brain
        logger.info("NyxBrain reference set for GoalManager.")
        
    # ==========================================================================
    # IMPROVEMENT 2: Persistence and Durability
    # ==========================================================================
    
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
        goal_ids = []
        
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
    
    # ==========================================================================
    # IMPROVEMENT 1: Concurrency and Performance Improvements
    # ==========================================================================
    
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
    
    # Optimized version of get_prioritized_goals
    async def get_prioritized_goals(self) -> List[Goal]:
        """
        Returns active and pending goals sorted by priority with optimized sorting.
        This method is now asynchronous to correctly handle async locks.
        """
        goals_copy = {}
        # CORRECTLY acquire the async lock
        async with self._goals_dict_lock:
            goals_copy = {
                g_id: g.model_copy() for g_id, g in self.goals.items()
                if g.status in ["pending", "active"]
            }

        if not goals_copy:
            return []
        
        # Use a cached sorting key function for better performance
        # NOTE: The LRU cache needs to be defined outside or cleared if Goal properties change its hash
        # Defining it inside means a new cache per call, defeating the purpose. Let's assume it's defined
        # at the class level or module level if appropriate, or accept the per-call overhead if necessary.
        # For this example, we keep it inside, acknowledging this limitation. If performance is critical,
        # move the cached function definition outside.
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

            # Added small random factor to break ties consistently but not predictably
            random.seed(goal_id) # Seed with goal_id for deterministic tie-breaking per goal
            tie_breaker = random.uniform(-0.001, 0.001)

            return (priority
                    + (age_penalty * 0.01 * status_boost)
                    + deadline_urgency
                    + time_horizon_factor
                    + motivation_boost
                    + tie_breaker)


        # Convert goals to tuples for faster sorting
        now = datetime.datetime.now()
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

    
    # ==========================================================================
    # IMPROVEMENT 3: Goal Conflict Resolution
    # ==========================================================================
    
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
            model="gpt-4o",
            tools=[
                function_tool(self._analyze_goal_similarity),
                function_tool(self._analyze_resource_conflicts),
                function_tool(self._get_goal_details)
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
            model="gpt-4o",
            tools=[
                function_tool(self._propose_goal_modifications),
                function_tool(self._evaluate_modification_impact),
                function_tool(self._get_shared_subgoals)
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
            model="gpt-4o",
            tools=[
                function_tool(self._get_goal_common_elements),
                function_tool(self._generate_merged_goal_description),
                function_tool(self._validate_merged_goal_coverage)
            ]
        )
    
    async def _analyze_goal_similarity(self, ctx: RunContextWrapper, goal_id1: str, goal_id2: str) -> Dict[str, Any]:
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
    
    async def _analyze_resource_conflicts(self, ctx: RunContextWrapper, goal_id1: str, goal_id2: str) -> Dict[str, Any]:
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
    
    async def _propose_goal_modifications(self, ctx: RunContextWrapper, goal_id: str, 
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
    
    async def _evaluate_modification_impact(self, ctx: RunContextWrapper, goal_id: str, 
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
    
    async def _get_goal_common_elements(self, ctx: RunContextWrapper, goal_id1: str, goal_id2: str) -> Dict[str, Any]:
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
    
    async def _generate_merged_goal_description(self, ctx: RunContextWrapper, goal_id1: str, goal_id2: str, 
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
        
        # Generate merged description - in a real system, this could use an LLM
        description1 = goal1.description
        description2 = goal2.description
        
        # Extract key verbs and objects
        words1 = description1.split()
        words2 = description2.split()
        
        # Simple combination for demonstration
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
    
    async def _validate_merged_goal_coverage(self, ctx: RunContextWrapper, merged_goal_data: Dict[str, Any], 
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
        
        # Check essential elements coverage
        # In a real system, this would be more sophisticated
        
        # Keywords that should be preserved
        keywords1 = set([w.lower() for w in goal1.description.split() if len(w) > 4])
        keywords2 = set([w.lower() for w in goal2.description.split() if len(w) > 4])
        merged_keywords = set([w.lower() for w in merged_goal_data["description"].split() if len(w) > 4])
        
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
            "missing_elements": []  # Would be populated in a real system
        }
    
    async def detect_goal_conflicts(self) -> List[Dict[str, Any]]:
        """Detect potential conflicts between goals"""
        # Get all active and pending goals
        goals = []
        async with self._goals_dict_lock:
            goals = [
                goal_id for goal_id, goal in self.goals.items()
                if goal.status in ["active", "pending"]
            ]
        
        conflicts = []
        
        # Compare each pair of goals
        for i, goal_id1 in enumerate(goals):
            for goal_id2 in goals[i+1:]:
                # Skip if goals are already in conflict resolution
                if await self._are_goals_in_conflict_resolution(goal_id1, goal_id2):
                    continue
                
                # Analyze similarity
                similarity = await self._analyze_goal_similarity(None, goal_id1, goal_id2)
                
                # If similarity suggests potential conflict
                if similarity.get("conflict_potential") in ["medium", "high"]:
                    # Analyze resource conflicts
                    resource_conflicts = await self._analyze_resource_conflicts(None, goal_id1, goal_id2)
                    
                    # If resource conflicts exist, add to conflicts list
                    if resource_conflicts.get("has_conflicts") or similarity.get("conflict_potential") == "high":
                        conflicts.append({
                            "goal_id1": goal_id1,
                            "goal_id2": goal_id2,
                            "similarity": similarity,
                            "resource_conflicts": resource_conflicts,
                            "detection_time": datetime.datetime.now().isoformat()
                        })
        
        logger.info(f"Detected {len(conflicts)} potential goal conflicts")
        return conflicts
    
    async def _are_goals_in_conflict_resolution(self, goal_id1: str, goal_id2: str) -> bool:
        """Check if two goals are already in conflict resolution"""
        # Check goal1
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        if goal1 and goal1.conflict_data and goal_id2 in goal1.conflict_data:
            return True
        
        # Check goal2
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        if goal2 and goal2.conflict_data and goal_id1 in goal2.conflict_data:
            return True
        
        return False
    
    async def resolve_goal_conflict(self, goal_id1: str, goal_id2: str, 
                                 conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a conflict between two goals
        
        Args:
            goal_id1: First goal ID
            goal_id2: Second goal ID
            conflict_data: Conflict detection data
            
        Returns:
            Resolution result
        """
        logger.info(f"Resolving conflict between goals {goal_id1} and {goal_id2}")
        
        # Default to priority-based resolution
        strategy = self.conflict_config.default_strategy
        
        try:
            # Use resolution agent to determine best strategy
            result = await Runner.run(
                self.conflict_resolution_agent,
                json.dumps({
                    "goal_id1": goal_id1,
                    "goal_id2": goal_id2,
                    "conflict_data": conflict_data
                }),
                run_config={
                    "workflow_name": "ConflictResolution",
                    "trace_metadata": {
                        "goal_id1": goal_id1,
                        "goal_id2": goal_id2
                    }
                }
            )
            
            # Extract recommended strategy
            resolution_output = result.final_output
            strategy = resolution_output.get("recommended_strategy", strategy)
            
            # Create conflict resolution record for both goals
            conflict_record = ConflictResolutionData(
                strategy=strategy,
                resolution_time=datetime.datetime.now()
            )
            
            # Apply resolution based on strategy
            if strategy == ConflictResolutionStrategy.PRIORITY_BASED:
                return await self._apply_priority_based_resolution(goal_id1, goal_id2, conflict_record)
                
            elif strategy == ConflictResolutionStrategy.TIME_HORIZON_BASED:
                return await self._apply_time_horizon_resolution(goal_id1, goal_id2, conflict_record)
                
            elif strategy == ConflictResolutionStrategy.NEGOTIATE:
                return await self._apply_negotiation_resolution(goal_id1, goal_id2, conflict_record, conflict_data)
                
            elif strategy == ConflictResolutionStrategy.MERGE:
                return await self._apply_merge_resolution(goal_id1, goal_id2, conflict_record, conflict_data)
                
            elif strategy == ConflictResolutionStrategy.USER_DECISION:
                # Mark as needing user input
                # In a real system, this would trigger a user notification
                async with self._get_goal_lock(goal_id1):
                    if goal_id1 in self.goals:
                        self.goals[goal_id1].negotiation_status = {
                            "status": "awaiting_user_input",
                            "conflict_with": goal_id2,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                
                async with self._get_goal_lock(goal_id2):
                    if goal_id2 in self.goals:
                        self.goals[goal_id2].negotiation_status = {
                            "status": "awaiting_user_input",
                            "conflict_with": goal_id1,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                
                return {
                    "resolution": "user_decision_required",
                    "goal_id1": goal_id1,
                    "goal_id2": goal_id2,
                    "strategy": strategy,
                    "needs_user_input": True
                }
            
            else:
                logger.warning(f"Unknown conflict resolution strategy: {strategy}")
                return {
                    "resolution": "failed",
                    "error": f"Unknown resolution strategy: {strategy}"
                }
                
        except Exception as e:
            logger.error(f"Error resolving conflict between goals {goal_id1} and {goal_id2}: {e}")
            return {
                "resolution": "failed",
                "error": str(e)
            }
    
    async def _apply_priority_based_resolution(self, goal_id1: str, goal_id2: str, 
                                           conflict_record: ConflictResolutionData) -> Dict[str, Any]:
        """Apply priority-based conflict resolution"""
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {"resolution": "failed", "error": "One or both goals not found"}
        
        # Determine which goal has higher priority
        if goal1.priority >= goal2.priority:
            higher_priority_id = goal_id1
            lower_priority_id = goal_id2
        else:
            higher_priority_id = goal_id2
            lower_priority_id = goal_id1
        
        # Update the lower priority goal to reduce priority further
        await self._update_goal_with_writer_lock(
            lower_priority_id,
            lambda goal: setattr(goal, "priority", max(0.1, goal.priority * 0.8))
        )
        
        # Add conflict data to both goals
        conflict_record.resolved = True
        conflict_record.resolution_reason = "priority_based"
        
        await self._update_goal_with_writer_lock(
            goal_id1,
            lambda goal: goal.conflict_data.update({goal_id2: conflict_record.model_copy()})
        )
        
        await self._update_goal_with_writer_lock(
            goal_id2,
            lambda goal: goal.conflict_data.update({goal_id1: conflict_record.model_copy()})
        )
        
        # Update relationship to reflect conflict
        await self._update_goal_with_writer_lock(
            goal_id1,
            lambda goal: goal.relationships.conflicts_with_goal_ids.append(goal_id2)
            if goal_id2 not in goal.relationships.conflicts_with_goal_ids else None
        )
        
        await self._update_goal_with_writer_lock(
            goal_id2,
            lambda goal: goal.relationships.conflicts_with_goal_ids.append(goal_id1)
            if goal_id1 not in goal.relationships.conflicts_with_goal_ids else None
        )
        
        # Update statistics
        self.goal_statistics["conflict_resolved"] += 1
        
        return {
            "resolution": "successful",
            "strategy": ConflictResolutionStrategy.PRIORITY_BASED,
            "higher_priority_goal": higher_priority_id,
            "lower_priority_goal": lower_priority_id,
            "action_taken": "reduced_priority"
        }
    
    async def _apply_time_horizon_resolution(self, goal_id1: str, goal_id2: str, 
                                         conflict_record: ConflictResolutionData) -> Dict[str, Any]:
        """Apply time horizon based conflict resolution"""
        goal1 = await self._get_goal_with_reader_lock(goal_id1)
        goal2 = await self._get_goal_with_reader_lock(goal_id2)
        
        if not goal1 or not goal2:
            return {"resolution": "failed", "error": "One or both goals not found"}
        
        # Determine which goal has shorter time horizon
        time_horizon_order = {
            TimeHorizon.SHORT_TERM: 0,
            TimeHorizon.MEDIUM_TERM: 1,
            TimeHorizon.LONG_TERM: 2
        }
        
        if time_horizon_order[goal1.time_horizon] <= time_horizon_order[goal2.time_horizon]:
            shorter_horizon_id = goal_id1
            longer_horizon_id = goal_id2
        else:
            shorter_horizon_id = goal_id2
            longer_horizon_id = goal_id1
        
        # Make the longer horizon goal even longer if possible
        if await self._get_goal_with_reader_lock(longer_horizon_id).time_horizon != TimeHorizon.LONG_TERM:
            await self._update_goal_with_writer_lock(
                longer_horizon_id,
                lambda goal: setattr(goal, "time_horizon", TimeHorizon.LONG_TERM)
            )
        
        # Reduce priority of longer horizon goal
        await self._update_goal_with_writer_lock(
            longer_horizon_id,
            lambda goal: setattr(goal, "priority", max(0.1, goal.priority * 0.9))
        )
        
        # Add conflict data to both goals
        conflict_record.resolved = True
        conflict_record.resolution_reason = "time_horizon_based"
        
        await self._update_goal_with_writer_lock(
            goal_id1,
            lambda goal: goal.conflict_data.update({goal_id2: conflict_record.model_copy()})
        )
        
        await self._update_goal_with_writer_lock(
            goal_id2,
            lambda goal: goal.conflict_data.update({goal_id1: conflict_record.model_copy()})
        )
        
        # Update statistics
        self.goal_statistics["conflict_resolved"] += 1
        
        return {
            "resolution": "successful",
            "strategy": ConflictResolutionStrategy.TIME_HORIZON_BASED,
            "shorter_horizon_goal": shorter_horizon_id,
            "longer_horizon_goal": longer_horizon_id,
            "action_taken": "adjusted_time_horizon"
        }
    
    async def _apply_negotiation_resolution(self, goal_id1: str, goal_id2: str, 
                                        conflict_record: ConflictResolutionData,
                                        conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply negotiation-based conflict resolution"""
        # Use negotiation agent to propose modifications
        try:
            # First, get modifications for goal1
            modifications1 = await Runner.run(
                self.negotiation_agent,
                json.dumps({
                    "goal_id": goal_id1,
                    "conflict_with": goal_id2,
                    "conflict_data": conflict_data
                })
            )
            
            # Then for goal2
            modifications2 = await Runner.run(
                self.negotiation_agent,
                json.dumps({
                    "goal_id": goal_id2,
                    "conflict_with": goal_id1,
                    "conflict_data": conflict_data
                })
            )
            
            # Extract proposed modifications
            mods1 = modifications1.final_output.get("proposed_modifications", [])
            mods2 = modifications2.final_output.get("proposed_modifications", [])
            
            # Apply modifications to both goals
            for goal_id, mods in [(goal_id1, mods1), (goal_id2, mods2)]:
                for mod in mods:
                    mod_type = mod.get("type")
                    
                    if mod_type == "priority_adjustment":
                        new_priority = mod.get("new_priority")
                        if new_priority is not None:
                            await self._update_goal_with_writer_lock(
                                goal_id,
                                lambda goal: setattr(goal, "priority", new_priority)
                            )
                    
                    elif mod_type == "deadline_adjustment":
                        new_deadline = mod.get("new_deadline")
                        if new_deadline:
                            await self._update_goal_with_writer_lock(
                                goal_id,
                                lambda goal: setattr(goal, "deadline", 
                                                    datetime.datetime.fromisoformat(new_deadline))
                            )
                    
                    elif mod_type == "time_horizon_adjustment":
                        new_horizon = mod.get("new_time_horizon")
                        if new_horizon:
                            await self._update_goal_with_writer_lock(
                                goal_id,
                                lambda goal: setattr(goal, "time_horizon", new_horizon)
                            )
            
            # Add conflict data to both goals
            conflict_record.resolved = True
            conflict_record.resolution_reason = "negotiated"
            conflict_record.negotiation_attempts = 1
            
            await self._update_goal_with_writer_lock(
                goal_id1,
                lambda goal: goal.conflict_data.update({goal_id2: conflict_record.model_copy()})
            )
            
            await self._update_goal_with_writer_lock(
                goal_id2,
                lambda goal: goal.conflict_data.update({goal_id1: conflict_record.model_copy()})
            )
            
            # Update statistics
            self.goal_statistics["conflict_resolved"] += 1
            
            return {
                "resolution": "successful",
                "strategy": ConflictResolutionStrategy.NEGOTIATE,
                "modifications_goal1": mods1,
                "modifications_goal2": mods2,
                "action_taken": "applied_negotiated_changes"
            }
            
        except Exception as e:
            logger.error(f"Error in negotiation resolution: {e}")
            
            # Fall back to priority-based resolution
            logger.info(f"Falling back to priority-based resolution for goals {goal_id1} and {goal_id2}")
            conflict_record.strategy = ConflictResolutionStrategy.PRIORITY_BASED
            return await self._apply_priority_based_resolution(goal_id1, goal_id2, conflict_record)
    
    async def _apply_merge_resolution(self, goal_id1: str, goal_id2: str, 
                                  conflict_record: ConflictResolutionData,
                                  conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply merge-based conflict resolution"""
        try:
            # Get common elements for merging
            common_elements = await self._get_goal_common_elements(None, goal_id1, goal_id2)
            
            # Generate merged goal description
            merged_description = await self._generate_merged_goal_description(
                None, goal_id1, goal_id2, common_elements
            )
            
            # Validate merged goal coverage
            coverage = await self._validate_merged_goal_coverage(
                None, merged_description["merged_goal_data"], goal_id1, goal_id2
            )
            
            # If coverage is sufficient, create the merged goal
            if coverage["is_sufficient"]:
                # Create a new goal that merges the two
                merged_data = merged_description["merged_goal_data"]
                
                # Set up relationships
                relationships = GoalRelationship(
                    supports_goal_ids=[goal_id1, goal_id2],
                    relationship_type="merged"
                )
                
                # Set up emotional motivation if available
                emotional_motivation = None
                if "emotional_motivation" in merged_data:
                    em_data = merged_data["emotional_motivation"]
                    emotional_motivation = EmotionalMotivation(
                        primary_need=em_data["primary_need"],
                        intensity=em_data["intensity"],
                        expected_satisfaction=em_data["expected_satisfaction"],
                        description=em_data["description"]
                    )
                
                # Create the merged goal
                merged_goal_id = await self.add_goal(
                    description=merged_data["description"],
                    priority=merged_data["priority"],
                    source="goal_merger",
                    associated_need=merged_data.get("associated_need"),
                    emotional_motivation=emotional_motivation
                )
                
                if not merged_goal_id:
                    raise Exception("Failed to create merged goal")
                
                # Set the goal's relationships
                await self._update_goal_with_writer_lock(
                    merged_goal_id,
                    lambda goal: setattr(goal, "relationships", relationships)
                )
                
                # Update the original goals to point to the merged goal
                conflict_record.resolved = True
                conflict_record.resolution_reason = "merged"
                conflict_record.merged_goal_id = merged_goal_id
                
                await self._update_goal_with_writer_lock(
                    goal_id1,
                    lambda goal: (
                        goal.conflict_data.update({goal_id2: conflict_record.model_copy()}),
                        setattr(goal, "status", "abandoned"),
                        setattr(goal, "completion_time", datetime.datetime.now()),
                        goal.execution_history.append({
                            "event": "merged",
                            "merged_into": merged_goal_id,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    )
                )
                
                await self._update_goal_with_writer_lock(
                    goal_id2,
                    lambda goal: (
                        goal.conflict_data.update({goal_id1: conflict_record.model_copy()}),
                        setattr(goal, "status", "abandoned"),
                        setattr(goal, "completion_time", datetime.datetime.now()),
                        goal.execution_history.append({
                            "event": "merged",
                            "merged_into": merged_goal_id,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    )
                )
                
                # Update statistics
                self.goal_statistics["conflict_resolved"] += 1
                self.goal_statistics["merged"] += 1
                
                # Remove original goals from active set
                async with self._active_goals_lock:
                    self.active_goals.discard(goal_id1)
                    self.active_goals.discard(goal_id2)
                
                return {
                    "resolution": "successful",
                    "strategy": ConflictResolutionStrategy.MERGE,
                    "merged_goal_id": merged_goal_id,
                    "original_goals": [goal_id1, goal_id2],
                    "action_taken": "created_merged_goal"
                }
            else:
                # If merging not feasible, fall back to negotiation
                logger.info(f"Merging not feasible for goals {goal_id1} and {goal_id2}, falling back to negotiation")
                conflict_record.strategy = ConflictResolutionStrategy.NEGOTIATE
                return await self._apply_negotiation_resolution(goal_id1, goal_id2, conflict_record, conflict_data)
            
        except Exception as e:
            logger.error(f"Error in merge resolution: {e}")
            
            # Fall back to priority-based resolution
            logger.info(f"Falling back to priority-based resolution for goals {goal_id1} and {goal_id2}")
            conflict_record.strategy = ConflictResolutionStrategy.PRIORITY_BASED
            return await self._apply_priority_based_resolution(goal_id1, goal_id2, conflict_record)
    
    # ==========================================================================
    # IMPROVEMENT 7: Integration Enhancements
    # ==========================================================================
    
    async def _init_external_systems(self):
        """Initialize connections to external systems"""
        # In a real implementation, this would establish connections
        # to external systems based on configuration
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
    
    async def send_goal_to_external_system(self, goal_id: str, system_name: str) -> Dict[str, Any]:
        """Send a goal to an external system"""
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return {"success": False, "error": "Goal not found"}
        
        if system_name not in self._external_system_clients:
            return {"success": False, "error": f"External system {system_name} not connected"}
        
        client = self._external_system_clients[system_name]
        
        try:
            # In a real system, this would make an API call to the external system
            # Simulating response for demo
            external_id = f"ext_{uuid.uuid4().hex[:8]}"
            
            # Update goal with external system ID
            await self._update_goal_with_writer_lock(
                goal_id,
                lambda g: g.external_system_ids.update({system_name: external_id})
            )
            
            return {
                "success": True,
                "goal_id": goal_id,
                "system": system_name,
                "external_id": external_id
            }
            
        except Exception as e:
            logger.error(f"Error sending goal {goal_id} to system {system_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_external_feedback(self, goal_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback from an external system about a goal"""
        if not await self._update_goal_with_writer_lock(
            goal_id,
            lambda goal: goal.external_feedback.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "source": feedback_data.get("source", "unknown"),
                "feedback": feedback_data
            })
        ):
            return {"success": False, "error": "Goal not found"}
        
        # Process the feedback based on its type
        feedback_type = feedback_data.get("type")
        
        if feedback_type == "priority_adjustment":
            # Adjust goal priority based on external feedback
            new_priority = feedback_data.get("new_priority")
            if new_priority is not None:
                await self._update_goal_with_writer_lock(
                    goal_id,
                    lambda goal: setattr(goal, "priority", max(0.1, min(1.0, new_priority)))
                )
        
        elif feedback_type == "emotional_response":
            # Record emotional response to the goal
            if feedback_data.get("emotional_data"):
                await self._update_goal_with_writer_lock(
                    goal_id,
                    lambda goal: goal.emotional_state_snapshots.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "data": feedback_data.get("emotional_data")
                    })
                )
                
                # If goal has emotional motivation, update it based on feedback
                if feedback_data.get("intensity_adjustment") and await self._get_goal_with_reader_lock(goal_id).emotional_motivation:
                    await self._update_goal_with_writer_lock(
                        goal_id,
                        lambda goal: setattr(
                            goal.emotional_motivation, 
                            "intensity",
                            max(0.1, min(1.0, goal.emotional_motivation.intensity + feedback_data.get("intensity_adjustment", 0)))
                        )
                    )
        
        # Trigger integration callbacks for the feedback
        await self.trigger_integration_callbacks("external_feedback", {
            "goal_id": goal_id,
            "feedback": feedback_data
        })
        
        return {"success": True, "goal_id": goal_id, "processed": True}
    
    async def _link_goal_to_emotional_system(self, goal_id: str, emotion_data: Dict[str, Any]) -> bool:
        """Creates an emotional link for a goal to influence the emotional state"""
        if not self.brain or not hasattr(self.brain, "emotional_core"):
            return False
        
        emotional_core = self.brain.emotional_core
        
        # Check if the goal exists
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return False
        
        # Enhanced integration with emotional system
        # Link based on emotional motivation if available
        if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
            # Set up chemical changes based on goal's emotional motivation
            chemicals = goal.emotional_motivation.associated_chemicals
            
            # Create emotional state snapshot before changes
            baseline_state = {}
            try:
                # Get current emotional state
                baseline_state = await emotional_core.get_emotional_state()
                
                # Store baseline in goal
                await self._update_goal_with_writer_lock(
                    goal_id,
                    lambda g: g.emotional_state_snapshots.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "type": "baseline",
                        "state": baseline_state
                    })
                )
            except Exception as e:
                logger.error(f"Error getting baseline emotional state: {e}")
            
            # Apply a small anticipatory boost with more detailed integration
            for chemical, value in chemicals.items():
                try:
                    # Small anticipatory boost for focusing on the goal
                    original_value = await emotional_core.get_neurochemical_level(chemical)
                    await emotional_core.update_neurochemical(chemical, value * 0.25)  # Increased from 0.2
                    new_value = await emotional_core.get_neurochemical_level(chemical)
                    
                    # Record the chemical change in goal
                    await self._update_goal_with_writer_lock(
                        goal_id,
                        lambda g: g.emotional_state_snapshots.append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "anticipatory_change",
                            "chemical": chemical,
                            "original_value": original_value,
                            "new_value": new_value,
                            "applied_change": value * 0.25
                        })
                    )
                    
                except Exception as e:
                    logger.error(f"Error updating neurochemical {chemical}: {e}")
            
            # NEW: Register for emotional state change notifications
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
        
        # NEW: Listen for reward signals related to this goal
        if self.brain and hasattr(self.brain, "reward_system"):
            try:
                reward_system = self.brain.reward_system
                
                # Define callback for reward signals
                async def reward_signal_callback(signal_data):
                    # Check if signal is related to this goal
                    if (signal_data.get("context") and 
                        signal_data["context"].get("goal_id") == goal_id):
                        
                        # Update goal with reward signal data
                        await self._update_goal_with_writer_lock(
                            goal_id,
                            lambda g: g.external_feedback.append({
                                "timestamp": datetime.datetime.now().isoformat(),
                                "type": "reward_signal",
                                "source": "reward_system",
                                "data": signal_data
                            })
                        )
                        
                        # If goal has emotional motivation, adjust expected satisfaction
                        if goal.emotional_motivation:
                            satisfaction_adjustment = signal_data.get("value", 0) * 0.1
                            
                            await self._update_goal_with_writer_lock(
                                goal_id,
                                lambda g: setattr(
                                    g.emotional_motivation,
                                    "expected_satisfaction",
                                    max(0.1, min(1.0, g.emotional_motivation.expected_satisfaction + satisfaction_adjustment))
                                )
                            )
                
                # Register callback with reward system
                if hasattr(reward_system, "register_signal_callback"):
                    await reward_system.register_signal_callback(
                        goal_id, reward_signal_callback, expiration=3600  # 1 hour expiration
                    )
            except Exception as e:
                logger.error(f"Error registering reward signal callback: {e}")
        
        return True
    
    async def _process_goal_completion_reward(self, goal_id: str, result: Any) -> Dict[str, Any]:
        """Processes reward signals when a goal is completed with enhanced integration"""
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
            if hasattr(goal, 'time_horizon'):
                if goal.time_horizon == TimeHorizon.SHORT_TERM:
                    time_horizon_factor = 1.2  # 20% boost for short-term goal completion
                elif goal.time_horizon == TimeHorizon.LONG_TERM:
                    time_horizon_factor = 0.9  # 10% reduction for long-term goals
            
            # Adjust based on emotional motivation if available
            satisfaction_factor = 1.0
            if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
                satisfaction_factor = goal.emotional_motivation.expected_satisfaction
            
            # Calculate final reward
            reward_value = base_reward * time_horizon_factor * satisfaction_factor
            
            # Create context for the reward
            context = {
                "goal_id": goal_id,
                "goal_description": goal.description,
                "time_horizon": goal.time_horizon.value if hasattr(goal, 'time_horizon') else "medium_term",
                "emotional_need": goal.emotional_motivation.primary_need if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation else None,
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
                if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
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
        
        # Create two-way relationship with emotions if this need has emotional components
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
                    await self._link_goal_to_emotional_system(goal_id, {
                        "source": "needs_system",
                        "need": need_name,
                        "components": emotional_components
                    })
                    
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
    
    # ==========================================================================
    # Agent creation methods (from original code)
    # ==========================================================================
    
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
                model="gpt-4o",
                model_settings=ModelSettings(response_format={"type": "json_object"}, temperature=0.1),
                tools=[
                    function_tool(self._get_available_actions),
                    function_tool(self._get_action_description),
                    function_tool(self._get_goal_details),
                    function_tool(self._get_recent_goals)
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
            model="gpt-4o",
            tools=[
                function_tool(self._get_active_goals), 
                function_tool(self._check_goal_conflicts),
                function_tool(self._verify_capabilities)
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
            model="gpt-4o",
            tools=[
                function_tool(self._validate_action_sequence),
                function_tool(self._check_parameter_references),
                function_tool(self._estimate_plan_efficiency)
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
            model="gpt-4o",
            tools=[
                function_tool(self._resolve_step_parameters_tool),
                function_tool(self._execute_action),
                function_tool(self._check_dominance_appropriateness),
                function_tool(self._log_execution_result)
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
                function_tool(self._get_prioritized_goals),
                function_tool(self._update_goal_status_tool),
                function_tool(self._notify_systems),
                function_tool(self._check_concurrency_limits)
            ],
            model="gpt-4o"
        )
    
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
    @function_tool
    async def _get_active_goals(self, ctx: RunContextWrapper) -> Dict[str, Any]:
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
    
    @function_tool
    async def _check_goal_conflicts(self, ctx: RunContextWrapper, goal_description: str) -> Dict[str, Any]:
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
    
    @function_tool
    async def _verify_capabilities(self, ctx: RunContextWrapper, required_actions: List[str]) -> Dict[str, Any]:
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
    @function_tool
    async def _validate_action_sequence(self, ctx: RunContextWrapper, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that actions are sequenced correctly with proper dependencies
        
        Args:
            plan: The plan to validate
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for valid action names
        available_actions = await self._get_available_actions(ctx)
        available_action_names = [a["name"] for a in available_actions["actions"]]
        
        for i, step in enumerate(plan):
            step_num = i + 1
            action = step.get("action", "")
            
            # Check if action exists
            if action not in available_action_names:
                issues.append(f"Step {step_num}: Action '{action}' is not available")
                
            # Check parameter references to previous steps
            for param_name, param_value in step.get("parameters", {}).items():
                if isinstance(param_value, str) and param_value.startswith("$step_"):
                    # Extract referenced step number
                    try:
                        ref_step = int(param_value.split("_")[1].split(".")[0])
                        if ref_step > step_num:
                            issues.append(f"Step {step_num}: References future step {ref_step}")
                        if ref_step == step_num:
                            issues.append(f"Step {step_num}: Self-reference detected")
                    except (ValueError, IndexError):
                        issues.append(f"Step {step_num}: Invalid step reference format: {param_value}")
        
        return {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues
        }
    
    @function_tool
    async def _check_parameter_references(self, ctx: RunContextWrapper, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if parameter references between steps are valid
        
        Args:
            plan: The plan to validate
            
        Returns:
            Parameter reference validation results
        """
        issues = []
        provided_outputs = {}  # Track what each step provides
        
        for i, step in enumerate(plan):
            step_num = i + 1
            # Analyze expected outputs
            action = step.get("action", "")
            provided_outputs[step_num] = self._estimate_action_output_fields(action)
            
            # Check parameter references
            for param_name, param_value in step.get("parameters", {}).items():
                if isinstance(param_value, str) and param_value.startswith("$step_"):
                    parts = param_value[1:].split('.')
                    if len(parts) < 2:
                        issues.append(f"Step {step_num}: Invalid reference format: {param_value}")
                        continue
                        
                    # Extract referenced step and field
                    try:
                        ref_step_str = parts[0]
                        ref_step = int(ref_step_str.replace("step_", ""))
                        field_path = '.'.join(parts[1:])
                        
                        if ref_step >= step_num:
                            issues.append(f"Step {step_num}: References non-executed step {ref_step}")
                            continue
                            
                        if ref_step not in provided_outputs:
                            issues.append(f"Step {step_num}: References unknown step {ref_step}")
                            continue
                            
                        # Check if the referenced field exists in the output
                        if not self._check_field_availability(provided_outputs[ref_step], field_path):
                            issues.append(f"Step {step_num}: Referenced field '{field_path}' may not exist in step {ref_step} output")
                    except (ValueError, IndexError):
                        issues.append(f"Step {step_num}: Invalid step reference: {param_value}")
        
        return {
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues
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
    
    @function_tool
    async def _estimate_plan_efficiency(self, ctx: RunContextWrapper, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate the efficiency of a plan
        
        Args:
            plan: The plan to evaluate
            
        Returns:
            Efficiency analysis
        """
        # Count actions by category
        action_categories = {}
        for step in plan:
            action = step.get("action", "")
            category = "unknown"
            
            if action.startswith(("query_", "retrieve_")):
                category = "retrieval"
            elif action.startswith(("add_", "create_")):
                category = "creation"
            elif action.startswith(("update_", "modify_")):
                category = "modification"
            elif action.startswith(("analyze_", "evaluate_", "reason_")):
                category = "analysis"
            elif action.startswith(("generate_", "express_")):
                category = "generation"
            
            action_categories[category] = action_categories.get(category, 0) + 1
        
        # Check for potential inefficiencies
        retrieval_heavy = action_categories.get("retrieval", 0) > len(plan) * 0.5
        creation_heavy = action_categories.get("creation", 0) > len(plan) * 0.4
        
        suggestions = []
        if retrieval_heavy:
            suggestions.append("Plan may benefit from combining multiple retrieval steps")
        if creation_heavy:
            suggestions.append("Plan has many creation steps; consider batching or combining some")
        if len(plan) > 10:
            suggestions.append("Plan is quite long; consider if some steps can be combined")
        
        return {
            "step_count": len(plan),
            "action_distribution": action_categories,
            "efficiency_score": 0.7 if suggestions else 0.9,  # Simple scoring
            "suggestions": suggestions
        }
    
    # Agent SDK tools for step execution agent
    @function_tool
    async def _resolve_step_parameters_tool(self, ctx: RunContextWrapper, goal_id: str, step_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves parameter placeholders for a step
        
        Args:
            goal_id: The goal ID
            step_parameters: The parameters to resolve
            
        Returns:
            Resolved parameters
        """
        resolved = await self._resolve_step_parameters(goal_id, step_parameters)
        
        # Check which parameters were successfully resolved
        resolution_status = {}
        for key, value in step_parameters.items():
            original = value
            resolved_value = resolved.get(key, None)
            
            if isinstance(original, str) and original.startswith("$step_"):
                resolution_status[key] = {
                    "original": original,
                    "resolved": resolved_value is not None,
                    "is_null": resolved_value is None
                }
        
        return {
            "resolved_parameters": resolved,
            "resolution_status": resolution_status,
            "all_resolved": all(status["resolved"] for status in resolution_status.values()) if resolution_status else True
        }
    
    @function_tool
    async def _execute_action(self, ctx: RunContextWrapper, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action with the given parameters
        
        Args:
            action: The action to execute
            parameters: The parameters for the action
            
        Returns:
            Execution result
        """
        if not self.brain:
            return {
                "success": False,
                "error": "NyxBrain reference not set in GoalManager"
            }
        
        try:
            action_method = getattr(self.brain, action, None)
            if not (action_method and callable(action_method)):
                return {
                    "success": False,
                    "error": f"Action '{action}' not found or not callable on NyxBrain"
                }
            
            # Execute the action
            start_time = datetime.datetime.now()
            result = await action_method(**parameters)
            end_time = datetime.datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "result": result,
                "duration": duration
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "exception_type": type(e).__name__
            }
    
    @function_tool
    async def _check_dominance_appropriateness(self, ctx: RunContextWrapper, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if a dominance-related action is appropriate
        
        Args:
            action: The action to check
            parameters: The parameters for the action
            
        Returns:
            Appropriateness check result
        """
        is_dominance_action = action in [
            "issue_command", "increase_control_intensity", "apply_consequence_simulated",
            "select_dominance_tactic", "trigger_dominance_gratification", "praise_submission"
        ]
        
        if not is_dominance_action:
            return {
                "is_dominance_action": False,
                "can_proceed": True,
                "action": "proceed"
            }
        
        user_id_param = parameters.get("user_id", parameters.get("target_user_id"))
        if not user_id_param:
            return {
                "is_dominance_action": True,
                "can_proceed": False,
                "action": "block",
                "reason": "Missing user_id for dominance action"
            }
        
        # If brain has dominance evaluation method, use it
        if self.brain and hasattr(self.brain, '_evaluate_dominance_step_appropriateness'):
            try:
                evaluation = await self.brain._evaluate_dominance_step_appropriateness(
                    action, parameters, user_id_param
                )
                return {
                    "is_dominance_action": True,
                    "evaluation_result": evaluation,
                    "can_proceed": evaluation.get("action") == "proceed",
                    "action": evaluation.get("action", "block"),
                    "reason": evaluation.get("reason")
                }
            except Exception as e:
                return {
                    "is_dominance_action": True,
                    "can_proceed": False,
                    "action": "block",
                    "reason": f"Evaluation error: {str(e)}"
                }
        
        # Default to blocking if no evaluation method
        return {
            "is_dominance_action": True,
            "can_proceed": False,
            "action": "block",
            "reason": "No dominance evaluation method available"
        }
    
    @function_tool
    async def _log_execution_result(self, ctx: RunContextWrapper, goal_id: str, step_id: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log the result of step execution
        
        Args:
            goal_id: The goal ID
            step_id: The step ID
            execution_result: The execution result
            
        Returns:
            Logging result
        """
        # Get the specific goal lock
        async with self._get_goal_lock(goal_id):
            if goal_id not in self.goals:
                return {
                    "success": False,
                    "error": f"Goal {goal_id} not found"
                }
            
            goal = self.goals[goal_id]
            step = None
            step_index = -1
            
            # Find the step by ID
            for i, s in enumerate(goal.plan):
                if s.step_id == step_id:
                    step = s
                    step_index = i
                    break
            
            if not step:
                return {
                    "success": False,
                    "error": f"Step {step_id} not found in goal {goal_id}"
                }
            
            # Update step with execution result
            step.status = "completed" if execution_result.get("success", False) else "failed"
            step.result = execution_result.get("result")
            step.error = execution_result.get("error")
            
            if not step.start_time:
                # If start time wasn't recorded earlier
                step.start_time = datetime.datetime.now() - datetime.timedelta(seconds=execution_result.get("duration", 0))
                
            step.end_time = datetime.datetime.now()
            
            # Add to execution history
            goal.execution_history.append({
                "step_id": step_id,
                "action": step.action,
                "status": step.status,
                "timestamp": step.end_time.isoformat(),
                "duration": execution_result.get("duration", 0),
                "error": step.error
            })
            
            # Mark goal as dirty for persistence
            await self.mark_goal_dirty(goal_id)
            
            return {
                "success": True,
                "step_index": step_index,
                "current_index": goal.current_step_index,
                "step_status": step.status
            }
    
    # Agent SDK tools for orchestration agent
    @function_tool
    async def _get_prioritized_goals(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get prioritized goals for execution (Agent Tool Wrapper).
        Calls the now asynchronous internal method.

        Returns:
            Prioritized goals
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
    @function_tool
    async def _update_goal_status_tool(self, ctx: RunContextWrapper, goal_id: str, status: str, 
                                    result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the status of a goal
        
        Args:
            goal_id: The goal ID
            status: The new status
            result: Optional result data
            error: Optional error message
            
        Returns:
            Status update result
        """
        if status not in ["pending", "active", "completed", "failed", "abandoned"]:
            return {
                "success": False,
                "error": f"Invalid status: {status}"
            }
        
        # Use the optimized update_goal_status method
        result = await self.update_goal_status(goal_id, status, result, error)
        
        return {
            "success": result.get("success", False),
            "goal_id": goal_id,
            "new_status": status,
            "old_status": result.get("old_status"),
            "notifications": result.get("notifications", {})
        }
    
    @function_tool
    async def _notify_systems(self, ctx: RunContextWrapper, goal_id: str, status: str, 
                           result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Notify relevant systems about goal status changes
        
        Args:
            goal_id: The goal ID
            status: The new status
            result: Optional result data
            error: Optional error message
            
        Returns:
            Notification results
        """
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            return {
                "success": False,
                "error": f"Goal {goal_id} not found"
            }
            
        notifications = {}
        
        # Notify NeedsSystem if applicable
        if goal.associated_need and self.brain and hasattr(self.brain, 'needs_system'):
            try:
                needs_system = getattr(self.brain, 'needs_system')
                if status == "completed" and hasattr(needs_system, 'satisfy_need'):
                    satisfaction_amount = goal.priority * 0.3 + 0.1  # Base + priority bonus
                    await needs_system.satisfy_need(goal.associated_need, satisfaction_amount)
                    notifications["needs_system"] = {
                        "success": True,
                        "need": goal.associated_need,
                        "amount": satisfaction_amount,
                        "action": "satisfy"
                    }
                elif status == "failed" and hasattr(needs_system, 'decrease_need'):
                    decrease_amount = goal.priority * 0.1  # Small decrease for failure
                    await needs_system.decrease_need(goal.associated_need, decrease_amount)
                    notifications["needs_system"] = {
                        "success": True,
                        "need": goal.associated_need,
                        "amount": decrease_amount,
                        "action": "decrease"
                    }
            except Exception as e:
                notifications["needs_system"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Notify RewardSystem if applicable
        if self.brain and hasattr(self.brain, 'reward_system'):
            try:
                reward_system = getattr(self.brain, 'reward_system')
                reward_value = 0.0
                if status == "completed": 
                    reward_value = goal.priority * 0.6  # Higher reward for completion
                elif status == "failed": 
                    reward_value = -goal.priority * 0.4  # Punish failure
                elif status == "abandoned": 
                    reward_value = -0.1  # Small punishment for abandoning

                if abs(reward_value) > 0.05 and hasattr(reward_system, 'process_reward_signal'):
                    # Import locally to avoid circular imports
                    from nyx.core.reward_system import RewardSignal
                    
                    reward_signal = RewardSignal(
                        value=reward_value, 
                        source="GoalManager",
                        context={
                            "goal_id": goal_id, 
                            "goal_description": goal.description, 
                            "outcome": status, 
                            "associated_need": goal.associated_need
                        },
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    await reward_system.process_reward_signal(reward_signal)
                    notifications["reward_system"] = {
                        "success": True,
                        "reward_value": reward_value,
                        "source": "GoalManager"
                    }
            except Exception as e:
                notifications["reward_system"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Notify MetaCore if applicable
        if self.brain and hasattr(self.brain, 'meta_core'):
            try:
                meta_core = getattr(self.brain, 'meta_core')
                if hasattr(meta_core, 'record_goal_outcome'):
                    await meta_core.record_goal_outcome(goal.model_dump())
                    notifications["meta_core"] = {
                        "success": True,
                        "recorded_goal": goal_id,
                        "status": status
                    }
            except Exception as e:
                notifications["meta_core"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Trigger integration callbacks
        callback_results = await self.trigger_integration_callbacks("goal_status_change", {
            "goal_id": goal_id,
            "old_status": goal.status,
            "new_status": status,
            "result": result,
            "error": error
        })
        
        if callback_results:
            notifications["integration_callbacks"] = callback_results
        
        # Notify external systems if goal has external system IDs
        if goal.external_system_ids:
            for system_name, external_id in goal.external_system_ids.items():
                if system_name in self._external_system_clients:
                    try:
                        # In a real system, this would call an external API
                        notifications[f"external_{system_name}"] = {
                            "success": True,
                            "external_id": external_id,
                            "status": status
                        }
                    except Exception as e:
                        notifications[f"external_{system_name}"] = {
                            "success": False,
                            "error": str(e)
                        }
        
        return {
            "success": True,
            "goal_id": goal_id,
            "status": status,
            "notifications": notifications
        }

    @function_tool
    async def _check_concurrency_limits(self, ctx: RunContextWrapper) -> Dict[str, Any]:
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
    @function_tool
    async def _get_available_actions(self, ctx: RunContextWrapper) -> Dict[str, Any]:
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
    
    @function_tool
    async def _get_action_description(self, ctx: RunContextWrapper, action: str) -> Dict[str, Any]:
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
    
    @function_tool
    async def _get_goal_details(self, ctx: RunContextWrapper, goal_id: str) -> Dict[str, Any]:
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
    
    @function_tool
    async def _get_recent_goals(self, ctx: RunContextWrapper, limit: int = 3) -> Dict[str, Any]:
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

    # Goal management methods
    async def derive_emotional_motivation(self, goal_description: str, need: Optional[str] = None) -> EmotionalMotivation:
        """
        Analyzes a goal description to derive likely emotional motivation,
        using the emotional core if available.
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
                # Simplified analysis - in a real implementation, you might use the emotional core's analysis tools
                
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
        """Creates a goal with emotional motivation and time horizon specifications"""
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
        await self._link_goal_to_emotional_system(goal_id, {
            "source": "goal_creation",
            "created_with_motivation": True
        })
        
        # If goal has associated need, integrate with needs system
        if goal_data.associated_need and self.brain and hasattr(self.brain, "needs_system"):
            await self.integrate_with_needs_system(
                goal_id, 
                goal_data.associated_need,
                goal_data.emotional_motivation.expected_satisfaction if goal_data.emotional_motivation else 0.3
            )
        
        return goal_id
    
    async def create_goal_hierarchy(self, root_goal_data: Dict[str, Any], subgoals_data: List[Dict[str, Any]]) -> str:
        """Creates a hierarchical structure of goals with a root goal and subgoals"""
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
    
    async def get_goal_hierarchy(self, root_goal_id: Optional[str] = None) -> List[GoalHierarchyNode]:
        """
        Retrieves the goal hierarchy as a tree structure.
        If root_goal_id is provided, returns that specific hierarchy.
        Otherwise, returns all top-level goals.
        """
        # If a specific root goal is requested
        if root_goal_id:
            root_goal = await self._get_goal_with_reader_lock(root_goal_id)
            if not root_goal:
                return []
            return [await self._build_goal_node(root_goal)]
            
        # Otherwise, find all top-level goals (goals without parents)
        top_level_goals = []
        
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
        
        # Find top-level goals
        for goal in all_goals:
            # Check if this goal has no parent
            if not hasattr(goal, 'relationships') or not goal.relationships or not goal.relationships.parent_goal_id:
                # This is a top-level goal
                top_level_goals.append(await self._build_goal_node(goal))
        
        return top_level_goals
    
    async def _build_goal_node(self, goal) -> GoalHierarchyNode:
        """Helper method to recursively build goal hierarchy nodes"""
        # Create the current node
        node = GoalHierarchyNode(
            goal_id=goal.id,
            description=goal.description,
            time_horizon=goal.time_horizon if hasattr(goal, 'time_horizon') else TimeHorizon.MEDIUM_TERM,
            status=goal.status,
            priority=goal.priority,
            children=[]
        )
        
        # Add child goals if any
        if hasattr(goal, 'relationships') and goal.relationships and goal.relationships.child_goal_ids:
            for child_id in goal.relationships.child_goal_ids:
                child_goal = await self._get_goal_with_reader_lock(child_id)
                if child_goal:
                    child_node = await self._build_goal_node(child_goal)
                    node.children.append(child_node)
        
        return node
    
    async def analyze_goal_motivations(self) -> GoalMotivationAnalysis:
        """Analyzes patterns in goal motivations across all goals"""
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
        
        # Initialize analysis data
        emotional_needs = {}
        chemical_associations = {}
        motivation_patterns = []
        
        # Scan all goals with emotional motivations
        for goal in all_goals:
            if hasattr(goal, 'emotional_motivation') and goal.emotional_motivation:
                # Count emotional needs
                primary_need = goal.emotional_motivation.primary_need
                emotional_needs[primary_need] = emotional_needs.get(primary_need, 0) + 1
                
                # Aggregate chemical associations
                for chemical, value in goal.emotional_motivation.associated_chemicals.items():
                    chemical_associations[chemical] = chemical_associations.get(chemical, 0.0) + value
                
                # Check for patterns based on time horizon
                time_horizon = goal.time_horizon if hasattr(goal, 'time_horizon') else TimeHorizon.MEDIUM_TERM
                
                # Add to pattern analysis
                motivation_patterns.append({
                    "need": primary_need,
                    "time_horizon": time_horizon,
                    "status": goal.status,
                    "intensity": goal.emotional_motivation.intensity,
                    "goal_id": goal.id
                })
        
        # Calculate primary motivations (top 3)
        primary_motivations = sorted(emotional_needs.keys(), 
                                    key=lambda x: emotional_needs[x], 
                                    reverse=True)[:3]
        
        # Normalize chemical associations
        for chemical in chemical_associations:
            chemical_associations[chemical] /= max(1, len(motivation_patterns))
        
        return GoalMotivationAnalysis(
            emotional_needs=emotional_needs,
            primary_motivations=primary_motivations,
            chemical_associations=chemical_associations,
            motivation_patterns=motivation_patterns
        )
    
    async def suggest_new_goals(self, based_on_need: Optional[str] = None, 
                             time_horizon: Optional[TimeHorizon] = None) -> List[Dict[str, Any]]:
        """
        Suggests new goals based on analysis of existing goals, emotional needs,
        and system state. Can focus on specific needs or time horizons.
        """
        # This method would use the GoalManager's AI planning agent to generate suggestions
        # based on analysis of existing goals and emotional needs
        
        # First, analyze current goal motivations
        motivation_analysis = await self.analyze_goal_motivations()
        
        # Get data about completed goals for learning patterns
        completed_goals = await self.get_all_goals(status_filter=["completed"])
        
        # Prepare context for the planning agent
        context = {
            "motivation_analysis": motivation_analysis.model_dump(),
            "completed_goals": completed_goals,
            "based_on_need": based_on_need,
            "time_horizon": time_horizon.value if time_horizon else None,
            "current_goals_count": len(self.goals)
        }
        
        # Use the planning agent to generate suggestions
        try:
            result = await Runner.run(
                self.planning_agent,
                json.dumps(context),
                context=RunContext(goal_id="goal_suggestion", brain_available=self.brain is not None),
                run_config={
                    "workflow_name": "GoalSuggestion",
                    "trace_metadata": {
                        "based_on_need": based_on_need,
                        "time_horizon": time_horizon.value if time_horizon else None
                    }
                }
            )
            
            # Extract suggestions from agent output
            suggested_goals = result.final_output.get("suggested_goals", [])
            
            return suggested_goals
        except Exception as e:
            logger.error(f"Error generating goal suggestions: {e}")
            return []
    
    async def add_goal(self, description: str, priority: float = 0.5, source: str = "unknown",
                     associated_need: Optional[str] = None, emotional_motivation: Optional[EmotionalMotivation] = None,
                     plan: Optional[List[Dict]] = None, user_id: Optional[str] = None, 
                     deadline: Optional[datetime.datetime] = None,
                     time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> str:
        """Adds a new goal, optionally generating a plan if none is provided."""
        if not description:
            raise ValueError("Goal description cannot be empty.")

        # Create the goal object with improved model
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
        """Generates and assigns a plan for a goal using the planning agent."""
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
        """Selects the highest priority goal to work on, respecting concurrency limits."""
        # Get prioritized goals
        prioritized = self.get_prioritized_goals()
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
        """Selects and executes the next step of the highest priority active goal."""
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
        """Helper method to update goal after step execution - used with writer lock"""
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
        """Resolves parameter placeholders like '$step_1.result' within the context of a specific goal."""
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

    async def update_goal_status(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Updates the status of a goal and notifies relevant systems."""
        # First get the goal with reader lock to check if it exists and get old status
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            logger.warning(f"Attempted to update status for unknown goal: {goal_id}")
            return {"success": False, "error": "Goal not found"}

        old_status = goal.status
        if old_status == status: 
            return {"success": True, "old_status": old_status, "new_status": status, "unchanged": True}  # No change

        # Update the goal with writer lock
        await self._update_goal_with_writer_lock(
            goal_id,
            lambda g: (
                setattr(g, "status", status),
                setattr(g, "last_error", error),
                setattr(g, "completion_time", datetime.datetime.now()) if status in ["completed", "failed", "abandoned"] else None
            )
        )
        
        # Update active goals set if needed
        if status in ["completed", "failed", "abandoned"]:
            async with self._active_goals_lock:
                self.active_goals.discard(goal_id)
            
            # Update statistics
            if status == "completed":
                self.goal_statistics["completed"] += 1
            elif status == "failed":
                self.goal_statistics["failed"] += 1
            elif status == "abandoned":
                self.goal_statistics["abandoned"] += 1

        logger.info(f"Goal '{goal_id}' status changed from {old_status} to {status}.")

        # Notify systems 
        try:
            notifications = await self._notify_systems(
                RunContextWrapper(context=RunContext(goal_id=goal_id)), 
                goal_id=goal_id, 
                status=status, 
                result=result, 
                error=error
            )
        except Exception as e:
            logger.error(f"Error in notifying systems about goal status change: {e}")
            notifications = {"error": str(e)}

        return {
            "success": True,
            "goal_id": goal_id,
            "old_status": old_status,
            "new_status": status,
            "notifications": notifications
        }

    async def update_goal_status_with_hierarchy(self, goal_id: str, status: str, result: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Updates goal status with hierarchy considerations - handles parent/child relationships"""
        # First get the goal with reader lock
        goal = await self._get_goal_with_reader_lock(goal_id)
        if not goal:
            logger.warning(f"Attempted to update status for unknown goal: {goal_id}")
            return {"success": False, "error": "Goal not found"}
    
        old_status = goal.status
        if old_status == status: 
            return {"success": True, "unchanged": True}  # No change
    
        # Update the goal's status with writer lock
        await self._update_goal_with_writer_lock(
            goal_id,
            lambda g: (
                setattr(g, "status", status),
                setattr(g, "last_error", error),
                setattr(g, "completion_time", datetime.datetime.now()) if status in ["completed", "failed", "abandoned"] else None
            )
        )
    
        # Update active goals set if needed
        if status in ["completed", "failed", "abandoned"]:
            async with self._active_goals_lock:
                self.active_goals.discard(goal_id)
            
            # Update statistics
            if status == "completed":
                self.goal_statistics["completed"] += 1
                # Process completion reward
                await self._process_goal_completion_reward(goal_id, result)
                
                # Handle parent/child relationships
                if goal.relationships and goal.relationships.parent_goal_id:
                    parent_id = goal.relationships.parent_goal_id
                    parent_goal = await self._get_goal_with_reader_lock(parent_id)
                    
                    if parent_goal:
                        # Update parent progress based on children completion
                        if parent_goal.relationships and parent_goal.relationships.child_goal_ids:
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
                                
                                # If all children completed, mark parent as completed
                                if completed_children == total_children:
                                    await self.update_goal_status(parent_id, "completed", result="All subgoals completed")
                
            elif status == "failed":
                self.goal_statistics["failed"] += 1
            elif status == "abandoned":
                self.goal_statistics["abandoned"] += 1
    
        logger.info(f"Goal '{goal_id}' status changed from {old_status} to {status}.")
    
        # Notify systems
        try:
            notifications = await self._notify_systems(
                RunContextWrapper(context=RunContext(goal_id=goal_id)), 
                goal_id=goal_id, 
                status=status, 
                result=result, 
                error=error
            )
        except Exception as e:
            logger.error(f"Error in notifying systems about goal status change: {e}")
            notifications = {"error": str(e)}
        
        return {
            "success": True, 
            "goal_id": goal_id, 
            "old_status": old_status, 
            "new_status": status,
            "notifications": notifications
        }
    
    async def abandon_goal(self, goal_id: str, reason: str) -> Dict[str, Any]:
        """Abandons an active or pending goal."""
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
        """Checks if there's an active goal associated with a specific need."""
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
        """Gets the status and plan of a specific goal."""
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
        """Gets all goals, optionally filtered by status."""
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
        """Gets statistics about goal execution."""
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
        
        # Enhanced statistics with new metrics
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
