# nyx/core/integration/action_selector.py

import logging
import asyncio
import datetime
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict
from enum import Enum

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context

logger = logging.getLogger(__name__)

class ActionPriority(Enum):
    """Priority levels for actions."""
    CRITICAL = 5   # Safety-critical actions
    HIGH = 4       # Urgent user-requested actions
    MEDIUM = 3     # Standard actions
    LOW = 2        # Background or optimization actions
    BACKGROUND = 1 # Non-essential actions

class ActionStatus(Enum):
    """Status of an action in the system."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

class ActionRequest:
    """Represents a request to perform an action."""
    def __init__(self, 
                action_type: str, 
                source_module: str, 
                parameters: Dict[str, Any], 
                priority: ActionPriority = ActionPriority.MEDIUM,
                deadline: Optional[datetime.datetime] = None):
        self.id = f"action_{uuid.uuid4().hex[:8]}"
        self.action_type = action_type
        self.source_module = source_module
        self.parameters = parameters
        self.priority = priority
        self.deadline = deadline
        self.created_at = datetime.datetime.now()
        self.status = ActionStatus.PENDING
        self.result = None
        self.error = None
        self.executed_at = None
        self.completed_at = None
    
    @property
    def is_expired(self) -> bool:
        """Check if the action request has expired."""
        if self.deadline:
            return datetime.datetime.now() > self.deadline
        return False
    
    @property
    def age(self) -> float:
        """Get the age of the action request in seconds."""
        return (datetime.datetime.now() - self.created_at).total_seconds()
    
    def __str__(self):
        return f"ActionRequest(id={self.id}, type={self.action_type}, source={self.source_module}, status={self.status.value})"

class ActionExecutionContext:
    """Context for action execution."""
    def __init__(self, action_request: ActionRequest, system_context: Any):
        self.action_request = action_request
        self.system_context = system_context
        self.execution_start = datetime.datetime.now()
        self.execution_data = {}  # Additional data during execution
    
    def set_data(self, key: str, value: Any) -> None:
        """Set context data."""
        self.execution_data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get context data."""
        return self.execution_data.get(key, default)
    
    @property
    def execution_time(self) -> float:
        """Get the execution time so far in seconds."""
        return (datetime.datetime.now() - self.execution_start).total_seconds()

class ActionConflictInfo:
    """Information about a conflict between actions."""
    def __init__(self, action1: ActionRequest, action2: ActionRequest, conflict_type: str, resolution: str):
        self.action1 = action1
        self.action2 = action2
        self.conflict_type = conflict_type
        self.resolution = resolution
        self.timestamp = datetime.datetime.now()

class ActionSelector:
    """
    Unified action selection mechanism for Nyx.
    
    Provides centralized action selection, prioritization, and execution
    to coordinate across modules.
    """
    def __init__(self, nyx_brain):
        self.brain = nyx_brain
        self.action_queue = []  # List of ActionRequest objects
        self.executing_actions = {}  # id -> ActionRequest
        self.completed_actions = []  # List of completed ActionRequest objects
        self.action_history = []  # Combined history of all actions
        self.max_history = 100
        self._lock = asyncio.Lock()  # For thread safety
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        
        # Module-specific executors
        self.action_executors = {}  # action_type -> executor_function
        
        # Conflict detection
        self.conflict_rules = []  # List of conflict detection functions
        self.conflict_history = []  # List of ActionConflictInfo objects
        
        # Concurrency control
        self.max_concurrent_actions = 3
        self.max_queue_size = 50
        
        # Statistics
        self.stats = {
            "actions_queued": 0,
            "actions_executed": 0,
            "actions_completed": 0,
            "actions_failed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0
        }
        
        logger.info("ActionSelector initialized")
    
    def register_executor(self, action_type: str, executor: Callable) -> None:
        """
        Register an executor function for an action type.
        
        Args:
            action_type: Type of action to register for
            executor: Async function that executes the action
        """
        self.action_executors[action_type] = executor
        logger.debug(f"Registered executor for action type: {action_type}")
    
    def register_conflict_rule(self, rule_function: Callable) -> None:
        """
        Register a conflict detection rule.
        
        Args:
            rule_function: Function that detects conflicts between actions
        """
        self.conflict_rules.append(rule_function)
        logger.debug(f"Registered conflict rule: {rule_function.__name__}")
    
    async def queue_action(self, action_request: ActionRequest) -> str:
        """
        Queue an action for execution.
        
        Args:
            action_request: Action request to queue
            
        Returns:
            Action ID
        """
        async with self._lock:
            # Check if queue is full
            if len(self.action_queue) >= self.max_queue_size:
                # Remove lowest priority action if necessary
                self.action_queue.sort(key=lambda a: (a.priority.value, -a.age))
                if action_request.priority.value > self.action_queue[0].priority.value:
                    removed = self.action_queue.pop(0)
                    removed.status = ActionStatus.CANCELLED
                    self.action_history.append(removed)
                    logger.warning(f"Removed low priority action {removed.id} from full queue to make room")
                else:
                    logger.warning(f"Action queue full, rejecting new action: {action_request}")
                    return None
            
            # Add to queue
            self.action_queue.append(action_request)
            self.stats["actions_queued"] += 1
            
            # Sort queue by priority (higher first) and then age (older first)
            self.action_queue.sort(key=lambda a: (-a.priority.value, a.age))
            
            logger.info(f"Queued action: {action_request}")
            
            # Start execution loop if not already running
            asyncio.create_task(self._process_queue())
            
            return action_request.id
    
    async def cancel_action(self, action_id: str) -> bool:
        """
        Cancel a pending or executing action.
        
        Args:
            action_id: ID of action to cancel
            
        Returns:
            True if cancelled, False otherwise
        """
        async with self._lock:
            # Check queue
            for i, action in enumerate(self.action_queue):
                if action.id == action_id:
                    action.status = ActionStatus.CANCELLED
                    self.action_history.append(action)
                    self.action_queue.pop(i)
                    logger.info(f"Cancelled queued action {action_id}")
                    return True
            
            # Check executing
            if action_id in self.executing_actions:
                action = self.executing_actions[action_id]
                action.status = ActionStatus.CANCELLED
                self.action_history.append(action)
                del self.executing_actions[action_id]
                logger.info(f"Cancelled executing action {action_id}")
                return True
            
            return False
    
    async def get_action_status(self, action_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an action.
        
        Args:
            action_id: Action ID
            
        Returns:
            Action status or None if not found
        """
        async with self._lock:
            # Check queue
            for action in self.action_queue:
                if action.id == action_id:
                    return {
                        "id": action.id,
                        "action_type": action.action_type,
                        "source_module": action.source_module,
                        "status": action.status.value,
                        "priority": action.priority.value,
                        "queued_at": action.created_at.isoformat(),
                        "queue_position": self.action_queue.index(action) + 1
                    }
            
            # Check executing
            if action_id in self.executing_actions:
                action = self.executing_actions[action_id]
                return {
                    "id": action.id,
                    "action_type": action.action_type,
                    "source_module": action.source_module,
                    "status": action.status.value,
                    "priority": action.priority.value,
                    "queued_at": action.created_at.isoformat(),
                    "executing_since": action.executed_at.isoformat() if action.executed_at else None
                }
            
            # Check completed
            for action in self.completed_actions:
                if action.id == action_id:
                    return {
                        "id": action.id,
                        "action_type": action.action_type,
                        "source_module": action.source_module,
                        "status": action.status.value,
                        "priority": action.priority.value,
                        "queued_at": action.created_at.isoformat(),
                        "executed_at": action.executed_at.isoformat() if action.executed_at else None,
                        "completed_at": action.completed_at.isoformat() if action.completed_at else None,
                        "result": action.result,
                        "error": action.error
                    }
            
            # Check history
            for action in self.action_history:
                if action.id == action_id:
                    return {
                        "id": action.id,
                        "action_type": action.action_type,
                        "source_module": action.source_module,
                        "status": action.status.value,
                        "priority": action.priority.value,
                        "queued_at": action.created_at.isoformat(),
                        "executed_at": action.executed_at.isoformat() if action.executed_at else None,
                        "completed_at": action.completed_at.isoformat() if action.completed_at else None,
                        "result": action.result,
                        "error": action.error
                    }
            
            return None
    
    async def _process_queue(self) -> None:
        """Process the action queue in priority order."""
        async with self._lock:
            # Clean up expired actions
            expired = [a for a in self.action_queue if a.is_expired]
            for action in expired:
                action.status = ActionStatus.CANCELLED
                action.error = "Deadline expired"
                self.action_history.append(action)
                self.action_queue.remove(action)
                logger.info(f"Removed expired action {action.id}")
            
            # Check if we can execute more actions
            if len(self.executing_actions) >= self.max_concurrent_actions:
                return
            
            # Get candidates for execution
            candidates = self.action_queue.copy()
            
            # Check for conflicts and block conflicting actions
            blocked_actions = set()
            for candidate in candidates:
                # Check against currently executing actions
                for _, executing in self.executing_actions.items():
                    conflict = self._check_conflicts(candidate, executing)
                    if conflict:
                        blocked_actions.add(candidate.id)
                        logger.info(f"Action {candidate.id} blocked due to conflict with executing action {executing.id}: {conflict.conflict_type}")
                        self.conflict_history.append(conflict)
                        self.stats["conflicts_detected"] += 1
            
            # Filter out blocked actions
            candidates = [c for c in candidates if c.id not in blocked_actions]
            
            # Sort by priority and age
            candidates.sort(key=lambda a: (-a.priority.value, a.age))
            
            # Start execution for non-blocked, highest priority actions
            available_slots = self.max_concurrent_actions - len(self.executing_actions)
            for i in range(min(available_slots, len(candidates))):
                action = candidates[i]
                self.action_queue.remove(action)
                action.status = ActionStatus.EXECUTING
                action.executed_at = datetime.datetime.now()
                self.executing_actions[action.id] = action
                logger.info(f"Starting execution of action {action.id}")
                
                # Start execution in background
                asyncio.create_task(self._execute_action(action))
        
        # When lock is released, scheduled actions will execute
    
    def _check_conflicts(self, action1: ActionRequest, action2: ActionRequest) -> Optional[ActionConflictInfo]:
        """
        Check for conflicts between two actions.
        
        Args:
            action1: First action
            action2: Second action
            
        Returns:
            Conflict info if conflict found, None otherwise
        """
        # Skip if same action
        if action1.id == action2.id:
            return None
        
        # Predefined conflict rules
        
        # Rule 1: Same action type with same target conflicting parameters (e.g., two dominance actions on same user)
        if action1.action_type == action2.action_type:
            # Check for dominance actions
            if "dominance" in action1.action_type.lower():
                # Check if same target user
                user1 = action1.parameters.get("user_id", action1.parameters.get("target_user_id"))
                user2 = action2.parameters.get("user_id", action2.parameters.get("target_user_id"))
                if user1 and user2 and user1 == user2:
                    # Resolve by keeping higher priority or newer
                    if action1.priority.value > action2.priority.value:
                        resolution = f"Prioritized {action1.id} (higher priority)"
                    elif action2.priority.value > action1.priority.value:
                        resolution = f"Prioritized {action2.id} (higher priority)"
                    elif action1.age < action2.age:
                        resolution = f"Prioritized {action1.id} (newer)"
                    else:
                        resolution = f"Prioritized {action2.id} (newer)"
                    
                    return ActionConflictInfo(
                        action1=action1,
                        action2=action2,
                        conflict_type="same_target_dominance",
                        resolution=resolution
                    )
        
        # Rule 2: Emotional expression conflict (e.g., expressing joy while attempting dominance)
        if (action1.action_type.lower() in ["express_emotion", "update_emotion"] and 
            action2.action_type.lower() in ["issue_command", "increase_control_intensity"]):
            
            # Get emotion
            emotion = action1.parameters.get("emotion", "").lower()
            
            # Check if conflicting emotion for dominance
            if emotion in ["joy", "playful", "excited"] and "dominance" in action2.action_type.lower():
                resolution = f"Block {action1.id} as emotion conflicts with dominance action"
                return ActionConflictInfo(
                    action1=action1,
                    action2=action2,
                    conflict_type="emotion_dominance_conflict",
                    resolution=resolution
                )
        
        # Rule 3: Motor control conflicts (e.g., simulate_physical_touch and express_attraction simultaneously)
        if ("physical" in action1.action_type.lower() and "express" in action2.action_type.lower()) or \
           ("physical" in action2.action_type.lower() and "express" in action1.action_type.lower()):
            if action1.priority.value > action2.priority.value:
                resolution = f"Prioritized {action1.id} (higher priority)"
            else:
                resolution = f"Prioritized {action2.id} (higher priority)"
            
            return ActionConflictInfo(
                action1=action1,
                action2=action2,
                conflict_type="motor_control_conflict",
                resolution=resolution
            )
        
        # Apply custom rules
        for rule in self.conflict_rules:
            conflict = rule(action1, action2)
            if conflict:
                self.stats["conflicts_detected"] += 1
                return conflict
        
        return None
    
    async def _execute_action(self, action: ActionRequest) -> None:
        """
        Execute an action.
        
        Args:
            action: Action to execute
        """
        try:
            # Skip if already cancelled
            if action.status == ActionStatus.CANCELLED:
                return
            
            # Choose executor
            executor = self.action_executors.get(action.action_type)
            if not executor:
                # Try to find method on brain
                if hasattr(self.brain, action.action_type) and callable(getattr(self.brain, action.action_type)):
                    executor = getattr(self.brain, action.action_type)
                else:
                    raise ValueError(f"No executor found for action type: {action.action_type}")
            
            # Create execution context
            context = ActionExecutionContext(action, self.system_context)
            
            logger.debug(f"Executing action {action.id} ({action.action_type})")
            
            # Execute action
            result = await executor(**action.parameters)
            
            # Update action status
            async with self._lock:
                if action.id in self.executing_actions:
                    action = self.executing_actions[action.id]
                    action.status = ActionStatus.COMPLETED
                    action.result = result
                    action.completed_at = datetime.datetime.now()
                    
                    # Move to completed
                    del self.executing_actions[action.id]
                    self.completed_actions.append(action)
                    self.action_history.append(action)
                    
                    # Trim history if needed
                    if len(self.completed_actions) > self.max_history:
                        self.completed_actions = self.completed_actions[-self.max_history:]
                    if len(self.action_history) > self.max_history:
                        self.action_history = self.action_history[-self.max_history:]
                    
                    self.stats["actions_executed"] += 1
                    self.stats["actions_completed"] += 1
                    
                    logger.info(f"Action {action.id} completed successfully")
                    
                    # Record in system context
                    await self.system_context.record_action(
                        action_type=action.action_type,
                        params=action.parameters,
                        result=result
                    )
                    
                    # Process queue for next action
                    asyncio.create_task(self._process_queue())
            
        except Exception as e:
            logger.error(f"Error executing action {action.id}: {e}")
            
            # Update action status
            async with self._lock:
                if action.id in self.executing_actions:
                    action = self.executing_actions[action.id]
                    action.status = ActionStatus.FAILED
                    action.error = str(e)
                    action.completed_at = datetime.datetime.now()
                    
                    # Move to completed
                    del self.executing_actions[action.id]
                    self.completed_actions.append(action)
                    self.action_history.append(action)
                    
                    self.stats["actions_executed"] += 1
                    self.stats["actions_failed"] += 1
                    
                    # Process queue for next action
                    asyncio.create_task(self._process_queue())
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the action queue.
        
        Returns:
            Queue status information
        """
        async with self._lock:
            return {
                "queued_actions": len(self.action_queue),
                "executing_actions": len(self.executing_actions),
                "completed_actions": len(self.completed_actions),
                "next_actions": [
                    {
                        "id": a.id,
                        "type": a.action_type,
                        "priority": a.priority.value,
                        "age": a.age
                    }
                    for a in self.action_queue[:3]
                ],
                "current_executing": [
                    {
                        "id": a.id,
                        "type": a.action_type,
                        "priority": a.priority.value,
                        "executing_for": (datetime.datetime.now() - a.executed_at).total_seconds() if a.executed_at else 0
                    }
                    for a in self.executing_actions.values()
                ],
                "stats": self.stats,
                "conflicts": len(self.conflict_history)
            }
    
    async def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the action execution history.
        
        Args:
            limit: Maximum number of actions to return
            
        Returns:
            List of action history entries
        """
        async with self._lock:
            return [
                {
                    "id": a.id,
                    "type": a.action_type,
                    "source": a.source_module,
                    "status": a.status.value,
                    "created_at": a.created_at.isoformat(),
                    "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                    "result": a.result if not isinstance(a.result, dict) or len(str(a.result)) < 100 else "< result data >",
                    "error": a.error
                }
                for a in list(reversed(self.action_history))[:limit]
            ]

    async def create_action(self, action_type: str, source_module: str, parameters: Dict[str, Any],
                          priority: ActionPriority = ActionPriority.MEDIUM) -> str:
        """
        Create and queue an action.
        
        Args:
            action_type: Type of action
            source_module: Module requesting the action
            parameters: Action parameters
            priority: Action priority
            
        Returns:
            Action ID
        """
        action = ActionRequest(
            action_type=action_type,
            source_module=source_module,
            parameters=parameters,
            priority=priority
        )
        
        return await self.queue_action(action)

# Create selector function
def create_action_selector(nyx_brain):
    """Create an action selector for the given brain."""
    return ActionSelector(nyx_brain)
