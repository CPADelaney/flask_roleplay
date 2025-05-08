# nyx/core/integration/action_selector.py

import logging
import asyncio
import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict
from enum import Enum

# --- Imports from your project structure ---
# These are now confirmed to be global getters
from nyx.core.integration.event_bus import Event, get_event_bus, EventBus
from nyx.core.integration.system_context import get_system_context, SystemContext



logger = logging.getLogger(__name__)

# --- Enums and Data Classes (ActionPriority, ActionStatus, ActionRequest, ActionExecutionContext, ActionConflictInfo) ---
# (Keep these exactly as you provided them in action_selector.py)
class ActionPriority(Enum):
    CRITICAL = 5; HIGH = 4; MEDIUM = 3; LOW = 2; BACKGROUND = 1

class ActionStatus(Enum):
    PENDING = "pending"; EXECUTING = "executing"; COMPLETED = "completed"
    FAILED = "failed"; CANCELLED = "cancelled"; BLOCKED = "blocked"

class ActionRequest:
    def __init__(self, action_type: str, source_module: str, parameters: Dict[str, Any], priority: ActionPriority = ActionPriority.MEDIUM, deadline: Optional[datetime.datetime] = None):
        self.id = f"action_{uuid.uuid4().hex[:8]}"
        self.action_type = action_type; self.source_module = source_module; self.parameters = parameters
        self.priority = priority; self.deadline = deadline; self.created_at = datetime.datetime.now()
        self.status = ActionStatus.PENDING; self.result = None; self.error = None
        self.executed_at = None; self.completed_at = None
    @property
    def is_expired(self) -> bool: return self.deadline and datetime.datetime.now() > self.deadline
    @property
    def age(self) -> float: return (datetime.datetime.now() - self.created_at).total_seconds()
    def __str__(self): return f"ActionRequest(id={self.id}, type={self.action_type}, source={self.source_module}, status={self.status.value})"

class ActionExecutionContext:
     def __init__(self, action_request: ActionRequest, system_context: SystemContext): # Added type hint
        self.action_request = action_request; self.system_context = system_context
        self.execution_start = datetime.datetime.now(); self.execution_data: Dict[str, Any] = {}
     def set_data(self, key: str, value: Any) -> None: self.execution_data[key] = value
     def get_data(self, key: str, default: Any = None) -> Any: return self.execution_data.get(key, default)
     @property
     def execution_time(self) -> float: return (datetime.datetime.now() - self.execution_start).total_seconds()

class ActionConflictInfo:
    def __init__(self, action1: ActionRequest, action2: ActionRequest, conflict_type: str, resolution: str):
        self.action1 = action1; self.action2 = action2; self.conflict_type = conflict_type
        self.resolution = resolution; self.timestamp = datetime.datetime.now()

# --- ActionSelector Class ---
class ActionSelector:
    """
    Unified action selection mechanism for Nyx. This is the 'action_selector' bridge.
    """
    def __init__(self, nyx_brain: Any):
        self.brain = nyx_brain # Keep reference if needed for executors, etc.
        self.action_queue: List[ActionRequest] = []
        self.executing_actions: Dict[str, ActionRequest] = {}
        self.completed_actions: List[ActionRequest] = []
        self.action_history: List[ActionRequest] = []
        self.max_history: int = 100
        self._lock = asyncio.Lock()

        # Dependencies acquired during initialize()
        self.event_bus: Optional[EventBus] = None
        self.system_context: Optional[SystemContext] = None # Use the imported type

        self.action_executors: Dict[str, Callable] = {}
        self.conflict_rules: List[Callable] = []
        self.conflict_history: List[ActionConflictInfo] = []

        self.max_concurrent_actions: int = 3
        self.max_queue_size: int = 50

        self.stats: Dict[str, int] = defaultdict(int)

        logger.info(f"ActionSelector instance created (pre-initialization).")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ THIS IS THE METHOD NEEDED TO FIX WARNING #4                        +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    async def initialize(self) -> bool:
        """
        Asynchronously initialize the ActionSelector bridge component.
        Called by the IntegrationManager. Acquires core dependencies.
        """
        logger.info("Initializing ActionSelector bridge...")
        try:
            # Get core dependencies using the global getters
            self.event_bus = get_event_bus()
            self.system_context = get_system_context()

            if not self.event_bus:
                logger.error("ActionSelector initialization failed: EventBus not available via get_event_bus().")
                return False
            if not self.system_context:
                 logger.error("ActionSelector initialization failed: SystemContext not available via get_system_context().")
                 return False

            # --- Add any other essential async setup for ActionSelector here ---
            # (If ActionSelector needs to load rules, subscribe async, etc.)
            # Example: await self.load_dynamic_conflict_rules()
            # --- End of other essential setup example ---

            logger.info("ActionSelector bridge initialized successfully.")
            return True # Signal success

        except Exception as e:
            logger.error(f"Critical error during ActionSelector initialization: {e}", exc_info=True)
            return False # Signal failure
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ END OF initialize() METHOD                                         +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # --- Other ActionSelector methods (keep your existing implementations) ---
    # (register_executor, register_conflict_rule, queue_action, cancel_action,
    #  get_action_status, _process_queue, _check_conflicts, _execute_action,
    #  get_queue_status, get_action_history, create_action)
    # Ensure async methods use `await` correctly internally.

    def register_executor(self, action_type: str, executor: Callable) -> None:
        self.action_executors[action_type] = executor
        logger.debug(f"Registered executor for action type: {action_type}")

    def register_conflict_rule(self, rule_function: Callable) -> None:
        self.conflict_rules.append(rule_function)
        logger.debug(f"Registered conflict rule: {rule_function.__name__}")

    async def queue_action(self, action_request: ActionRequest) -> Optional[str]:
        async with self._lock:
            if len(self.action_queue) >= self.max_queue_size:
                self.action_queue.sort(key=lambda a: (a.priority.value, -a.age))
                if action_request.priority.value > self.action_queue[0].priority.value:
                    removed = self.action_queue.pop(0)
                    removed.status = ActionStatus.CANCELLED
                    self.action_history.append(removed)
                    logger.warning(f"Queue full. Removed {removed.id} to make room for {action_request.id}.")
                else:
                    logger.warning(f"Queue full. Rejecting {action_request.id}.")
                    return None
            self.action_queue.append(action_request)
            self.stats["actions_queued"] += 1
            self.action_queue.sort(key=lambda a: (-a.priority.value, a.age))
            logger.info(f"Queued: {action_request}")
            asyncio.create_task(self._process_queue())
            return action_request.id

    async def cancel_action(self, action_id: str) -> bool:
        async with self._lock:
            action_in_queue = next((a for a in self.action_queue if a.id == action_id), None)
            if action_in_queue:
                action_in_queue.status = ActionStatus.CANCELLED
                self.action_history.append(action_in_queue)
                self.action_queue.remove(action_in_queue)
                logger.info(f"Cancelled queued action {action_id}")
                return True
            if action_id in self.executing_actions:
                action = self.executing_actions.pop(action_id)
                action.status = ActionStatus.CANCELLED
                self.action_history.append(action)
                logger.info(f"Marked executing action {action_id} as cancelled.")
                return True
            logger.debug(f"Action {action_id} not found to cancel.")
            return False

    async def get_action_status(self, action_id: str) -> Optional[Dict[str, Any]]:
         async with self._lock:
             if action_id in self.executing_actions:
                 action = self.executing_actions[action_id]
                 return {"id": action.id, "type": action.action_type, "status": action.status.value, "priority": action.priority.value, "executing_since": action.executed_at.isoformat() if action.executed_at else None}
             action_in_queue = next(((a, i) for i, a in enumerate(self.action_queue) if a.id == action_id), None)
             if action_in_queue:
                 action, idx = action_in_queue
                 return {"id": action.id, "type": action.action_type, "status": action.status.value, "priority": action.priority.value, "queue_pos": idx + 1, "queued_at": action.created_at.isoformat()}
             for action in reversed(self.action_history):
                  if action.id == action_id:
                     return {"id": action.id, "type": action.action_type, "status": action.status.value, "priority": action.priority.value, "completed_at": action.completed_at.isoformat() if action.completed_at else None, "error": action.error}
             return None

    async def _process_queue(self) -> None:
        async with self._lock:
            if not self.action_queue: return
            now = datetime.datetime.now()
            expired_indices = [i for i, a in enumerate(self.action_queue) if a.deadline and now > a.deadline]
            for i in sorted(expired_indices, reverse=True):
                action = self.action_queue.pop(i); action.status = ActionStatus.CANCELLED; action.error = "Deadline expired"
                self.action_history.append(action); logger.info(f"Removed expired {action.id}")
            if len(self.executing_actions) >= self.max_concurrent_actions: return

            actions_to_start = []
            available_slots = self.max_concurrent_actions - len(self.executing_actions)
            candidates = self.action_queue.copy()
            for candidate in candidates:
                if available_slots <= 0: break
                is_conflicted = any(self._check_conflicts(candidate, executing) for executing in self.executing_actions.values()) or \
                                any(self._check_conflicts(candidate, selected) for selected in actions_to_start)
                if not is_conflicted:
                    actions_to_start.append(candidate); available_slots -= 1
                else: logger.debug(f"Conflict prevents starting {candidate.id}")

            for action in actions_to_start:
                try:
                    self.action_queue.remove(action)
                    action.status = ActionStatus.EXECUTING; action.executed_at = datetime.datetime.now()
                    self.executing_actions[action.id] = action
                    logger.info(f"Starting execution: {action}")
                    asyncio.create_task(self._execute_action(action))
                except ValueError: logger.warning(f"Action {action.id} gone before start.")
                except Exception as start_err: logger.error(f"Error starting {action.id}: {start_err}", exc_info=True)

    def _check_conflicts(self, action1: ActionRequest, action2: ActionRequest) -> Optional[ActionConflictInfo]:
         # --- Your synchronous conflict logic ---
         if action1.id == action2.id: return None
         # Example rule:
         if action1.action_type == action2.action_type and "move_to" in action1.action_type:
             return ActionConflictInfo(action1, action2, "simultaneous_move", "Prioritize higher priority")
         # --- Add your other rules ---
         for rule in self.conflict_rules:
             if conflict := rule(action1, action2):
                 self.stats["conflicts_detected"] += 1; return conflict
         return None

    async def _execute_action(self, action: ActionRequest) -> None:
        result_data, error_data = None, None
        try:
            if action.status == ActionStatus.CANCELLED: return
            executor = self.action_executors.get(action.action_type) or getattr(self.brain, action.action_type, None)
            if not callable(executor): raise ValueError(f"No executor/method for {action.action_type}")

            if asyncio.iscoroutinefunction(executor):
                result_data = await executor(**action.parameters)
            else: # Handle sync executor (consider run_in_executor for blocking calls)
                result_data = executor(**action.parameters)
        except Exception as e:
            logger.error(f"Execution error action {action.id}: {e}", exc_info=True)
            error_data = str(e)
        finally:
            async with self._lock:
                if action.id in self.executing_actions: # Check if still considered executing
                    executed_action = self.executing_actions.pop(action.id)
                    executed_action.completed_at = datetime.datetime.now()
                    executed_action.status = ActionStatus.FAILED if error_data else ActionStatus.COMPLETED
                    executed_action.result = result_data
                    executed_action.error = error_data
                    self.completed_actions.append(executed_action)
                    self.action_history.append(executed_action)
                    self.stats["actions_executed"] += 1
                    self.stats["actions_completed" if not error_data else "actions_failed"] += 1
                    # Trim history
                    if len(self.completed_actions) > self.max_history: self.completed_actions.pop(0)
                    if len(self.action_history) > self.max_history: self.action_history.pop(0)
                    # Record in system context (if available and has record_action)
                    if self.system_context and hasattr(self.system_context, 'record_action') and callable(self.system_context.record_action):
                        record_func = self.system_context.record_action
                        record_params = {"action_type": executed_action.action_type, "params": executed_action.parameters, "status": executed_action.status.value}
                        if error_data: record_params["error"] = error_data
                        else: record_params["result"] = result_data
                        try: # Handle sync/async record_action
                            if asyncio.iscoroutinefunction(record_func): await record_func(**record_params)
                            else: record_func(**record_params)
                        except Exception as record_err: logger.error(f"Failed to record action {executed_action.id} outcome: {record_err}")

            # Trigger next processing cycle regardless of outcome
            asyncio.create_task(self._process_queue())

    async def get_queue_status(self) -> Dict[str, Any]:
        async with self._lock:
            now = datetime.datetime.now()
            return {
                "queued": len(self.action_queue), "executing": len(self.executing_actions),
                "history_size": len(self.action_history),
                "next": [{"id": a.id, "type": a.action_type} for a in self.action_queue[:3]],
                "current": [{"id": a.id, "type": a.action_type, "running_for": round((now - a.executed_at).total_seconds(), 1) if a.executed_at else 0} for a in self.executing_actions.values()],
                "stats": dict(self.stats)
            }

    async def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
         async with self._lock:
             history_slice = self.action_history[-limit:]
             return [{"id": a.id, "type": a.action_type, "status": a.status.value, "completed_at": a.completed_at.isoformat() if a.completed_at else None, "error": a.error} for a in reversed(history_slice)]

    async def create_action(self, action_type: str, source_module: str, parameters: Dict[str, Any],
                          priority: ActionPriority = ActionPriority.MEDIUM) -> Optional[str]:
        action = ActionRequest(action_type=action_type, source_module=source_module, parameters=parameters, priority=priority)
        return await self.queue_action(action)

# --- Factory Function ---
def create_action_selector(nyx_brain: Any) -> ActionSelector:
    """Factory function to create an ActionSelector instance."""
    return ActionSelector(nyx_brain)
