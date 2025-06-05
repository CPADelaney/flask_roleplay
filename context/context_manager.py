# context/context_manager.py

import asyncio
import logging
import json
import time
import hashlib
import copy
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Agent SDK imports
from agents import Agent, function_tool, RunContextWrapper
from agents.tracing import trace, custom_span

logger = logging.getLogger(__name__)

class ContextData(BaseModel):
    """Full context data"""
    full_context: 'ContextContent'
    is_incremental: bool = False
    version: int
    no_changes: Optional[bool] = None
    
    class Config:
        extra = "forbid"


class DeltaContextData(BaseModel):
    """Delta context data"""
    delta_context: List['ContextDiffModel']
    is_incremental: bool = True
    version: int
    
    class Config:
        extra = "forbid"


class ContextContent(BaseModel):
    """Generic context content container"""
    data: Dict[str, Union[str, int, float, bool, List[str]]]
    
    class Config:
        extra = "forbid"


class ContextDiffModel(BaseModel):
    """Model for context diff"""
    path: str
    op: str  # "add", "remove", "replace"
    value: Optional[Union[str, int, float, bool, List[str]]] = None
    old_value: Optional[Union[str, int, float, bool, List[str]]] = None
    priority: int
    timestamp: float
    
    class Config:
        extra = "forbid"


class PrioritizedContext(BaseModel):
    """Prioritized context with metadata"""
    context: ContextContent
    priority_metadata: 'PriorityMetadata'
    
    class Config:
        extra = "forbid"


class PriorityMetadata(BaseModel):
    """Priority metadata"""
    scores: Dict[str, float]
    timestamp: float
    
    class Config:
        extra = "forbid"

class ContextDiff:
    """Represents a context difference with change priority information"""
    
    def __init__(
        self, 
        path: str, 
        operation: str, 
        value: Any = None, 
        old_value: Any = None, 
        priority: int = 5
    ):
        """
        Initialize a context difference.
        
        Args:
            path: Path to the changed element
            operation: "add", "remove", or "replace"
            value: New value (for add/replace)
            old_value: Previous value (for remove/replace)
            priority: Importance of this change (1-10)
        """
        self.path = path
        self.operation = operation
        self.value = value
        self.old_value = old_value
        self.priority = priority
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "path": self.path,
            "op": self.operation,
            "value": self.value,
            "old_value": self.old_value,
            "priority": self.priority,
            "timestamp": self.timestamp
        }


class ContextManager:
    """
    Context manager that handles context state and change tracking,
    integrated with OpenAI Agents SDK (but we do NOT use @function_tool on these instance methods).
    """

    def __init__(self, component_id: str = "main_context"):
        self.component_id = component_id

        # The main context dictionary
        self.context: Dict[str, Any] = {}
        self.context_hash: str = self._hash_context({})
        
        # Change log, versioning, and pending-changes
        self.change_log: List[ContextDiff] = []
        self.max_change_log_size = 20
        self.pending_changes: List[ContextDiff] = []
        self.version = 0
        
        # Subscription system
        self.change_subscriptions = defaultdict(list)
        
        # Batch processing for pending changes
        self.batch_task: Optional[asyncio.Task] = None
        self.batch_interval = 0.5  # seconds between each batch run

        # Nyx directive handling
        self.nyx_directives: Dict[str, Any] = {}
        self.nyx_overrides: Dict[str, Any] = {}
        self.nyx_prohibitions: Dict[str, Any] = {}

        # Governance integration objects
        self.governance = None
        self.directive_handler = None

    # ---------------------------------------------------------------------
    # Internal: Hashing & Diff
    # ---------------------------------------------------------------------

    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a hash representation of `context` for quick comparison."""
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()
    
    def _detect_changes(
        self, 
        old_context: Dict[str, Any], 
        new_context: Dict[str, Any]
    ) -> List[ContextDiff]:
        """
        Detect changes between two context objects in a naive (top-level) manner.
        
        For a deeper or more sophisticated diff, you'd traverse nested dictionaries, etc.
        """
        changes = []
        
        # Check new or changed keys
        for key, value in new_context.items():
            if key not in old_context:
                changes.append(ContextDiff(
                    path=f"/{key}",
                    operation="add",
                    value=value,
                    priority=5
                ))
            elif old_context[key] != value:
                changes.append(ContextDiff(
                    path=f"/{key}",
                    operation="replace",
                    value=value,
                    old_value=old_context[key],
                    priority=5
                ))
        
        # Check removed keys
        for key in old_context:
            if key not in new_context:
                changes.append(ContextDiff(
                    path=f"/{key}",
                    operation="remove",
                    old_value=old_context[key],
                    priority=5
                ))
        
        return changes

    # ---------------------------------------------------------------------
    # Internal: Setting, Removing, & Applying diffs
    # ---------------------------------------------------------------------
    
    def _get_value_at_path(
        self, 
        context: Dict[str, Any], 
        path: str
    ) -> Any:
        """Get the value from `context` at a slash-separated path, e.g. '/someKey/0/nested'."""
        if not path or path == "/":
            return context
        parts = path.strip("/").split("/")
        current = context
        try:
            for part in parts:
                if isinstance(current, dict):
                    current = current[part]
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    # Can't navigate further
                    return None
            return current
        except (KeyError, IndexError):
            return None
    
    def _set_value_at_path(
        self, 
        context: Dict[str, Any], 
        path: str, 
        value: Any
    ) -> Dict[str, Any]:
        """Return a copy of `context` with `value` set at `path` (slash-separated)."""
        if not path or path == "/":
            return value  # Replace entire context

        result = copy.deepcopy(context)
        parts = path.strip("/").split("/")
        current = result
        
        # Navigate to the parent container
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if index >= len(current):
                    current.extend([None]*(index - len(current)+1))
                if current[index] is None:
                    current[index] = {}
                current = current[index]
        
        last_part = parts[-1]
        if isinstance(current, dict):
            current[last_part] = value
        elif isinstance(current, list) and last_part.isdigit():
            index = int(last_part)
            if index >= len(current):
                current.extend([None]*(index - len(current)+1))
            current[index] = value
        return result
    
    def _remove_value_at_path(
        self,
        context: Dict[str, Any], 
        path: str
    ) -> Dict[str, Any]:
        """Return a copy of `context` with the value at `path` removed."""
        if not path or path == "/":
            return {}  # remove entire context

        result = copy.deepcopy(context)
        parts = path.strip("/").split("/")
        current = result
        
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    return result
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                if idx >= len(current):
                    return result
                current = current[idx]
        
        last_part = parts[-1]
        if isinstance(current, dict) and last_part in current:
            del current[last_part]
        elif isinstance(current, list) and last_part.isdigit():
            idx = int(last_part)
            if idx < len(current):
                current.pop(idx)
        
        return result
    
    def _apply_diff(
        self, 
        context: Dict[str, Any], 
        diff: ContextDiff
    ) -> Dict[str, Any]:
        """Apply a single ContextDiff to produce a new context dictionary."""
        if diff.operation in ("add", "replace"):
            return self._set_value_at_path(context, diff.path, diff.value)
        elif diff.operation == "remove":
            return self._remove_value_at_path(context, diff.path)
        return context

    # ---------------------------------------------------------------------
    # Internal: Batching & Processing
    # ---------------------------------------------------------------------
    
    def _start_batch_processor(self):
        """Ensure the batch processing task is running."""
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._process_batches())
    
    async def _process_batches(self):
        """Background loop to process pending changes at intervals."""
        try:
            while True:
                await asyncio.sleep(self.batch_interval)
                await self._process_pending_batch()
        except asyncio.CancelledError:
            logger.debug("ContextManager batch processor cancelled.")
    
    async def _process_pending_batch(self):
        """Apply all pending changes as a batch, update version, and notify subscribers."""
        if not self.pending_changes:
            return

        # Grab all changes
        changes = self.pending_changes
        self.pending_changes = []

        # Apply them to produce a new context
        modified = self.context
        for diff in changes:
            modified = self._apply_diff(modified, diff)
        
        # Update version & context
        self.version += 1
        self.context = modified
        self.context_hash = self._hash_context(modified)
        
        # Save changes to the log
        self.change_log.extend(changes)
        while len(self.change_log) > self.max_change_log_size:
            self.change_log.pop(0)
        
        # Notify subscribers
        await self._notify_subscribers(changes)

    async def _notify_subscribers(self, changes: List[ContextDiff]):
        """Call any subscriber callbacks that care about the changed paths."""
        for path, subs in self.change_subscriptions.items():
            # For each subscription path, see if any diffs match that path
            relevant = [d for d in changes if d.path.startswith(path)]
            if relevant:
                for cb in subs:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(relevant)
                        else:
                            cb(relevant)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")

    # ---------------------------------------------------------------------
    # INTERNAL (PRIVATE) Methods that used to be @function_tool
    # ---------------------------------------------------------------------

    async def _get_context(self, source_version: Optional[int] = None) -> Union[ContextData, DeltaContextData]:
        """
        Return the full context or a delta if `source_version` is older.
        """
        # Make sure pending changes are processed
        await self._process_pending_batch()
        
        if source_version is None:
            # Return everything if no version is specified
            return ContextData(
                full_context=ContextContent(data=self.context),
                is_incremental=False,
                version=self.version
            )
        
        if source_version >= self.version:
            # The requester is up-to-date
            return ContextData(
                full_context=ContextContent(data=self.context),
                is_incremental=False,
                no_changes=True,
                version=self.version
            )
        
        # Attempt to build a delta from change_log if not too large
        if len(self.change_log) == 0 or len(self.change_log) > 20:
            return ContextData(
                full_context=ContextContent(data=self.context),
                is_incremental=False,
                version=self.version
            )
        
        delta_models = [ContextDiffModel(**diff.to_dict()) for diff in self.change_log]
        return DeltaContextData(
            delta_context=delta_models,
            is_incremental=True,
            version=self.version
        )

    async def _update_context(self, new_context: Dict[str, Any]) -> bool:
        """
        Replace the entire context with `new_context`.
        Return True if changed, otherwise False.
        """
        new_hash = self._hash_context(new_context)
        if new_hash == self.context_hash:
            return False
        
        diffs = self._detect_changes(self.context, new_context)
        self.pending_changes.extend(diffs)
        self._start_batch_processor()
        
        return True

    async def _apply_targeted_change(
        self, 
        path: str, 
        value: Any, 
        operation: str
    ) -> bool:
        """
        Apply a single diff (add, remove, or replace) to `self.context`.
        """
        old_value = self._get_value_at_path(self.context, path)
        diff = ContextDiff(
            path=path,
            operation=operation,
            value=value if operation != "remove" else None,
            old_value=old_value if operation != "add" else None
        )
        self.pending_changes.append(diff)
        self._start_batch_processor()
        return True

    def subscribe_to_changes(self, path: str, callback: Callable) -> None:
        """Subscribe a callback to changes at `path`."""
        self.change_subscriptions[path].append(callback)

    def unsubscribe_from_changes(self, path: str, callback: Callable) -> bool:
        """Unsubscribe a callback previously subscribed."""
        if path in self.change_subscriptions:
            if callback in self.change_subscriptions[path]:
                self.change_subscriptions[path].remove(callback)
                return True
        return False

    def _prioritize_context(self, context: Dict[str, Any]) -> PrioritizedContext:
        """
        Internal method: reorder or annotate context by priority.
        """
        try:
            # Some sample type-based scoring
            type_scores = {
                "npcs": 8,
                "memories": 7,
                "quests": 6,
                "location": 5,
                "relationships": 7,
                "narrative": 8,
                "conflicts": 7,
                "lore": 6,
                "events": 5,
                "items": 4
            }
    
            priority_scores = {}
    
            # For each key in the context, compute a "score"
            for key, value in context.items():
                # Start with a base from type_scores
                score = type_scores.get(key, 3)
                priority_scores[key] = score
    
            # Sort items in descending order of the computed score
            sorted_items = sorted(
                context.items(),
                key=lambda x: priority_scores.get(x[0], 0),
                reverse=True
            )
    
            # Build a new dictionary with the high-score items first
            prioritized_dict = dict(sorted_items)
            
            # Return typed result
            return PrioritizedContext(
                context=ContextContent(data=prioritized_dict),
                priority_metadata=PriorityMetadata(
                    scores=priority_scores,
                    timestamp=time.time()
                )
            )
    
        except Exception as e:
            logger.error(f"Error prioritizing context: {e}")
            return PrioritizedContext(
                context=ContextContent(data=context),
                priority_metadata=PriorityMetadata(scores={}, timestamp=time.time())
            )

    # ---------------------------------------------------------------------
    # Nyx / Governance integration
    # ---------------------------------------------------------------------
    
    async def initialize_nyx_integration(self, user_id: int, conversation_id: int):
        """
        Asynchronously initialize Nyx governance, directive handlers, etc.
        """
        try:
            from nyx.integrate import get_central_governance
            from nyx.directive_handler import DirectiveHandler
            from nyx.nyx_governance import AgentType
            
            self.governance = await get_central_governance(user_id, conversation_id)
            self.directive_handler = DirectiveHandler(
                user_id,
                conversation_id,
                AgentType.CONTEXT_MANAGER,
                self.component_id
            )
            # Register directive handlers
            self.directive_handler.register_handler("action", self._handle_action_directive)
            self.directive_handler.register_handler("override", self._handle_override_directive)
            self.directive_handler.register_handler("prohibition", self._handle_prohibition_directive)
            
            logger.info(f"Initialized Nyx integration for context manager {self.component_id}")
        except Exception as e:
            logger.error(f"Error initializing Nyx integration: {e}")

    async def _handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx."""
        instruction = directive.get("instruction", "")
        logging.info(f"[ContextManager] Processing action directive: {instruction}")
        
        # Example stubs:
        if "prioritize_context" in instruction.lower():
            self.context = self._prioritize_context(self.context)
            return {"result": "context_prioritized"}
        elif "consolidate_context" in instruction.lower():
            # Possibly run some consolidation logic
            return {"result": "context_consolidated"}
        
        return {"result": "action_not_recognized"}

    async def _handle_override_directive(self, directive: dict) -> dict:
        logging.info("[ContextManager] Processing override directive")
        # Example stub
        return {"result": "override_stored"}

    async def _handle_prohibition_directive(self, directive: dict) -> dict:
        logging.info("[ContextManager] Processing prohibition directive")
        # Example stub
        return {"result": "prohibition_stored"}


# ---------------------------------------------------------------------
# Singleton accessor for your ContextManager
# ---------------------------------------------------------------------
_context_manager: Optional[ContextManager] = None

def get_context_manager() -> ContextManager:
    """Obtain (or create) the singleton ContextManager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


# ---------------------------------------------------------------------
# STANDALONE TOOL FUNCTIONS (these have `ctx: RunContextWrapper` first!)
# ---------------------------------------------------------------------

@function_tool
async def get_context_tool(
    ctx: RunContextWrapper,
    source_version: Optional[int] = None
) -> Union[ContextData, DeltaContextData]:
    """
    Tool function: get context or a delta from the singleton context manager.
    """
    cm = get_context_manager()
    return await cm._get_context(source_version)


@function_tool
async def update_context_tool(
    ctx: RunContextWrapper,
    new_context: ContextContent
) -> bool:
    """
    Tool function: replace the entire context with `new_context`.
    """
    cm = get_context_manager()
    return await cm._update_context(new_context.data)


@function_tool
def prioritize_context_tool(
    ctx: RunContextWrapper,
    user_context: ContextContent
) -> PrioritizedContext:
    """
    Tool function: reorder/annotate a context dictionary by priority.
    """
    cm = get_context_manager()
    return cm._prioritize_context(user_context.data)


@function_tool
async def apply_targeted_change_tool(
    ctx: RunContextWrapper,
    path: str,
    value: Any,
    operation: str = "replace"
) -> bool:
    """
    Tool function: apply a single add/remove/replace diff on the context.
    """
    cm = get_context_manager()
    return await cm._apply_targeted_change(path, value, operation)

# ---------------------------------------------------------------------
# Agent creation referencing the *standalone* tool functions
# ---------------------------------------------------------------------

def create_context_manager_agent() -> Agent:
    """Create an agent for the context manager"""
    
    # Create an agent for the context manager
    agent = Agent(
        name="ContextManager",
        instructions="""
        You are a context manager for RPG interactions. Your responsibilities include:
        
        1. Managing the context state
        2. Tracking changes to the context
        3. Providing relevant context for the current interaction
        4. Optimizing context for token efficiency
        
        When handling context operations, prioritize important information and ensure
        efficient use of the token budget.
        """,
        tools=[
            get_context_tool,
            update_context_tool,
            apply_targeted_change_tool,
            prioritize_context_tool,
        ],
        model="gpt-4.1-nano"
    )
    
    return agent


# Global instance
_context_manager = None

def get_context_manager() -> ContextManager:
    """Get the singleton context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def get_context_manager_agent() -> Agent:
    """Get the context manager agent"""
    return create_context_manager_agent()
