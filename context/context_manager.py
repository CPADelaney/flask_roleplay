# context/context_manager.py

import asyncio
import logging
import json
import time
import math
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import copy

from context.unified_cache import context_cache
from context.context_config import get_config

logger = logging.getLogger(__name__)

class ContextDiff:
    """Represents a context difference with change priority information"""
    
    def __init__(self, path: str, operation: str, value: Any = None, old_value: Any = None, priority: int = 5):
        """
        Initialize a context difference.
        
        Args:
            path: JSON path to the changed element
            operation: "add", "remove", "replace", or "move"
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextDiff':
        """Create from dictionary"""
        diff = cls(
            path=data["path"],
            operation=data["op"],
            value=data.get("value"),
            old_value=data.get("old_value"),
            priority=data.get("priority", 5)
        )
        diff.timestamp = data.get("timestamp", time.time())
        return diff


class UnifiedContextManager:
    """
    Unified context manager that combines features from multiple previous implementations
    with delta-based updates, change prioritization, and relevance tracking.
    """
    
    def __init__(self, component_id: str = "main_context", batch_interval: float = 1.0):
        self.component_id = component_id
        self.context = {}
        self.context_hash = self._hash_context({})
        self.change_log = []
        self.max_change_log_size = 20
        self.pending_changes = []
        self.batch_interval = batch_interval
        self.last_batch_time = time.time()
        self.batch_task = None
        self.change_subscriptions = defaultdict(list)
        
        # Tracking for access patterns and relevance
        self.access_patterns = defaultdict(int)
        self.relevance_scores = {}
        self.last_accessed = {}
        
        # Version tracking (simplified)
        self.version = 0
        self.versions_seen = {component_id: 0}
        
        # Config
        self.config = get_config()
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a hash representation of context to detect changes"""
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()
    
    def _get_value_at_path(self, context: Dict[str, Any], path: str) -> Any:
        """Get a value at the specified JSON path"""
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
                    return None
            return current
        except (KeyError, IndexError):
            return None
    
    def _set_value_at_path(self, context: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
        """Set a value at the specified JSON path"""
        if not path or path == "/":
            return value  # Replace entire context
            
        result = copy.deepcopy(context)
        parts = path.strip("/").split("/")
        current = result
        
        # Navigate to the parent of the target
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, dict):
                if part not in current:
                    # Create missing intermediate objects
                    current[part] = {} if i < len(parts) - 2 else {}
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if index >= len(current):
                    # Extend the list if needed
                    current.extend([None] * (index - len(current) + 1))
                if current[index] is None:
                    current[index] = {} if i < len(parts) - 2 else {}
                current = current[index]
            else:
                # Cannot navigate further
                return result
        
        # Set the value
        last_part = parts[-1]
        if isinstance(current, dict):
            current[last_part] = value
        elif isinstance(current, list) and last_part.isdigit():
            index = int(last_part)
            if index >= len(current):
                current.extend([None] * (index - len(current) + 1))
            current[index] = value
        
        return result
    
    def _remove_value_at_path(self, context: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Remove a value at the specified JSON path"""
        if not path or path == "/":
            return {}  # Remove entire context
            
        result = copy.deepcopy(context)
        parts = path.strip("/").split("/")
        current = result
        
        # Navigate to the parent of the target
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, dict):
                if part not in current:
                    # Path doesn't exist, nothing to remove
                    return result
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if index >= len(current):
                    # Path doesn't exist, nothing to remove
                    return result
                current = current[index]
            else:
                # Cannot navigate further
                return result
        
        # Remove the value
        last_part = parts[-1]
        if isinstance(current, dict) and last_part in current:
            del current[last_part]
        elif isinstance(current, list) and last_part.isdigit():
            index = int(last_part)
            if index < len(current):
                current.pop(index)
        
        return result
    
    def _apply_diff(self, context: Dict[str, Any], diff: ContextDiff) -> Dict[str, Any]:
        """Apply a single diff to a context"""
        if diff.operation == "add" or diff.operation == "replace":
            return self._set_value_at_path(context, diff.path, diff.value)
        elif diff.operation == "remove":
            return self._remove_value_at_path(context, diff.path)
        elif diff.operation == "move":
            # Remove from source path and add to target path
            if isinstance(diff.old_value, dict) and "from" in diff.old_value and "path" in diff.old_value:
                source_path = diff.old_value["from"]
                target_path = diff.old_value["path"]
                value_to_move = self._get_value_at_path(context, source_path)
                if value_to_move is not None:
                    # First remove from source
                    context = self._remove_value_at_path(context, source_path)
                    # Then add to target
                    return self._set_value_at_path(context, target_path, value_to_move)
        return context  # No change for unknown operations
    
    def _extract_diff_paths(self, old_context: Dict[str, Any], new_context: Dict[str, Any], base_path: str = "/") -> List[ContextDiff]:
        """
        Extract detailed difference paths between two contexts
        """
        diffs = []
        
        if isinstance(old_context, dict) and isinstance(new_context, dict):
            # Handle dictionaries
            for key in set(old_context.keys()) | set(new_context.keys()):
                path = f"{base_path}{key}"
                if path.startswith("//"):
                    path = path[1:]  # Fix double slashes
                
                if key not in new_context:
                    # Key was removed
                    diffs.append(ContextDiff(
                        path=path,
                        operation="remove",
                        old_value=old_context[key],
                        priority=self._estimate_priority(old_context[key])
                    ))
                elif key not in old_context:
                    # Key was added
                    diffs.append(ContextDiff(
                        path=path,
                        operation="add",
                        value=new_context[key],
                        priority=self._estimate_priority(new_context[key])
                    ))
                elif old_context[key] != new_context[key]:
                    if isinstance(old_context[key], (dict, list)) and isinstance(new_context[key], (dict, list)):
                        # Recurse into nested structures
                        if path.endswith("/"):
                            nested_path = path
                        else:
                            nested_path = f"{path}/"
                        diffs.extend(self._extract_diff_paths(old_context[key], new_context[key], nested_path))
                    else:
                        # Value was changed
                        diffs.append(ContextDiff(
                            path=path,
                            operation="replace",
                            value=new_context[key],
                            old_value=old_context[key],
                            priority=self._estimate_priority(new_context[key], old_context[key])
                        ))
        
        elif isinstance(old_context, list) and isinstance(new_context, list):
            # Handle lists
            if len(old_context) != len(new_context) or old_context != new_context:
                # For simplicity, just replace the whole list if it changed
                diffs.append(ContextDiff(
                    path=base_path.rstrip("/"),
                    operation="replace",
                    value=new_context,
                    old_value=old_context,
                    priority=5  # Medium priority for list changes
                ))
        
        else:
            # Different types, just replace
            diffs.append(ContextDiff(
                path=base_path.rstrip("/"),
                operation="replace",
                value=new_context,
                old_value=old_context,
                priority=5  # Medium priority for type changes
            ))
        
        return diffs
    
    def _estimate_priority(self, value: Any, old_value: Any = None) -> int:
        """Estimate priority of a change based on value type and content"""
        # Start with medium priority
        priority = 5
        
        if old_value is not None:
            # Prioritize significant changes
            try:
                if isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
                    # Calculate percentage change for numeric values
                    if old_value != 0:
                        pct_change = abs(value - old_value) / abs(old_value)
                        if pct_change > 0.5:  # Over 50% change
                            priority += 2
                        elif pct_change > 0.2:  # Over 20% change
                            priority += 1
            except (TypeError, ValueError):
                pass
        
        # Prioritize by type and structure
        if isinstance(value, dict):
            # Changes to dictionaries are often important
            priority += min(len(value) // 5, 2)  # More properties = higher priority, up to +2
            
            # Check for specific high-priority keys
            important_keys = {"npc_id", "conflict_id", "quest_id", "error", "name", "type", "status"}
            if any(key in value for key in important_keys):
                priority += 1
                
        elif isinstance(value, list):
            # Changes to lists may be important
            priority += min(len(value) // 10, 2)  # More items = higher priority (up to +2)
        
        # Clamp to valid range
        return max(1, min(priority, 10))
    
    def _start_batch_processor(self):
        """Start the background batch processor if not already running"""
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._process_batches())
    
    async def _process_batches(self):
        """Process batches of changes at regular intervals"""
        while True:
            try:
                await asyncio.sleep(self.batch_interval)
                await self._process_pending_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                # Continue processing despite errors
    
    async def _process_pending_batch(self):
        """Process any pending changes as a single batch"""
        if not self.pending_changes:
            return
            
        # Get all pending changes
        changes = self.pending_changes
        self.pending_changes = []
        
        # Apply all changes
        modified_context = self.context
        for diff in changes:
            modified_context = self._apply_diff(modified_context, diff)
        
        # Update the version
        self.version += 1
        
        # Update context and hash
        self.context = modified_context
        self.context_hash = self._hash_context(modified_context)
        
        # Add to change log
        self.change_log.extend(changes)
        
        # Trim change log if needed
        while len(self.change_log) > self.max_change_log_size:
            self.change_log.pop(0)
        
        # Notify subscribers
        await self._notify_subscribers(changes)
    
    async def _notify_subscribers(self, changes: List[ContextDiff]):
        """Notify all subscribers of changes"""
        paths_changed = {diff.path for diff in changes}
        
        # Group changes by subscription path
        for path, subscribers in self.change_subscriptions.items():
            matching_changes = [diff for diff in changes if diff.path.startswith(path)]
            if matching_changes:
                # Notify all subscribers for this path
                for subscriber in subscribers:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            await subscriber(matching_changes)
                        else:
                            subscriber(matching_changes)
                    except Exception as e:
                        logger.error(f"Error in subscriber: {e}")
    
    async def get_context(self, source_version: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the current context or delta based on version.
        
        Args:
            source_version: Version of the requester (if any)
            
        Returns:
            Dictionary containing full context or delta
        """
        # Process any pending changes first
        await self._process_pending_batch()
        
        # If no version provided, return full context
        if source_version is None:
            return {
                "full_context": self.context,
                "is_incremental": False,
                "version": self.version
            }
        
        # If the source is newer or equal to us, no need to send anything
        if source_version >= self.version:
            return {
                "full_context": self.context,
                "is_incremental": False,
                "no_changes": True,
                "version": self.version
            }
        
        # Find changes that are not reflected in the source's version
        relevant_changes = []
        for diff in self.change_log:
            # Calculate relevance - is this change not reflected in source's version?
            is_relevant = True
            
            if is_relevant:
                relevant_changes.append(diff)
        
        # If no relevant changes or too many, send full context
        if not relevant_changes or len(relevant_changes) > 50:  # Arbitrary threshold
            return {
                "full_context": self.context,
                "is_incremental": False,
                "version": self.version
            }
        
        # Otherwise, return delta
        return {
            "delta_context": [diff.to_dict() for diff in relevant_changes],
            "full_context": self.context,  # Include full context as fallback
            "is_incremental": True,
            "version": self.version
        }
    
    async def update_context(self, new_context: Dict[str, Any]) -> Tuple[bool, List[ContextDiff]]:
        """
        Update the entire context with a new version.
        
        Args:
            new_context: New context dictionary
            
        Returns:
            Tuple of (was_changed, changes)
        """
        # If same hash, nothing changed
        new_hash = self._hash_context(new_context)
        if new_hash == self.context_hash:
            return False, []
        
        # Compute detailed diffs
        changes = self._extract_diff_paths(self.context, new_context)
        
        # Add to pending changes
        self.pending_changes.extend(changes)
        
        # Ensure batch processor is running
        self._start_batch_processor()
        
        return True, changes
    
    async def apply_delta(self, delta: Dict[str, Any], source_version: Optional[int] = None) -> bool:
        """
        Apply a delta update from another component.
        
        Args:
            delta: Delta information containing changes
            source_version: Version of the source component
            
        Returns:
            Whether the context was modified
        """
        # Handle full context update
        if "full_context" in delta and not delta.get("is_incremental", False):
            return await self.update_context(delta["full_context"])
        
        # Handle delta update
        if "delta_context" in delta and delta.get("is_incremental", False):
            changes = []
            for diff_dict in delta["delta_context"]:
                diff = ContextDiff.from_dict(diff_dict)
                changes.append(diff)
            
            # Add to pending changes
            self.pending_changes.extend(changes)
            
            # Ensure batch processor is running
            self._start_batch_processor()
            
            # Update our version if provided
            if source_version is not None and source_version > self.version:
                self.version = source_version
            
            return len(changes) > 0
        
        return False
    
    async def apply_targeted_change(self, path: str, value: Any, operation: str = "replace", priority: int = 5) -> bool:
        """
        Apply a specific change to a path in the context.
        
        Args:
            path: JSON path to change
            value: New value
            operation: "add", "remove", or "replace"
            priority: Importance of this change (1-10)
            
        Returns:
            Whether the change was applied
        """
        # Create the diff
        old_value = self._get_value_at_path(self.context, path)
        diff = ContextDiff(
            path=path,
            operation=operation,
            value=value if operation != "remove" else None,
            old_value=old_value if operation != "add" else None,
            priority=priority
        )
        
        # Add to pending changes
        self.pending_changes.append(diff)
        
        # Ensure batch processor is running
        self._start_batch_processor()
        
        return True
    
    def subscribe_to_changes(self, path: str, callback: Callable) -> None:
        """
        Subscribe to changes at a specific path.
        
        Args:
            path: JSON path to monitor
            callback: Function to call when changes occur
        """
        self.change_subscriptions[path].append(callback)
    
    def unsubscribe_from_changes(self, path: str, callback: Callable) -> bool:
        """
        Unsubscribe from changes.
        
        Args:
            path: JSON path that was being monitored
            callback: Function to remove
            
        Returns:
            Whether the subscription was removed
        """
        if path in self.change_subscriptions:
            if callback in self.change_subscriptions[path]:
                self.change_subscriptions[path].remove(callback)
                return True
        return False
    
    def track_access(self, path: str, score: float = 0.1) -> None:
        """Track access to a specific path to update relevance scores"""
        if not path:
            return
            
        self.access_patterns[path] = self.access_patterns.get(path, 0) + score
        self.last_accessed[path] = time.time()
        
        # Update overall path
        parts = path.split("/")
        for i in range(1, len(parts)):
            parent_path = "/".join(parts[:i])
            if parent_path:
                self.access_patterns[parent_path] = self.access_patterns.get(parent_path, 0) + (score * 0.5)
                self.last_accessed[parent_path] = time.time()
    
    def update_relevance(self, path: str, score: float) -> None:
        """Update the relevance score for a specific path"""
        self.relevance_scores[path] = score
    
    def get_relevance(self, path: str) -> float:
        """Get the relevance score for a specific path"""
        # Check direct score
        if path in self.relevance_scores:
            return self.relevance_scores[path]
            
        # Check from access patterns with time decay
        if path in self.access_patterns:
            access_score = self.access_patterns[path]
            last_access = self.last_accessed.get(path, 0)
            age_hours = (time.time() - last_access) / 3600
            
            # Apply time decay (half-life of ~24 hours)
            decay = math.exp(-0.029 * age_hours)  # ln(2)/24 ≈ 0.029
            return access_score * decay
            
        return 0.0


# Global instance
_context_manager = None

def get_context_manager() -> UnifiedContextManager:
    """Get the singleton context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = UnifiedContextManager()
    return _context_manager
