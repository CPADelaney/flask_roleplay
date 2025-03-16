# context/smart_incredmental_context.py

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Set, Optional, Tuple, Callable
import hashlib
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)

class VersionVector:
    """Version vector for tracking state across distributed components"""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.versions: Dict[str, int] = {component_id: 0}
    
    def increment(self) -> None:
        """Increment this component's version"""
        self.versions[self.component_id] = self.versions.get(self.component_id, 0) + 1
    
    def update(self, other_vector: 'VersionVector') -> bool:
        """
        Update from another vector. Return True if changed.
        """
        changed = False
        for component, version in other_vector.versions.items():
            if component not in self.versions or self.versions[component] < version:
                self.versions[component] = version
                changed = True
        return changed
    
    def is_newer_than(self, other_vector: 'VersionVector') -> bool:
        """Check if this vector is strictly newer than another"""
        # Must be at least as new for all components
        for component, version in other_vector.versions.items():
            if component in self.versions and self.versions[component] < version:
                return False
        
        # And strictly newer for at least one component
        for component, version in self.versions.items():
            other_version = other_vector.versions.get(component, 0)
            if version > other_version:
                return True
                
        return False
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization"""
        return self.versions
    
    @classmethod
    def from_dict(cls, component_id: str, versions: Dict[str, int]) -> 'VersionVector':
        """Create from dictionary"""
        vector = cls(component_id)
        vector.versions = versions
        return vector


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


class SmartIncrementalContext:
    """
    Smart incremental context manager with delta-based updates,
    change prioritization, and efficient batching.
    """
    
    def __init__(self, component_id: str, batch_interval: float = 1.0):
        self.component_id = component_id
        self.version_vector = VersionVector(component_id)
        self.context: Dict[str, Any] = {}
        self.context_hash = self._hash_context({})
        self.change_log: List[ContextDiff] = []
        self.max_change_log_size = 100
        self.pending_changes: List[ContextDiff] = []
        self.batch_interval = batch_interval
        self.last_batch_time = time.time()
        self.batch_task = None
        self.change_subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self.components_seen: Set[str] = {component_id}
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a hash representation of context to detect changes"""
        serialized = json.dumps(context, sort_keys=True)
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
            if "from" in diff.old_value and "path" in diff.old_value:
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
                # A more sophisticated implementation might do element-by-element diffing
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
            priority += len(value) // 5  # More properties = higher priority
            
            # Check for specific high-priority keys
            important_keys = {"npc_id", "conflict_id", "quest_id", "error", "name", "type"}
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
        self.version_vector.increment()
        
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
    
    async def get_context(self, source_component_id: str = None, source_version_vector: VersionVector = None) -> Dict[str, Any]:
        """
        Get the current context or delta based on version vector.
        
        Args:
            source_component_id: ID of the requesting component (if any)
            source_version_vector: Version vector of the requester (if any)
            
        Returns:
            Dictionary containing full context or delta
        """
        # Process any pending changes first
        await self._process_pending_batch()
        
        # If no version vector provided, return full context
        if source_version_vector is None:
            return {
                "full_context": self.context,
                "is_incremental": False,
                "version_vector": self.version_vector.to_dict()
            }
        
        # If the source is newer or equal to us, no need to send anything
        source_vector = VersionVector.from_dict(
            source_component_id or "unknown",
            source_version_vector.to_dict() if isinstance(source_version_vector, VersionVector) else source_version_vector
        )
        
        if not self.version_vector.is_newer_than(source_vector):
            return {
                "full_context": self.context,
                "is_incremental": False,
                "no_changes": True,
                "version_vector": self.version_vector.to_dict()
            }
        
        # Find changes that are not reflected in the source's version
        relevant_changes = []
        for diff in self.change_log:
            # Check if this change is from a component the source doesn't know about
            # or if it's from a component that the source has an older version of
            component_id = self.component_id  # Default to our ID
            
            # Calculate relevance - is this change not reflected in source's version?
            is_relevant = True
            
            if is_relevant:
                relevant_changes.append(diff)
        
        # If no relevant changes or too many, send full context
        if not relevant_changes or len(relevant_changes) > 50:  # Arbitrary threshold
            return {
                "full_context": self.context,
                "is_incremental": False,
                "version_vector": self.version_vector.to_dict()
            }
        
        # Otherwise, return delta
        return {
            "delta_context": [diff.to_dict() for diff in relevant_changes],
            "full_context": self.context,  # Include full context as fallback
            "is_incremental": True,
            "version_vector": self.version_vector.to_dict()
        }
    
    async def update_context(self, new_context: Dict[str, Any], source_component_id: str = None) -> Tuple[bool, List[ContextDiff]]:
        """
        Update the entire context with a new version.
        
        Args:
            new_context: New context dictionary
            source_component_id: ID of the component that generated the update
            
        Returns:
            Tuple of (was_changed, changes)
        """
        if source_component_id:
            self.components_seen.add(source_component_id)
        
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
    
    async def apply_delta(self, delta: Dict[str, Any], source_component_id: str = None, version_vector: Dict[str, int] = None) -> bool:
        """
        Apply a delta update from another component.
        
        Args:
            delta: Delta information containing changes
            source_component_id: ID of the component that generated the delta
            version_vector: Version vector of the source component
            
        Returns:
            Whether the context was modified
        """
        if source_component_id:
            self.components_seen.add(source_component_id)
        
        # Handle full context update
        if "full_context" in delta and not delta.get("is_incremental", False):
            return await self.update_context(delta["full_context"], source_component_id)
        
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
            
            # Update our version vector if provided
            if version_vector:
                source_vector = VersionVector.from_dict(
                    source_component_id or "unknown",
                    version_vector
                )
                self.version_vector.update(source_vector)
            
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


# Example usage
import asyncio

async def example_usage():
    # Create two context managers for different components
    context_manager1 = SmartIncrementalContext("component1")
    context_manager2 = SmartIncrementalContext("component2")
    
    # Set up a change subscriber
    def on_changes(changes):
        print(f"Received {len(changes)} changes:")
        for change in changes:
            print(f"  {change.operation} at {change.path}: {change.value}")
    
    # Subscribe to changes
    context_manager1.subscribe_to_changes("/player", on_changes)
    
    # Make some changes to the first context manager
    await context_manager1.apply_targeted_change("/player/name", "Chase", "add", priority=8)
    await context_manager1.apply_targeted_change("/player/stats/corruption", 30, "add", priority=7)
    await context_manager1.apply_targeted_change("/location/name", "Forest", "add", priority=6)
    
    # Get the current context
    context1 = await context_manager1.get_context()
    print("Context 1:", context1["full_context"])
    
    # Send update to context manager 2
    await context_manager2.apply_delta(context1)
    
    # Get context from manager 2
    context2 = await context_manager2.get_context()
    print("Context 2:", context2["full_context"])
    
    # Make a change in context manager 2
    await context_manager2.apply_targeted_change("/player/stats/corruption", 40, "replace", priority=9)
    
    # Get incremental update
    delta = await context_manager2.get_context("component1", context1["version_vector"])
    print("Delta:", delta)
    
    # Apply the delta back to context manager 1
    await context_manager1.apply_delta(delta, "component2", delta["version_vector"])
    
    # Check final state
    final_context1 = await context_manager1.get_context()
    print("Final Context 1:", final_context1["full_context"])

# Real-world usage examples for RPG context

class RPGContextManager:
    """Manager for handling RPG-specific context with incremental updates"""
    
    def __init__(self, user_id, conversation_id, component_id="game_manager"):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context_manager = SmartIncrementalContext(component_id)
        self.initialize_subscriptions()
    
    def initialize_subscriptions(self):
        """Set up subscriptions for important context changes"""
        # Subscribe to changes in critical paths
        self.context_manager.subscribe_to_changes("/player/stats", self.on_player_stats_change)
        self.context_manager.subscribe_to_changes("/npcs", self.on_npcs_change)
        self.context_manager.subscribe_to_changes("/location", self.on_location_change)
        self.context_manager.subscribe_to_changes("/quests", self.on_quests_change)
        self.context_manager.subscribe_to_changes("/time", self.on_time_change)
    
    async def on_player_stats_change(self, changes):
        """React to player stats changes"""
        for change in changes:
            if "corruption" in change.path or "dependency" in change.path:
                # Check for narrative advancement triggers
                # High priority changes can trigger immediate reactions
                if change.priority >= 8:
                    await self.check_narrative_advancement()
    
    async def on_npcs_change(self, changes):
        """React to NPC changes"""
        # Handle NPC location changes, relationship updates, etc.
        pass
    
    async def on_location_change(self, changes):
        """React to location changes"""
        # Handle scene changes, NPC encounters, etc.
        pass
    
    async def on_quests_change(self, changes):
        """React to quest changes"""
        # Handle quest progress, completions, etc.
        pass
    
    async def on_time_change(self, changes):
        """React to time changes"""
        # Handle day/night cycle events, timed events, etc.
        pass
    
    async def check_narrative_advancement(self):
        """Check if narrative should advance based on context"""
        # Logic to determine if a narrative stage change is needed
        pass
    
    async def update_from_database(self):
        """Fetch latest data from database and update context"""
        # Fetch player data
        player_data = await fetch_player_data(self.user_id, self.conversation_id)
        await self.context_manager.apply_targeted_change("/player", player_data, "replace")
        
        # Fetch location data
        location_data = await fetch_location_data(self.user_id, self.conversation_id)
        await self.context_manager.apply_targeted_change("/location", location_data, "replace")
        
        # Fetch NPCs in the vicinity
        current_location = location_data.get("name", "Unknown")
        nearby_npcs = await fetch_nearby_npcs(self.user_id, self.conversation_id, current_location)
        await self.context_manager.apply_targeted_change("/npcs/nearby", nearby_npcs, "replace")
        
        # Fetch active quests
        active_quests = await fetch_active_quests(self.user_id, self.conversation_id)
        await self.context_manager.apply_targeted_change("/quests/active", active_quests, "replace")
    
    async def get_context_for_agent(self, agent_type, current_input=None):
        """
        Get optimized context for a specific agent type
        
        Args:
            agent_type: Type of agent (narrator, npc_handler, etc.)
            current_input: Current user input for relevance filtering
        """
        # Get full context
        context_data = await self.context_manager.get_context()
        full_context = context_data["full_context"]
        
        # Create optimized view based on agent type
        if agent_type == "narrator":
            # Narrator needs everything
            return full_context
        
        elif agent_type == "npc_handler":
            # NPC handler only needs NPC and player data
            return {
                "player": full_context.get("player", {}),
                "npcs": full_context.get("npcs", {}),
                "location": full_context.get("location", {}),
                "time": full_context.get("time", {})
            }
        
        elif agent_type == "conflict_analyst":
            # Conflict analyst needs conflict and relationship data
            return {
                "player": full_context.get("player", {}),
                "conflicts": full_context.get("conflicts", {}),
                "npcs": full_context.get("npcs", {}),
                "relationships": full_context.get("relationships", {})
            }
        
        # Default to returning the full context
        return full_context
    
    async def apply_agent_updates(self, agent_type, updates):
        """
        Apply updates from an agent
        
        Args:
            agent_type: Type of agent providing updates
            updates: Dictionary of updates to apply
        """
        # Apply updates based on the agent type
        if agent_type == "narrator":
            # Narrator can update any part of the context
            for path, value in updates.items():
                await self.context_manager.apply_targeted_change(
                    path,
                    value,
                    "replace",
                    priority=8  # Narrator updates are high priority
                )
        
        elif agent_type == "npc_handler":
            # NPC handler can only update NPC data
            for npc_id, npc_data in updates.items():
                await self.context_manager.apply_targeted_change(
                    f"/npcs/{npc_id}",
                    npc_data,
                    "replace",
                    priority=7
                )
        
        elif agent_type == "universal_updater":
            # Universal updater can update game state
            if "player_stats" in updates:
                await self.context_manager.apply_targeted_change(
                    "/player/stats",
                    updates["player_stats"],
                    "replace",
                    priority=9  # High priority for stat changes
                )
            
            if "npcs" in updates:
                for npc_update in updates["npcs"]:
                    npc_id = npc_update.get("npc_id")
                    if npc_id:
                        await self.context_manager.apply_targeted_change(
                            f"/npcs/{npc_id}",
                            npc_update,
                            "replace",
                            priority=6
                        )
            
            # Apply other universal updates as needed
        
        # Trigger an immediate batch process for critical updates
        await self.context_manager._process_pending_batch()

# Async utility function to run the example
async def run_example():
    # Set up the event loop
    await example_usage()

# Run the example
if __name__ == "__main__":
    asyncio.run(run_example())
