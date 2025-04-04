# context/context_manager.py

import asyncio
import logging
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)

class ContextDiff:
    """Represents a context difference with change priority information"""
    
    def __init__(self, path: str, operation: str, value: Any = None, old_value: Any = None, priority: int = 5):
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
    Simplified context manager that handles context state and change tracking
    """
    
    def __init__(self, component_id: str = "main_context"):
        self.component_id = component_id
        self.context = {}
        self.context_hash = self._hash_context({})
        self.change_log = []
        self.max_change_log_size = 20
        self.pending_changes = []
        self.change_subscriptions = defaultdict(list)
        self.version = 0
        
        # Initialize the processing task
        self.batch_task = None
        self.batch_interval = 0.5  # seconds
        
        # NEW: Nyx directive handling
        self.nyx_directives = {}
        self.nyx_overrides = {}
        self.nyx_prohibitions = {}
        
        # NEW: Nyx governance integration
        self.governance = None
        self.directive_handler = None
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a hash representation of context to detect changes"""
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()
    
    def _get_value_at_path(self, context: Dict[str, Any], path: str) -> Any:
        """Get a value at the specified path"""
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
        """Set a value at the specified path"""
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
                    current[part] = {}
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if index >= len(current):
                    # Extend the list if needed
                    current.extend([None] * (index - len(current) + 1))
                if current[index] is None:
                    current[index] = {}
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
        """Remove a value at the specified path"""
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
        return context  # No change for unknown operations
    
    def _detect_changes(self, old_context: Dict[str, Any], new_context: Dict[str, Any]) -> List[ContextDiff]:
        """Detect changes between two context objects"""
        # For simplicity, we'll just check top-level keys
        changes = []
        
        # Check for added or modified keys
        for key, value in new_context.items():
            if key not in old_context:
                # Key was added
                changes.append(ContextDiff(
                    path=f"/{key}",
                    operation="add",
                    value=value,
                    priority=self._estimate_priority(value)
                ))
            elif old_context[key] != value:
                # Key was modified
                changes.append(ContextDiff(
                    path=f"/{key}",
                    operation="replace",
                    value=value,
                    old_value=old_context[key],
                    priority=self._estimate_priority(value, old_context[key])
                ))
        
        # Check for removed keys
        for key in old_context:
            if key not in new_context:
                # Key was removed
                changes.append(ContextDiff(
                    path=f"/{key}",
                    operation="remove",
                    old_value=old_context[key],
                    priority=self._estimate_priority(old_context[key])
                ))
        
        return changes
    
    def _estimate_priority(self, value: Any, old_value: Any = None) -> int:
        """Estimate priority of a change based on content"""
        # Default priority
        priority = 5
        
        # Adjust based on value type
        if isinstance(value, dict):
            # More complex dictionaries get higher priority
            keys = value.keys()
            if any(k in keys for k in ["error", "critical", "important"]):
                priority += 3
            elif any(k in keys for k in ["player", "npc", "quest"]):
                priority += 2
            elif len(keys) > 5:
                priority += 1
        elif isinstance(value, list):
            # Longer lists get higher priority
            if len(value) > 10:
                priority += 2
            elif len(value) > 5:
                priority += 1
        
        # Adjust based on change size
        if old_value is not None:
            # Significant changes get higher priority
            if type(value) != type(old_value):
                priority += 2
            elif isinstance(value, dict) and isinstance(old_value, dict):
                # Major dictionary changes
                if len(value) - len(old_value) > 5:
                    priority += 2
                elif len(value) != len(old_value):
                    priority += 1
            elif isinstance(value, list) and isinstance(old_value, list):
                # Major list changes
                if len(value) - len(old_value) > 5:
                    priority += 2
                elif len(value) != len(old_value):
                    priority += 1
            elif isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
                # Major numeric changes
                if abs(value - old_value) > 10:
                    priority += 2
                elif abs(value - old_value) > 5:
                    priority += 1
        
        # Ensure priority is within valid range
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
        relevant_changes = [diff for diff in self.change_log]
        
        # If no relevant changes or too many, send full context
        if not relevant_changes or len(relevant_changes) > 20:
            return {
                "full_context": self.context,
                "is_incremental": False,
                "version": self.version
            }
        
        # Otherwise, return delta
        return {
            "delta_context": [diff.to_dict() for diff in relevant_changes],
            "is_incremental": True,
            "version": self.version
        }
    
    async def update_context(self, new_context: Dict[str, Any]) -> bool:
        """
        Update the entire context with a new version.
        
        Args:
            new_context: New context dictionary
            
        Returns:
            Whether the context was changed
        """
        # If same hash, nothing changed
        new_hash = self._hash_context(new_context)
        if new_hash == self.context_hash:
            return False
        
        # Compute changes
        changes = self._detect_changes(self.context, new_context)
        
        # Add to pending changes
        self.pending_changes.extend(changes)
        
        # Ensure batch processor is running
        self._start_batch_processor()
        
        return True
    
    async def apply_targeted_change(self, path: str, value: Any, operation: str = "replace") -> bool:
        """
        Apply a specific change to a path in the context.
        
        Args:
            path: Path to change
            value: New value
            operation: "add", "remove", or "replace"
            
        Returns:
            Whether the change was applied
        """
        # Create the diff
        old_value = self._get_value_at_path(self.context, path)
        diff = ContextDiff(
            path=path,
            operation=operation,
            value=value if operation != "remove" else None,
            old_value=old_value if operation != "add" else None
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
            path: Path to monitor
            callback: Function to call when changes occur
        """
        self.change_subscriptions[path].append(callback)
    
    def unsubscribe_from_changes(self, path: str, callback: Callable) -> bool:
        """
        Unsubscribe from changes.
        
        Args:
            path: Path that was being monitored
            callback: Function to remove
            
        Returns:
            Whether the subscription was removed
        """
        if path in self.change_subscriptions:
            if callback in self.change_subscriptions[path]:
                self.change_subscriptions[path].remove(callback)
                return True
        return False

    def _prioritize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize context elements based on relevance and importance."""
        try:
            # Calculate priority scores
            priority_scores = {}
            for key, value in context.items():
                score = 0
                
                # Base score by type
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
                score += type_scores.get(key, 3)
                
                # Adjust for recency
                if hasattr(value, "timestamp"):
                    age = time.time() - value.timestamp
                    recency = max(0, 1 - (age / 86400))  # 24-hour decay
                    score *= (1 + recency)
                
                # Adjust for relevance
                if hasattr(value, "relevance_score"):
                    score *= (1 + value.relevance_score)
                
                # Adjust for complexity
                if isinstance(value, dict):
                    # More complex dictionaries get higher priority
                    if any(k in value for k in ["error", "critical", "important"]):
                        score += 3
                    elif any(k in value for k in ["player", "npc", "quest"]):
                        score += 2
                    elif len(value) > 5:
                        score += 1
                elif isinstance(value, list):
                    # Longer lists get higher priority
                    if len(value) > 10:
                        score += 2
                    elif len(value) > 5:
                        score += 1
                
                # Adjust for relationships
                if key == "relationships" and isinstance(value, dict):
                    # More complex relationships get higher priority
                    relationship_count = len(value)
                    if relationship_count > 5:
                        score += 2
                    elif relationship_count > 2:
                        score += 1
                
                # Adjust for narrative importance
                if key == "narrative_metadata":
                    if value.get("coherence_score", 0) > 0.8:
                        score += 2
                    elif value.get("coherence_score", 0) > 0.6:
                        score += 1
                
                priority_scores[key] = score
            
            # Sort by priority
            sorted_items = sorted(
                context.items(),
                key=lambda x: priority_scores.get(x[0], 0),
                reverse=True
            )
            
            # Create prioritized context
            prioritized_context = dict(sorted_items)
            
            # Add priority metadata
            prioritized_context["_priority_metadata"] = {
                "scores": priority_scores,
                "timestamp": time.time()
            }
            
            return prioritized_context
        except Exception as e:
            logger.error(f"Error prioritizing context: {e}")
            return context

    async def initialize_nyx_integration(self, user_id: int, conversation_id: int):
        """Initialize Nyx governance integration."""
        try:
            from nyx.integrate import get_central_governance
            from nyx.directive_handler import DirectiveHandler
            from nyx.nyx_governance import AgentType
            
            # Get governance system
            self.governance = await get_central_governance(user_id, conversation_id)
            
            # Initialize directive handler
            self.directive_handler = DirectiveHandler(
                user_id,
                conversation_id,
                AgentType.CONTEXT_MANAGER,
                self.component_id,
                governance=governance  # pass the object here
            )
            
            # Register handlers
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
        
        if "prioritize_context" in instruction.lower():
            # Apply context prioritization
            params = directive.get("parameters", {})
            priority_rules = params.get("priority_rules", {})
            
            # Update priority rules
            self._update_priority_rules(priority_rules)
            
            # Re-prioritize current context
            self.context = self._prioritize_context(self.context)
            
            return {"result": "context_prioritized"}
            
        elif "consolidate_context" in instruction.lower():
            # Consolidate context based on rules
            params = directive.get("parameters", {})
            consolidation_rules = params.get("consolidation_rules", {})
            
            # Apply consolidation
            self.context = await self._consolidate_context(consolidation_rules)
            
            return {"result": "context_consolidated"}
            
        elif "track_changes" in instruction.lower():
            # Enable/disable change tracking
            params = directive.get("parameters", {})
            enabled = params.get("enabled", True)
            
            self.change_tracking_enabled = enabled
            return {"result": "change_tracking_updated", "enabled": enabled}
        
        return {"result": "action_not_recognized"}

    async def _handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx."""
        logging.info(f"[ContextManager] Processing override directive")
        
        # Extract override details
        override_action = directive.get("override_action", {})
        applies_to = directive.get("applies_to", [])
        
        # Store override
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_overrides[directive_id] = {
                "action": override_action,
                "applies_to": applies_to
            }
        
        return {"result": "override_stored"}

    async def _handle_prohibition_directive(self, directive: dict) -> dict:
        """Handle a prohibition directive from Nyx."""
        logging.info(f"[ContextManager] Processing prohibition directive")
        
        # Extract prohibition details
        prohibited_actions = directive.get("prohibited_actions", [])
        reason = directive.get("reason", "No reason provided")
        
        # Store prohibition
        directive_id = directive.get("id")
        if directive_id:
            self.nyx_prohibitions[directive_id] = {
                "prohibited_actions": prohibited_actions,
                "reason": reason
            }
        
        return {"result": "prohibition_stored"}

    def _update_priority_rules(self, rules: Dict[str, Any]) -> None:
        """Update priority rules based on Nyx directive."""
        try:
            # Update type scores
            if "type_scores" in rules:
                self.type_scores.update(rules["type_scores"])
            
            # Update priority adjustments
            if "priority_adjustments" in rules:
                self.priority_adjustments.update(rules["priority_adjustments"])
            
            # Update relationship weights
            if "relationship_weights" in rules:
                self.relationship_weights.update(rules["relationship_weights"])
            
            logger.info("Updated priority rules from Nyx directive")
        except Exception as e:
            logger.error(f"Error updating priority rules: {e}")

    async def _consolidate_context(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate context based on Nyx rules."""
        try:
            consolidated = self.context.copy()
            
            # Apply consolidation rules
            if "group_by" in rules:
                for group_key in rules["group_by"]:
                    if group_key in consolidated:
                        consolidated[group_key] = self._consolidate_group(
                            consolidated[group_key],
                            rules.get("group_rules", {}).get(group_key, {})
                        )
            
            # Apply memory consolidation
            if "consolidate_memories" in rules and "memories" in consolidated:
                consolidated["memories"] = await self._consolidate_memories(
                    consolidated["memories"],
                    rules["consolidate_memories"]
                )
            
            # Apply relationship consolidation
            if "consolidate_relationships" in rules and "relationships" in consolidated:
                consolidated["relationships"] = await self._consolidate_relationships(
                    consolidated["relationships"],
                    rules["consolidate_relationships"]
                )
            
            return consolidated
        except Exception as e:
            logger.error(f"Error consolidating context: {e}")
            return self.context

    def _consolidate_group(self, group: Any, rules: Dict[str, Any]) -> Any:
        """Consolidate a group of items based on rules."""
        try:
            if isinstance(group, list):
                # Apply list consolidation rules
                if "max_items" in rules:
                    group = group[:rules["max_items"]]
                if "group_by" in rules:
                    # Group items by specified key
                    grouped = defaultdict(list)
                    for item in group:
                        key = item.get(rules["group_by"])
                        if key:
                            grouped[key].append(item)
                    # Convert back to list with summaries
                    group = [
                        self._summarize_group(items, rules.get("summary_rules", {}))
                        for items in grouped.values()
                    ]
            elif isinstance(group, dict):
                # Apply dictionary consolidation rules
                if "max_keys" in rules:
                    # Keep only top N keys by importance
                    sorted_keys = sorted(
                        group.keys(),
                        key=lambda k: self._calculate_key_importance(k, group[k]),
                        reverse=True
                    )[:rules["max_keys"]]
                    group = {k: group[k] for k in sorted_keys}
            
            return group
        except Exception as e:
            logger.error(f"Error consolidating group: {e}")
            return group

    def _summarize_group(self, items: List[Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of a group of items."""
        try:
            summary = {
                "count": len(items),
                "items": items[:rules.get("max_summary_items", 3)],
                "timestamp": datetime.now().isoformat()
            }
            
            # Add additional summary fields based on rules
            if "include_stats" in rules:
                summary["stats"] = self._calculate_group_stats(items, rules["include_stats"])
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing group: {e}")
            return {"count": len(items), "items": items[:3]}

    def _calculate_key_importance(self, key: str, value: Any) -> float:
        """Calculate importance of a dictionary key."""
        try:
            importance = 0.0
            
            # Base importance by key name
            if key in ["error", "critical", "important"]:
                importance += 3.0
            elif key in ["player", "npc", "quest"]:
                importance += 2.0
            
            # Adjust by value type and size
            if isinstance(value, dict):
                importance += min(2.0, len(value) * 0.2)
            elif isinstance(value, list):
                importance += min(2.0, len(value) * 0.1)
            
            return importance
        except Exception as e:
            logger.error(f"Error calculating key importance: {e}")
            return 0.0

    def _calculate_group_stats(self, items: List[Any], stat_types: List[str]) -> Dict[str, Any]:
        """Calculate statistics for a group of items."""
        try:
            stats = {}
            
            for stat_type in stat_types:
                if stat_type == "count":
                    stats["count"] = len(items)
                elif stat_type == "importance":
                    # Calculate average importance
                    importance_values = [
                        item.get("importance", 0.5)
                        for item in items
                        if isinstance(item, dict)
                    ]
                    if importance_values:
                        stats["avg_importance"] = sum(importance_values) / len(importance_values)
                elif stat_type == "recency":
                    # Calculate average recency
                    timestamps = [
                        datetime.fromisoformat(item["timestamp"])
                        for item in items
                        if isinstance(item, dict) and "timestamp" in item
                    ]
                    if timestamps:
                        now = datetime.now()
                        avg_age = sum(
                            (now - ts).total_seconds()
                            for ts in timestamps
                        ) / len(timestamps)
                        stats["avg_age_seconds"] = avg_age
            
            return stats
        except Exception as e:
            logger.error(f"Error calculating group stats: {e}")
            return {}


# Global instance
_context_manager = None

def get_context_manager() -> ContextManager:
    """Get the singleton context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
