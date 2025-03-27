# nyx/core/integration/integrated_tracer.py

import logging
import asyncio
import datetime
import uuid
import traceback
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from contextlib import contextmanager
import time
import threading
from enum import Enum

from nyx.core.integration.event_bus import get_event_bus, Event

logger = logging.getLogger(__name__)

class TraceLevel(Enum):
    """Trace detail level for entries."""
    DEBUG = 0
    INFO = 1
    IMPORTANT = 2
    CRITICAL = 3

class TraceEntry:
    """A single trace entry."""
    def __init__(self, 
                source_module: str, 
                operation: str, 
                level: TraceLevel = TraceLevel.INFO,
                data: Optional[Dict[str, Any]] = None):
        self.id = f"trace_{uuid.uuid4().hex[:8]}"
        self.timestamp = datetime.datetime.now()
        self.source_module = source_module
        self.operation = operation
        self.level = level
        self.data = data or {}
        self.duration = None
        self.status = "started"
        self.parent_id = None
        self.child_ids = []
    
    def end(self, status: str = "completed", result: Any = None, error: Optional[Exception] = None) -> None:
        """Mark the trace entry as ended."""
        self.duration = (datetime.datetime.now() - self.timestamp).total_seconds()
        self.status = status
        if result is not None:
            # If result is complex, store a summary
            if isinstance(result, dict) and len(str(result)) > 200:
                # Summarize the dict
                result_summary = {}
                for k, v in result.items():
                    if isinstance(v, (str, int, float, bool)):
                        result_summary[k] = v
                    elif isinstance(v, dict):
                        result_summary[k] = f"<dict with {len(v)} items>"
                    elif isinstance(v, list):
                        result_summary[k] = f"<list with {len(v)} items>"
                    else:
                        result_summary[k] = f"<{type(v).__name__}>"
                self.data["result_summary"] = result_summary
            else:
                self.data["result"] = result
        if error:
            self.data["error"] = str(error)
            self.data["error_type"] = type(error).__name__
            self.data["stack_trace"] = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source_module": self.source_module,
            "operation": self.operation,
            "level": self.level.name,
            "data": self.data,
            "duration": self.duration,
            "status": self.status,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids
        }

class TraceGroup:
    """A group of related trace entries."""
    def __init__(self, group_id: str, description: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.id = group_id
        self.description = description
        self.metadata = metadata or {}
        self.entries: Dict[str, TraceEntry] = {}
        self.start_time = datetime.datetime.now()
        self.end_time = None
    
    def add_entry(self, entry: TraceEntry) -> None:
        """Add an entry to the group."""
        self.entries[entry.id] = entry
    
    def end(self) -> None:
        """Mark the group as ended."""
        self.end_time = datetime.datetime.now()
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the group in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "entry_count": len(self.entries)
        }

class TraceStorage:
    """Storage for trace entries and groups."""
    def __init__(self, max_entries: int = 10000, max_groups: int = 100):
        self.entries: Dict[str, TraceEntry] = {}
        self.groups: Dict[str, TraceGroup] = {}
        self.max_entries = max_entries
        self.max_groups = max_groups
        self.current_trace_id = threading.local()
        self._lock = asyncio.Lock()
    
    async def add_entry(self, entry: TraceEntry, group_id: Optional[str] = None) -> str:
        """
        Add an entry to storage.
        
        Args:
            entry: Trace entry to add
            group_id: Optional group ID to add to
            
        Returns:
            Entry ID
        """
        async with self._lock:
            # Check if we need to trim
            if len(self.entries) >= self.max_entries:
                # Remove oldest entries
                oldest_keys = sorted(self.entries.keys(), key=lambda k: self.entries[k].timestamp)[:100]
                for key in oldest_keys:
                    del self.entries[key]
            
            # Add entry
            self.entries[entry.id] = entry
            
            # Add to group if specified
            if group_id and group_id in self.groups:
                self.groups[group_id].add_entry(entry)
            
            # Set parent ID if we have a current trace
            if hasattr(self.current_trace_id, "value") and self.current_trace_id.value:
                parent_id = self.current_trace_id.value
                if parent_id in self.entries:
                    entry.parent_id = parent_id
                    self.entries[parent_id].child_ids.append(entry.id)
            
            return entry.id
    
    async def add_group(self, group: TraceGroup) -> None:
        """Add a group to storage."""
        async with self._lock:
            # Check if we need to trim
            if len(self.groups) >= self.max_groups:
                # Remove oldest groups
                oldest_keys = sorted(self.groups.keys(), key=lambda k: self.groups[k].start_time)[:10]
                for key in oldest_keys:
                    del self.groups[key]
            
            # Add group
            self.groups[group.id] = group
    
    async def update_entry(self, entry_id: str, status: str, result: Any = None, error: Optional[Exception] = None) -> None:
        """Update an existing entry."""
        async with self._lock:
            if entry_id in self.entries:
                self.entries[entry_id].end(status, result, error)
    
    async def get_entry(self, entry_id: str) -> Optional[TraceEntry]:
        """Get an entry by ID."""
        async with self._lock:
            return self.entries.get(entry_id)
    
    async def get_group(self, group_id: str) -> Optional[TraceGroup]:
        """Get a group by ID."""
        async with self._lock:
            return self.groups.get(group_id)
    
    async def get_group_entries(self, group_id: str) -> List[TraceEntry]:
        """Get all entries in a group."""
        async with self._lock:
            if group_id in self.groups:
                return list(self.groups[group_id].entries.values())
            return []
    
    async def end_group(self, group_id: str) -> None:
        """Mark a group as ended."""
        async with self._lock:
            if group_id in self.groups:
                self.groups[group_id].end()
    
    async def get_recent_entries(self, limit: int = 50, level: Optional[TraceLevel] = None) -> List[TraceEntry]:
        """Get recent entries, optionally filtered by level."""
        async with self._lock:
            entries = list(self.entries.values())
            
            # Filter by level if specified
            if level:
                entries = [e for e in entries if e.level.value >= level.value]
            
            # Sort by timestamp (most recent first)
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            
            return entries[:limit]
    
    async def get_recent_groups(self, limit: int = 10) -> List[TraceGroup]:
        """Get recent groups."""
        async with self._lock:
            groups = list(self.groups.values())
            
            # Sort by start time (most recent first)
            groups.sort(key=lambda g: g.start_time, reverse=True)
            
            return groups[:limit]
    
    def set_current_trace(self, trace_id: str) -> None:
        """Set the current trace for this thread."""
        self.current_trace_id.value = trace_id
    
    def clear_current_trace(self) -> None:
        """Clear the current trace for this thread."""
        if hasattr(self.current_trace_id, "value"):
            delattr(self.current_trace_id, "value")

class IntegratedTracer:
    """
    Unified tracing system for cross-module operations.
    
    Provides context managers and decorators for tracing operations across module boundaries.
    """
    def __init__(self, storage: Optional[TraceStorage] = None, use_event_bus: bool = True):
        self.storage = storage or TraceStorage()
        self.event_bus = get_event_bus() if use_event_bus else None
        logger.info("IntegratedTracer initialized")
    
    def create_group(self, group_id: str, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> TraceGroup:
        """
        Create a new trace group.
        
        Args:
            group_id: Unique ID for the group
            description: Description of the group
            metadata: Optional metadata for the group
            
        Returns:
            The created trace group
        """
        group = TraceGroup(group_id, description, metadata)
        asyncio.create_task(self.storage.add_group(group))
        return group
    
    @contextmanager
    def trace(self, source_module: str, operation: str, level: TraceLevel = TraceLevel.INFO, 
             group_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Context manager for tracing operations.
        
        Args:
            source_module: Module name
            operation: Operation name
            level: Trace detail level
            group_id: Optional group ID
            data: Optional data to include
        """
        entry = TraceEntry(source_module, operation, level, data)
        
        # Store the current parent trace ID so we can restore it later
        old_trace_id = None
        if hasattr(self.storage.current_trace_id, "value"):
            old_trace_id = self.storage.current_trace_id.value
        
        # Set this trace as the current one for this thread
        self.storage.set_current_trace(entry.id)
        
        # Add entry to storage
        asyncio.create_task(self.storage.add_entry(entry, group_id))
        
        try:
            # Publish event if available
            if self.event_bus:
                asyncio.create_task(self.event_bus.publish(Event(
                    event_type="trace_started",
                    source="integrated_tracer",
                    data={
                        "trace_id": entry.id,
                        "source_module": source_module,
                        "operation": operation,
                        "level": level.name,
                        "group_id": group_id
                    }
                )))
            
            # Yield control
            yield entry
            
            # Complete the trace
            entry.end("completed")
            asyncio.create_task(self.storage.update_entry(entry.id, "completed"))
            
            # Publish event if available
            if self.event_bus:
                asyncio.create_task(self.event_bus.publish(Event(
                    event_type="trace_completed",
                    source="integrated_tracer",
                    data={
                        "trace_id": entry.id,
                        "source_module": source_module,
                        "operation": operation,
                        "duration": entry.duration
                    }
                )))
            
        except Exception as e:
            # Log the error
            entry.end("failed", error=e)
            asyncio.create_task(self.storage.update_entry(entry.id, "failed", error=e))
            
            # Publish event if available
            if self.event_bus:
                asyncio.create_task(self.event_bus.publish(Event(
                    event_type="trace_failed",
                    source="integrated_tracer",
                    data={
                        "trace_id": entry.id,
                        "source_module": source_module,
                        "operation": operation,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )))
            
            # Re-raise the exception
            raise
        finally:
            # Restore the old trace ID if there was one
            if old_trace_id:
                self.storage.set_current_trace(old_trace_id)
            else:
                self.storage.clear_current_trace()
    
    @contextmanager
    def cross_module_trace(self, source_module: str, target_module: str, operation: str, 
                         level: TraceLevel = TraceLevel.INFO, group_id: Optional[str] = None, 
                         data: Optional[Dict[str, Any]] = None) -> None:
        """
        Context manager for tracing operations that cross module boundaries.
        
        Args:
            source_module: Source module name
            target_module: Target module name
            operation: Operation name
            level: Trace detail level
            group_id: Optional group ID
            data: Optional data to include
        """
        # Create combined operation name
        cross_operation = f"{source_module}->{target_module}:{operation}"
        
        # Use regular trace with combined operation name
        with self.trace(source_module, cross_operation, level, group_id, data) as entry:
            if data is None:
                data = {}
            
            # Add cross-module specific data
            entry.data["target_module"] = target_module
            entry.data["cross_operation"] = True
            
            # Yield entry
            yield entry
    
    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trace entry by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace entry or None if not found
        """
        entry = await self.storage.get_entry(trace_id)
        if entry:
            return entry.to_dict()
        return None
    
    async def get_recent_traces(self, limit: int = 50, level: Optional[TraceLevel] = None) -> List[Dict[str, Any]]:
        """
        Get recent trace entries.
        
        Args:
            limit: Maximum number of entries to return
            level: Optional minimum level to filter by
            
        Returns:
            List of trace entries
        """
        entries = await self.storage.get_recent_entries(limit, level)
        return [entry.to_dict() for entry in entries]
    
    async def get_group_traces(self, group_id: str) -> Dict[str, Any]:
        """
        Get all trace entries in a group.
        
        Args:
            group_id: Group ID
            
        Returns:
            Group information and trace entries
        """
        group = await self.storage.get_group(group_id)
        if not group:
            return {"error": f"Group {group_id} not found"}
        
        entries = await self.storage.get_group_entries(group_id)
        
        return {
            "group": group.to_dict(),
            "entries": [entry.to_dict() for entry in entries]
        }
    
    async def get_related_traces(self, trace_id: str) -> Dict[str, Any]:
        """
        Get a trace entry and all related entries (parent and children).
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace entry and related entries
        """
        entry = await self.storage.get_entry(trace_id)
        if not entry:
            return {"error": f"Trace {trace_id} not found"}
        
        result = {"entry": entry.to_dict(), "children": [], "parent": None}
        
        # Get children
        for child_id in entry.child_ids:
            child = await self.storage.get_entry(child_id)
            if child:
                result["children"].append(child.to_dict())
        
        # Get parent
        if entry.parent_id:
            parent = await self.storage.get_entry(entry.parent_id)
            if parent:
                result["parent"] = parent.to_dict()
        
        return result
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about traces.
        
        Returns:
            Trace statistics
        """
        async with self.storage._lock:
            entries = list(self.storage.entries.values())
            groups = list(self.storage.groups.values())
            
            # Count entries by module
            module_counts = {}
            for entry in entries:
                if entry.source_module not in module_counts:
                    module_counts[entry.source_module] = 0
                module_counts[entry.source_module] += 1
            
            # Count entries by level
            level_counts = {}
            for entry in entries:
                level = entry.level.name
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += 1
            
            # Count errors
            error_count = sum(1 for entry in entries if entry.status == "failed")
            
            return {
                "total_entries": len(entries),
                "total_groups": len(groups),
                "entries_by_module": module_counts,
                "entries_by_level": level_counts,
                "error_count": error_count,
                "cross_module_count": sum(1 for entry in entries if entry.data.get("cross_operation", False))
            }

# Singleton instance
_instance = None

def get_tracer() -> IntegratedTracer:
    """Get the singleton tracer instance."""
    global _instance
    if _instance is None:
        _instance = IntegratedTracer()
    return _instance

# Decorator for easy method tracing
def trace_method(level: TraceLevel = TraceLevel.INFO, group_id: Optional[str] = None):
    """
    Decorator for tracing methods.
    
    Args:
        level: Trace detail level
        group_id: Optional group ID
        
    Returns:
        Decorated method
    """
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Get module name from self
            module_name = self.__class__.__module__
            class_name = self.__class__.__name__
            method_name = func.__name__
            
            # Create operation name
            operation = f"{class_name}.{method_name}"
            
            # Create data from args and kwargs
            data = {
                "args": str(args) if args else None,
                "kwargs": str(kwargs) if kwargs else None
            }
            
            # Get tracer instance
            tracer = get_tracer()
            
            # Use trace context manager
            with tracer.trace(module_name, operation, level, group_id, data) as entry:
                result = await func(self, *args, **kwargs)
                return result
        
        return wrapper
    
    return decorator
