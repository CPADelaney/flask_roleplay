# utils/performance.py

import time
import logging
import functools
from typing import Dict, Any, Optional, Callable

class PerformanceTracker:
    """Track performance metrics for operations."""
    
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.phases = {}
        self.current_phase = None
        self.current_phase_start = None
    
    def start_phase(self, phase_name):
        """Start timing a new phase."""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.current_phase_start = time.time()
        return self
    
    def end_phase(self):
        """End timing the current phase."""
        if self.current_phase and self.current_phase_start:
            elapsed = time.time() - self.current_phase_start
            self.phases[self.current_phase] = elapsed
            self.current_phase = None
            self.current_phase_start = None
        return self
    
    def get_metrics(self):
        """Get all timing metrics."""
        # Make sure any current phase is ended
        if self.current_phase:
            self.end_phase()
            
        # Add total time
        total_time = time.time() - self.start_time
        metrics = {
            "total_time": total_time,
            "phases": self.phases
        }
        
        # Log all metrics
        phase_metrics = ", ".join([f"{k}={v:.3f}s" for k, v in self.phases.items()])
        logging.info(f"{self.operation_name} completed in {total_time:.3f}s: {phase_metrics}")
        
        return metrics

def timed_function(func=None, *, name=None):
    """
    Decorator to time function execution.
    
    Args:
        func: The function to decorate
        name: Optional name for the function (defaults to function name)
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                function_name = name or func.__name__
                logging.info(f"{function_name} completed in {elapsed:.3f}s")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                function_name = name or func.__name__
                logging.info(f"{function_name} completed in {elapsed:.3f}s")
        
        # Use the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    if func is None:
        # Called with parameters
        return decorator
    # Called without parameters
    return decorator(func)

class MonitoringStats:
    """Collect and report system-wide stats."""
    
    def __init__(self):
        self.npc_interaction_times = []
        self.db_query_times = []
        self.memory_access_times = []
        self.request_counts = {}
        self.error_counts = {}
    
    def record_interaction_time(self, time_ms):
        """Record NPC interaction time."""
        self.npc_interaction_times.append(time_ms)
        if len(self.npc_interaction_times) > 1000:
            self.npc_interaction_times = self.npc_interaction_times[-1000:]
    
    def record_db_query_time(self, time_ms):
        """Record database query time."""
        self.db_query_times.append(time_ms)
        if len(self.db_query_times) > 1000:
            self.db_query_times = self.db_query_times[-1000:]
    
    def record_memory_access_time(self, time_ms):
        """Record memory access time."""
        self.memory_access_times.append(time_ms)
        if len(self.memory_access_times) > 1000:
            self.memory_access_times = self.memory_access_times[-1000:]
    
    def record_request(self, endpoint):
        """Record an API request."""
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
    
    def record_error(self, error_type):
        """Record an error."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_stats(self):
        """Get all monitoring stats."""
        return {
            "npc_interaction": {
                "count": len(self.npc_interaction_times),
                "avg_time_ms": sum(self.npc_interaction_times) / max(1, len(self.npc_interaction_times)),
                "max_time_ms": max(self.npc_interaction_times, default=0)
            },
            "db_queries": {
                "count": len(self.db_query_times),
                "avg_time_ms": sum(self.db_query_times) / max(1, len(self.db_query_times)),
                "max_time_ms": max(self.db_query_times, default=0)
            },
            "memory_access": {
                "count": len(self.memory_access_times),
                "avg_time_ms": sum(self.memory_access_times) / max(1, len(self.memory_access_times)),
                "max_time_ms": max(self.memory_access_times, default=0)
            },
            "requests": self.request_counts,
            "errors": self.error_counts
        }

# Global stats collector
STATS = MonitoringStats()
