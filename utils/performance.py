# utils/performance.py

import time
import logging
import functools
import asyncio
import psutil
import os
from typing import Dict, Any, Optional, Callable

class PerformanceTracker:
    """Track performance metrics for operations."""
    
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.phases = {}
        self.current_phase = None
        self.current_phase_start = None
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss
    
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
        
        # Get memory usage
        end_memory = self.process.memory_info().rss
        memory_diff_mb = (end_memory - self.start_memory) / (1024 * 1024)
        
        metrics = {
            "total_time": total_time,
            "phases": self.phases,
            "memory": {
                "start_mb": self.start_memory / (1024 * 1024),
                "end_mb": end_memory / (1024 * 1024),
                "diff_mb": memory_diff_mb
            }
        }
        
        # Log all metrics
        phase_metrics = ", ".join([f"{k}={v:.3f}s" for k, v in self.phases.items()])
        logging.info(f"{self.operation_name} completed in {total_time:.3f}s: {phase_metrics} (Memory: {memory_diff_mb:.2f}MB)")
        
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
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss
            start_time = time.time()
            
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                end_memory = process.memory_info().rss
                memory_diff_mb = (end_memory - start_memory) / (1024 * 1024)
                function_name = name or func.__name__
                logging.info(f"{function_name} completed in {elapsed:.3f}s (Memory: {memory_diff_mb:.2f}MB)")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss
            start_time = time.time()
            
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                end_memory = process.memory_info().rss
                memory_diff_mb = (end_memory - start_memory) / (1024 * 1024)
                function_name = name or func.__name__
                logging.info(f"{function_name} completed in {elapsed:.3f}s (Memory: {memory_diff_mb:.2f}MB)")
        
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
        self.last_memory_check = time.time()
        self.memory_measurements = []
        # Start memory monitoring
        self._start_memory_monitor()
    
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
    
    def _start_memory_monitor(self):
        """Start background thread to monitor memory usage."""
        def memory_monitor():
            while True:
                try:
                    # Check memory every 30 seconds
                    time.sleep(30)
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_percent = process.memory_percent()
                    
                    # Record memory info
                    # connections() can be deprecated; guard and fall back
                    try:
                        conn_count = len(process.connections())
                    except Exception:
                        conn_count = 0
                    self.memory_measurements.append({
                        "timestamp": time.time(),
                        "rss_mb": memory_info.rss / (1024 * 1024),
                        "vms_mb": memory_info.vms / (1024 * 1024),
                        "percent": memory_percent,
                        "connections": conn_count
                    })
                    
                    # Keep only the last 100 measurements
                    if len(self.memory_measurements) > 100:
                        self.memory_measurements = self.memory_measurements[-100:]
                        
                    # Log memory if it's getting high
                    if memory_percent > 70:
                        logging.warning(f"High memory usage: {memory_percent:.1f}% ({memory_info.rss / (1024 * 1024):.1f}MB)")
                except Exception as e:
                    logging.error(f"Error in memory monitor: {e}")
        
        # Start monitoring in background thread
        import threading
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
    
    def get_stats(self):
        """Get all monitoring stats."""
        # Calculate memory stats
        memory_stats = {
            "current": None,
            "average": None,
            "peak": None
        }
        
        if self.memory_measurements:
            current = self.memory_measurements[-1]["rss_mb"]
            average = sum(m["rss_mb"] for m in self.memory_measurements) / len(self.memory_measurements)
            peak = max(m["rss_mb"] for m in self.memory_measurements)
            
            memory_stats = {
                "current_mb": current,
                "average_mb": average,
                "peak_mb": peak,
                "percent": self.memory_measurements[-1]["percent"]
            }
        
        return {
            "memory": memory_stats,
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
