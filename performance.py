# context/performance.py

import time
import logging
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Simplified performance monitoring for context operations.
    
    Tracks response times, token usage, and memory usage with minimal overhead.
    """
    
    _instances = {}  # Singleton registry
    
    @classmethod
    def get_instance(cls, user_id: int, conversation_id: int):
        """Get or create a performance monitor instance"""
        key = f"{user_id}:{conversation_id}"
        if key not in cls._instances:
            cls._instances[key] = cls(user_id, conversation_id)
        return cls._instances[key]
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize with tracking for key metrics"""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Track only the most important metrics
        self.response_times = deque(maxlen=20)  # Last 20 response times
        self.token_usage = deque(maxlen=20)     # Last 20 token usages
        self.memory_usage = deque(maxlen=10)    # Last 10 memory samples
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Active timers
        self.active_timers = {}
        
        # Start time for this session
        self.start_time = time.time()
    
    def start_timer(self, operation: str) -> str:
        """Start a timer for an operation"""
        timer_id = f"{operation}_{time.time()}"
        self.active_timers[timer_id] = {
            "operation": operation,
            "start_time": time.time()
        }
        return timer_id
    
    def stop_timer(self, timer_id: str) -> float:
        """Stop a timer and record the elapsed time"""
        if timer_id not in self.active_timers:
            return 0.0
        
        timer = self.active_timers.pop(timer_id)
        elapsed = time.time() - timer["start_time"]
        
        # Record time for the context operation
        operation = timer["operation"]
        if operation == "get_context":
            self.response_times.append(elapsed)
        
        return elapsed
    
    def record_token_usage(self, tokens: int):
        """Record token usage"""
        self.token_usage.append(tokens)
    
    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Record in MB
            self.memory_usage.append(memory_info.rss / (1024 * 1024))
        except Exception as e:
            logger.warning(f"Error recording memory usage: {e}")
    
    def record_cache_access(self, hit: bool):
        """Record a cache access"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        # Calculate averages
        avg_response_time = sum(self.response_times) / max(1, len(self.response_times))
        avg_token_usage = sum(self.token_usage) / max(1, len(self.token_usage))
        avg_memory_mb = sum(self.memory_usage) / max(1, len(self.memory_usage))
        
        # Calculate cache hit rate
        total_cache_accesses = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(1, total_cache_accesses)
        
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        
        return {
            "response_time": {
                "avg_seconds": avg_response_time,
                "max_seconds": max(self.response_times) if self.response_times else 0,
            },
            "token_usage": {
                "avg": avg_token_usage,
                "max": max(self.token_usage) if self.token_usage else 0,
            },
            "cache": {
                "hit_rate": cache_hit_rate,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            },
            "memory_usage": {
                "avg_mb": avg_memory_mb,
                "current_mb": self.memory_usage[-1] if self.memory_usage else 0
            },
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.now().isoformat()
        }


def track_performance(operation: str):
    """
    Decorator for tracking performance of a function.
    
    Args:
        operation: Operation name
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(user_id, conversation_id, *args, **kwargs):
            # Get monitor
            monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
            
            # Start timer
            timer_id = monitor.start_timer(operation)
            
            try:
                # Call function
                result = await func(user_id, conversation_id, *args, **kwargs)
                
                # Record token usage if available
                if isinstance(result, dict) and "token_usage" in result:
                    token_usage = result["token_usage"]
                    if isinstance(token_usage, dict):
                        total_tokens = sum(token_usage.values())
                        monitor.record_token_usage(total_tokens)
                    elif isinstance(token_usage, int):
                        monitor.record_token_usage(token_usage)
                
                # Record memory usage
                monitor.record_memory_usage()
                
                return result
            except Exception as e:
                # Record error
                logger.error(f"Error during {operation}: {e}")
                raise
            finally:
                # Stop timer
                elapsed = monitor.stop_timer(timer_id)
                
                # Log performance
                if elapsed > 1.0:  # Only log slow operations
                    logger.info(f"{operation} took {elapsed:.3f}s")
        
        return wrapper
    
    return decorator
