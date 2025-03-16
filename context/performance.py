# context/performance.py

import asyncio
import logging
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from collections import deque
from functools import wraps

from context.context_config import get_config

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Unified performance monitoring with reduced complexity"""
    
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
        self.config = get_config()
        
        # Core metrics
        self.response_times = deque(maxlen=100)
        self.token_usage = deque(maxlen=100)
        self.vector_search_times = deque(maxlen=50)
        self.memory_usage = deque(maxlen=50)
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Active timers for current operations
        self.active_timers = {}
        
        # Config
        self.enabled = self.config.get("performance", "analytics_enabled", True)
        self.thresholds = {
            "max_response_time": 1.0,  # seconds
            "max_token_usage": 6000,   # tokens
            "min_cache_hit_rate": 0.5  # ratio
        }
    
    def start_timer(self, operation: str) -> str:
        """Start a timer for an operation"""
        if not self.enabled:
            return "disabled"
            
        timer_id = f"{operation}_{time.time()}"
        self.active_timers[timer_id] = {
            "operation": operation,
            "start_time": time.time()
        }
        return timer_id
    
    def stop_timer(self, timer_id: str) -> float:
        """Stop a timer and record the elapsed time"""
        if not self.enabled or timer_id == "disabled" or timer_id not in self.active_timers:
            return 0.0
        
        timer = self.active_timers.pop(timer_id)
        elapsed = time.time() - timer["start_time"]
        
        # Record based on operation type
        operation = timer["operation"]
        
        if operation == "get_context":
            self.response_times.append(elapsed)
        elif operation == "vector_search":
            self.vector_search_times.append(elapsed)
        
        return elapsed
    
    def record_cache_access(self, hit: bool):
        """Record a cache access"""
        if not self.enabled:
            return
            
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_token_usage(self, tokens: int):
        """Record token usage"""
        if not self.enabled:
            return
            
        self.token_usage.append(tokens)
    
    def record_memory_usage(self):
        """Record current memory usage"""
        if not self.enabled:
            return
            
        try:
            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Record in MB
            self.memory_usage.append(memory_info.rss / (1024 * 1024))
        except Exception as e:
            logger.warning(f"Error recording memory usage: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.enabled:
            return {"enabled": False}
            
        # Calculate averages
        avg_response_time = sum(self.response_times) / max(1, len(self.response_times))
        avg_token_usage = sum(self.token_usage) / max(1, len(self.token_usage))
        avg_vector_time = sum(self.vector_search_times) / max(1, len(self.vector_search_times))
        avg_memory_mb = sum(self.memory_usage) / max(1, len(self.memory_usage))
        
        # Calculate cache hit rate
        total_cache_accesses = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(1, total_cache_accesses)
        
        return {
            "enabled": True,
            "response_time": {
                "avg_seconds": avg_response_time,
                "max_seconds": max(self.response_times) if self.response_times else 0,
                "threshold_seconds": self.thresholds["max_response_time"]
            },
            "token_usage": {
                "avg": avg_token_usage,
                "max": max(self.token_usage) if self.token_usage else 0,
                "threshold": self.thresholds["max_token_usage"]
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate,
                "threshold": self.thresholds["min_cache_hit_rate"]
            },
            "vector_search": {
                "avg_seconds": avg_vector_time,
                "count": len(self.vector_search_times)
            },
            "memory_usage": {
                "avg_mb": avg_memory_mb,
                "current_mb": self.memory_usage[-1] if self.memory_usage else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.response_times.clear()
        self.token_usage.clear()
        self.vector_search_times.clear()
        self.memory_usage.clear()
        self.cache_hits = 0
        self.cache_misses = 0


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
                    if isinstance(token_usage, dict) and "total" in token_usage:
                        monitor.record_token_usage(token_usage["total"])
                    elif isinstance(token_usage, int):
                        monitor.record_token_usage(token_usage)
                
                # Record cache hit if available
                if isinstance(result, dict) and "source" in result:
                    source = result["source"]
                    if source == "cache":
                        monitor.record_cache_access(hit=True)
                    else:
                        monitor.record_cache_access(hit=False)
                
                return result
            except Exception as e:
                # Record error
                logger.error(f"Error during {operation}: {e}")
                raise
            finally:
                # Stop timer
                monitor.stop_timer(timer_id)
        
        return wrapper
    
    return decorator
