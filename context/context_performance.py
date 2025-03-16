# context/context_performance.py

"""
Performance monitoring for the context optimization system.

This module provides tools to track and analyze the performance of the
context optimization system, including metrics on:
- Response times
- Token usage
- Cache effectiveness
- Memory usage
- Vector search quality
"""

import time
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque

from context_config import get_config

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Performance Data Structures
# -------------------------------------------------------------------------------

class PerformanceMetrics:
    """
    Container for performance metrics.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize with empty metrics.
        
        Args:
            max_history: Maximum number of historical entries to keep
        """
        # Response time metrics
        self.response_times = deque(maxlen=max_history)
        self.response_time_avg = 0.0
        
        # Token usage metrics
        self.token_usage = deque(maxlen=max_history)
        self.token_usage_avg = 0
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_hit_rate = 0.0
        
        # Vector search metrics
        self.vector_searches = 0
        self.vector_search_times = deque(maxlen=max_history)
        self.vector_search_time_avg = 0.0
        
        # Memory usage metrics
        self.memory_usage = deque(maxlen=max_history)
        self.memory_usage_avg = 0
        
        # Additional stats
        self.total_requests = 0
        self.errors = 0
        
        # Timestamp
        self.last_updated = datetime.now()
    
    def add_response_time(self, response_time: float):
        """
        Add a response time measurement.
        
        Args:
            response_time: Response time in seconds
        """
        self.response_times.append(response_time)
        self.response_time_avg = sum(self.response_times) / len(self.response_times)
        self.last_updated = datetime.now()
    
    def add_token_usage(self, tokens: int):
        """
        Add a token usage measurement.
        
        Args:
            tokens: Number of tokens used
        """
        self.token_usage.append(tokens)
        self.token_usage_avg = sum(self.token_usage) / len(self.token_usage)
        self.last_updated = datetime.now()
    
    def record_cache_access(self, hit: bool):
        """
        Record a cache access.
        
        Args:
            hit: Whether the access was a hit or miss
        """
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        self.total_requests += 1
        
        # Update hit rate
        total_cache_accesses = self.cache_hits + self.cache_misses
        if total_cache_accesses > 0:
            self.cache_hit_rate = self.cache_hits / total_cache_accesses
        
        self.last_updated = datetime.now()
    
    def add_vector_search_time(self, search_time: float):
        """
        Add a vector search time measurement.
        
        Args:
            search_time: Search time in seconds
        """
        self.vector_searches += 1
        self.vector_search_times.append(search_time)
        self.vector_search_time_avg = sum(self.vector_search_times) / len(self.vector_search_times)
        self.last_updated = datetime.now()
    
    def add_memory_usage(self, usage: int):
        """
        Add a memory usage measurement.
        
        Args:
            usage: Memory usage in bytes
        """
        self.memory_usage.append(usage)
        self.memory_usage_avg = sum(self.memory_usage) / len(self.memory_usage)
        self.last_updated = datetime.now()
    
    def record_error(self):
        """Record an error"""
        self.errors += 1
        self.last_updated = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the metrics.
        
        Returns:
            Dictionary with metric summary
        """
        return {
            "response_time": {
                "avg": self.response_time_avg,
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "p90": self._percentile(self.response_times, 90) if self.response_times else 0
            },
            "token_usage": {
                "avg": self.token_usage_avg,
                "min": min(self.token_usage) if self.token_usage else 0,
                "max": max(self.token_usage) if self.token_usage else 0,
                "p90": self._percentile(self.token_usage, 90) if self.token_usage else 0
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hit_rate
            },
            "vector_search": {
                "count": self.vector_searches,
                "avg_time": self.vector_search_time_avg
            },
            "memory_usage": {
                "avg": self.memory_usage_avg,
                "last": self.memory_usage[-1] if self.memory_usage else 0
            },
            "requests": self.total_requests,
            "errors": self.errors,
            "last_updated": self.last_updated.isoformat()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.__init__(max_history=self.response_times.maxlen)
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """
        Calculate the given percentile from the data.
        
        Args:
            data: Data points
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not data:
            return 0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1 if f < len(sorted_data) - 1 else f
        
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

# -------------------------------------------------------------------------------
# Performance Monitor
# -------------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Monitor for tracking context optimization performance.
    """
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, user_id: int, conversation_id: int) -> 'PerformanceMonitor':
        """
        Get or create a performance monitor instance.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            PerformanceMonitor instance
        """
        key = f"{user_id}:{conversation_id}"
        
        if key not in cls._instances:
            cls._instances[key] = PerformanceMonitor(user_id, conversation_id)
        
        return cls._instances[key]
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the performance monitor.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize metrics
        self.config = get_config()
        max_history = self.config.get("performance", "metrics_history", 100)
        self.metrics = PerformanceMetrics(max_history=max_history)
        
        # Logging
        self.logging_enabled = self.config.get("performance", "log_metrics", True)
        self.log_interval = self.config.get("performance", "log_interval_seconds", 300)  # 5 minutes
        self.last_log_time = time.time()
        
        # Recording
        self.record_to_db = self.config.get("performance", "record_to_db", False)
        
        # Active timers for current operations
        self.active_timers = {}
    
    def start_timer(self, operation: str) -> str:
        """
        Start a timer for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Timer ID
        """
        timer_id = f"{operation}_{time.time()}"
        self.active_timers[timer_id] = {
            "operation": operation,
            "start_time": time.time()
        }
        return timer_id
    
    def stop_timer(self, timer_id: str) -> float:
        """
        Stop a timer and record the elapsed time.
        
        Args:
            timer_id: Timer ID from start_timer
            
        Returns:
            Elapsed time in seconds
        """
        if timer_id not in self.active_timers:
            return 0.0
        
        timer = self.active_timers.pop(timer_id)
        elapsed = time.time() - timer["start_time"]
        
        # Record based on operation type
        operation = timer["operation"]
        
        if operation == "get_context":
            self.metrics.add_response_time(elapsed)
        elif operation == "vector_search":
            self.metrics.add_vector_search_time(elapsed)
        
        # Maybe log metrics
        self._maybe_log_metrics()
        
        return elapsed
    
    def record_cache_access(self, hit: bool):
        """
        Record a cache access.
        
        Args:
            hit: Whether the access was a hit or miss
        """
        self.metrics.record_cache_access(hit)
        self._maybe_log_metrics()
    
    def record_token_usage(self, tokens: int):
        """
        Record token usage.
        
        Args:
            tokens: Number of tokens used
        """
        self.metrics.add_token_usage(tokens)
        self._maybe_log_metrics()
    
    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            import psutil
            
            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Record in bytes
            self.metrics.add_memory_usage(memory_info.rss)
        except ImportError:
            logger.debug("psutil not available, skipping memory usage tracking")
        except Exception as e:
            logger.warning(f"Error recording memory usage: {e}")
    
    def record_error(self, error: Optional[Exception] = None):
        """
        Record an error.
        
        Args:
            error: Optional exception
        """
        self.metrics.record_error()
        
        if error and self.logging_enabled:
            logger.error(f"Error in context optimization: {error}")
    
    async def record_to_database(self):
        """Record metrics to database if enabled"""
        if not self.record_to_db:
            return
        
        try:
            from db.connection import get_db_connection
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                # Get metrics summary
                summary = self.metrics.get_summary()
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO ContextPerformanceMetrics
                    (user_id, conversation_id, metrics_data, recorded_at)
                    VALUES (%s, %s, %s, NOW())
                """, (self.user_id, self.conversation_id, json.dumps(summary)))
                
                conn.commit()
                
                logger.debug(f"Recorded performance metrics to database for user {self.user_id}")
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            logger.error(f"Error recording metrics to database: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary with metrics
        """
        # Update memory usage before returning
        self.record_memory_usage()
        
        return self.metrics.get_summary()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.reset()
        logger.info(f"Reset performance metrics for user {self.user_id}, conversation {self.conversation_id}")
    
    def _maybe_log_metrics(self):
        """Log metrics if enough time has passed"""
        if not self.logging_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            metrics = self.metrics.get_summary()
            
            logger.info(
                f"Context performance metrics for user {self.user_id}:"
                f" response_time={metrics['response_time']['avg']:.4f}s,"
                f" token_usage={metrics['token_usage']['avg']:.0f},"
                f" cache_hit_rate={metrics['cache']['hit_rate']*100:.1f}%"
            )
            
            # Schedule database recording
            asyncio.create_task(self.record_to_database())
            
            self.last_log_time = current_time

# -------------------------------------------------------------------------------
# Decorator for Performance Tracking
# -------------------------------------------------------------------------------

def track_performance(operation: str):
    """
    Decorator for tracking performance of a function.
    
    Args:
        operation: Operation name
    """
    def decorator(func):
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
                monitor.record_error(e)
                raise
            finally:
                # Stop timer
                monitor.stop_timer(timer_id)
        
        return wrapper
    
    return decorator

# -------------------------------------------------------------------------------
# API Functions
# -------------------------------------------------------------------------------

def get_performance_metrics(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Get performance metrics for a user and conversation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary with performance metrics
    """
    monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    return monitor.get_metrics()

async def get_historical_metrics(
    user_id: int, 
    conversation_id: int, 
    hours: int = 24
) -> List[Dict[str, Any]]:
    """
    Get historical performance metrics from the database.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        hours: Number of hours to look back
        
    Returns:
        List of historical metrics
    """
    try:
        from db.connection import get_db_connection
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get historical metrics
            cursor.execute("""
                SELECT metrics_data, recorded_at
                FROM ContextPerformanceMetrics
                WHERE user_id = %s AND conversation_id = %s
                  AND recorded_at > NOW() - INTERVAL %s HOUR
                ORDER BY recorded_at DESC
            """, (user_id, conversation_id, hours))
            
            metrics = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                data["recorded_at"] = row[1].isoformat()
                metrics.append(data)
            
            return metrics
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        return []

def reset_performance_metrics(user_id: int, conversation_id: int) -> bool:
    """
    Reset performance metrics for a user and conversation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
        monitor.reset_metrics()
        return True
    except Exception as e:
        logger.error(f"Error resetting performance metrics: {e}")
        return False

# -------------------------------------------------------------------------------
# Enhanced context functions with performance tracking
# -------------------------------------------------------------------------------

@track_performance("get_context")
async def get_context_with_tracking(
    user_id: int, 
    conversation_id: int, 
    input_text: str, 
    location: Optional[str] = None,
    context_budget: int = 4000
) -> Dict[str, Any]:
    """
    Get optimized context with performance tracking.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        input_text: User input text
        location: Optional current location
        context_budget: Maximum token budget
        
    Returns:
        Optimized context
    """
    from context_optimization import get_comprehensive_context
    
    return await get_comprehensive_context(
        user_id=user_id,
        conversation_id=conversation_id,
        input_text=input_text,
        location=location,
        context_budget=context_budget
    )

@track_performance("vector_search")
async def get_vector_context_with_tracking(
    user_id: int, 
    conversation_id: int, 
    query_text: str, 
    current_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get vector-enhanced context with performance tracking.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        current_location: Optional current location
        
    Returns:
        Vector-enhanced context
    """
    from vector_integration import get_vector_enhanced_context
    
    return await get_vector_enhanced_context(
        user_id=user_id,
        conversation_id=conversation_id,
        query_text=query_text,
        current_location=current_location
    )
