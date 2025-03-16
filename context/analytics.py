# context/analytics.py

"""
Performance analytics and monitoring for the context optimization system.

This module provides unified tools for tracking, analyzing, and visualizing 
the performance of the context system, combining detailed analytics and 
real-time monitoring capabilities.
"""

import asyncio
import logging
import time
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import hashlib
import math
import os
import csv
from collections import deque
from functools import wraps

from context.context_config import get_config

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Performance Metrics Classes
# -------------------------------------------------------------------------------

class PerformanceMetrics:
    """Container for performance metrics"""
    
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
# Context Analytics Class
# -------------------------------------------------------------------------------

class ContextAnalytics:
    """
    Analytics system for monitoring context performance
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.metrics = {}
        self.history = []
        self.max_history_size = 1000
        self.last_save = time.time()
        self.save_interval = 3600  # 1 hour
        self.enabled = self.config.get("performance", "analytics_enabled", True)
        
        # Ensure storage directory exists
        self.storage_dir = self._get_storage_dir()
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Performance metrics
        self.performance_metrics = PerformanceMetrics()
        
        # Thresholds
        self.thresholds = {
            "max_retrieval_time_ms": self.config.get("performance", "max_retrieval_time_ms", 1000),
            "max_token_usage": self.config.get("performance", "max_token_usage", 6000),
            "min_cache_hit_rate": self.config.get("performance", "min_cache_hit_rate", 0.3),
            "max_memory_usage_mb": self.config.get("performance", "max_memory_usage_mb", 500)
        }
        
        # Active timers for current operations
        self.active_timers = {}
        
        # Alerts
        self.alerts = []
        self.max_alerts = 100
        
        # Logging configuration
        self.logging_enabled = self.config.get("performance", "log_metrics", True)
        self.log_interval = self.config.get("performance", "log_interval_seconds", 300)
        self.last_log_time = time.time()
    
    def _get_storage_dir(self) -> str:
        """Get the directory to store analytics data"""
        base_dir = self.config.get("performance", "analytics_dir", "analytics")
        user_dir = f"{base_dir}/user_{self.user_id}/conversation_{self.conversation_id}"
        return user_dir
    
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
            self.performance_metrics.add_response_time(elapsed)
        elif operation == "vector_search":
            self.performance_metrics.add_vector_search_time(elapsed)
        
        # Maybe log metrics
        self._maybe_log_metrics()
        
        return elapsed
    
    def record_cache_access(self, hit: bool):
        """
        Record a cache access.
        
        Args:
            hit: Whether the access was a hit or miss
        """
        self.performance_metrics.record_cache_access(hit)
        self._maybe_log_metrics()
    
    def record_token_usage(self, tokens: int):
        """
        Record token usage.
        
        Args:
            tokens: Number of tokens used
        """
        self.performance_metrics.add_token_usage(tokens)
        self._maybe_log_metrics()
    
    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Record in bytes
            self.performance_metrics.add_memory_usage(memory_info.rss)
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
        self.performance_metrics.record_error()
        
        if error and self.logging_enabled:
            logger.error(f"Error in context optimization: {error}")
    
    def record_context_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Record a context-related event for analytics
        
        Args:
            event_type: Type of event (e.g., retrieval, update, maintenance)
            data: Event data to record
        """
        if not self.enabled:
            return
            
        # Create event record
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "event_type": event_type,
            "data": data
        }
        
        # Add to history
        self.history.append(event)
        
        # Limit history size
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
        
        # Update metrics
        self._update_metrics(event_type, data)
        
        # Check if we should save analytics
        if time.time() - self.last_save > self.save_interval:
            asyncio.create_task(self.save_analytics())
    
    def record_context_retrieval(
        self,
        input_text: str,
        location: Optional[str],
        context_size: int,
        token_usage: Dict[str, int],
        duration_ms: int,
        is_delta: bool,
        is_cached: bool,
        used_vector: bool
    ) -> None:
        """
        Record a context retrieval event
        
        Args:
            input_text: User input text (or digest of it)
            location: Current location
            context_size: Size of context in bytes
            token_usage: Token usage by component
            duration_ms: Duration in milliseconds
            is_delta: Whether it was a delta update
            is_cached: Whether it was cached
            used_vector: Whether vector search was used
        """
        # Create a digest of input text for privacy
        input_digest = hashlib.md5(input_text.encode()).hexdigest()[:8]
        
        # Record the event
        self.record_context_event("retrieval", {
            "input_digest": input_digest,
            "location": location,
            "context_size": context_size,
            "token_usage": token_usage,
            "duration_ms": duration_ms,
            "is_delta": is_delta,
            "is_cached": is_cached,
            "used_vector": used_vector
        })
        
        # Check retrieval performance
        self.check_retrieval_performance(
            duration_ms=duration_ms,
            token_usage=token_usage,
            is_cached=is_cached
        )
    
    def record_maintenance_event(
        self,
        maintenance_type: str,
        results: Dict[str, Any],
        duration_ms: int
    ) -> None:
        """
        Record a maintenance event
        
        Args:
            maintenance_type: Type of maintenance (e.g., memory_consolidation)
            results: Maintenance results
            duration_ms: Duration in milliseconds
        """
        self.record_context_event("maintenance", {
            "maintenance_type": maintenance_type,
            "results": results,
            "duration_ms": duration_ms
        })
    
    def _update_metrics(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Update metrics based on an event
        
        Args:
            event_type: Type of event
            data: Event data
        """
        # Initialize metrics if needed
        if "retrieval" not in self.metrics:
            self.metrics["retrieval"] = {
                "count": 0,
                "durations": [],
                "sizes": [],
                "token_usages": [],
                "delta_count": 0,
                "cache_hit_count": 0,
                "vector_count": 0
            }
        
        if "maintenance" not in self.metrics:
            self.metrics["maintenance"] = {
                "count": 0,
                "durations": [],
                "types": {}
            }
        
        # Update metrics based on event type
        if event_type == "retrieval":
            # Update retrieval metrics
            metrics = self.metrics["retrieval"]
            metrics["count"] += 1
            metrics["durations"].append(data.get("duration_ms", 0))
            metrics["sizes"].append(data.get("context_size", 0))
            
            # Add token usage if available
            if "token_usage" in data:
                metrics["token_usages"].append(sum(data["token_usage"].values()))
            
            # Update flag counts
            if data.get("is_delta", False):
                metrics["delta_count"] += 1
            if data.get("is_cached", False):
                metrics["cache_hit_count"] += 1
            if data.get("used_vector", False):
                metrics["vector_count"] += 1
                
            # Limit array sizes
            max_samples = 100
            if len(metrics["durations"]) > max_samples:
                metrics["durations"] = metrics["durations"][-max_samples:]
            if len(metrics["sizes"]) > max_samples:
                metrics["sizes"] = metrics["sizes"][-max_samples:]
            if len(metrics["token_usages"]) > max_samples:
                metrics["token_usages"] = metrics["token_usages"][-max_samples:]
        
        elif event_type == "maintenance":
            # Update maintenance metrics
            metrics = self.metrics["maintenance"]
            metrics["count"] += 1
            metrics["durations"].append(data.get("duration_ms", 0))
            
            # Track by maintenance type
            maint_type = data.get("maintenance_type", "unknown")
            if maint_type not in metrics["types"]:
                metrics["types"][maint_type] = {
                    "count": 0,
                    "durations": []
                }
                
            type_metrics = metrics["types"][maint_type]
            type_metrics["count"] += 1
            type_metrics["durations"].append(data.get("duration_ms", 0))
            
            # Limit array sizes
            max_samples = 20
            if len(metrics["durations"]) > max_samples:
                metrics["durations"] = metrics["durations"][-max_samples:]
            if len(type_metrics["durations"]) > max_samples:
                type_metrics["durations"] = type_metrics["durations"][-max_samples:]
    
    def check_retrieval_performance(
        self,
        duration_ms: int,
        token_usage: Dict[str, int],
        is_cached: bool
    ) -> List[Dict[str, Any]]:
        """
        Check retrieval performance against thresholds
        
        Args:
            duration_ms: Retrieval duration in milliseconds
            token_usage: Token usage by component
            is_cached: Whether the retrieval was cached
            
        Returns:
            List of alerts generated
        """
        if not self.enabled:
            return []
            
        new_alerts = []
        
        # Check retrieval time
        if duration_ms > self.thresholds["max_retrieval_time_ms"]:
            alert = {
                "level": "warning",
                "message": f"Slow context retrieval: {duration_ms}ms (threshold: {self.thresholds['max_retrieval_time_ms']}ms)",
                "type": "performance",
                "timestamp": datetime.now().isoformat()
            }
            new_alerts.append(alert)
        
        # Check token usage
        total_tokens = sum(token_usage.values())
        if total_tokens > self.thresholds["max_token_usage"]:
            alert = {
                "level": "warning",
                "message": f"High token usage: {total_tokens} tokens (threshold: {self.thresholds['max_token_usage']})",
                "type": "resource",
                "timestamp": datetime.now().isoformat()
            }
            new_alerts.append(alert)
        
        # Add alerts to history
        self.alerts.extend(new_alerts)
        
        # Limit alerts history
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        return new_alerts
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of analytics data
        
        Returns:
            Dictionary with analytics summary
        """
        summary = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "start_time": self.history[0]["timestamp"] if self.history else None,
            "end_time": self.history[-1]["timestamp"] if self.history else None,
            "retrieval": {},
            "maintenance": {}
        }
        
        # Calculate retrieval metrics
        if "retrieval" in self.metrics:
            retrieval = self.metrics["retrieval"]
            retrieval_count = retrieval["count"]
            
            if retrieval_count > 0:
                # Calculate averages and percentiles
                durations = retrieval["durations"]
                sizes = retrieval["sizes"]
                token_usages = retrieval["token_usages"]
                
                summary["retrieval"] = {
                    "count": retrieval_count,
                    "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "avg_context_size": sum(sizes) / len(sizes) if sizes else 0,
                    "avg_token_usage": sum(token_usages) / len(token_usages) if token_usages else 0,
                    "delta_percentage": (retrieval["delta_count"] / retrieval_count) * 100 if retrieval_count else 0,
                    "cache_hit_percentage": (retrieval["cache_hit_count"] / retrieval_count) * 100 if retrieval_count else 0,
                    "vector_percentage": (retrieval["vector_count"] / retrieval_count) * 100 if retrieval_count else 0
                }
        
        # Calculate maintenance metrics
        if "maintenance" in self.metrics:
            maintenance = self.metrics["maintenance"]
            maintenance_count = maintenance["count"]
            
            if maintenance_count > 0:
                # Calculate averages
                durations = maintenance["durations"]
                
                summary["maintenance"] = {
                    "count": maintenance_count,
                    "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "types": {}
                }
                
                # Add type-specific metrics
                for mtype, metrics in maintenance["types"].items():
                    type_durations = metrics["durations"]
                    summary["maintenance"]["types"][mtype] = {
                        "count": metrics["count"],
                        "avg_duration_ms": sum(type_durations) / len(type_durations) if type_durations else 0
                    }
        
        # Add performance metrics
        summary["performance"] = self.performance_metrics.get_summary()
        
        return summary

    async def save_analytics(self) -> bool:
        """
        Save analytics data to files
        
        Returns:
            Whether the save was successful
        """
        if not self.enabled:
            return False
            
        try:
            # Update last save time
            self.last_save = time.time()
            
            # Get timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics summary
            summary = self.get_analytics_summary()
            summary_file = f"{self.storage_dir}/metrics_summary_{timestamp}.json"
            
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            # Save recent events
            events_file = f"{self.storage_dir}/recent_events_{timestamp}.json"
            with open(events_file, "w") as f:
                # Save last 100 events
                json.dump(self.history[-100:], f, indent=2)
            
            # Save CSV data for analysis
            self._save_csv_data(timestamp)
            
            logger.info(f"Saved analytics data to {self.storage_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analytics: {e}")
            return False
    
    def _save_csv_data(self, timestamp: str) -> None:
        """
        Save analytics data in CSV format for easier analysis
        
        Args:
            timestamp: Timestamp for filenames
        """
        # Save retrieval events
        retrieval_events = [e for e in self.history if e["event_type"] == "retrieval"]
        if retrieval_events:
            retrieval_file = f"{self.storage_dir}/retrieval_events_{timestamp}.csv"
            
            with open(retrieval_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "timestamp", "input_digest", "location", "context_size",
                    "token_usage_total", "duration_ms", "is_delta", "is_cached", "used_vector"
                ])
                
                # Write data
                for event in retrieval_events:
                    data = event["data"]
                    token_usage = sum(data.get("token_usage", {}).values())
                    
                    writer.writerow([
                        event["timestamp"],
                        data.get("input_digest", ""),
                        data.get("location", ""),
                        data.get("context_size", 0),
                        token_usage,
                        data.get("duration_ms", 0),
                        data.get("is_delta", False),
                        data.get("is_cached", False),
                        data.get("used_vector", False)
                    ])
        
        # Save maintenance events
        maintenance_events = [e for e in self.history if e["event_type"] == "maintenance"]
        if maintenance_events:
            maintenance_file = f"{self.storage_dir}/maintenance_events_{timestamp}.csv"
            
            with open(maintenance_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "timestamp", "maintenance_type", "duration_ms", "success"
                ])
                
                # Write data
                for event in maintenance_events:
                    data = event["data"]
                    
                    writer.writerow([
                        event["timestamp"],
                        data.get("maintenance_type", "unknown"),
                        data.get("duration_ms", 0),
                        "results" in data and data["results"].get("success", False)
                    ])
    
    def _maybe_log_metrics(self):
        """Log metrics if enough time has passed"""
        if not self.logging_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            metrics = self.performance_metrics.get_summary()
            
            logger.info(
                f"Context performance metrics for user {self.user_id}:"
                f" response_time={metrics['response_time']['avg']:.4f}s,"
                f" token_usage={metrics['token_usage']['avg']:.0f},"
                f" cache_hit_rate={metrics['cache']['hit_rate']*100:.1f}%"
            )
            
            # Schedule database recording
            if self.config.get("performance", "record_to_db", False):
                asyncio.create_task(self._record_metrics_to_db(metrics))
            
            self.last_log_time = current_time
    
    async def _record_metrics_to_db(self, metrics: Dict[str, Any]) -> None:
        """Record metrics to database"""
        try:
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Insert into database
                await conn.execute("""
                    INSERT INTO ContextPerformanceMetrics
                    (user_id, conversation_id, metrics_data, recorded_at)
                    VALUES ($1, $2, $3, NOW())
                """, self.user_id, self.conversation_id, json.dumps(metrics))
                
                logger.debug(f"Recorded performance metrics to database for user {self.user_id}")
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error recording metrics to database: {e}")
    
    async def run_health_check(self) -> Dict[str, Any]:
        """
        Run a comprehensive health check of the context system
        
        Returns:
            Dictionary with health check results
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        results = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {},
            "alerts": []
        }
        
        try:
            # 1. Check cache hit rate
            analytics_summary = self.get_analytics_summary()
            retrieval = analytics_summary.get("retrieval", {})
            
            if "cache_hit_percentage" in retrieval:
                cache_hit_rate = retrieval["cache_hit_percentage"] / 100
                
                if cache_hit_rate < self.thresholds["min_cache_hit_rate"]:
                    alert = {
                        "level": "warning",
                        "message": f"Low cache hit rate: {cache_hit_rate:.2f} (threshold: {self.thresholds['min_cache_hit_rate']:.2f})",
                        "type": "performance",
                        "timestamp": datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                    results["alerts"].append(alert)
                
                results["checks"]["cache_hit_rate"] = {
                    "status": "ok" if cache_hit_rate >= self.thresholds["min_cache_hit_rate"] else "warning",
                    "value": cache_hit_rate,
                    "threshold": self.thresholds["min_cache_hit_rate"]
                }
            
            # 2. Check memory usage
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                if memory_mb > self.thresholds["max_memory_usage_mb"]:
                    alert = {
                        "level": "warning",
                        "message": f"High memory usage: {memory_mb:.2f}MB (threshold: {self.thresholds['max_memory_usage_mb']}MB)",
                        "type": "resource",
                        "timestamp": datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                    results["alerts"].append(alert)
                
                results["checks"]["memory_usage"] = {
                    "status": "ok" if memory_mb <= self.thresholds["max_memory_usage_mb"] else "warning",
                    "value": memory_mb,
                    "threshold": self.thresholds["max_memory_usage_mb"]
                }
            except ImportError:
                results["checks"]["memory_usage"] = {
                    "status": "skipped",
                    "message": "psutil not available"
                }
            
            # 3. Check database connection
            try:
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                await conn.execute("SELECT 1")
                await conn.close()
                
                results["checks"]["database"] = {
                    "status": "ok",
                    "message": "Connection successful"
                }
            except Exception as e:
                alert = {
                    "level": "error",
                    "message": f"Database connection error: {str(e)}",
                    "type": "connectivity",
                    "timestamp": datetime.now().isoformat()
                }
                self.alerts.append(alert)
                results["alerts"].append(alert)
                
                results["checks"]["database"] = {
                    "status": "error",
                    "message": str(e)
                }
            
            # 4. Check vector service if enabled
            if self.config.is_enabled("use_vector_search"):
                try:
                    from context.vector_service import get_vector_service
                    
                    vector_service = await get_vector_service(self.user_id, self.conversation_id)
                    await vector_service.initialize()
                    
                    results["checks"]["vector_service"] = {
                        "status": "ok",
                        "message": "Vector service initialized"
                    }
                except Exception as e:
                    alert = {
                        "level": "warning",
                        "message": f"Vector service error: {str(e)}",
                        "type": "service",
                        "timestamp": datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                    results["alerts"].append(alert)
                    
                    results["checks"]["vector_service"] = {
                        "status": "warning",
                        "message": str(e)
                    }
            
            # 5. Check unified cache status
            try:
                from context.unified_cache import context_cache
                
                cache_size = len(context_cache.l1_cache) + len(context_cache.l2_cache) + len(context_cache.l3_cache)
                
                results["checks"]["unified_cache"] = {
                    "status": "ok",
                    "items": cache_size,
                    "l1_items": len(context_cache.l1_cache),
                    "l2_items": len(context_cache.l2_cache),
                    "l3_items": len(context_cache.l3_cache)
                }
            except Exception as e:
                results["checks"]["unified_cache"] = {
                    "status": "warning",
                    "message": str(e)
                }
            
            # Set overall status
            if any(check.get("status") == "error" for check in results["checks"].values()):
                results["status"] = "error"
            elif any(check.get("status") == "warning" for check in results["checks"].values()):
                results["status"] = "warning"
            
            return results
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": str(e)
            }
    
    def get_alerts(
        self,
        level: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            level: Optional level filter ('error', 'warning', 'info')
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        if not self.enabled:
            return []
            
        # Filter by level if specified
        if level:
            filtered = [a for a in self.alerts if a["level"] == level]
        else:
            filtered = self.alerts
        
        # Sort by timestamp (newest first)
        sorted_alerts = sorted(filtered, key=lambda a: a["timestamp"], reverse=True)
        
        # Apply limit
        return sorted_alerts[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary with metrics
        """
        # Update memory usage before returning
        self.record_memory_usage()
        
        return self.performance_metrics.get_summary()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.performance_metrics.reset()
        logger.info(f"Reset performance metrics for user {self.user_id}, conversation {self.conversation_id}")

# -------------------------------------------------------------------------------
# Performance Tracking Decorator
# -------------------------------------------------------------------------------

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
            monitor = get_performance_monitor(user_id, conversation_id)
            
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
# Global Registry and API Functions
# -------------------------------------------------------------------------------

# Global registry for monitors
_performance_monitors = {}

def get_performance_monitor(user_id: int, conversation_id: int) -> ContextAnalytics:
    """
    Get or create a performance monitor for the specified user and conversation
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        ContextAnalytics instance
    """
    key = f"{user_id}:{conversation_id}"
    
    if key not in _performance_monitors:
        _performance_monitors[key] = ContextAnalytics(user_id, conversation_id)
    
    return _performance_monitors[key]

async def run_health_check(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Run a health check for the specified user and conversation
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Health check results
    """
    monitor = get_performance_monitor(user_id, conversation_id)
    return await monitor.run_health_check()

def record_context_retrieval(
    user_id: int,
    conversation_id: int,
    input_text: str,
    location: Optional[str],
    context_size: int,
    token_usage: Dict[str, int],
    duration_ms: int,
    is_delta: bool,
    is_cached: bool,
    used_vector: bool
) -> None:
    """
    Record a context retrieval event for analytics and health monitoring
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        input_text: User input text
        location: Current location
        context_size: Size of context in bytes
        token_usage: Token usage by component
        duration_ms: Duration in milliseconds
        is_delta: Whether it was a delta update
        is_cached: Whether it was cached
        used_vector: Whether vector search was used
    """
    # Get monitor and record event
    monitor = get_performance_monitor(user_id, conversation_id)
    
    # Record in analytics
    monitor.record_context_retrieval(
        input_text=input_text,
        location=location,
        context_size=context_size,
        token_usage=token_usage,
        duration_ms=duration_ms,
        is_delta=is_delta,
        is_cached=is_cached,
        used_vector=used_vector
    )

def get_performance_metrics(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Get performance metrics for a user and conversation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary with performance metrics
    """
    monitor = get_performance_monitor(user_id, conversation_id)
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
        import asyncpg
        
        conn = await asyncpg.connect(dsn=get_db_connection())
        try:
            # Get historical metrics
            rows = await conn.fetch("""
                SELECT metrics_data, recorded_at
                FROM ContextPerformanceMetrics
                WHERE user_id = $1 AND conversation_id = $2
                  AND recorded_at > NOW() - INTERVAL '$3 HOUR'
                ORDER BY recorded_at DESC
            """, user_id, conversation_id, hours)
            
            metrics = []
            for row in rows:
                data = json.loads(row[0])
                data["recorded_at"] = row[1].isoformat()
                metrics.append(data)
            
            return metrics
        finally:
            await conn.close()
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
        monitor = get_performance_monitor(user_id, conversation_id)
        monitor.reset_metrics()
        return True
    except Exception as e:
        logger.error(f"Error resetting performance metrics: {e}")
        return False

# Enhanced context functions with performance tracking
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
    from context.context_optimization import get_comprehensive_context
    
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
    from context.vector_service import get_vector_enhanced_context
    
    return await get_vector_enhanced_context(
        user_id=user_id,
        conversation_id=conversation_id,
        query_text=query_text,
        current_location=current_location
    )

async def cleanup_monitors():
    """Close and clean up all monitors"""
    global _performance_monitors
    
    # Save analytics for each monitor
    for key, monitor in _performance_monitors.items():
        try:
            await monitor.save_analytics()
        except Exception as e:
            logger.error(f"Error saving analytics for monitor {key}: {e}")
    
    _performance_monitors.clear()
