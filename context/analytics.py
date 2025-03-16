# context/analytics.py

"""
Performance analytics and monitoring for the context optimization system.

This module provides tools to track, analyze, and visualize the performance
of the context system over time, helping to identify bottlenecks and
optimization opportunities for long-running games.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib
import math
import os
import csv

from context.context_config import get_config

logger = logging.getLogger(__name__)

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
    
    def _get_storage_dir(self) -> str:
        """Get the directory to store analytics data"""
        base_dir = self.config.get("performance", "analytics_dir", "analytics")
        user_dir = f"{base_dir}/user_{self.user_id}/conversation_{self.conversation_id}"
        return user_dir
    
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

class ContextHealthMonitor:
    """
    Monitor for tracking context system health and detecting issues
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.analytics = ContextAnalytics(user_id, conversation_id)
        self.thresholds = {
            "max_retrieval_time_ms": self.config.get("performance", "max_retrieval_time_ms", 1000),
            "max_token_usage": self.config.get("performance", "max_token_usage", 6000),
            "min_cache_hit_rate": self.config.get("performance", "min_cache_hit_rate", 0.3),
            "max_memory_usage_mb": self.config.get("performance", "max_memory_usage_mb", 500)
        }
        self.alerts = []
        self.max_alerts = 100
        self.enabled = self.config.get("performance", "health_monitoring_enabled", True)
    
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
            analytics_summary = self.analytics.get_analytics_summary()
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
                import psutil
                
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

# Global registry for monitors
_health_monitors = {}

def get_health_monitor(user_id: int, conversation_id: int) -> ContextHealthMonitor:
    """
    Get or create a health monitor for the specified user and conversation
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        ContextHealthMonitor instance
    """
    key = f"{user_id}:{conversation_id}"
    
    if key not in _health_monitors:
        _health_monitors[key] = ContextHealthMonitor(user_id, conversation_id)
    
    return _health_monitors[key]

async def run_health_check(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Run a health check for the specified user and conversation
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Health check results
    """
    monitor = get_health_monitor(user_id, conversation_id)
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
    # Get health monitor and record event
    monitor = get_health_monitor(user_id, conversation_id)
    
    # Record in analytics
    monitor.analytics.record_context_retrieval(
        input_text=input_text,
        location=location,
        context_size=context_size,
        token_usage=token_usage,
        duration_ms=duration_ms,
        is_delta=is_delta,
        is_cached=is_cached,
        used_vector=used_vector
    )
    
    # Check performance
    monitor.check_retrieval_performance(
        duration_ms=duration_ms,
        token_usage=token_usage,
        is_cached=is_cached
    )

async def cleanup_monitors():
    """Close and clean up all health monitors"""
    global _health_monitors
    
    # Save analytics for each monitor
    for key, monitor in _health_monitors.items():
        try:
            await monitor.analytics.save_analytics()
        except Exception as e:
            logger.error(f"Error saving analytics for monitor {key}: {e}")
    
    _health_monitors.clear()
