"""
Unified Monitoring with Resource Management

This module provides unified monitoring capabilities with integrated resource management.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List, Set
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, field
import psutil
import asyncio
from collections import defaultdict
import json
import threading
from prometheus_client import Counter, Histogram, Gauge, Summary
from .base_manager import BaseManager
from .resource_manager import resource_manager

logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Configuration for metrics collection"""
    enabled: bool = True
    collection_interval: int = 60  # seconds
    retention_period: int = 86400  # 24 hours in seconds
    max_samples: int = 1000
    buckets: List[float] = field(default_factory=lambda: [
        0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0
    ])

class MetricsManager:
    """Manages system-wide metrics collection"""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self._initialize_metrics()
        self._metrics_data = defaultdict(list)
        self._last_cleanup = datetime.utcnow()
        self._lock = threading.Lock()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            'lore_requests_total',
            'Total number of requests',
            ['method', 'endpoint']
        )
        self.request_latency = Histogram(
            'lore_request_duration_seconds',
            'Request latency in seconds',
            ['method', 'endpoint'],
            buckets=self.config.buckets
        )
        
        # Error metrics
        self.error_counter = Counter(
            'lore_errors_total',
            'Total number of errors',
            ['type', 'component']
        )
        
        # Resource metrics
        self.memory_gauge = Gauge(
            'lore_memory_usage_bytes',
            'Memory usage in bytes'
        )
        self.cpu_gauge = Gauge(
            'lore_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Cache metrics
        self.cache_size = Gauge(
            'lore_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type']
        )
        self.cache_hits = Counter(
            'lore_cache_hits_total',
            'Total number of cache hits',
            ['cache_type']
        )
        self.cache_misses = Counter(
            'lore_cache_misses_total',
            'Total number of cache misses',
            ['cache_type']
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'lore_db_connections_active',
            'Number of active database connections'
        )
        self.query_latency = Histogram(
            'lore_query_duration_seconds',
            'Query execution time in seconds',
            ['query_type'],
            buckets=self.config.buckets
        )
        
        # Business metrics
        self.entity_counter = Counter(
            'lore_entities_total',
            'Total number of entities',
            ['entity_type']
        )
        self.operation_counter = Counter(
            'lore_operations_total',
            'Total number of operations',
            ['operation_type']
        )
    
    async def collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # Memory usage
                memory = psutil.Process().memory_info()
                self.memory_gauge.set(memory.rss)
                
                # CPU usage
                cpu_percent = psutil.Process().cpu_percent()
                self.cpu_gauge.set(cpu_percent)
                
                # Cleanup old metrics if needed
                self._cleanup_old_metrics()
                
                await asyncio.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        with self._lock:
            now = datetime.utcnow()
            if (now - self._last_cleanup).total_seconds() < self.config.collection_interval:
                return
            
            cutoff = now - timedelta(seconds=self.config.retention_period)
            for metric_type in self._metrics_data:
                self._metrics_data[metric_type] = [
                    m for m in self._metrics_data[metric_type]
                    if m['timestamp'] > cutoff
                ]
            
            self._last_cleanup = now
    
    def track_request(self, method: str, endpoint: str):
        """Decorator for tracking request metrics"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    self.request_counter.labels(method=method, endpoint=endpoint).inc()
                    return result
                finally:
                    duration = time.time() - start_time
                    self.request_latency.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
            return wrapper
        return decorator
    
    def track_operation(self, operation_type: str):
        """Decorator for tracking operation metrics"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    self.operation_counter.labels(
                        operation_type=operation_type
                    ).inc()
                    return result
                except Exception as e:
                    self.error_counter.labels(
                        type=type(e).__name__,
                        component=operation_type
                    ).inc()
                    raise
            return wrapper
        return decorator
    
    def track_query(self, query_type: str):
        """Decorator for tracking database query metrics"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.query_latency.labels(
                        query_type=query_type
                    ).observe(duration)
            return wrapper
        return decorator
    
    def record_cache_operation(
        self,
        cache_type: str,
        hit: bool,
        size: Optional[int] = None
    ):
        """Record cache operation metrics"""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
        
        if size is not None:
            self.cache_size.labels(cache_type=cache_type).set(size)
    
    def record_entity_operation(self, entity_type: str):
        """Record entity operation metrics"""
        self.entity_counter.labels(entity_type=entity_type).inc()
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'memory_usage': self.memory_gauge._value.get(),
                'cpu_usage': self.cpu_gauge._value.get()
            },
            'requests': {
                'total': self.request_counter._value.get(),
                'latency_avg': self.request_latency._sum.get() / max(self.request_latency._count.get(), 1)
            },
            'errors': self.error_counter._value.get(),
            'cache': {
                'hits': self.cache_hits._value.get(),
                'misses': self.cache_misses._value.get()
            },
            'database': {
                'connections': self.db_connections._value.get(),
                'query_latency_avg': self.query_latency._sum.get() / max(self.query_latency._count.get(), 1)
            }
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        data = self.get_metrics_snapshot()
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'prometheus':
            # Convert to Prometheus format
            lines = []
            for metric, value in data.items():
                if isinstance(value, dict):
                    for submetric, subvalue in value.items():
                        lines.append(f"{metric}_{submetric} {subvalue}")
                else:
                    lines.append(f"{metric} {value}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global metrics manager instance
metrics_manager = MetricsManager(MetricConfig())

# Convenience decorators
def track_request(method: str, endpoint: str):
    """Decorator for tracking request metrics"""
    return metrics_manager.track_request(method, endpoint)

def track_operation(operation_type: str):
    """Decorator for tracking operation metrics"""
    return metrics_manager.track_operation(operation_type)

def track_query(query_type: str):
    """Decorator for tracking query metrics"""
    return metrics_manager.track_query(query_type)

class UnifiedMonitoring(BaseManager):
    """Manager for unified monitoring with resource management support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.monitoring_data = {}
        self.resource_manager = resource_manager
    
    async def start(self):
        """Start the unified monitoring manager and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the unified monitoring manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_monitoring_data(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get monitoring data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('monitoring', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting monitoring data: {e}")
            return None
    
    async def set_monitoring_data(
        self,
        data_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set monitoring data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('monitoring', data_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting monitoring data: {e}")
            return False
    
    async def invalidate_monitoring_data(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate monitoring data cache."""
        try:
            await self.invalidate_cached_data('monitoring', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating monitoring data: {e}")
    
    async def get_monitoring_history(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get monitoring history from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('monitoring_history', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting monitoring history: {e}")
            return None
    
    async def set_monitoring_history(
        self,
        data_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set monitoring history in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('monitoring_history', data_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting monitoring history: {e}")
            return False
    
    async def invalidate_monitoring_history(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate monitoring history cache."""
        try:
            await self.invalidate_cached_data('monitoring_history', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating monitoring history: {e}")
    
    async def get_monitoring_metadata(
        self,
        data_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get monitoring metadata from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('monitoring_metadata', data_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting monitoring metadata: {e}")
            return None
    
    async def set_monitoring_metadata(
        self,
        data_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set monitoring metadata in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('monitoring_metadata', data_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting monitoring metadata: {e}")
            return False
    
    async def invalidate_monitoring_metadata(
        self,
        data_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate monitoring metadata cache."""
        try:
            await self.invalidate_cached_data('monitoring_metadata', data_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating monitoring metadata: {e}")
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

# Create a singleton instance for easy access
unified_monitoring = UnifiedMonitoring() 