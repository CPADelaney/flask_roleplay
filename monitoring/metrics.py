# monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge, Summary
import time
import psutil
import logging
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Request metrics
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

REQUEST_COUNT = Counter(
    'request_count_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

# Memory system metrics
MEMORY_CLEANUP_DURATION = Histogram(
    'memory_cleanup_duration_seconds',
    'Time taken for memory cleanup operation'
)

MEMORY_OPERATION_LATENCY = Histogram(
    'memory_operation_latency_seconds',
    'Memory operation latency in seconds',
    ['operation_type']
)

MEMORY_SIZE = Gauge(
    'memory_size_total',
    'Total number of memories stored'
)

# NPC metrics
NPC_OPERATIONS = Counter(
    'npc_operations_total',
    'Total number of NPC operations',
    ['operation_type']
)

NPC_INTERACTION_LATENCY = Histogram(
    'npc_interaction_latency_seconds',
    'NPC interaction latency in seconds',
    ['interaction_type']
)

NPC_COUNT = Gauge(
    'npc_count_total',
    'Total number of active NPCs'
)

# System metrics
SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'Current system memory usage'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

DB_CONNECTION_COUNT = Gauge(
    'db_connection_count',
    'Number of active database connections'
)

# Cache metrics
CACHE_HIT_COUNT = Counter(
    'cache_hit_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISS_COUNT = Counter(
    'cache_miss_total',
    'Total number of cache misses',
    ['cache_type']
)

CACHE_SIZE = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type']
)

def track_latency(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """Decorator to track operation latency."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
                raise
        return wrapper
    return decorator

def update_system_metrics():
    """Update system-level metrics."""
    try:
        # Update memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory.used)
        
        # Update CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        SYSTEM_CPU_USAGE.set(cpu_percent)
        
        # Log warnings for high resource usage
        if memory.percent > 85:
            logger.warning(f"High memory usage: {memory.percent}%")
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent}%")
            
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")

def record_request_metric(method: str, endpoint: str, status: int, duration: float):
    """Record metrics for an HTTP request."""
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=status
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)

def record_memory_operation(operation_type: str, duration: float):
    """Record metrics for a memory operation."""
    MEMORY_OPERATION_LATENCY.labels(
        operation_type=operation_type
    ).observe(duration)

def record_npc_operation(operation_type: str, duration: float):
    """Record metrics for an NPC operation."""
    NPC_OPERATIONS.labels(
        operation_type=operation_type
    ).inc()
    
    NPC_INTERACTION_LATENCY.labels(
        interaction_type=operation_type
    ).observe(duration)

def record_cache_operation(cache_type: str, hit: bool, size_bytes: Optional[int] = None):
    """Record metrics for a cache operation."""
    if hit:
        CACHE_HIT_COUNT.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISS_COUNT.labels(cache_type=cache_type).inc()
    
    if size_bytes is not None:
        CACHE_SIZE.labels(cache_type=cache_type).set(size_bytes)

def get_current_metrics() -> Dict[str, Any]:
    """Get current values for all metrics."""
    return {
        "system": {
            "memory_usage_bytes": SYSTEM_MEMORY_USAGE._value.get(),
            "cpu_usage_percent": SYSTEM_CPU_USAGE._value.get(),
            "db_connections": DB_CONNECTION_COUNT._value.get()
        },
        "requests": {
            "total": sum(REQUEST_COUNT._metrics.values()),
            "latency_avg": REQUEST_LATENCY._metrics.get("avg", 0)
        },
        "memory_operations": {
            "cleanup_duration_avg": MEMORY_CLEANUP_DURATION._metrics.get("avg", 0),
            "total_memories": MEMORY_SIZE._value.get()
        },
        "npc_operations": {
            "total": sum(NPC_OPERATIONS._metrics.values()),
            "active_npcs": NPC_COUNT._value.get()
        },
        "cache": {
            "hits": sum(CACHE_HIT_COUNT._metrics.values()),
            "misses": sum(CACHE_MISS_COUNT._metrics.values())
        }
    } 
