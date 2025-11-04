# monitoring/metrics.py

import logging
import time
from functools import lru_cache, wraps
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Type, Union

import psutil
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Summary

# Import database connections
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

CollectorType = Union[Counter, Gauge, Histogram, Summary]

_COLLECTOR_DEFINITIONS: Dict[str, Tuple[Type[CollectorType], Tuple[Any, ...], Dict[str, Any]]] = {
    # Request metrics
    "REQUEST_LATENCY": (
        Histogram,
        ("request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]),
        {},
    ),
    "REQUEST_COUNT": (
        Counter,
        ("request_count_total", "Total number of requests", ["method", "endpoint", "status"]),
        {},
    ),
    # Memory system metrics
    "MEMORY_CLEANUP_DURATION": (
        Histogram,
        ("memory_cleanup_duration_seconds", "Time taken for memory cleanup operation"),
        {},
    ),
    "MEMORY_OPERATION_LATENCY": (
        Histogram,
        ("memory_operation_latency_seconds", "Memory operation latency in seconds", ["operation_type"]),
        {},
    ),
    "MEMORY_OPERATIONS": (
        Counter,
        ("memory_operations_total", "Total number of memory operations", ["operation_type"]),
        {},
    ),
    "MEMORY_SIZE": (
        Gauge,
        ("memory_size_total", "Total number of memories stored"),
        {},
    ),
    # NPC metrics
    "NPC_OPERATIONS": (
        Counter,
        ("npc_operations_total", "Total number of NPC operations", ["operation_type"]),
        {},
    ),
    "NPC_INTERACTION_LATENCY": (
        Histogram,
        ("npc_interaction_latency_seconds", "NPC interaction latency in seconds", ["interaction_type"]),
        {},
    ),
    "NPC_COUNT": (
        Gauge,
        ("npc_count_total", "Total number of active NPCs"),
        {},
    ),
    # System metrics
    "SYSTEM_MEMORY_USAGE": (
        Gauge,
        ("system_memory_usage_bytes", "Current system memory usage"),
        {},
    ),
    "SYSTEM_CPU_USAGE": (
        Gauge,
        ("system_cpu_usage_percent", "Current CPU usage percentage"),
        {},
    ),
    "DB_CONNECTION_COUNT": (
        Gauge,
        ("db_connection_count", "Number of active database connections"),
        {},
    ),
    # Cache metrics
    "CACHE_HIT_COUNT": (
        Counter,
        ("cache_hit_total", "Total number of cache hits", ["cache_type"]),
        {},
    ),
    "CACHE_MISS_COUNT": (
        Counter,
        ("cache_miss_total", "Total number of cache misses", ["cache_type"]),
        {},
    ),
    "CACHE_SIZE": (
        Gauge,
        ("cache_size_bytes", "Current cache size in bytes", ["cache_type"]),
        {},
    ),
    "CONFLICT_ROUTER_DECISIONS": (
        Counter,
        (
            "conflict_router_decisions_total",
            "Conflict subsystem routing decisions by source",
            ["source"],
        ),
        {},
    ),
    "CONFLICT_ROUTER_TIMEOUTS": (
        Counter,
        (
            "conflict_router_timeouts_total",
            "Number of orchestrator routing timeouts",
            ["source"],
        ),
        {},
    ),
    "MODE_RECOMMENDATION_SOURCE": (
        Counter,
        (
            "mode_recommendation_source_total",
            "Count of mode recommendation selections by source",
            ["source"],
        ),
        {},
    ),
    "CONFLICT_TEMPLATE_WARMUPS": (
        Counter,
        (
            "conflict_template_warmups_total",
            "Total number of conflict template warmup executions",
            ["stage", "result"],
        ),
        {},
    ),
    "CONFLICT_TEMPLATE_CACHE_PENDING": (
        Gauge,
        (
            "conflict_template_cache_pending",
            "Pending conflict template cache entries",
            ["stage"],
        ),
        {},
    ),
    "LOCATION_ROUTER_DECISIONS": (
        Counter,
        (
            "location_router_decisions_total",
            "Location router decision outcomes",
            ["outcome"],
        ),
        {},
    ),
}


def _get_registry_collectors() -> Dict[str, CollectorType]:
    """Return the registry collectors mapping for reuse."""

    return getattr(REGISTRY, "_names_to_collectors", {})


def _get_or_create_collector(
    collector_cls: Type[CollectorType],
    *args: Any,
    **kwargs: Any,
) -> CollectorType:
    """Fetch an existing collector or create a new one."""

    collectors = _get_registry_collectors()
    name = args[0] if args else kwargs.get("name")
    if name:
        existing = collectors.get(name)
        if existing is not None:
            if not isinstance(existing, collector_cls):
                raise TypeError(
                    f"Collector '{name}' already registered with type {type(existing).__name__}, "
                    f"expected {collector_cls.__name__}."
                )
            return existing

    return collector_cls(*args, **kwargs)


@lru_cache(maxsize=1)
def metrics() -> SimpleNamespace:
    """Return a singleton namespace containing all Prometheus collectors."""

    namespace: Dict[str, CollectorType] = {}
    for attr_name, (collector_cls, collector_args, collector_kwargs) in _COLLECTOR_DEFINITIONS.items():
        namespace[attr_name] = _get_or_create_collector(
            collector_cls, *collector_args, **collector_kwargs
        )

    return SimpleNamespace(**namespace)


def _resolve_metric(metric_ref: Union[str, CollectorType]) -> CollectorType:
    """Resolve a metric reference to an actual Prometheus collector."""

    if isinstance(metric_ref, str):
        collector = getattr(metrics(), metric_ref, None)
        if collector is None:
            raise AttributeError(f"Metric '{metric_ref}' is not defined in monitoring.metrics")
        return collector
    return metric_ref


def track_latency(metric: Union[str, Histogram], labels: Optional[Dict[str, str]] = None):
    """Decorator to track operation latency."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            collector = _resolve_metric(metric)
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
            except Exception:
                duration = time.time() - start_time
                if labels:
                    collector.labels(**labels).observe(duration)
                else:
                    collector.observe(duration)
                raise
            else:
                duration = time.time() - start_time
                if labels:
                    collector.labels(**labels).observe(duration)
                else:
                    collector.observe(duration)
                return result

        return wrapper

    return decorator


async def update_system_metrics():
    """Update system-level metrics."""

    try:
        # Update memory usage
        memory = psutil.virtual_memory()
        metrics().SYSTEM_MEMORY_USAGE.set(memory.used)

        # Update CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics().SYSTEM_CPU_USAGE.set(cpu_percent)

        # Log warnings for high resource usage
        if memory.percent > 85:
            logger.warning(f"High memory usage: {memory.percent}%")
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent}%")

    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")


async def record_request_metric(method: str, endpoint: str, status: int, duration: float):
    """Record metrics for an HTTP request."""

    namespace = metrics()
    namespace.REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=status
    ).inc()

    namespace.REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


async def record_memory_operation(operation_type: str, duration: float):
    """Record metrics for a memory operation."""

    namespace = metrics()
    namespace.MEMORY_OPERATIONS.labels(
        operation_type=operation_type
    ).inc()

    namespace.MEMORY_OPERATION_LATENCY.labels(
        operation_type=operation_type
    ).observe(duration)


async def record_npc_operation(operation_type: str, duration: float):
    """Record metrics for an NPC operation."""

    namespace = metrics()
    namespace.NPC_OPERATIONS.labels(
        operation_type=operation_type
    ).inc()

    namespace.NPC_INTERACTION_LATENCY.labels(
        interaction_type=operation_type
    ).observe(duration)


async def record_cache_operation(cache_type: str, hit: bool, size_bytes: Optional[int] = None):
    """Record metrics for a cache operation."""

    namespace = metrics()
    if hit:
        namespace.CACHE_HIT_COUNT.labels(cache_type=cache_type).inc()
    else:
        namespace.CACHE_MISS_COUNT.labels(cache_type=cache_type).inc()

    if size_bytes is not None:
        namespace.CACHE_SIZE.labels(cache_type=cache_type).set(size_bytes)


async def get_current_metrics() -> Dict[str, Any]:
    """Get current values for all metrics."""

    namespace = metrics()

    # Get DB connection count
    db_connection_count = 0
    try:
        async with get_db_connection_context() as conn:
            result = await conn.fetchrow("SELECT COUNT(*) FROM pg_stat_activity")
            if result and result[0]:
                db_connection_count = result[0]
    except Exception as e:
        logger.error(f"Error getting DB connection count: {e}")

    # Update DB_CONNECTION_COUNT gauge
    namespace.DB_CONNECTION_COUNT.set(db_connection_count)

    return {
        "system": {
            "memory_usage_bytes": namespace.SYSTEM_MEMORY_USAGE._value.get(),
            "cpu_usage_percent": namespace.SYSTEM_CPU_USAGE._value.get(),
            "db_connections": db_connection_count
        },
        "requests": {
            "total": sum(namespace.REQUEST_COUNT._metrics.values()),
            "latency_avg": namespace.REQUEST_LATENCY._metrics.get("avg", 0)
        },
        "memory_operations": {
            "cleanup_duration_avg": namespace.MEMORY_CLEANUP_DURATION._metrics.get("avg", 0),
            "total_memories": namespace.MEMORY_SIZE._value.get(),
            "operations_total": sum(namespace.MEMORY_OPERATIONS._metrics.values()),
        },
        "npc_operations": {
            "total": sum(namespace.NPC_OPERATIONS._metrics.values()),
            "active_npcs": namespace.NPC_COUNT._value.get()
        },
        "cache": {
            "hits": sum(namespace.CACHE_HIT_COUNT._metrics.values()),
            "misses": sum(namespace.CACHE_MISS_COUNT._metrics.values())
        }
    }
