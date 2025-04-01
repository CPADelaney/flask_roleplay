# lore/metrics.py

"""
Unified Metrics System

Centralizes all metrics collection and reporting for the lore system.
Supports Prometheus, OpenTelemetry, and custom metrics.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Dict, Any, Optional, List, Union
from functools import wraps
import time
import psutil
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import opentelemetry as otel
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
import structlog

logger = structlog.get_logger(__name__)

# System Metrics
SYSTEM_MEMORY_USAGE = Gauge('lore_system_memory_bytes', 'Current system memory usage in bytes')
SYSTEM_CPU_USAGE = Gauge('lore_system_cpu_percent', 'Current CPU usage percentage')
SYSTEM_DISK_USAGE = Gauge('lore_system_disk_percent', 'Current disk usage percentage')
SYSTEM_NETWORK_IO = Gauge('lore_system_network_bytes', 'Network I/O in bytes', ['direction'])
SYSTEM_PROCESS_COUNT = Gauge('lore_system_process_count', 'Number of running processes')
SYSTEM_THREAD_COUNT = Gauge('lore_system_thread_count', 'Number of active threads')

# Request Metrics
REQUEST_COUNT = Counter('lore_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('lore_request_duration_seconds', 'Request latency in seconds', ['endpoint'])
REQUEST_ERRORS = Counter('lore_request_errors_total', 'Total number of request errors', ['error_type'])

# Validation Metrics
VALIDATION_COUNT = Counter('lore_validations_total', 'Total number of validations', ['status'])
VALIDATION_TIME = Histogram('lore_validation_duration_seconds', 'Validation duration in seconds')
VALIDATION_ERRORS = Counter('lore_validation_errors_total', 'Total number of validation errors', ['error_type'])

# Cache Metrics
CACHE_HITS = Counter('lore_cache_hits_total', 'Total number of cache hits', ['cache_type'])
CACHE_MISSES = Counter('lore_cache_misses_total', 'Total number of cache misses', ['cache_type'])
CACHE_SIZE = Gauge('lore_cache_size_bytes', 'Current cache size in bytes', ['cache_type'])
CACHE_EVICTIONS = Counter('lore_cache_evictions_total', 'Total number of cache evictions', ['cache_type', 'reason'])
CACHE_LATENCY = Histogram('lore_cache_operation_seconds', 'Cache operation latency in seconds', ['operation'])

# Resource Metrics
MEMORY_USAGE = Gauge('lore_memory_usage_bytes', 'Current memory usage in bytes')
CPU_USAGE = Gauge('lore_cpu_usage_percent', 'Current CPU usage percentage')
THREAD_COUNT = Gauge('lore_thread_count', 'Current number of threads')
FILE_DESCRIPTORS = Gauge('lore_file_descriptors', 'Current number of open file descriptors')

# Circuit Breaker Metrics
CIRCUIT_BREAKER_STATE = Gauge('lore_circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open)')
CIRCUIT_BREAKER_TRIPS = Counter('lore_circuit_breaker_trips_total', 'Total number of circuit breaker trips', ['breaker_id'])

# Rate Limiting Metrics
RATE_LIMIT_HITS = Counter('lore_rate_limit_hits_total', 'Total number of rate limit hits', ['limit_key'])
RATE_LIMIT_REQUESTS = Counter('lore_rate_limit_requests_total', 'Total number of rate-limited requests', ['limit_key'])

# Error Metrics
ERROR_COUNT = Counter('lore_errors_total', 'Total number of errors', ['error_type', 'recovered'])
ERROR_RECOVERY_TIME = Histogram('lore_error_recovery_seconds', 'Error recovery time in seconds')

# Performance Metrics
OPERATION_LATENCY = Summary('lore_operation_latency_seconds', 'Operation latency in seconds', ['operation'])
BATCH_SIZE = Gauge('lore_batch_size', 'Current batch size', ['operation'])
QUEUE_SIZE = Gauge('lore_queue_size', 'Current queue size', ['queue_name'])

# Task Metrics
TASK_LATENCY = Histogram('lore_task_latency_seconds', 'Task execution time in seconds', ['task_name'])
TASK_FAILURES = Counter('lore_task_failures_total', 'Number of failed tasks', ['task_name'])
TASK_RETRIES = Counter('lore_task_retries_total', 'Number of task retries', ['task_name'])
TASK_SUCCESS = Counter('lore_task_success_total', 'Number of successful tasks', ['task_name'])
TASKS_QUEUED = Gauge('lore_tasks_queued', 'Number of tasks in queue', ['queue_name'])

@dataclass
class SystemMetrics:
    """System metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    cache_size: int
    active_connections: int

class MetricsManager:
    """Manages metrics collection and reporting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_monitoring = False
        self._monitoring_tasks = []
        self._metrics_history: Dict[str, List[Any]] = defaultdict(list)
        self._max_history_size = 1000
        self._operation_timings: Dict[str, float] = {}
        self._batch_sizes: Dict[str, int] = {}
        self._queue_sizes: Dict[str, int] = {}
        
        # Initialize OpenTelemetry
        self._setup_tracing()
        self._setup_metrics()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        resource = Resource.create({"service.name": "lore_system"})
        tracer_provider = TracerProvider(resource=resource)
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.get('jaeger_host', 'localhost'),
            agent_port=self.config.get('jaeger_port', 6831)
        )
        tracer_provider.add_span_processor(
            otel.sdk.trace.BatchSpanProcessor(jaeger_exporter)
        )
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
    
    def _setup_metrics(self):
        """Setup OpenTelemetry metrics"""
        resource = Resource.create({"service.name": "lore_system"})
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)
    
    async def initialize(self):
        """Initialize the metrics manager"""
        try:
            self._is_monitoring = True
            self._monitoring_tasks = [
                asyncio.create_task(self._collect_system_metrics()),
                asyncio.create_task(self._cleanup_old_metrics())
            ]
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup metrics resources"""
        self._is_monitoring = False
        for task in self._monitoring_tasks:
            task.cancel()
        try:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self._is_monitoring:
            try:
                metrics = await self._gather_system_metrics()
                self._update_prometheus_metrics(metrics)
                self._store_metrics(metrics)
                await asyncio.sleep(self.config.get('collection_interval', 60))
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)
    
    async def _gather_system_metrics(self) -> SystemMetrics:
        """Gather current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io={
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            },
            process_count=len(psutil.pids()),
            thread_count=sum(p.num_threads() for p in psutil.process_iter(['num_threads'])),
            cache_size=self._get_cache_size(),
            active_connections=self._get_active_connections()
        )
    
    def _update_prometheus_metrics(self, metrics: SystemMetrics):
        """Update Prometheus metrics"""
        SYSTEM_MEMORY_USAGE.set(metrics.memory_percent)
        SYSTEM_CPU_USAGE.set(metrics.cpu_percent)
        SYSTEM_DISK_USAGE.set(metrics.disk_percent)
        SYSTEM_NETWORK_IO.labels(direction='sent').set(metrics.network_io['bytes_sent'])
        SYSTEM_NETWORK_IO.labels(direction='received').set(metrics.network_io['bytes_recv'])
        SYSTEM_PROCESS_COUNT.set(metrics.process_count)
        SYSTEM_THREAD_COUNT.set(metrics.thread_count)
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in history"""
        for key, value in metrics.__dict__.items():
            if key != 'timestamp':
                self._metrics_history[key].append(value)
                if len(self._metrics_history[key]) > self._max_history_size:
                    self._metrics_history[key].pop(0)
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics periodically"""
        while self._is_monitoring:
            try:
                cutoff = datetime.utcnow() - timedelta(hours=24)
                for key in list(self._metrics_history.keys()):
                    self._metrics_history[key] = [
                        m for m in self._metrics_history[key]
                        if m.timestamp > cutoff
                    ]
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error cleaning up metrics: {e}")
                await asyncio.sleep(300)
    
    def record_operation_start(self, operation: str):
        """Record the start of an operation"""
        self._operation_timings[operation] = time.time()
    
    def record_operation_end(self, operation: str):
        """Record the end of an operation and update metrics"""
        if operation in self._operation_timings:
            duration = time.time() - self._operation_timings[operation]
            OPERATION_LATENCY.labels(operation=operation).observe(duration)
            del self._operation_timings[operation]
    
    def update_batch_size(self, operation: str, size: int):
        """Update batch size metric"""
        self._batch_sizes[operation] = size
        BATCH_SIZE.labels(operation=operation).set(size)
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size metric"""
        self._queue_sizes[queue_name] = size
        QUEUE_SIZE.labels(queue_name=queue_name).set(size)
    
    def record_request(self, endpoint: str, method: str, duration: float, status: int):
        """Record a request metric"""
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    
    def record_error(self, error_type: str, recovered: bool = False):
        """Record an error metric"""
        ERROR_COUNT.labels(
            error_type=error_type,
            recovered=recovered
        ).inc()
    
    def record_cache_operation(self, cache_type: str, hit: bool, size_bytes: Optional[int] = None):
        """Record a cache operation"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
        
        if size_bytes is not None:
            CACHE_SIZE.labels(cache_type=cache_type).set(size_bytes)
    
    def record_task_metrics(self, task_name: str, duration: float, success: bool, retries: int = 0):
        """Record task execution metrics"""
        if success:
            TASK_SUCCESS.labels(task_name=task_name).inc()
        else:
            TASK_FAILURES.labels(task_name=task_name).inc()
        
        if retries > 0:
            TASK_RETRIES.labels(task_name=task_name).inc(retries)
        
        TASK_LATENCY.labels(task_name=task_name).observe(duration)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        return {
            'system': {
                'memory_usage': SYSTEM_MEMORY_USAGE._value.get(),
                'cpu_usage': SYSTEM_CPU_USAGE._value.get(),
                'disk_usage': SYSTEM_DISK_USAGE._value.get(),
                'network_io': {
                    'sent': SYSTEM_NETWORK_IO.labels(direction='sent')._value.get(),
                    'received': SYSTEM_NETWORK_IO.labels(direction='received')._value.get()
                },
                'process_count': SYSTEM_PROCESS_COUNT._value.get(),
                'thread_count': SYSTEM_THREAD_COUNT._value.get()
            },
            'validation': {
                'total_validations': VALIDATION_COUNT._value.sum(),
                'average_validation_time': VALIDATION_TIME._sum.sum() / VALIDATION_TIME._count.sum() if VALIDATION_TIME._count.sum() > 0 else 0,
                'error_count': VALIDATION_ERRORS._value.sum()
            },
            'cache': {
                'hits': CACHE_HITS._value.sum(),
                'misses': CACHE_MISSES._value.sum(),
                'total_size': CACHE_SIZE._value.sum(),
                'evictions': CACHE_EVICTIONS._value.sum()
            },
            'resources': {
                'memory_usage': MEMORY_USAGE._value.get(),
                'cpu_usage': CPU_USAGE._value.get(),
                'thread_count': THREAD_COUNT._value.get(),
                'file_descriptors': FILE_DESCRIPTORS._value.get()
            },
            'circuit_breakers': {
                'active_breakers': len(CIRCUIT_BREAKER_STATE._value),
                'total_trips': CIRCUIT_BREAKER_TRIPS._value.sum()
            },
            'rate_limits': {
                'total_hits': RATE_LIMIT_HITS._value.sum(),
                'total_requests': RATE_LIMIT_REQUESTS._value.sum()
            },
            'errors': {
                'total_errors': ERROR_COUNT._value.sum(),
                'average_recovery_time': ERROR_RECOVERY_TIME._sum.sum() / ERROR_RECOVERY_TIME._count.sum() if ERROR_RECOVERY_TIME._count.sum() > 0 else 0
            },
            'performance': {
                'operation_latencies': {
                    op: OPERATION_LATENCY._sum[op] / OPERATION_LATENCY._count[op] if op in OPERATION_LATENCY._count else 0
                    for op in OPERATION_LATENCY._sum
                },
                'batch_sizes': self._batch_sizes.copy(),
                'queue_sizes': self._queue_sizes.copy()
            },
            'tasks': {
                'success': TASK_SUCCESS._value.sum(),
                'failures': TASK_FAILURES._value.sum(),
                'retries': TASK_RETRIES._value.sum(),
                'queued': TASKS_QUEUED._value.sum()
            }
        }
    
    def _get_cache_size(self) -> int:
        """Get current cache size"""
        return CACHE_SIZE._value.sum()
    
    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        return ACTIVE_CONNECTIONS._value.get()

def track_operation(operation: str):
    """Decorator to track operation timing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = MetricsManager()
            metrics.record_operation_start(operation)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                metrics.record_operation_end(operation)
        return wrapper
    return decorator

def track_batch(operation: str):
    """Decorator to track batch operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = MetricsManager()
            result = await func(*args, **kwargs)
            if isinstance(result, (list, tuple)):
                metrics.update_batch_size(operation, len(result))
            return result
        return wrapper
    return decorator 
