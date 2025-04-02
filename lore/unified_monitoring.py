"""
Unified Monitoring System

Centralizes all metrics collection, monitoring, and alerting for the lore system.
Supports Prometheus, OpenTelemetry, and custom metrics.
"""

import logging
import psutil
import asyncio
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import prometheus_client as prom
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.exposition import start_http_server
import json
import aiohttp
import backoff
import os
import gc
import sys
import time
import hashlib
from functools import wraps

# OpenTelemetry imports
import opentelemetry as otel
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource

from .config import config

logger = logging.getLogger(__name__)

#---------------------------
# Dataclasses for metrics
#---------------------------

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

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    operation: str
    duration: float
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    trace_id: Optional[str]

#---------------------------
# Prometheus Metrics
#---------------------------

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
ACTIVE_CONNECTIONS = Gauge('lore_active_connections', 'Number of active connections')

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

# Business Metrics
ACTIVE_USERS = Gauge('lore_active_users', 'Number of currently active users')
WORLD_COUNT = Gauge('lore_worlds_total', 'Total number of worlds')
NPC_COUNT = Gauge('lore_npcs_total', 'Total number of NPCs')
LOCATION_COUNT = Gauge('lore_locations_total', 'Total number of locations')

# Database Metrics
DB_QUERY_COUNT = Counter('lore_db_queries_total', 'Total number of database queries', ['operation', 'table'])
DB_QUERY_LATENCY = Histogram('lore_db_query_duration_seconds', 'Database query latency in seconds', ['operation', 'table'])
DB_CONNECTION_POOL = Gauge('lore_db_connections', 'Number of database connections in pool', ['state'])

#---------------------------
# Alert Manager
#---------------------------

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._alerts: Dict[str, List[Dict]] = defaultdict(list)
        self._alert_history: List[Dict] = []
        self._max_history_size = config.get('max_alert_history', 1000)
    
    async def check_alerts(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check metrics against alert thresholds"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.config.get('cpu_threshold', 80):
            alerts.append({
                'type': 'high_cpu',
                'value': metrics.cpu_percent,
                'threshold': self.config.get('cpu_threshold', 80),
                'message': f"High CPU usage: {metrics.cpu_percent:.1f}% exceeds threshold of {self.config.get('cpu_threshold', 80)}%"
            })
        
        # Memory alert
        if metrics.memory_percent > self.config.get('memory_threshold', 85):
            alerts.append({
                'type': 'high_memory',
                'value': metrics.memory_percent,
                'threshold': self.config.get('memory_threshold', 85),
                'message': f"High memory usage: {metrics.memory_percent:.1f}% exceeds threshold of {self.config.get('memory_threshold', 85)}%"
            })
        
        # Disk alert
        if metrics.disk_percent > self.config.get('disk_threshold', 90):
            alerts.append({
                'type': 'high_disk',
                'value': metrics.disk_percent,
                'threshold': self.config.get('disk_threshold', 90),
                'message': f"High disk usage: {metrics.disk_percent:.1f}% exceeds threshold of {self.config.get('disk_threshold', 90)}%"
            })
        
        # Store alerts in history
        for alert in alerts:
            self._store_alert(alert)
            
        return alerts
    
    def _store_alert(self, alert: Dict[str, Any]):
        """Store alert in history"""
        alert_with_timestamp = {
            **alert,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._alert_history.append(alert_with_timestamp)
        
        # Trim history if needed
        if len(self._alert_history) > self._max_history_size:
            self._alert_history = self._alert_history[-self._max_history_size:]
            
        # Group by type for easy access
        self._alerts[alert['type']].append(alert_with_timestamp)
        
        # Trim type-specific history as well
        if len(self._alerts[alert['type']]) > self._max_history_size:
            self._alerts[alert['type']] = self._alerts[alert['type']][-self._max_history_size:]
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert to configured notification channels"""
        try:
            if 'webhook_url' in self.config:
                await self._send_webhook_alert(alert)
                
            if 'email' in self.config:
                await self._send_email_alert(alert)
                
            if 'log_alerts' in self.config and self.config['log_alerts']:
                self._log_alert(alert)
                
            return True
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
            return False
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _send_webhook_alert(self, alert: Dict[str, Any]):
        """Send alert to webhook with retry logic"""
        async with aiohttp.ClientSession() as session:
            webhook_data = {
                **alert,
                'timestamp': datetime.utcnow().isoformat(),
                'system': self.config.get('system_name', 'lore-system')
            }
            
            await session.post(
                self.config['webhook_url'],
                json=webhook_data,
                headers={'Content-Type': 'application/json'}
            )
    
    async def _send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email"""
        # Email sending implementation would go here
        # This is a placeholder for actual email sending logic
        pass
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to system logs"""
        logger.warning(f"SYSTEM ALERT: {alert['type']} - {alert.get('message', '')}")
    
    def get_recent_alerts(self, alert_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts, optionally filtered by type"""
        if alert_type:
            return self._alerts.get(alert_type, [])[-limit:]
        else:
            return self._alert_history[-limit:]

#---------------------------
# Unified Metrics Manager
#---------------------------

class MetricsManager:
    """
    Unified metrics and monitoring system.
    Provides metrics collection, monitoring, and alerting capabilities.
    """
    
    def __init__(self):
        self.config = config.get_metrics_config()
        self._initialize_metrics()
        self._is_monitoring = False
        self._monitoring_tasks = []
        self._metrics_history = defaultdict(list)
        self._max_history_size = self.config.get('max_metrics_history', 1000)
        self._operation_timings = {}
        self._batch_sizes = {}
        self._queue_sizes = {}
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        
        # Initialize OpenTelemetry
        self._setup_tracing()
        self._setup_metrics()
        
        if self.config.get('enabled', True):
            self._start_server()
    
    def _initialize_metrics(self):
        """Initialize all metrics"""
        # All metrics are already defined as global variables
        pass
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        if not self.config.get('otel_enabled', False):
            return
            
        resource = Resource.create({"service.name": "lore_system"})
        tracer_provider = TracerProvider(resource=resource)
        
        if self.config.get('jaeger_enabled', False):
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
        if not self.config.get('otel_enabled', False):
            return
            
        resource = Resource.create({"service.name": "lore_system"})
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)
    
    def _start_server(self):
        """Start the Prometheus metrics server."""
        try:
            port = self.config.get('port', 9090)
            host = self.config.get('host', '0.0.0.0')
            start_http_server(port, addr=host)
            logger.info(f"Metrics server started on {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
    
    async def initialize(self):
        """Initialize the metrics manager"""
        try:
            self._is_monitoring = True
            self._monitoring_tasks = [
                asyncio.create_task(self._collect_system_metrics()),
                asyncio.create_task(self._cleanup_old_metrics()),
                asyncio.create_task(self._check_alerts())
            ]
            logger.info("Metrics and monitoring system initialized")
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
        logger.info("Metrics and monitoring system shutdown complete")
    
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
        
        process = psutil.Process()
        
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
        MEMORY_USAGE.set(metrics.memory_percent)
        CPU_USAGE.set(metrics.cpu_percent)
        THREAD_COUNT.set(metrics.thread_count)
        
        # Attempt to get open file descriptors if platform supports it
        try:
            FILE_DESCRIPTORS.set(len(psutil.Process().open_files()))
        except (AttributeError, NotImplementedError):
            # Not supported on all platforms
            pass
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in history"""
        for key, value in metrics.__dict__.items():
            if key != 'timestamp':
                if isinstance(value, dict):
                    # Store each sub-item in the dictionary
                    for sub_key, sub_value in value.items():
                        self._metrics_history[f"{key}.{sub_key}"].append({
                            'timestamp': metrics.timestamp,
                            'value': sub_value
                        })
                else:
                    # Store the value directly
                    self._metrics_history[key].append({
                        'timestamp': metrics.timestamp,
                        'value': value
                    })
                    
                # Trim history if needed
                if len(self._metrics_history[key]) > self._max_history_size:
                    self._metrics_history[key] = self._metrics_history[key][-self._max_history_size:]
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics periodically"""
        while self._is_monitoring:
            try:
                cutoff = datetime.utcnow() - timedelta(hours=24)
                for key in list(self._metrics_history.keys()):
                    self._metrics_history[key] = [
                        m for m in self._metrics_history[key]
                        if m['timestamp'] > cutoff
                    ]
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error cleaning up metrics: {e}")
                await asyncio.sleep(300)
    
    async def _check_alerts(self):
        """Check for alerts periodically"""
        while self._is_monitoring:
            try:
                metrics = await self._gather_system_metrics()
                alerts = await self.alert_manager.check_alerts(metrics)
                for alert in alerts:
                    await self.alert_manager.send_alert(alert)
                await asyncio.sleep(self.config.get('alert_check_interval', 300))
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(60)
    
    def _get_cache_size(self) -> int:
        """Get current cache size"""
        # Add implementation based on your cache system
        # This is a placeholder
        try:
            cache_sum = sum(CACHE_SIZE._value.get(label=l) for l in CACHE_SIZE._labelnames)
            return cache_sum
        except:
            return 0
    
    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        # Add implementation based on your database connection pool
        # This is a placeholder
        try:
            return ACTIVE_CONNECTIONS._value.get()
        except:
            return 0
    
    def record_operation_start(self, operation: str):
        """Record the start of an operation"""
        self._operation_timings[operation] = time.time()
    
    def record_operation_end(self, operation: str):
        """Record the end of an operation and update metrics"""
        if operation in self._operation_timings:
            duration = time.time() - self._operation_timings[operation]
            OPERATION_LATENCY.labels(operation=operation).observe(duration)
            del self._operation_timings[operation]
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        
        if status >= 400:
            error_type = 'server_error' if status >= 500 else 'client_error'
            REQUEST_ERRORS.labels(error_type=error_type).inc()
    
    def record_error(self, error_type: str, recovered: bool = False):
        """Record an error metric"""
        ERROR_COUNT.labels(error_type=error_type, recovered=str(recovered)).inc()
    
    def record_cache_operation(self, cache_type: str, hit: bool, size_bytes: Optional[int] = None):
        """Record cache operation metrics"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
        
        if size_bytes is not None:
            CACHE_SIZE.labels(cache_type=cache_type).set(size_bytes)
    
    def record_db_query(self, operation: str, table: str, duration: float):
        """Record database query metrics"""
        DB_QUERY_COUNT.labels(
            operation=operation,
            table=table
        ).inc()
        
        DB_QUERY_LATENCY.labels(
            operation=operation,
            table=table
        ).observe(duration)
    
    def update_db_connections(self, active: int, idle: int):
        """Update database connection pool metrics"""
        DB_CONNECTION_POOL.labels(state='active').set(active)
        DB_CONNECTION_POOL.labels(state='idle').set(idle)
    
    def update_batch_size(self, operation: str, size: int):
        """Update batch size metric"""
        self._batch_sizes[operation] = size
        BATCH_SIZE.labels(operation=operation).set(size)
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size metric"""
        self._queue_sizes[queue_name] = size
        QUEUE_SIZE.labels(queue_name=queue_name).set(size)
    
    def record_task_metrics(self, task_name: str, duration: float, success: bool, retries: int = 0):
        """Record task execution metrics"""
        if success:
            TASK_SUCCESS.labels(task_name=task_name).inc()
        else:
            TASK_FAILURES.labels(task_name=task_name).inc()
        
        if retries > 0:
            TASK_RETRIES.labels(task_name=task_name).inc(retries)
        
        TASK_LATENCY.labels(task_name=task_name).observe(duration)
    
    def update_business_metrics(self, metrics: Dict[str, int]):
        """Update business-related metrics"""
        if 'active_users' in metrics:
            ACTIVE_USERS.set(metrics['active_users'])
        if 'world_count' in metrics:
            WORLD_COUNT.set(metrics['world_count'])
        if 'npc_count' in metrics:
            NPC_COUNT.set(metrics['npc_count'])
        if 'location_count' in metrics:
            LOCATION_COUNT.set(metrics['location_count'])
    
    def update_resource_metrics(self):
        """Update system resource metrics"""
        process = psutil.Process()
        MEMORY_USAGE.set(process.memory_info().rss)
        CPU_USAGE.set(process.cpu_percent())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
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
                'average_validation_time': self._calculate_average(VALIDATION_TIME),
                'error_count': VALIDATION_ERRORS._value.sum()
            },
            'cache': {
                'hits': CACHE_HITS._value.sum(),
                'misses': CACHE_MISSES._value.sum(),
                'hit_rate': self._calculate_hit_rate(),
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
                'average_recovery_time': self._calculate_average(ERROR_RECOVERY_TIME)
            },
            'performance': {
                'operation_latencies': self._get_operation_latencies(),
                'batch_sizes': self._batch_sizes.copy(),
                'queue_sizes': self._queue_sizes.copy()
            },
            'tasks': {
                'success': TASK_SUCCESS._value.sum(),
                'failures': TASK_FAILURES._value.sum(),
                'retries': TASK_RETRIES._value.sum(),
                'queued': TASKS_QUEUED._value.sum()
            },
            'database': {
                'queries_total': DB_QUERY_COUNT._value.sum(),
                'average_query_time': self._calculate_average(DB_QUERY_LATENCY)
            }
        }
    
    def _calculate_average(self, histogram) -> float:
        """Calculate average from a Histogram"""
        try:
            return histogram._sum.sum() / histogram._count.sum() if histogram._count.sum() > 0 else 0
        except (AttributeError, ZeroDivisionError):
            return 0
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            hits = CACHE_HITS._value.sum()
            misses = CACHE_MISSES._value.sum()
            total = hits + misses
            return hits / total if total > 0 else 0
        except (AttributeError, ZeroDivisionError):
            return 0
    
    def _get_operation_latencies(self) -> Dict[str, float]:
        """Get average latencies for operations"""
        try:
            return {
                op: OPERATION_LATENCY._sum.get(op) / OPERATION_LATENCY._count.get(op) 
                if OPERATION_LATENCY._count.get(op, 0) > 0 else 0
                for op in OPERATION_LATENCY._sum._values
            }
        except (AttributeError, ZeroDivisionError):
            return {}

#---------------------------
# Decorator utilities
#---------------------------

def track_operation(operation: str):
    """Decorator to track operation timing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = metrics_manager
            metrics.record_operation_start(operation)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                metrics.record_operation_end(operation)
        return wrapper
    return decorator

def track_db_query(operation: str, table: str):
    """Decorator to track database query timing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics_manager.record_db_query(operation, table, duration)
        return wrapper
    return decorator

def track_batch(operation: str):
    """Decorator to track batch operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            if isinstance(result, (list, tuple)):
                metrics_manager.update_batch_size(operation, len(result))
            return result
        return wrapper
    return decorator

def track_request(endpoint: str, method: str):
    """Decorator to track HTTP request timing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = getattr(result, 'status_code', 200)
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = time.time() - start_time
                metrics_manager.record_request(method, endpoint, status, duration)
        return wrapper
    return decorator

def track_task(task_name: str):
    """Decorator to track task execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            retries = 0
            success = False
            try:
                # Get retries from kwargs if provided
                retries = kwargs.pop('__retries', 0)
                result = await func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                metrics_manager.record_task_metrics(task_name, duration, success, retries)
        return wrapper
    return decorator

#---------------------------
# Global instance
#---------------------------

# Create global metrics instance
metrics_manager = MetricsManager()
