# lore/monitoring.py

"""System monitoring and metrics collection for the lore system."""

import logging
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import prometheus_client as prom
from prometheus_client import Counter, Gauge, Histogram, Summary
import opentelemetry as otel
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
import json
import aiohttp
import backoff
from prometheus_client.exposition import start_http_server
from .config import config

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('lore_requests_total', 'Total number of requests', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('lore_request_duration_seconds', 'Request latency in seconds', ['endpoint'])
ERROR_COUNT = Counter('lore_errors_total', 'Total number of errors', ['type'])
MEMORY_USAGE = Gauge('lore_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('lore_cpu_usage_percent', 'CPU usage percentage')
CACHE_SIZE = Gauge('lore_cache_size_bytes', 'Cache size in bytes')
ACTIVE_CONNECTIONS = Gauge('lore_active_connections', 'Number of active connections')

@dataclass
class SystemMetrics:
    """System metrics data structure."""
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
    """Performance metrics data structure."""
    timestamp: datetime
    operation: str
    duration: float
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    trace_id: Optional[str]

class AlertManager:
    """Manages system alerts and notifications"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._alerts: Dict[str, List[Dict]] = defaultdict(list)
        self._alert_history: List[Dict] = []
        self._max_history_size = 1000
    
    async def check_alerts(self, metrics: SystemMetrics):
        """Check metrics against alert thresholds"""
        alerts = []
        
        if metrics.cpu_percent > self.config.get('cpu_threshold', 80):
            alerts.append({
                'type': 'high_cpu',
                'value': metrics.cpu_percent,
                'threshold': self.config.get('cpu_threshold', 80)
            })
        
        if metrics.memory_percent > self.config.get('memory_threshold', 85):
            alerts.append({
                'type': 'high_memory',
                'value': metrics.memory_percent,
                'threshold': self.config.get('memory_threshold', 85)
            })
        
        if metrics.disk_percent > self.config.get('disk_threshold', 90):
            alerts.append({
                'type': 'high_disk',
                'value': metrics.disk_percent,
                'threshold': self.config.get('disk_threshold', 90)
            })
        
        return alerts
    
    async def send_alert(self, alert: Dict):
        """Send alert to configured notification channels"""
        if 'webhook_url' in self.config:
            await self._send_webhook_alert(alert)
        if 'email' in self.config:
            await self._send_email_alert(alert)
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError)
    async def _send_webhook_alert(self, alert: Dict):
        """Send alert to webhook"""
        async with aiohttp.ClientSession() as session:
            await session.post(
                self.config['webhook_url'],
                json=alert
            )
    
    async def _send_email_alert(self, alert: Dict):
        """Send alert via email"""
        # Implement email sending logic here
        pass

class SystemMonitor:
    """System monitoring and metrics collection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_monitoring = False
        self._monitoring_tasks = []
        self._metrics_history: Dict[str, List[Any]] = defaultdict(list)
        self._max_history_size = 1000
        self._alert_manager = AlertManager(config)
        
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
        """Initialize the system monitor"""
        try:
            self._is_monitoring = True
            self._monitoring_tasks = [
                asyncio.create_task(self._collect_system_metrics()),
                asyncio.create_task(self._cleanup_old_metrics()),
                asyncio.create_task(self._check_alerts())
            ]
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup monitoring resources"""
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
        MEMORY_USAGE.set(metrics.memory_percent)
        CPU_USAGE.set(metrics.cpu_percent)
        CACHE_SIZE.set(metrics.cache_size)
        ACTIVE_CONNECTIONS.set(metrics.active_connections)
    
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
    
    async def _check_alerts(self):
        """Check for alerts periodically"""
        while self._is_monitoring:
            try:
                metrics = await self._gather_system_metrics()
                alerts = await self._alert_manager.check_alerts(metrics)
                for alert in alerts:
                    await self._alert_manager.send_alert(alert)
                await asyncio.sleep(self.config.get('alert_check_interval', 300))
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(60)
    
    def _get_cache_size(self) -> int:
        """Get current cache size"""
        # Implement cache size calculation
        return 0
    
    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        # Implement connection counting
        return 0
    
    def record_request(self, endpoint: str, method: str, duration: float, success: bool):
        """Record a request metric"""
        REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        if not success:
            ERROR_COUNT.labels(type='request').inc()
    
    def record_error(self, error_type: str):
        """Record an error metric"""
        ERROR_COUNT.labels(type=error_type).inc()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        return {
            'cpu_usage': CPU_USAGE._value.get(),
            'memory_usage': MEMORY_USAGE._value.get(),
            'cache_size': CACHE_SIZE._value.get(),
            'active_connections': ACTIVE_CONNECTIONS._value.get(),
            'request_count': REQUEST_COUNT._value.get(),
            'error_count': ERROR_COUNT._value.get()
        }

class MetricsManager:
    """
    Unified metrics and monitoring system.
    Provides Prometheus metrics collection and monitoring capabilities.
    """
    
    def __init__(self):
        self.config = config.get_metrics_config()
        self._initialize_metrics()
        if self.config['enabled']:
            self._start_server()
            
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        # Request metrics
        self.request_count = Counter(
            'lore_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_latency = Histogram(
            'lore_request_duration_seconds',
            'Request latency in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Database metrics
        self.db_query_count = Counter(
            'lore_db_queries_total',
            'Total number of database queries',
            ['operation', 'table']
        )
        
        self.db_query_latency = Histogram(
            'lore_db_query_duration_seconds',
            'Database query latency in seconds',
            ['operation', 'table'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.db_connection_pool = Gauge(
            'lore_db_connections',
            'Number of database connections in pool',
            ['state']
        )
        
        # Cache metrics
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
        
        self.cache_size = Gauge(
            'lore_cache_size_bytes',
            'Current size of cache in bytes',
            ['cache_type']
        )
        
        # Error metrics
        self.error_count = Counter(
            'lore_errors_total',
            'Total number of errors',
            ['error_type', 'severity']
        )
        
        # Resource metrics
        self.memory_usage = Gauge(
            'lore_memory_usage_bytes',
            'Current memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'lore_cpu_usage_percent',
            'Current CPU usage percentage'
        )
        
        # Business metrics
        self.active_users = Gauge(
            'lore_active_users',
            'Number of currently active users'
        )
        
        self.world_count = Gauge(
            'lore_worlds_total',
            'Total number of worlds'
        )
        
        self.npc_count = Gauge(
            'lore_npcs_total',
            'Total number of NPCs'
        )
        
        self.location_count = Gauge(
            'lore_locations_total',
            'Total number of locations'
        )
        
    def _start_server(self):
        """Start the Prometheus metrics server."""
        try:
            start_http_server(
                self.config['port'],
                addr=self.config['host']
            )
            logger.info(f"Metrics server started on {self.config['host']}:{self.config['port']}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
            
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.request_latency.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    def record_db_query(self, operation: str, table: str, duration: float):
        """Record database query metrics."""
        self.db_query_count.labels(
            operation=operation,
            table=table
        ).inc()
        
        self.db_query_latency.labels(
            operation=operation,
            table=table
        ).observe(duration)
        
    def update_db_connections(self, active: int, idle: int):
        """Update database connection pool metrics."""
        self.db_connection_pool.labels(state='active').set(active)
        self.db_connection_pool.labels(state='idle').set(idle)
        
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics."""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
            
    def update_cache_size(self, cache_type: str, size_bytes: int):
        """Update cache size metric."""
        self.cache_size.labels(cache_type=cache_type).set(size_bytes)
        
    def record_error(self, error_type: str, severity: str):
        """Record error metrics."""
        self.error_count.labels(
            error_type=error_type,
            severity=severity
        ).inc()
        
    def update_resource_metrics(self):
        """Update system resource metrics."""
        import psutil
        
        process = psutil.Process()
        self.memory_usage.set(process.memory_info().rss)
        self.cpu_usage.set(process.cpu_percent())
        
    def update_business_metrics(self, metrics: Dict[str, int]):
        """Update business-related metrics."""
        if 'active_users' in metrics:
            self.active_users.set(metrics['active_users'])
        if 'world_count' in metrics:
            self.world_count.set(metrics['world_count'])
        if 'npc_count' in metrics:
            self.npc_count.set(metrics['npc_count'])
        if 'location_count' in metrics:
            self.location_count.set(metrics['location_count'])
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            'requests': {
                'total': self.request_count._value.sum(),
                'latency_avg': self.request_latency._sum.sum() / self.request_latency._count.sum()
                if self.request_latency._count.sum() > 0 else 0
            },
            'database': {
                'queries_total': self.db_query_count._value.sum(),
                'latency_avg': self.db_query_latency._sum.sum() / self.db_query_latency._count.sum()
                if self.db_query_latency._count.sum() > 0 else 0
            },
            'cache': {
                'hits': self.cache_hits._value.sum(),
                'misses': self.cache_misses._value.sum(),
                'hit_rate': self.cache_hits._value.sum() / (self.cache_hits._value.sum() + self.cache_misses._value.sum())
                if (self.cache_hits._value.sum() + self.cache_misses._value.sum()) > 0 else 0
            },
            'errors': {
                'total': self.error_count._value.sum(),
                'by_type': dict(zip(self.error_count._labels, self.error_count._value))
            },
            'resources': {
                'memory_usage': self.memory_usage._value.get(),
                'cpu_usage': self.cpu_usage._value.get()
            }
        }

# Create global metrics instance
metrics = MetricsManager() 
