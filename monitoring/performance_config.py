"""
Performance monitoring configuration and setup.
Integrates Prometheus metrics, DataDog, and NewRelic monitoring.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

# Import database connections
from db.connection import get_db_connection_context
from monitoring.metrics import metrics

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    enable_prometheus: bool = True
    enable_datadog: bool = False
    enable_newrelic: bool = False
    slow_request_threshold: float = 1.0  # seconds
    memory_warning_threshold: float = 85.0  # percentage
    cpu_warning_threshold: float = 80.0  # percentage
    request_timeout: float = 30.0  # seconds

# DataDog configuration
DATADOG_CONFIG = {
    'api_key': os.getenv('DD_API_KEY'),
    'app_key': os.getenv('DD_APP_KEY'),
    'service_name': 'npc-roleplay',
    'env': os.getenv('FLASK_ENV', 'development'),
    'version': '1.0.0',
    'tags': [
        'service:npc-roleplay',
        f"env:{os.getenv('FLASK_ENV', 'development')}"
    ]
}

# NewRelic configuration
NEWRELIC_CONFIG = {
    'license_key': os.getenv('NEW_RELIC_LICENSE_KEY'),
    'app_name': 'NPC Roleplay',
    'environment': os.getenv('FLASK_ENV', 'development'),
    'distributed_tracing': {
        'enabled': True
    },
    'transaction_tracer': {
        'enabled': True,
        'transaction_threshold': 0.5
    },
    'error_collector': {
        'enabled': True
    }
}

async def init_performance_monitoring(app) -> None:
    """Initialize performance monitoring for the application."""
    config = PerformanceConfig()
    
    if config.enable_prometheus:
        logger.info("Initializing Prometheus monitoring")
        from prometheus_flask_exporter import PrometheusMetrics
        metrics = PrometheusMetrics(app)
        metrics.info('app_info', 'Application info', version='1.0.0')
    
    if config.enable_datadog and DATADOG_CONFIG['api_key']:
        logger.info("Initializing DataDog monitoring")
        from ddtrace import patch_all, tracer
        patch_all()
        tracer.configure(
            hostname=os.getenv('DD_AGENT_HOST', 'localhost'),
            port=int(os.getenv('DD_AGENT_PORT', 8126))
        )
    
    if config.enable_newrelic and NEWRELIC_CONFIG['license_key']:
        logger.info("Initializing NewRelic monitoring")
        import newrelic.agent
        newrelic.agent.initialize()

async def record_operation_metrics(operation_type: str, duration: float, success: bool) -> None:
    """Record operation metrics."""
    namespace = metrics()

    if operation_type.startswith('npc_'):
        namespace.NPC_OPERATIONS.labels(operation_type=operation_type).inc()
    elif operation_type.startswith('memory_'):
        namespace.MEMORY_OPERATIONS.labels(operation_type=operation_type).inc()
        namespace.MEMORY_OPERATION_LATENCY.labels(operation_type=operation_type).observe(duration)

    namespace.REQUEST_LATENCY.labels(
        method='operation',
        endpoint=operation_type
    ).observe(duration)

async def update_system_metrics() -> None:
    """Update system metrics (memory and CPU usage)."""
    import psutil

    namespace = metrics()

    # Update memory usage
    memory = psutil.virtual_memory()
    namespace.SYSTEM_MEMORY_USAGE.set(memory.used)

    # Update CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    namespace.SYSTEM_CPU_USAGE.set(cpu_percent)
    
    # Check thresholds
    config = PerformanceConfig()
    if memory.percent > config.memory_warning_threshold:
        logger.warning(f"High memory usage: {memory.percent}%")
    
    if cpu_percent > config.cpu_warning_threshold:
        logger.warning(f"High CPU usage: {cpu_percent}%")
        
    # Check database connection count
    try:
        async with get_db_connection_context() as conn:
            result = await conn.fetchrow("SELECT COUNT(*) FROM pg_stat_activity")
            if result and result[0]:
                db_connections = result[0]
                logger.info(f"Current database connections: {db_connections}")
                
                # Check if connections are high
                if db_connections > 50:  # Threshold for warning
                    logger.warning(f"High number of database connections: {db_connections}")
    except Exception as e:
        logger.error(f"Error checking database connections: {e}")
