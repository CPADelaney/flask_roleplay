# lore/metrics.py

"""
Metrics Collection and Monitoring System

This module provides metrics collection, monitoring, and analysis capabilities
for the lore system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import time
from functools import wraps

logger = logging.getLogger(__name__)

class MetricsManager:
    """
    Manages metrics collection and analysis.
    """
    
    def __init__(self):
        self.metrics = {
            'request_count': {},
            'latency': {},
            'errors': {},
            'cache': {
                'hits': 0,
                'misses': 0
            },
            'resources': {
                'memory': [],
                'cpu': []
            }
        }
        self.monitors = {}
        self.interval = 5  # Default collection interval
        self._lock = asyncio.Lock()  # Thread-safety for async operations
    
    async def record_request(self, method: str, endpoint: str, status: int, duration: float = 0):
        """Record HTTP request metrics"""
        key = f"{method}:{endpoint}"
        
        async with self._lock:
            if key not in self.metrics['request_count']:
                self.metrics['request_count'][key] = 0
                self.metrics['latency'][key] = []
            
            self.metrics['request_count'][key] += 1
            self.metrics['latency'][key].append(duration)
            
            # Track errors
            if status >= 400:
                error_type = 'server_error' if status >= 500 else 'client_error'
                if error_type not in self.metrics['errors']:
                    self.metrics['errors'][error_type] = 0
                self.metrics['errors'][error_type] += 1
    
    async def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics"""
        async with self._lock:
            if hit:
                self.metrics['cache']['hits'] += 1
            else:
                self.metrics['cache']['misses'] += 1
    
    async def record_resource_usage(self, memory: float, cpu: float):
        """Record resource usage metrics"""
        async with self._lock:
            self.metrics['resources']['memory'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'value': memory
            })
            
            self.metrics['resources']['cpu'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'value': cpu
            })
            
            # Keep only the most recent 100 samples
            if len(self.metrics['resources']['memory']) > 100:
                self.metrics['resources']['memory'] = self.metrics['resources']['memory'][-100:]
            
            if len(self.metrics['resources']['cpu']) > 100:
                self.metrics['resources']['cpu'] = self.metrics['resources']['cpu'][-100:]
    
    async def clear_metrics(self, older_than: Optional[int] = None):
        """Clear metrics older than the specified time in seconds"""
        async with self._lock:
            if older_than is None:
                # Clear all metrics
                self.metrics = {
                    'request_count': {},
                    'latency': {},
                    'errors': {},
                    'cache': {
                        'hits': 0,
                        'misses': 0
                    },
                    'resources': {
                        'memory': [],
                        'cpu': []
                    }
                }
            else:
                # Clear only old metric data
                cutoff_time = datetime.utcnow() - timedelta(seconds=older_than)
                
                # Clear resource metrics
                self.metrics['resources']['memory'] = [
                    m for m in self.metrics['resources']['memory']
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ]
                
                self.metrics['resources']['cpu'] = [
                    m for m in self.metrics['resources']['cpu']
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ]
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        async with self._lock:
            summary = {
                'requests': {
                    'total': sum(self.metrics['request_count'].values()),
                    'by_endpoint': self.metrics['request_count'].copy()
                },
                'latency': {
                    endpoint: sum(values) / len(values) if values else 0
                    for endpoint, values in self.metrics['latency'].items()
                },
                'errors': self.metrics['errors'].copy(),
                'cache': {
                    'hits': self.metrics['cache']['hits'],
                    'misses': self.metrics['cache']['misses'],
                    'hit_rate': self.metrics['cache']['hits'] / (self.metrics['cache']['hits'] + self.metrics['cache']['misses'])
                    if (self.metrics['cache']['hits'] + self.metrics['cache']['misses']) > 0 else 0
                }
            }
            
            # Add resource usage
            if self.metrics['resources']['memory']:
                summary['resources'] = {
                    'memory': {
                        'current': self.metrics['resources']['memory'][-1]['value'],
                        'avg': sum(item['value'] for item in self.metrics['resources']['memory']) / len(self.metrics['resources']['memory'])
                    },
                    'cpu': {
                        'current': self.metrics['resources']['cpu'][-1]['value'] if self.metrics['resources']['cpu'] else 0,
                        'avg': sum(item['value'] for item in self.metrics['resources']['cpu']) / len(self.metrics['resources']['cpu'])
                        if self.metrics['resources']['cpu'] else 0
                    }
                }
            
            return summary
    
    async def start_monitor(self, name: str, metric_func: Callable, interval: int = 5):
        """Start a new metric monitor"""
        async with self._lock:
            if name in self.monitors:
                return
            
            async def monitor_task():
                while True:
                    try:
                        value = await metric_func()
                        # Store the metric with timestamp
                        if name not in self.metrics:
                            self.metrics[name] = []
                        
                        self.metrics[name].append({
                            'timestamp': datetime.utcnow().isoformat(),
                            'value': value
                        })
                        
                        # Keep only recent values
                        if len(self.metrics[name]) > 100:
                            self.metrics[name] = self.metrics[name][-100:]
                            
                    except Exception as e:
                        logger.error(f"Error in metrics monitor {name}: {e}")
                    
                    await asyncio.sleep(interval)
            
            # Start the monitor task
            self.monitors[name] = asyncio.create_task(monitor_task())
    
    async def stop_monitor(self, name: str):
        """Stop a metric monitor"""
        async with self._lock:
            if name in self.monitors:
                self.monitors[name].cancel()
                try:
                    await self.monitors[name]
                except asyncio.CancelledError:
                    pass
                del self.monitors[name]
    
    async def stop_all_monitors(self):
        """Stop all metric monitors"""
        async with self._lock:
            for name in list(self.monitors.keys()):
                await self.stop_monitor(name)

# Create global metrics manager instance
metrics_manager = MetricsManager()

#---------------------------
# Tracking Decorators
#---------------------------

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
                await metrics_manager.record_request(method, endpoint, status, duration)
        return wrapper
    return decorator

def track_operation(operation: str):
    """Decorator to track operation timing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record operation timing under custom metrics
                async with metrics_manager._lock:
                    if 'operations' not in metrics_manager.metrics:
                        metrics_manager.metrics['operations'] = {}
                    
                    if operation not in metrics_manager.metrics['operations']:
                        metrics_manager.metrics['operations'][operation] = []
                    
                    metrics_manager.metrics['operations'][operation].append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'duration': duration,
                        'success': True
                    })
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed operation
                async with metrics_manager._lock:
                    if 'operations' not in metrics_manager.metrics:
                        metrics_manager.metrics['operations'] = {}
                    
                    if operation not in metrics_manager.metrics['operations']:
                        metrics_manager.metrics['operations'][operation] = []
                    
                    metrics_manager.metrics['operations'][operation].append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'duration': duration,
                        'success': False,
                        'error': str(e)
                    })
                
                raise
        return wrapper
    return decorator

# Function to record DB query metrics (simplified from resource_manager.py)
def record_db_query(operation: str, table: str, duration: float):
    """Record database query metrics (synchronous version for backward compatibility)"""
    # This is a synchronous wrapper that can be used in synchronous contexts
    # It will create a task to run the async version
    async def _async_record():
        async with metrics_manager._lock:
            if 'db_queries' not in metrics_manager.metrics:
                metrics_manager.metrics['db_queries'] = {}
            
            key = f"{operation}:{table}"
            if key not in metrics_manager.metrics['db_queries']:
                metrics_manager.metrics['db_queries'][key] = []
            
            metrics_manager.metrics['db_queries'][key].append({
                'timestamp': datetime.utcnow().isoformat(),
                'duration': duration
            })
    
    # Create task to run async function
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_async_record())
    else:
        # For synchronous contexts where no event loop is running
        asyncio.run(_async_record())
