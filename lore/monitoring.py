# lore/monitoring.py

"""
Monitoring System

Provides metrics collection, monitoring, and alerting capabilities for the lore system.
"""

import logging
from typing import Dict, Any, Callable
from datetime import datetime
import time
from functools import wraps

logger = logging.getLogger(__name__)

class MetricsManager:
    """
    Simplified metrics and monitoring system.
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
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float = 0):
        """Record HTTP request metrics"""
        key = f"{method}:{endpoint}"
        
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
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics"""
        if hit:
            self.metrics['cache']['hits'] += 1
        else:
            self.metrics['cache']['misses'] += 1
    
    def record_resource_usage(self, memory: float, cpu: float):
        """Record resource usage metrics"""
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
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
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

# Create global metrics manager
metrics_manager = MetricsManager()

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

def track_operation(operation: str):
    """Decorator to track operation timing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            # Record operation timing in a real system
            return result
        return wrapper
    return decorator
