# lore/resource_manager.py

"""
Unified Resource Management and Monitoring System

This module provides comprehensive resource management and monitoring capabilities
for the lore system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime, timedelta
import time
from dataclasses import dataclass, field
from functools import wraps
import psutil
import os
import json
import sys
import gc
from .base_manager import BaseManager

logger = logging.getLogger(__name__)

#---------------------------
# Configuration Dataclasses
#---------------------------

@dataclass
class CacheConfig:
    """Cache configuration"""
    name: str
    max_size: int
    ttl: int
    eviction_policy: str = "lru"
    max_memory_usage: float = 0.8
    monitoring_enabled: bool = True

@dataclass
class ResourceConfig:
    """Resource management configuration"""
    caches: Dict[str, CacheConfig] = field(default_factory=dict)
    cleanup_interval: int = 300
    validation_batch_size: int = 50
    performance_monitoring: bool = True
    rate_limit_requests: int = 100
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    monitoring_interval: int = 60

#---------------------------
# Metrics Collection Classes
#---------------------------

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
    
    def clear_metrics(self, older_than: Optional[int] = None):
        """Clear metrics older than the specified time in seconds"""
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
            cutoff_timestamp = cutoff_time.timestamp()
            
            # Clear resource metrics
            self.metrics['resources']['memory'] = [
                m for m in self.metrics['resources']['memory']
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            self.metrics['resources']['cpu'] = [
                m for m in self.metrics['resources']['cpu']
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            # Other metrics don't have timestamps, can't filter by age
    
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
    
    def start_monitor(self, name: str, metric_func: Callable, interval: int = 5):
        """Start a new metric monitor"""
        if name in self.monitors:
            return
        
        async def monitor_task():
            while True:
                try:
                    value = metric_func()
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
    
    def stop_monitor(self, name: str):
        """Stop a metric monitor"""
        if name in self.monitors:
            self.monitors[name].cancel()
            del self.monitors[name]
    
    def stop_all_monitors(self):
        """Stop all metric monitors"""
        for name in list(self.monitors.keys()):
            self.stop_monitor(name)

# Create global metrics manager
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
                metrics_manager.record_request(method, endpoint, status, duration)
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

#---------------------------
# Resource Manager Class
#---------------------------

class ResourceManager(BaseManager):
    """Unified Resource Management System with monitoring and optimization capabilities."""
    
    def __init__(
        self,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None,
        config: Optional[ResourceConfig] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.config = config or ResourceConfig()
        self.resource_metrics = {
            "memory": {
                "current": 0,
                "peak": 0,
                "threshold": 0.8,  # 80% threshold
                "history": []
            },
            "cpu": {
                "current": 0,
                "peak": 0,
                "threshold": 0.7,  # 70% threshold
                "history": []
            },
            "storage": {
                "current": 0,
                "peak": 0,
                "threshold": 0.9,  # 90% threshold
                "history": []
            }
        }
        self.optimization_strategies = {
            "memory": self._optimize_memory,
            "cpu": self._optimize_cpu,
            "storage": self._optimize_storage
        }
        self.cleanup_tasks = []
        self.resource_locks = {}
        self.optimization_queue = asyncio.Queue()
        self.cleanup_queue = asyncio.Queue()
        self.metrics_manager = metrics_manager
        self._tasks = []
        
    async def start(self):
        """Start the resource manager and monitoring."""
        await super().start()
        self._start_monitoring()
        self._start_optimization()
        self._start_cleanup()
        
        # Start standard metrics monitors
        self.metrics_manager.start_monitor('memory', self._get_memory_usage)
        self.metrics_manager.start_monitor('cpu', self._get_cpu_usage)
        self.metrics_manager.start_monitor('disk', self._get_disk_usage)
        
        return True
        
    async def stop(self):
        """Stop the resource manager and cleanup."""
        await super().stop()
        await self._stop_monitoring()
        await self._stop_optimization()
        await self._stop_cleanup()
        
        # Stop metrics monitors
        self.metrics_manager.stop_all_monitors()
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        return True
        
    def _start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring_task = asyncio.create_task(self._monitor_resources())
        self._tasks.append(self.monitoring_task)
        
    def _start_optimization(self):
        """Start resource optimization."""
        self.optimization_task = asyncio.create_task(self._process_optimization_queue())
        self._tasks.append(self.optimization_task)
        
    def _start_cleanup(self):
        """Start resource cleanup."""
        self.cleanup_task = asyncio.create_task(self._process_cleanup_queue())
        self._tasks.append(self.cleanup_task)
        
    async def _stop_monitoring(self):
        """Stop resource monitoring."""
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
    async def _stop_optimization(self):
        """Stop resource optimization."""
        if hasattr(self, 'optimization_task'):
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
                
    async def _stop_cleanup(self):
        """Stop resource cleanup."""
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage."""
        try:
            return psutil.Process().memory_percent()
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage as percentage."""
        try:
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Get current disk usage as percentage."""
        try:
            return psutil.disk_usage('/').percent
        except:
            return 0.0
                
    async def _monitor_resources(self):
        """Monitor system resources."""
        while True:
            try:
                # Get current resource usage
                memory_usage = self._get_memory_usage()
                cpu_usage = self._get_cpu_usage()
                storage_usage = self._get_disk_usage()
                
                # Update metrics
                self._update_resource_metrics("memory", memory_usage)
                self._update_resource_metrics("cpu", cpu_usage)
                self._update_resource_metrics("storage", storage_usage)
                
                # Record in metrics manager too
                self.metrics_manager.record_resource_usage(memory_usage, cpu_usage)
                
                # Check thresholds and trigger optimization if needed
                await self._check_resource_thresholds()
                
                # Wait before next check
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
                
    def _update_resource_metrics(self, resource_type: str, current_value: float):
        """Update resource metrics."""
        metrics = self.resource_metrics[resource_type]
        metrics["current"] = current_value
        metrics["peak"] = max(metrics["peak"], current_value)
        metrics["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": current_value
        })
        
        # Keep history limited
        if len(metrics["history"]) > 1000:
            metrics["history"] = metrics["history"][-1000:]
            
    async def _check_resource_thresholds(self):
        """Check if any resources exceed their thresholds."""
        for resource_type, metrics in self.resource_metrics.items():
            if metrics["current"] > metrics["threshold"]:
                await self.optimization_queue.put({
                    "type": resource_type,
                    "current_value": metrics["current"],
                    "threshold": metrics["threshold"]
                })
                
    async def _process_optimization_queue(self):
        """Process optimization requests from the queue."""
        while True:
            try:
                request = await self.optimization_queue.get()
                resource_type = request["type"]
                
                if resource_type in self.optimization_strategies:
                    await self.optimization_strategies[resource_type](request)
                    
                self.optimization_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing optimization request: {e}")
                
    async def _process_cleanup_queue(self):
        """Process cleanup requests from the queue."""
        while True:
            try:
                request = await self.cleanup_queue.get()
                resource_type = request["type"]
                
                if resource_type in self.optimization_strategies:
                    await self._cleanup_resources(resource_type, request)
                    
                self.cleanup_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing cleanup request: {e}")
    
    async def _check_resource_availability(self, resource_type: str) -> bool:
        """
        Check if a resource is available for use.
        
        Args:
            resource_type: Type of resource (memory, cpu, storage)
            
        Returns:
            True if resource is available, False otherwise
        """
        if resource_type == "memory":
            current_memory = self._get_memory_usage()
            return current_memory < self.resource_metrics["memory"]["threshold"]
        elif resource_type == "cpu":
            current_cpu = self._get_cpu_usage()
            return current_cpu < self.resource_metrics["cpu"]["threshold"]
        elif resource_type == "storage":
            current_storage = self._get_disk_usage()
            return current_storage < self.resource_metrics["storage"]["threshold"]
        
        return True
                
    async def _optimize_memory(self, request: Dict[str, Any]):
        """Optimize memory usage."""
        try:
            # Get current memory usage
            current_memory = request["current_value"]
            
            # Check if we need to optimize
            if current_memory > self.resource_metrics["memory"]["threshold"]:
                # Clear caches
                await self._clear_memory_caches()
                
                # Force garbage collection
                gc.collect()
                
                # Check if we need more aggressive optimization
                if self._get_memory_usage() > self.resource_metrics["memory"]["threshold"]:
                    await self._aggressive_memory_optimization()
                    
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            
    async def _optimize_cpu(self, request: Dict[str, Any]):
        """Optimize CPU usage."""
        try:
            # Get current CPU usage
            current_cpu = request["current_value"]
            
            # Check if we need to optimize
            if current_cpu > self.resource_metrics["cpu"]["threshold"]:
                # Adjust task priorities
                await self._adjust_task_priorities()
                
                # Check if we need more aggressive optimization
                if self._get_cpu_usage() > self.resource_metrics["cpu"]["threshold"]:
                    await self._aggressive_cpu_optimization()
                    
        except Exception as e:
            logger.error(f"Error optimizing CPU: {e}")
            
    async def _optimize_storage(self, request: Dict[str, Any]):
        """Optimize storage usage."""
        try:
            # Get current storage usage
            current_storage = request["current_value"]
            
            # Check if we need to optimize
            if current_storage > self.resource_metrics["storage"]["threshold"]:
                # Clean up temporary files
                await self._cleanup_temp_files()
                
                # Check if we need more aggressive optimization
                if self._get_disk_usage() > self.resource_metrics["storage"]["threshold"]:
                    await self._aggressive_storage_optimization()
                    
        except Exception as e:
            logger.error(f"Error optimizing storage: {e}")
            
    async def _clear_memory_caches(self):
        """Clear memory caches."""
        try:
            # Clear Redis cache if available
            if hasattr(self, 'redis_client'):
                await self.redis_client.flushdb()
                
            # Clear in-memory caches
            self.resource_metrics["memory"]["history"] = []
            
            # Clear other caches
            for cache in self._get_active_caches():
                await cache.clear()
                
        except Exception as e:
            logger.error(f"Error clearing memory caches: {e}")
    
    def _get_active_caches(self) -> List[Any]:
        """Get list of active caches."""
        # This would be implemented based on your cache management
        return []
            
    async def _aggressive_memory_optimization(self):
        """Perform aggressive memory optimization."""
        try:
            # Clear all caches
            await self._clear_memory_caches()
            
            # Force garbage collection
            gc.collect()
            
            # Clear unused objects
            await self._clear_unused_objects()
            
            # Check if we need to restart the process
            if self._get_memory_usage() > 0.95:  # 95% threshold
                await self._request_process_restart("High memory usage")
                
        except Exception as e:
            logger.error(f"Error in aggressive memory optimization: {e}")
    
    async def _clear_unused_objects(self):
        """Clear unused objects from memory."""
        # Implementation would depend on your object tracking
        pass
            
    async def _adjust_task_priorities(self):
        """Adjust task priorities to reduce CPU usage."""
        try:
            # Get current tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            
            # Sort tasks by priority
            tasks.sort(key=lambda t: t.get_name().startswith('high_priority'), reverse=True)
            
            # Adjust priorities
            for task in tasks:
                if not task.get_name().startswith('high_priority'):
                    task.set_name(f"low_priority_{task.get_name()}")
                    
        except Exception as e:
            logger.error(f"Error adjusting task priorities: {e}")
            
    async def _aggressive_cpu_optimization(self):
        """Perform aggressive CPU optimization."""
        try:
            # Cancel non-essential tasks
            await self._cancel_non_essential_tasks()
            
            # Adjust task priorities
            await self._adjust_task_priorities()
            
            # Check if we need to restart the process
            if self._get_cpu_usage() > 0.95:  # 95% threshold
                await self._request_process_restart("High CPU usage")
                
        except Exception as e:
            logger.error(f"Error in aggressive CPU optimization: {e}")
    
    async def _cancel_non_essential_tasks(self):
        """Cancel non-essential tasks to free resources."""
        try:
            # Get all tasks
            all_tasks = asyncio.all_tasks()
            
            # Identify non-essential tasks (those not marked as high_priority)
            non_essential = [
                t for t in all_tasks
                if not t.get_name().startswith('high_priority') and
                t is not asyncio.current_task() and
                not t.done()
            ]
            
            # Cancel non-essential tasks
            for task in non_essential:
                task.cancel()
                
        except Exception as e:
            logger.error(f"Error cancelling non-essential tasks: {e}")
            
    async def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            # Get temp directory
            temp_dir = os.path.join(os.getcwd(), "temp")
            
            if os.path.exists(temp_dir):
                # Remove old files
                current_time = time.time()
                for filename in os.listdir(temp_dir):
                    filepath = os.path.join(temp_dir, filename)
                    if os.path.getmtime(filepath) < current_time - 3600:  # 1 hour old
                        try:
                            os.remove(filepath)
                        except Exception as e:
                            logger.error(f"Error removing temp file {filepath}: {e}")
                            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
            
    async def _aggressive_storage_optimization(self):
        """Perform aggressive storage optimization."""
        try:
            # Clean up temp files
            await self._cleanup_temp_files()
            
            # Clean up old logs
            await self._cleanup_old_logs()
            
            # Clean up old data
            await self._cleanup_old_data()
            
            # Check if we need to restart the process
            if self._get_disk_usage() > 0.95:  # 95% threshold
                await self._request_process_restart("High storage usage")
                
        except Exception as e:
            logger.error(f"Error in aggressive storage optimization: {e}")
    
    async def _cleanup_resources(self, resource_type: str, request: Dict[str, Any] = None):
        """Cleanup specific resources."""
        try:
            if resource_type == "memory":
                await self._clear_memory_caches()
                gc.collect()
            elif resource_type == "storage":
                await self._cleanup_temp_files()
                await self._cleanup_old_logs()
                await self._cleanup_old_data()
        except Exception as e:
            logger.error(f"Error cleaning up {resource_type} resources: {e}")
            
    async def _cleanup_all_resources(self):
        """Cleanup all resources."""
        try:
            await self._cleanup_resources("memory")
            await self._cleanup_resources("storage")
        except Exception as e:
            logger.error(f"Error cleaning up all resources: {e}")
            
    async def _cleanup_old_logs(self):
        """Clean up old log files."""
        try:
            # Get log directory
            log_dir = os.path.join(os.getcwd(), "logs")
            
            if os.path.exists(log_dir):
                # Remove old log files
                current_time = time.time()
                for filename in os.listdir(log_dir):
                    if filename.endswith('.log'):
                        filepath = os.path.join(log_dir, filename)
                        if os.path.getmtime(filepath) < current_time - 86400:  # 1 day old
                            try:
                                os.remove(filepath)
                            except Exception as e:
                                logger.error(f"Error removing log file {filepath}: {e}")
                                
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
            
    async def _cleanup_old_data(self):
        """Clean up old data files."""
        try:
            # Get data directory
            data_dir = os.path.join(os.getcwd(), "data")
            
            if os.path.exists(data_dir):
                # Remove old data files
                current_time = time.time()
                for filename in os.listdir(data_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(data_dir, filename)
                        if os.path.getmtime(filepath) < current_time - 604800:  # 1 week old
                            try:
                                os.remove(filepath)
                            except Exception as e:
                                logger.error(f"Error removing data file {filepath}: {e}")
                                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def _request_process_restart(self, reason: str):
        """
        Request a process restart due to resource issues.
        
        Instead of directly restarting, this signals that a restart is needed.
        The actual restart would be handled by the process manager.
        """
        try:
            # Save state for recovery
            await self._save_state()
            
            logger.warning(f"Process restart requested: {reason}")
            
            # Signal the need for restart
            # In a real system, this might write to a status file or send a signal
            restart_file = os.path.join(os.getcwd(), "data", "restart_requested")
            with open(restart_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "reason": reason,
                    "resource_metrics": {
                        "memory": self._get_memory_usage(),
                        "cpu": self._get_cpu_usage(),
                        "storage": self._get_disk_usage()
                    }
                }, f)
                
        except Exception as e:
            logger.error(f"Error requesting process restart: {e}")
            
    async def _save_state(self):
        """Save current state before restart."""
        try:
            # Save resource metrics
            metrics_file = os.path.join(os.getcwd(), "data", "resource_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.resource_metrics, f)
                
            # Save other state
            state_file = os.path.join(os.getcwd(), "data", "process_state.json")
            with open(state_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_tasks": len(asyncio.all_tasks()),
                    "resource_usage": {
                        "memory": self._get_memory_usage(),
                        "cpu": self._get_cpu_usage(),
                        "storage": self._get_disk_usage()
                    }
                }, f)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize system resources."""
        try:
            # Get current resource usage
            memory_usage = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            storage_usage = self._get_disk_usage()
            
            # Check each resource
            memory_result = await self._optimize_memory({"current_value": memory_usage})
            cpu_result = await self._optimize_cpu({"current_value": cpu_usage})
            storage_result = await self._optimize_storage({"current_value": storage_usage})
            
            return {
                "success": True,
                "memory": memory_usage,
                "cpu": cpu_usage,
                "storage": storage_usage
            }
            
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def cleanup_resources(self) -> Dict[str, Any]:
        """Clean up system resources."""
        try:
            # Clean up memory
            await self._clear_memory_caches()
            
            # Clean up storage
            await self._cleanup_temp_files()
            await self._cleanup_old_logs()
            await self._cleanup_old_data()
            
            # Force garbage collection
            gc.collect()
            
            # Get final metrics after cleanup
            current_metrics = {
                "memory": self._get_memory_usage(),
                "storage": self._get_disk_usage()
            }
            
            return {
                "success": True,
                "metrics": current_metrics
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get detailed resource statistics."""
        try:
            return {
                "current": {
                    "memory": self._get_memory_usage(),
                    "cpu": self._get_cpu_usage(),
                    "storage": self._get_disk_usage()
                },
                "peak": {
                    "memory": self.resource_metrics["memory"]["peak"],
                    "cpu": self.resource_metrics["cpu"]["peak"],
                    "storage": self.resource_metrics["storage"]["peak"]
                },
                "thresholds": {
                    "memory": self.resource_metrics["memory"]["threshold"],
                    "cpu": self.resource_metrics["cpu"]["threshold"],
                    "storage": self.resource_metrics["storage"]["threshold"]
                },
                "process": {
                    "memory_info": dict(psutil.Process().memory_info()._asdict()),
                    "cpu_times": dict(psutil.Process().cpu_times()._asdict()),
                    "io_counters": dict(psutil.Process().io_counters()._asdict()) if hasattr(psutil.Process(), 'io_counters') else None,
                    "num_threads": psutil.Process().num_threads(),
                    "nice": psutil.Process().nice()
                },
                "system": {
                    "total_memory": psutil.virtual_memory().total,
                    "available_memory": psutil.virtual_memory().available,
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_count": psutil.cpu_count(),
                    "disk_usage": dict(psutil.disk_usage('/')._asdict())
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Create a singleton instance for easy access
resource_manager = ResourceManager()
