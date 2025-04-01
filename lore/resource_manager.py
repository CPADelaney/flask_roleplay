# lore/resource_manager.py

"""
Resource Management System

This module provides comprehensive resource management capabilities for the lore system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from .base_manager import BaseManager
import psutil
import os
import time
import json
import sys

logger = logging.getLogger(__name__)

class ResourceManager(BaseManager):
    """Manages system resources with sophisticated tracking and optimization."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
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
        
    async def start(self):
        """Start the resource manager and monitoring."""
        await super().start()
        self._start_monitoring()
        self._start_optimization()
        self._start_cleanup()
        return True
        
    async def stop(self):
        """Stop the resource manager and cleanup."""
        await super().stop()
        await self._stop_monitoring()
        await self._stop_optimization()
        await self._stop_cleanup()
        return True
        
    def _start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring_task = asyncio.create_task(self._monitor_resources())
        
    def _start_optimization(self):
        """Start resource optimization."""
        self.optimization_task = asyncio.create_task(self._process_optimization_queue())
        
    def _start_cleanup(self):
        """Start resource cleanup."""
        self.cleanup_task = asyncio.create_task(self._process_cleanup_queue())
        
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
                
    async def _monitor_resources(self):
        """Monitor system resources."""
        while True:
            try:
                # Get current resource usage
                memory_usage = psutil.Process().memory_percent()
                cpu_usage = psutil.cpu_percent()
                storage_usage = psutil.disk_usage('/').percent
                
                # Update metrics
                self._update_resource_metrics("memory", memory_usage)
                self._update_resource_metrics("cpu", cpu_usage)
                self._update_resource_metrics("storage", storage_usage)
                
                # Check thresholds and trigger optimization if needed
                await self._check_resource_thresholds()
                
                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
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
                import gc
                gc.collect()
                
                # Check if we need more aggressive optimization
                if psutil.Process().memory_percent() > self.resource_metrics["memory"]["threshold"]:
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
                if psutil.cpu_percent() > self.resource_metrics["cpu"]["threshold"]:
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
                if psutil.disk_usage('/').percent > self.resource_metrics["storage"]["threshold"]:
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
            
    async def _aggressive_memory_optimization(self):
        """Perform aggressive memory optimization."""
        try:
            # Clear all caches
            await self._clear_memory_caches()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear unused objects
            await self._clear_unused_objects()
            
            # Check if we need to restart the process
            if psutil.Process().memory_percent() > 0.95:  # 95% threshold
                await self._restart_process()
                
        except Exception as e:
            logger.error(f"Error in aggressive memory optimization: {e}")
            
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
            if psutil.cpu_percent() > 0.95:  # 95% threshold
                await self._restart_process()
                
        except Exception as e:
            logger.error(f"Error in aggressive CPU optimization: {e}")
            
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
            if psutil.disk_usage('/').percent > 0.95:  # 95% threshold
                await self._restart_process()
                
        except Exception as e:
            logger.error(f"Error in aggressive storage optimization: {e}")
            
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
            
    async def _restart_process(self):
        """Restart the current process."""
        try:
            # Save current state
            await self._save_state()
            
            # Restart the process
            os.execv(sys.executable, ['python'] + sys.argv)
            
        except Exception as e:
            logger.error(f"Error restarting process: {e}")
            
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
                        "memory": psutil.Process().memory_percent(),
                        "cpu": psutil.cpu_percent(),
                        "storage": psutil.disk_usage('/').percent
                    }
                }, f)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize system resources."""
        try:
            # Get current resource usage
            memory_usage = psutil.Process().memory_percent()
            cpu_usage = psutil.cpu_percent()
            storage_usage = psutil.disk_usage('/').percent
            
            # Check each resource
            results = {
                "memory": await self._optimize_memory({"current_value": memory_usage}),
                "cpu": await self._optimize_cpu({"current_value": cpu_usage}),
                "storage": await self._optimize_storage({"current_value": storage_usage})
            }
            
            return {
                "success": True,
                "results": results,
                "metrics": {
                    "memory": memory_usage,
                    "cpu": cpu_usage,
                    "storage": storage_usage
                }
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
            import gc
            gc.collect()
            
            return {
                "success": True,
                "metrics": {
                    "memory": psutil.Process().memory_percent(),
                    "storage": psutil.disk_usage('/').percent
                }
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Create a singleton instance for easy access
resource_manager = ResourceManager() 
