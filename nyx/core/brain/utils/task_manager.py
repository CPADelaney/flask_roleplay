# nyx/core/brain/utils/task_manager.py
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Coroutine

logger = logging.getLogger(__name__)

class TaskManager:
    """
    Manages asynchronous tasks with dependencies for distributed processing
    """
    
    def __init__(self, max_parallel_tasks: int = 5):
        """
        Initialize task manager
        
        Args:
            max_parallel_tasks: Maximum number of tasks to run in parallel
        """
        self.tasks = {}
        self.dependencies = {}
        self.priorities = {}
        self.groups = {}
        self.max_parallel_tasks = max_parallel_tasks
        self.semaphore = asyncio.Semaphore(max_parallel_tasks)
    
    def register_task(self, 
                     task_id: str, 
                     coroutine: Coroutine,
                     dependencies: List[str] = None,
                     priority: int = 1,
                     group: str = None) -> None:
        """
        Register a task for execution
        
        Args:
            task_id: Unique identifier for the task
            coroutine: Coroutine to execute
            dependencies: List of task IDs that must complete before this task
            priority: Task priority (higher numbers = higher priority)
            group: Optional group name for related tasks
        """
        self.tasks[task_id] = coroutine
        self.dependencies[task_id] = dependencies or []
        self.priorities[task_id] = priority
        
        if group:
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(task_id)
    
    async def execute_tasks(self) -> Dict[str, Any]:
        """
        Execute all tasks respecting dependencies
        
        Returns:
            Dictionary of task results
        """
        start_time = asyncio.get_event_loop().time()
        
        # Create a copy of tasks to track which ones are left
        pending_tasks = set(self.tasks.keys())
        completed_tasks = set()
        results = {}
        performance_metrics = {
            "task_times": {},
            "group_times": {},
            "total_tasks": len(pending_tasks),
            "successful_tasks": 0
        }
        
        # Keep processing until all tasks are done
        while pending_tasks:
            # Find tasks that can be executed (all dependencies satisfied)
            executable_tasks = []
            for task_id in pending_tasks:
                if all(dep in completed_tasks for dep in self.dependencies[task_id]):
                    executable_tasks.append(task_id)
            
            # If no tasks can be executed, we have a dependency cycle
            if not executable_tasks:
                logger.error(f"Dependency cycle detected in tasks: {pending_tasks}")
                remaining_tasks = list(pending_tasks)
                for task_id in remaining_tasks:
                    results[task_id] = {"error": "Dependency cycle detected"}
                break
            
            # Sort by priority (higher numbers = higher priority)
            executable_tasks.sort(key=lambda x: self.priorities[x], reverse=True)
            
            # Execute tasks in parallel (up to max_parallel_tasks)
            task_start_time = asyncio.get_event_loop().time()
            
            tasks_to_run = []
            for task_id in executable_tasks:
                task_start_time = asyncio.get_event_loop().time()
                task_to_run = asyncio.create_task(self._execute_task_with_timing(
                    task_id, self.tasks[task_id], performance_metrics
                ))
                tasks_to_run.append(task_to_run)
            
            # Wait for all tasks to complete
            completed_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
            # Process results
            for i, task_id in enumerate(executable_tasks):
                result = completed_results[i]
                
                # If exception, store error
                if isinstance(result, Exception):
                    results[task_id] = {"error": str(result)}
                else:
                    results[task_id] = result
                    performance_metrics["successful_tasks"] += 1
                
                # Mark as completed
                pending_tasks.remove(task_id)
                completed_tasks.add(task_id)
                
                # Record group times
                for group, group_tasks in self.groups.items():
                    if task_id in group_tasks:
                        if group not in performance_metrics["group_times"]:
                            performance_metrics["group_times"][group] = 0
                        performance_metrics["group_times"][group] += performance_metrics["task_times"].get(task_id, 0)
        
        # Calculate total time
        total_time = asyncio.get_event_loop().time() - start_time
        performance_metrics["total_time"] = total_time
        
        # Add performance metrics to results
        results["_performance"] = performance_metrics
        
        return results
    
    async def _execute_task_with_timing(self, task_id: str, coro: Coroutine, metrics: Dict[str, Any]) -> Any:
        """
        Execute a task with timing measurement
        
        Args:
            task_id: Task ID
            coro: Coroutine to execute
            metrics: Metrics dictionary to update
            
        Returns:
            Task result
        """
        start_time = asyncio.get_event_loop().time()
        
        # Execute task
        async with self.semaphore:
            try:
                result = await coro
            except Exception as e:
                logger.error(f"Error executing task {task_id}: {e}")
                raise
        
        # Record execution time
        execution_time = asyncio.get_event_loop().time() - start_time
        metrics["task_times"][task_id] = execution_time
        
        return result
