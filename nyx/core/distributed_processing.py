# nyx/core/distributed_processing.py

import asyncio
import concurrent.futures
import datetime
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Coroutine

logger = logging.getLogger(__name__)

class SubsystemTask:
    """Represents a parallel task for a subsystem"""
    
    def __init__(self, 
               subsystem_name: str, 
               coroutine: Coroutine, 
               dependencies: List[str] = None,
               priority: int = 1):
        self.subsystem_name = subsystem_name
        self.coroutine = coroutine
        self.dependencies = dependencies or []
        self.priority = priority
        self.result = None
        self.completed = False
        self.started = False
        self.start_time = None
        self.end_time = None
        self.task = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate duration if task completed"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def __repr__(self):
        return f"SubsystemTask({self.subsystem_name}, completed={self.completed})"

class DistributedProcessingManager:
    """
    Manager for distributed parallel processing across subsystems
    
    Handles dependency resolution, task scheduling, and resource allocation
    for parallel processing of cognitive tasks.
    """
    
    def __init__(self, max_parallel_tasks: int = 10):
        self.max_parallel_tasks = max_parallel_tasks
        self.task_registry = {}
        self.subsystem_performance = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_tasks)
        
        # Task groups for different cognitive processes
        self.task_groups = {
            "perception": [],
            "memory": [],
            "emotion": [],
            "reasoning": [],
            "adaptation": [],
            "reflection": [],
            "meta": []
        }
        
        # Resource allocation for different task groups
        self.resource_allocation = {
            "perception": 0.2,
            "memory": 0.25,
            "emotion": 0.1,
            "reasoning": 0.15,
            "adaptation": 0.1,
            "reflection": 0.1,
            "meta": 0.1
        }
        
        # Performance tracking
        self.performance_metrics = {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "avg_task_duration": 0.0,
            "max_parallel_achieved": 0,
            "bottlenecks": {}
        }
    
    def register_task(self, 
                    task_id: str, 
                    subsystem_name: str, 
                    coroutine: Coroutine, 
                    dependencies: List[str] = None,
                    priority: int = 1,
                    group: str = None) -> None:
        """
        Register a task for parallel execution
        
        Args:
            task_id: Unique task identifier
            subsystem_name: Name of the subsystem responsible for the task
            coroutine: Async coroutine to execute
            dependencies: List of task IDs that must complete before this task
            priority: Task priority (higher = more important)
            group: Task group for resource allocation
        """
        task = SubsystemTask(
            subsystem_name=subsystem_name,
            coroutine=coroutine,
            dependencies=dependencies or [],
            priority=priority
        )
        
        self.task_registry[task_id] = task
        
        # Add to task group if specified
        if group and group in self.task_groups:
            self.task_groups[group].append(task_id)
    
    async def execute_tasks(self) -> Dict[str, Any]:
        """
        Execute all registered tasks with dependency resolution
        
        Returns:
            Dictionary of task results by task ID
        """
        start_time = datetime.datetime.now()
        
        # Track tasks that are ready to run, running, and completed
        ready_tasks = []
        running_tasks = []
        completed_tasks = []
        
        # Initialize ready tasks (those with no dependencies)
        for task_id, task in self.task_registry.items():
            if not task.dependencies:
                ready_tasks.append(task_id)
        
        # Process tasks until all are completed
        while len(completed_tasks) < len(self.task_registry):
            # Calculate available slots based on resource allocation
            available_slots = min(self.max_parallel_tasks - len(running_tasks), len(ready_tasks))
            
            if available_slots > 0:
                # Track maximum parallel achieved
                current_parallel = len(running_tasks) + available_slots
                self.performance_metrics["max_parallel_achieved"] = max(
                    self.performance_metrics["max_parallel_achieved"],
                    current_parallel
                )
                
                # Prioritize ready tasks
                ready_tasks.sort(
                    key=lambda tid: self.task_registry[tid].priority,
                    reverse=True
                )
                
                # Start new tasks
                for _ in range(available_slots):
                    if not ready_tasks:
                        break
                        
                    task_id = ready_tasks.pop(0)
                    task = self.task_registry[task_id]
                    
                    # Create and start the task
                    task.task = asyncio.create_task(task.coroutine)
                    task.started = True
                    task.start_time = datetime.datetime.now()
                    running_tasks.append(task_id)
                    
                    # Update performance metrics
                    self.performance_metrics["tasks_processed"] += 1
            
            # Check for completed tasks
            newly_completed = []
            
            for task_id in running_tasks:
                task = self.task_registry[task_id]
                
                if task.task.done():
                    # Task completed
                    task.completed = True
                    task.end_time = datetime.datetime.now()
                    
                    try:
                        task.result = await task.task
                    except Exception as e:
                        logger.error(f"Error in task {task_id}: {str(e)}")
                        task.result = {"error": str(e)}
                    
                    newly_completed.append(task_id)
                    completed_tasks.append(task_id)
                    
                    # Update performance metrics
                    self.performance_metrics["tasks_completed"] += 1
                    
                    # Update average task duration
                    if task.duration is not None:
                        alpha = 0.1  # Exponential moving average weight
                        self.performance_metrics["avg_task_duration"] = (
                            (1 - alpha) * self.performance_metrics["avg_task_duration"] +
                            alpha * task.duration
                        )
                    
                    # Update subsystem performance
                    if task.subsystem_name not in self.subsystem_performance:
                        self.subsystem_performance[task.subsystem_name] = {
                            "tasks_completed": 0,
                            "avg_duration": 0.0
                        }
                    
                    self.subsystem_performance[task.subsystem_name]["tasks_completed"] += 1
                    
                    if task.duration is not None:
                        perf = self.subsystem_performance[task.subsystem_name]
                        perf["avg_duration"] = (
                            (perf["tasks_completed"] - 1) * perf["avg_duration"] + task.duration
                        ) / perf["tasks_completed"]
            
            # Remove completed tasks from running
            for task_id in newly_completed:
                running_tasks.remove(task_id)
            
            # Update ready tasks based on completed dependencies
            for task_id, task in self.task_registry.items():
                if not task.started and not task_id in ready_tasks:
                    # Check if all dependencies are completed
                    dependencies_completed = all(
                        self.task_registry[dep_id].completed 
                        for dep_id in task.dependencies 
                        if dep_id in self.task_registry
                    )
                    
                    if dependencies_completed:
                        ready_tasks.append(task_id)
            
            # Short delay before next check if nothing happened
            if not newly_completed and available_slots == 0:
                await asyncio.sleep(0.01)
        
        # Calculate total duration
        total_duration = (datetime.datetime.now() - start_time).total_seconds()
        
        # Identify bottlenecks
        for subsystem, perf in self.subsystem_performance.items():
            if perf["avg_duration"] > self.performance_metrics["avg_task_duration"] * 1.5:
                self.performance_metrics["bottlenecks"][subsystem] = {
                    "avg_duration": perf["avg_duration"],
                    "relative_slowdown": perf["avg_duration"] / self.performance_metrics["avg_task_duration"]
                }
        
        # Prepare results
        results = {}
        
        for task_id, task in self.task_registry.items():
            results[task_id] = task.result
        
        # Add performance info
        results["_performance"] = {
            "total_duration": total_duration,
            "task_count": len(self.task_registry),
            "max_parallel": self.performance_metrics["max_parallel_achieved"],
            "bottlenecks": self.performance_metrics["bottlenecks"]
        }
        
        # Reset task registry for next use
        self.task_registry = {}
        for group in self.task_groups:
            self.task_groups[group] = []
        
        return results
