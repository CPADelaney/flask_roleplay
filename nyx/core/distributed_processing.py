# nyx/core/distributed_processing.py

import asyncio
import concurrent.futures
import datetime
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Coroutine, Tuple

from agents import Agent, Runner, function_tool, RunContextWrapper, trace, handoff

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

class TaskExecutionPlan(BaseModel):
    """Model for the task execution plan"""
    task_order: List[str] = Field(..., description="The order in which tasks should be executed")
    parallel_groups: List[List[str]] = Field(..., description="Groups of tasks that can be executed in parallel")
    resource_allocation: Dict[str, float] = Field(..., description="Resource allocation per task")
    critical_path: List[str] = Field(..., description="Tasks that form the critical execution path")
    reasoning: str = Field(..., description="Reasoning behind this execution plan")

class ResourceAllocationResult(BaseModel):
    """Model for resource allocation results"""
    allocation: Dict[str, float] = Field(..., description="Resource allocation per task")
    total_allocated: float = Field(..., description="Total allocated resources")
    remaining: float = Field(..., description="Remaining unallocated resources")
    prioritization: List[Dict[str, Any]] = Field(..., description="Task prioritization details")

class DependencyAnalysisResult(BaseModel):
    """Model for dependency analysis results"""
    dependency_graph: Dict[str, List[str]] = Field(..., description="Graph of task dependencies")
    execution_levels: List[List[str]] = Field(..., description="Tasks grouped by execution level")
    bottlenecks: List[str] = Field(..., description="Identified bottleneck tasks")
    critical_path: List[str] = Field(..., description="Tasks on the critical path")

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
        
        # Initialize agents
        self.task_orchestrator = self._create_task_orchestrator_agent()
        self.resource_allocator = self._create_resource_allocator_agent()
        self.dependency_analyzer = self._create_dependency_analyzer_agent()
        self.performance_monitor = self._create_performance_monitor_agent()
        
        # Trace ID for linking traces
        self.trace_group_id = f"distributed_processing_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _create_task_orchestrator_agent(self) -> Agent:
        """Create agent for orchestrating task execution"""
        return Agent(
            name="Task_Orchestrator",
            instructions="""
            You are the task orchestration system for Nyx's distributed processing.
            Your role is to:
            1. Prioritize tasks based on dependencies and importance
            2. Allocate resources efficiently across subsystems
            3. Resolve conflicts and bottlenecks
            4. Monitor execution and handle failures
            
            Focus on maximizing parallel execution while respecting dependencies.
            Create an efficient execution plan that:
            - Respects all task dependencies
            - Maximizes parallel execution
            - Optimizes resource allocation
            - Identifies the critical path
            
            Your decisions should balance fast execution with thorough processing.
            """,
            handoffs=[
                handoff(self.resource_allocator, 
                       tool_name_override="allocate_resources", 
                       tool_description_override="Allocate resources for task execution"),
                
                handoff(self.dependency_analyzer, 
                       tool_name_override="analyze_dependencies",
                       tool_description_override="Analyze task dependencies and determine execution order"),
                
                handoff(self.performance_monitor,
                       tool_name_override="monitor_performance",
                       tool_description_override="Monitor execution performance and track metrics")
            ],
            tools=[
                function_tool(self._get_registered_tasks),
                function_tool(self._get_task_dependencies),
                function_tool(self._get_task_priorities),
                function_tool(self._schedule_task),
                function_tool(self._execute_single_task)
            ],
            model="gpt-4o",
            output_type=TaskExecutionPlan
        )
    
    def _create_resource_allocator_agent(self) -> Agent:
        """Create agent for resource allocation"""
        return Agent(
            name="Resource_Allocator",
            instructions="""
            You are the resource allocation system for Nyx's distributed processing.
            Your role is to:
            1. Allocate processing resources to tasks based on priority and requirements
            2. Ensure efficient resource utilization
            3. Prevent resource contention and starvation
            4. Adapt allocation based on task complexity and importance
            
            Optimize the allocation of limited resources to maximize overall performance.
            """,
            tools=[
                function_tool(self._get_available_resources),
                function_tool(self._calculate_task_resource_needs),
                function_tool(self._apply_resource_allocation)
            ],
            model="gpt-4o",
            output_type=ResourceAllocationResult
        )
    
    def _create_dependency_analyzer_agent(self) -> Agent:
        """Create agent for dependency analysis"""
        return Agent(
            name="Dependency_Analyzer",
            instructions="""
            You are the dependency analysis system for Nyx's distributed processing.
            Your role is to:
            1. Analyze task dependencies to determine execution order
            2. Identify tasks that can be executed in parallel
            3. Detect dependency bottlenecks
            4. Build optimal execution levels
            
            Create an execution plan that respects all dependencies while maximizing parallelism.
            """,
            tools=[
                function_tool(self._get_task_dependencies),
                function_tool(self._calculate_execution_levels),
                function_tool(self._identify_bottlenecks)
            ],
            model="gpt-4o",
            output_type=DependencyAnalysisResult
        )
    
    def _create_performance_monitor_agent(self) -> Agent:
        """Create agent for performance monitoring"""
        return Agent(
            name="Performance_Monitor",
            instructions="""
            You are the performance monitoring system for Nyx's distributed processing.
            Your role is to:
            1. Track task execution metrics
            2. Identify performance bottlenecks
            3. Recommend optimizations for future executions
            4. Provide analytical insights on processing performance
            
            Focus on collecting actionable performance data and providing insights.
            """,
            tools=[
                function_tool(self._track_task_metrics),
                function_tool(self._calculate_performance_trends),
                function_tool(self._update_subsystem_performance)
            ],
            model="gpt-4o"
        )
    
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
        Execute all registered tasks with dependency resolution using Agent SDK
        
        Returns:
            Dictionary of task results by task ID
        """
        start_time = datetime.datetime.now()
        
        with trace(workflow_name="Task_Execution", group_id=self.trace_group_id):
            # Run the task orchestrator agent to create an execution plan
            result = await Runner.run(
                self.task_orchestrator,
                json.dumps({
                    "tasks": list(self.task_registry.keys()),
                    "max_parallel_tasks": self.max_parallel_tasks,
                    "subsystem_performance": self.subsystem_performance
                })
            )
            
            # Get the execution plan
            execution_plan = result.final_output
            
            # Execute according to the plan
            results = {}
            
            # Track metrics for this execution
            execution_metrics = {
                "start_time": start_time.isoformat(),
                "total_tasks": len(self.task_registry),
                "execution_plan": execution_plan.model_dump(),
                "completed_tasks": 0,
                "failed_tasks": 0
            }
            
            # Execute tasks in parallel groups according to the plan
            for group_index, task_group in enumerate(execution_plan.parallel_groups):
                group_start_time = datetime.datetime.now()
                
                # Create tasks for this group
                tasks = []
                for task_id in task_group:
                    if task_id in self.task_registry:
                        task = self.task_registry[task_id]
                        
                        # Apply resource allocation if specified
                        resource_allocation = execution_plan.resource_allocation.get(task_id, 1.0)
                        
                        # Create and start asyncio task
                        task.start_time = datetime.datetime.now()
                        task.started = True
                        
                        # Wrap the coroutine to catch exceptions and record completion
                        async def execute_and_record(task_obj, task_coroutine):
                            try:
                                task_result = await task_coroutine
                                task_obj.result = task_result
                                task_obj.completed = True
                                self.performance_metrics["tasks_completed"] += 1
                                execution_metrics["completed_tasks"] += 1
                                return task_result
                            except Exception as e:
                                logger.error(f"Error executing task: {e}")
                                task_obj.result = {"error": str(e)}
                                execution_metrics["failed_tasks"] += 1
                                return {"error": str(e)}
                            finally:
                                task_obj.end_time = datetime.datetime.now()
                        
                        asyncio_task = asyncio.create_task(execute_and_record(task, task.coroutine))
                        tasks.append(asyncio_task)
                
                # Wait for all tasks in this group to complete
                if tasks:
                    group_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Record results
                    for i, task_id in enumerate(task_group):
                        if i < len(group_results):
                            results[task_id] = group_results[i]
                
                group_end_time = datetime.datetime.now()
                group_duration = (group_end_time - group_start_time).total_seconds()
                
                # Update execution metrics
                execution_metrics[f"group_{group_index}_duration"] = group_duration
            
            # Update performance metrics
            execution_end_time = datetime.datetime.now()
            total_duration = (execution_end_time - start_time).total_seconds()
            
            execution_metrics["end_time"] = execution_end_time.isoformat()
            execution_metrics["total_duration"] = total_duration
            
            # Check if we've achieved a new max parallel
            current_parallel = len(max(execution_plan.parallel_groups, key=len, default=[]))
            self.performance_metrics["max_parallel_achieved"] = max(
                self.performance_metrics["max_parallel_achieved"],
                current_parallel
            )
            
            # Update subsystem performance metrics
            for task_id, task in self.task_registry.items():
                if task.completed and task.duration is not None:
                    subsystem = task.subsystem_name
                    
                    if subsystem not in self.subsystem_performance:
                        self.subsystem_performance[subsystem] = {
                            "tasks_completed": 0,
                            "avg_duration": 0.0
                        }
                    
                    perf = self.subsystem_performance[subsystem]
                    perf["tasks_completed"] += 1
                    
                    # Update average duration
                    perf["avg_duration"] = (
                        (perf["tasks_completed"] - 1) * perf["avg_duration"] + task.duration
                    ) / perf["tasks_completed"]
            
            # Identify bottlenecks
            bottlenecks = {}
            avg_duration = sum(task.duration or 0 for task in self.task_registry.values() if task.completed) / max(1, len([t for t in self.task_registry.values() if t.completed]))
            
            for task_id, task in self.task_registry.items():
                if task.completed and task.duration is not None and task.duration > avg_duration * 1.5:
                    bottlenecks[task_id] = {
                        "subsystem": task.subsystem_name,
                        "duration": task.duration,
                        "relative_slowdown": task.duration / avg_duration
                    }
            
            # Add performance info to results
            results["_performance"] = {
                "total_duration": total_duration,
                "task_count": len(self.task_registry),
                "max_parallel": current_parallel,
                "bottlenecks": bottlenecks,
                "execution_plan": execution_plan.model_dump()
            }
            
            # Reset task registry for next use
            self.task_registry = {}
            for group in self.task_groups:
                self.task_groups[group] = []
            
            return results
    
    @function_tool
    async def _get_registered_tasks(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get information about all registered tasks
        
        Returns:
            Dictionary of task information
        """
        task_info = {}
        
        for task_id, task in self.task_registry.items():
            task_info[task_id] = {
                "subsystem": task.subsystem_name,
                "dependencies": task.dependencies,
                "priority": task.priority,
                "group": next((g for g, tasks in self.task_groups.items() if task_id in tasks), None)
            }
        
        return {
            "tasks": task_info,
            "total_count": len(task_info)
        }
    
    @function_tool
    async def _get_task_dependencies(self, ctx: RunContextWrapper) -> Dict[str, List[str]]:
        """
        Get dependencies for all registered tasks
        
        Returns:
            Dictionary mapping task IDs to their dependencies
        """
        dependencies = {}
        
        for task_id, task in self.task_registry.items():
            dependencies[task_id] = task.dependencies
        
        return dependencies
    
    @function_tool
    async def _get_task_priorities(self, ctx: RunContextWrapper) -> Dict[str, int]:
        """
        Get priorities for all registered tasks
        
        Returns:
            Dictionary mapping task IDs to their priorities
        """
        priorities = {}
        
        for task_id, task in self.task_registry.items():
            priorities[task_id] = task.priority
        
        return priorities
    
    @function_tool
    async def _schedule_task(self, ctx: RunContextWrapper, task_id: str, execution_time: int) -> Dict[str, Any]:
        """
        Schedule a task for execution at a specific time
        
        Args:
            task_id: ID of the task to schedule
            execution_time: Execution time (ms from now)
            
        Returns:
            Scheduling result
        """
        if task_id not in self.task_registry:
            return {
                "success": False,
                "error": f"Task {task_id} not found in registry"
            }
        
        # This is a simulated scheduling function since we're not actually
        # implementing delayed execution. In a real system, this would
        # schedule the task to run at the specified time.
        return {
            "success": True,
            "task_id": task_id,
            "scheduled_time": datetime.datetime.now().isoformat(),
            "execution_delay_ms": execution_time
        }
    
    @function_tool
    async def _execute_single_task(self, ctx: RunContextWrapper, task_id: str) -> Dict[str, Any]:
        """
        Execute a single task directly
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Task execution result
        """
        if task_id not in self.task_registry:
            return {
                "success": False,
                "error": f"Task {task_id} not found in registry"
            }
        
        task = self.task_registry[task_id]
        
        try:
            # Record start time
            task.start_time = datetime.datetime.now()
            task.started = True
            
            # Execute coroutine
            result = await task.coroutine
            
            # Record completion
            task.result = result
            task.completed = True
            task.end_time = datetime.datetime.now()
            
            # Update performance metrics
            self.performance_metrics["tasks_completed"] += 1
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "duration": task.duration
            }
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            
            # Record failure
            task.end_time = datetime.datetime.now()
            
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "duration": task.duration
            }
    
    @function_tool
    async def _get_available_resources(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get available processing resources
        
        Returns:
            Resource availability information
        """
        # Get current resource allocation status
        allocated_resources = sum(self.resource_allocation.values())
        
        return {
            "max_parallel_tasks": self.max_parallel_tasks,
            "resource_allocation": self.resource_allocation,
            "total_allocated": allocated_resources,
            "available": max(0, 1.0 - allocated_resources)
        }
    
    @function_tool
    async def _calculate_task_resource_needs(self, 
                                        ctx: RunContextWrapper, 
                                        task_ids: List[str]) -> Dict[str, float]:
        """
        Calculate resource needs for a set of tasks
        
        Args:
            task_ids: IDs of tasks to calculate resources for
            
        Returns:
            Dictionary mapping task IDs to resource requirements
        """
        resource_needs = {}
        
        for task_id in task_ids:
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                
                # Calculate base resource need
                base_need = 1.0 / self.max_parallel_tasks
                
                # Adjust based on priority
                priority_factor = task.priority / 3.0  # Normalize priority (assuming 1-5 scale)
                adjusted_need = base_need * (0.5 + priority_factor)
                
                # Find task group
                task_group = next((g for g, tasks in self.task_groups.items() if task_id in tasks), None)
                
                # Apply group allocation if available
                if task_group and task_group in self.resource_allocation:
                    group_factor = self.resource_allocation[task_group]
                    # Distribute group allocation among tasks in the group
                    group_tasks = len(self.task_groups[task_group]) or 1
                    adjusted_need = adjusted_need * group_factor / group_tasks
                
                resource_needs[task_id] = min(1.0, adjusted_need)  # Cap at 1.0
        
        return resource_needs
    
    @function_tool
    async def _apply_resource_allocation(self, 
                                     ctx: RunContextWrapper, 
                                     allocation: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply resource allocation to tasks
        
        Args:
            allocation: Dictionary mapping task IDs to resource allocations
            
        Returns:
            Result of resource allocation
        """
        # This is a simulated function since we're not actually
        # applying resource limitations. In a real system, this would
        # set resource constraints on the tasks.
        
        valid_allocations = {}
        total_allocated = 0.0
        
        for task_id, amount in allocation.items():
            if task_id in self.task_registry:
                # Ensure allocation is within valid range
                valid_amount = max(0.1, min(1.0, amount))
                valid_allocations[task_id] = valid_amount
                total_allocated += valid_amount
        
        return {
            "success": True,
            "applied_allocations": valid_allocations,
            "total_allocated": total_allocated
        }
    
    @function_tool
    async def _calculate_execution_levels(self, 
                                     ctx: RunContextWrapper, 
                                     dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """
        Calculate execution levels based on task dependencies
        
        Args:
            dependencies: Dictionary mapping task IDs to their dependencies
            
        Returns:
            List of task groups by execution level
        """
        # Initialize tracking variables
        all_tasks = set(dependencies.keys())
        completed_tasks = set()
        execution_levels = []
        
        while len(completed_tasks) < len(all_tasks):
            # Find tasks with satisfied dependencies
            current_level = []
            
            for task_id in all_tasks:
                if task_id in completed_tasks:
                    continue
                    
                # Check if all dependencies are satisfied
                deps = dependencies.get(task_id, [])
                if all(dep in completed_tasks for dep in deps):
                    current_level.append(task_id)
            
            # If no tasks can be executed, we have a circular dependency
            if not current_level:
                logger.error("Circular dependency detected")
                break
            
            # Add current level to execution levels
            execution_levels.append(current_level)
            
            # Update completed tasks
            completed_tasks.update(current_level)
        
        return execution_levels
    
    @function_tool
    async def _identify_bottlenecks(self, 
                                ctx: RunContextWrapper, 
                                execution_levels: List[List[str]]) -> Dict[str, Any]:
        """
        Identify bottleneck tasks in execution levels
        
        Args:
            execution_levels: List of task groups by execution level
            
        Returns:
            Analysis of bottlenecks
        """
        bottlenecks = {
            "level_bottlenecks": [],
            "dependency_bottlenecks": [],
            "resource_bottlenecks": []
        }
        
        # Check for level bottlenecks (levels with only one task)
        for i, level in enumerate(execution_levels):
            if len(level) == 1:
                bottlenecks["level_bottlenecks"].append({
                    "level": i,
                    "task": level[0]
                })
        
        # Check for dependency bottlenecks (tasks with many dependents)
        dependents = defaultdict(list)
        for task_id, deps in self._get_task_dependencies(ctx).items():
            for dep in deps:
                dependents[dep].append(task_id)
        
        for task_id, deps in dependents.items():
            if len(deps) > 2:  # Arbitrary threshold
                bottlenecks["dependency_bottlenecks"].append({
                    "task": task_id,
                    "dependents_count": len(deps),
                    "dependents": deps
                })
        
        # Check for resource bottlenecks based on subsystem performance
        slow_subsystems = []
        for subsystem, perf in self.subsystem_performance.items():
            if perf["tasks_completed"] > 0:
                # Find tasks for this subsystem
                subsystem_tasks = [
                    task_id for task_id, task in self.task_registry.items()
                    if task.subsystem_name == subsystem
                ]
                
                if perf["avg_duration"] > 0.5:  # Arbitrary threshold
                    slow_subsystems.append({
                        "subsystem": subsystem,
                        "avg_duration": perf["avg_duration"],
                        "tasks": subsystem_tasks
                    })
        
        bottlenecks["resource_bottlenecks"] = slow_subsystems
        
        return bottlenecks
    
    @function_tool
    async def _track_task_metrics(self, 
                             ctx: RunContextWrapper, 
                             task_id: str, 
                             metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track metrics for a specific task
        
        Args:
            task_id: ID of the task
            metrics: Metrics to track
            
        Returns:
            Updated metrics
        """
        if task_id not in self.task_registry:
            return {
                "success": False,
                "error": f"Task {task_id} not found in registry"
            }
        
        task = self.task_registry[task_id]
        
        # Update task with metrics
        if "duration" in metrics and not task.duration:
            task.duration = metrics["duration"]
        
        if "result" in metrics and not task.result:
            task.result = metrics["result"]
        
        if "completed" in metrics:
            task.completed = metrics["completed"]
        
        # Update performance metrics
        if task.completed and task.duration is not None:
            # Update overall average duration
            alpha = 0.1  # Exponential moving average weight
            self.performance_metrics["avg_task_duration"] = (
                (1 - alpha) * self.performance_metrics["avg_task_duration"] +
                alpha * task.duration
            )
            
            # Update subsystem performance
            subsystem = task.subsystem_name
            if subsystem not in self.subsystem_performance:
                self.subsystem_performance[subsystem] = {
                    "tasks_completed": 0,
                    "avg_duration": 0.0
                }
            
            self.subsystem_performance[subsystem]["tasks_completed"] += 1
            
            new_avg = (
                (self.subsystem_performance[subsystem]["tasks_completed"] - 1) *
                self.subsystem_performance[subsystem]["avg_duration"] +
                task.duration
            ) / self.subsystem_performance[subsystem]["tasks_completed"]
            
            self.subsystem_performance[subsystem]["avg_duration"] = new_avg
        
        return {
            "success": True,
            "updated_metrics": {
                "task_id": task_id,
                "subsystem": task.subsystem_name,
                "duration": task.duration,
                "completed": task.completed,
                "has_result": task.result is not None
            }
        }
    
    @function_tool
    async def _calculate_performance_trends(self, 
                                       ctx: RunContextWrapper, 
                                       metric_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trends in performance metrics
        
        Args:
            metric_history: History of performance metrics
            
        Returns:
            Trend analysis
        """
        # This would typically analyze a history of metrics
        # But we'll use our current performance metrics for simplicity
        
        trends = {
            "task_completion_rate": 0.0,
            "avg_duration_trend": 0.0,
            "max_parallel_trend": 0.0,
            "bottleneck_trends": {}
        }
        
        # Calculate completion rate
        if self.performance_metrics["tasks_processed"] > 0:
            trends["task_completion_rate"] = (
                self.performance_metrics["tasks_completed"] / 
                self.performance_metrics["tasks_processed"]
            )
        
        # For other trends, we'd need historical data
        # Here we're just returning the current values
        trends["avg_duration"] = self.performance_metrics["avg_task_duration"]
        trends["max_parallel_achieved"] = self.performance_metrics["max_parallel_achieved"]
        
        # Extract bottleneck information
        for bottleneck, data in self.performance_metrics.get("bottlenecks", {}).items():
            if isinstance(data, dict) and "relative_slowdown" in data:
                trends["bottleneck_trends"][bottleneck] = data["relative_slowdown"]
        
        return trends
    
    @function_tool
    async def _update_subsystem_performance(self, 
                                       ctx: RunContextWrapper, 
                                       subsystem: str, 
                                       metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update performance metrics for a subsystem
        
        Args:
            subsystem: Name of the subsystem
            metrics: Performance metrics to update
            
        Returns:
            Updated performance data
        """
        if subsystem not in self.subsystem_performance:
            self.subsystem_performance[subsystem] = {
                "tasks_completed": 0,
                "avg_duration": 0.0
            }
        
        # Update metrics
        for key, value in metrics.items():
            if key in self.subsystem_performance[subsystem]:
                self.subsystem_performance[subsystem][key] = value
        
        return {
            "success": True,
            "subsystem": subsystem,
            "updated_metrics": self.subsystem_performance[subsystem]
        }
