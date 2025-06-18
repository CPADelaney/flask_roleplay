# nyx/core/distributed_processing.py

import asyncio
import datetime
import json
import logging
import time
from typing import Dict, List, Any, Optional, Coroutine
from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, trace, function_tool, RunContextWrapper, handoff, 
    ModelSettings, InputGuardrail, GuardrailFunctionOutput
)

logger = logging.getLogger(__name__)

# Reusable JSON result wrapper for all tools
class JSONResult(BaseModel, extra="forbid"):
    json: str

# Input parameter wrappers for strict schema compliance
class ValidateResourceReqParams(BaseModel, extra="forbid"):
    task_id: str
    resource_requirements: Dict[str, float]

class CalcExecLevelsParams(BaseModel, extra="forbid"):
    dependencies: Dict[str, List[str]]

class SubsystemTask(BaseModel):
    """Represents a parallel task for a subsystem"""
    subsystem_name: str = Field(..., description="Name of the subsystem")
    task_id: str = Field(..., description="Unique identifier for the task")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must complete before this one")
    priority: int = Field(1, description="Task priority (higher = more important)")
    started: bool = Field(False, description="Whether the task has started")
    completed: bool = Field(False, description="Whether the task has completed")
    result: Optional[Any] = Field(None, description="Result from task execution")
    error: Optional[str] = Field(None, description="Error message if task failed")
    start_time: Optional[datetime.datetime] = Field(None, description="When task started")
    end_time: Optional[datetime.datetime] = Field(None, description="When task completed")

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

class TaskContext(BaseModel):
    """Context for task execution"""
    user_id: Optional[str] = Field(None, description="User ID if relevant")
    session_id: Optional[str] = Field(None, description="Session ID if relevant")
    max_retries: int = Field(3, description="Maximum number of task retries")
    timeout_seconds: float = Field(60.0, description="Task timeout in seconds")
    resource_limit: float = Field(1.0, description="Resource usage limit (0.0-1.0)")

class TaskValidationResult(BaseModel):
    """Output from task validation guardrail"""
    is_valid: bool = Field(..., description="Whether the task is valid")
    reason: Optional[str] = Field(None, description="Reason if task is invalid")
    recommended_priority: Optional[int] = Field(None, description="Recommended priority adjustment")

class DistributedProcessingManager:
    """
    Manager for distributed parallel processing across subsystems
    
    Handles dependency resolution, task scheduling, and resource allocation
    for parallel processing of cognitive tasks.
    """
    
    def __init__(self, max_parallel_tasks: int = 10):
        """
        Initialize the distributed processing manager
        
        Args:
            max_parallel_tasks: Maximum number of tasks to run in parallel
        """
        self.max_parallel_tasks = max_parallel_tasks
        self.task_registry: Dict[str, SubsystemTask] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.subsystem_performance: Dict[str, Dict[str, Any]] = {}
        
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
        self._init_agents()
        
        # Trace ID for linking traces
        self.trace_group_id = f"distributed_processing_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info("DistributedProcessingManager initialized")
    
    def _init_agents(self):
        """Initialize all agent components"""
        # Task validation agent
        self.validation_agent = self._create_validation_agent()
        
        # Resource allocation agent
        self.resource_allocator = self._create_resource_allocator()
        
        # Dependency analysis agent
        self.dependency_analyzer = self._create_dependency_analyzer()
        
        # Performance monitoring agent
        self.performance_monitor = self._create_performance_monitor()
        
        # Main orchestration agent
        self.task_orchestrator = self._create_task_orchestrator()
    
    def _create_validation_agent(self) -> Agent:
        """Create an agent for validating task configurations"""
        return Agent(
            name="Task_Validator",
            instructions="""
            You validate task configurations before they are added to the processing queue.
            
            Your responsibilities:
            1. Verify task dependencies exist and don't create circular dependencies
            2. Check that task priorities are appropriate for their importance
            3. Validate resource requirements against available system resources
            4. Ensure tasks have all required metadata and parameters
            
            Reject tasks that would cause system issues, and suggest fixes when possible.
            """,
            tools=[
                self._check_dependency_existence,
                self._check_circular_dependencies,
                self._validate_resource_requirements
            ],
            output_type=TaskValidationResult,
            model="gpt-4.1-nano"
        )
    
    def _create_resource_allocator(self) -> Agent:
        """Create an agent for allocating resources to tasks"""
        return Agent(
            name="Resource_Allocator",
            instructions="""
            You allocate computational resources to tasks based on their priority and requirements.
            
            Your responsibilities:
            1. Analyze task resource requirements
            2. Balance resource allocation across different task types
            3. Ensure high-priority tasks receive adequate resources
            4. Prevent resource starvation for lower-priority tasks
            5. Adapt allocation based on system load and performance data
            
            Produce an optimal resource allocation plan that maximizes throughput.
            """,
            tools=[
                self._get_available_resources,
                self._calculate_task_resource_needs,
                self._get_system_load
            ],
            output_type=ResourceAllocationResult,
            model="gpt-4.1-nano"
        )
    
    def _create_dependency_analyzer(self) -> Agent:
        """Create an agent for analyzing task dependencies"""
        return Agent(
            name="Dependency_Analyzer",
            instructions="""
            You analyze dependencies between tasks to determine execution order.
            
            Your responsibilities:
            1. Build a dependency graph from task relationships
            2. Identify execution levels for parallel processing
            3. Detect bottlenecks in the execution flow
            4. Find the critical path that determines minimum execution time
            5. Identify opportunities to increase parallelism
            
            Produce a detailed analysis that enables optimal execution planning.
            """,
            tools=[
                self._get_task_dependencies,
                self._calculate_execution_levels,
                self._identify_bottlenecks
            ],
            output_type=DependencyAnalysisResult,
            model="gpt-4.1-nano"
        )
    
    def _create_performance_monitor(self) -> Agent:
        """Create an agent for monitoring performance metrics"""
        return Agent(
            name="Performance_Monitor",
            instructions="""
            You monitor and analyze the performance of the distributed processing system.
            
            Your responsibilities:
            1. Track execution metrics for tasks and subsystems
            2. Identify performance bottlenecks and inefficiencies
            3. Analyze trends in task execution times
            4. Recommend system optimizations based on performance data
            5. Provide insights on resource utilization patterns
            
            Focus on actionable insights that can improve system performance.
            """,
            tools=[
                self._get_performance_metrics,
                self._analyze_execution_trends,
                self._identify_performance_bottlenecks
            ],
            model="gpt-4.1-nano"
        )
    
    def _create_task_orchestrator(self) -> Agent:
        """Create the main orchestration agent for task execution"""
        return Agent(
            name="Task_Orchestrator",
            instructions="""
            You are the main orchestrator for distributed task processing.
            
            Your responsibilities:
            1. Create execution plans based on dependency analysis
            2. Allocate resources to tasks based on priority and requirements
            3. Monitor task execution and handle failures or delays
            4. Optimize parallel execution while respecting dependencies
            5. Balance system load across different subsystems
            
            Make intelligent decisions to maximize throughput while ensuring
            correct execution order and handling failures gracefully.
            """,
            handoffs=[
                handoff(self.validation_agent, 
                       tool_name_override="validate_task", 
                       tool_description_override="Validate a task before registration"),
                
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
                self._get_registered_tasks,
                self._schedule_task,
                self._execute_single_task,
                self._handle_task_failure
            ],
            output_type=TaskExecutionPlan,
            model="gpt-4.1-nano"
        )

    @function_tool
    async def _check_dependency_existence(self, ctx: RunContextWrapper, dependencies: List[str]) -> JSONResult:
        """
        Check if all dependencies exist in the task registry
        
        Args:
            dependencies: List of task IDs to check
            
        Returns:
            Dictionary with existence check results
        """
        missing_dependencies = []
        for dep_id in dependencies:
            if dep_id not in self.task_registry:
                missing_dependencies.append(dep_id)
        
        payload = {
            "all_exist": len(missing_dependencies) == 0,
            "missing_dependencies": missing_dependencies,
            "total_dependencies": len(dependencies),
            "existing_dependencies": len(dependencies) - len(missing_dependencies)
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _check_circular_dependencies(self, ctx: RunContextWrapper, task_id: str, dependencies: List[str]) -> JSONResult:
        """
        Check for circular dependencies in the task graph
        
        Args:
            task_id: The ID of the task being checked
            dependencies: Direct dependencies of the task
            
        Returns:
            Dictionary with circular dependency check results
        """
        # Build dependency graph for existing tasks
        dependency_graph = {}
        for tid, task in self.task_registry.items():
            dependency_graph[tid] = task.dependencies
        
        # Add the new task's dependencies
        dependency_graph[task_id] = dependencies
        
        # Check for circular dependencies
        visited = set()
        path = set()
        circular_path = []
        
        def dfs(node):
            visited.add(node)
            path.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        circular_path.append(neighbor)
                        return True
                elif neighbor in path:
                    circular_path.append(neighbor)
                    return True
            
            path.remove(node)
            return False
        
        has_cycle = dfs(task_id)
        
        payload = {
            "has_circular_dependency": has_cycle,
            "circular_path": circular_path[::-1] if has_cycle else [],
            "task_id": task_id
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _validate_resource_requirements(self, ctx: RunContextWrapper, params: ValidateResourceReqParams) -> JSONResult:
        """
        Validate that resource requirements are within system constraints
        
        Args:
            params: Parameters containing task_id and resource_requirements
            
        Returns:
            Dictionary with validation results
        """
        task_id = params.task_id
        resource_requirements = params.resource_requirements
        
        # Get system resource limits
        system_resources = {
            "cpu": 1.0,
            "memory": 1.0,
            "network": 1.0,
            "io": 1.0
        }
        
        # Check requirements against limits
        issues = []
        for resource, required in resource_requirements.items():
            if resource not in system_resources:
                issues.append(f"Unknown resource: {resource}")
            elif required > system_resources[resource]:
                issues.append(f"Excessive {resource} requirement: {required} > {system_resources[resource]}")
        
        valid = len(issues) == 0
        
        # Calculate total resource load
        total_load = sum(resource_requirements.values()) / len(resource_requirements) if resource_requirements else 0
        
        payload = {
            "is_valid": valid,
            "issues": issues,
            "total_resource_load": total_load,
            "resource_requirements": resource_requirements
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _get_available_resources(self, ctx: RunContextWrapper) -> JSONResult:
        """
        Get information about available system resources
        
        Returns:
            Dictionary with resource availability information
        """
        # Get current resource allocation status
        allocated_resources = sum(self.resource_allocation.values())
        
        # Calculate current usage from active tasks
        active_task_count = len(self.active_tasks)
        active_task_load = min(1.0, active_task_count / max(1, self.max_parallel_tasks))
        
        payload = {
            "max_parallel_tasks": self.max_parallel_tasks,
            "current_active_tasks": active_task_count,
            "resource_allocation": self.resource_allocation,
            "total_allocated": allocated_resources,
            "system_load": active_task_load,
            "available_capacity": max(0, 1.0 - active_task_load)
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _calculate_task_resource_needs(self, ctx: RunContextWrapper, task_ids: List[str]) -> JSONResult:
        """
        Calculate resource needs for a set of tasks
        
        Args:
            task_ids: List of task IDs to calculate resources for
            
        Returns:
            Dictionary mapping task IDs to resource requirements
        """
        resource_needs = {}
        
        for task_id in task_ids:
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                
                # Calculate base resource need
                base_need = 1.0 / self.max_parallel_tasks
                
                # Adjust based on priority (higher priority = more resources)
                priority_factor = task.priority / 3.0  # Normalize priority (assuming 1-5 scale)
                adjusted_need = base_need * (0.5 + priority_factor)
                
                # Find task group (if any)
                task_group = next((g for g, tasks in self.task_groups.items() if task_id in tasks), None)
                
                # Apply group allocation if available
                if task_group and task_group in self.resource_allocation:
                    group_factor = self.resource_allocation[task_group]
                    group_tasks = len(self.task_groups[task_group]) or 1
                    adjusted_need = adjusted_need * group_factor / group_tasks
                
                resource_needs[task_id] = min(1.0, adjusted_need)  # Cap at 1.0
        
        return JSONResult(json=json.dumps(resource_needs))

    @function_tool
    async def _get_system_load(self, ctx: RunContextWrapper) -> JSONResult:
        """
        Get current system load metrics
        
        Returns:
            Dictionary with system load information
        """
        active_task_count = len(self.active_tasks)
        pending_task_count = len([t for t in self.task_registry.values() if not t.started and not t.completed])
        
        # Calculate per-subsystem load
        subsystem_load = {}
        for task_id, task in self.task_registry.items():
            if task_id in self.active_tasks:
                subsystem = task.subsystem_name
                subsystem_load[subsystem] = subsystem_load.get(subsystem, 0) + 1
        
        payload = {
            "active_tasks": active_task_count,
            "pending_tasks": pending_task_count,
            "system_capacity": self.max_parallel_tasks,
            "load_percentage": (active_task_count / self.max_parallel_tasks) * 100 if self.max_parallel_tasks > 0 else 0,
            "subsystem_load": subsystem_load
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _get_task_dependencies(self, ctx: RunContextWrapper) -> JSONResult:
        """
        Get dependencies for all registered tasks
        
        Returns:
            Dictionary mapping task IDs to their dependencies
        """
        dependencies = {}
        
        for task_id, task in self.task_registry.items():
            dependencies[task_id] = task.dependencies
        
        return JSONResult(json=json.dumps(dependencies))

    @function_tool
    async def _calculate_execution_levels(self, ctx: RunContextWrapper, params: CalcExecLevelsParams) -> JSONResult:
        """
        Calculate execution levels based on task dependencies
        
        Args:
            params: Parameters containing the dependencies dictionary
            
        Returns:
            List of task groups by execution level
        """
        dependencies = params.dependencies
        
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
        
        return JSONResult(json=json.dumps(execution_levels))

    @function_tool
    async def _identify_bottlenecks(self, ctx: RunContextWrapper, execution_levels: List[List[str]]) -> JSONResult:
        """
        Identify bottleneck tasks in execution levels
        
        Args:
            execution_levels: List of task groups by execution level
            
        Returns:
            Analysis of bottlenecks
        """
        bottlenecks = {
            "level_bottlenecks": [],
            "dependency_bottlenecks": []
        }
        
        # Check for level bottlenecks (levels with only one task)
        for i, level in enumerate(execution_levels):
            if len(level) == 1:
                bottlenecks["level_bottlenecks"].append({
                    "level": i,
                    "task": level[0]
                })
        
        # Build dependency mapping
        dependents = {}
        for task_id, task in self.task_registry.items():
            for dep in task.dependencies:
                if dep not in dependents:
                    dependents[dep] = []
                dependents[dep].append(task_id)
        
        # Check for dependency bottlenecks (tasks with many dependents)
        for task_id, deps in dependents.items():
            if len(deps) > 2:  # Tasks with multiple dependents
                bottlenecks["dependency_bottlenecks"].append({
                    "task": task_id,
                    "dependents_count": len(deps),
                    "dependents": deps
                })
        
        return JSONResult(json=json.dumps(bottlenecks))

    @function_tool
    async def _get_performance_metrics(self, ctx: RunContextWrapper) -> JSONResult:
        """
        Get current performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate average task duration for completed tasks
        completed_tasks = [t for t in self.task_registry.values() if t.completed]
        
        avg_duration = 0.0
        if completed_tasks:
            durations = [
                (t.end_time - t.start_time).total_seconds() 
                for t in completed_tasks 
                if t.start_time and t.end_time
            ]
            avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate per-subsystem metrics
        subsystem_metrics = {}
        for task in completed_tasks:
            subsystem = task.subsystem_name
            if subsystem not in subsystem_metrics:
                subsystem_metrics[subsystem] = {
                    "task_count": 0,
                    "total_duration": 0,
                    "avg_duration": 0
                }
            
            if task.start_time and task.end_time:
                duration = (task.end_time - task.start_time).total_seconds()
                subsystem_metrics[subsystem]["task_count"] += 1
                subsystem_metrics[subsystem]["total_duration"] += duration
        
        # Calculate averages
        for metrics in subsystem_metrics.values():
            if metrics["task_count"] > 0:
                metrics["avg_duration"] = metrics["total_duration"] / metrics["task_count"]
        
        payload = {
            "tasks_processed": self.performance_metrics["tasks_processed"],
            "tasks_completed": self.performance_metrics["tasks_completed"],
            "avg_task_duration": avg_duration,
            "max_parallel_achieved": self.performance_metrics["max_parallel_achieved"],
            "subsystem_metrics": subsystem_metrics,
            "bottlenecks": self.performance_metrics.get("bottlenecks", {})
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _analyze_execution_trends(self, ctx: RunContextWrapper) -> JSONResult:
        """
        Analyze trends in task execution
        
        Returns:
            Dictionary with trend analysis
        """
        # This would be more sophisticated in a real implementation
        completed_tasks = [t for t in self.task_registry.values() if t.completed]
        
        # Sort by completion time
        sorted_tasks = sorted(
            [t for t in completed_tasks if t.end_time], 
            key=lambda t: t.end_time
        )
        
        # Calculate running averages
        window_size = 5
        running_durations = []
        
        for i in range(len(sorted_tasks)):
            window = sorted_tasks[max(0, i-window_size+1):i+1]
            durations = [
                (t.end_time - t.start_time).total_seconds() 
                for t in window 
                if t.start_time and t.end_time
            ]
            avg = sum(durations) / len(durations) if durations else 0
            running_durations.append(avg)
        
        # Calculate efficiency trend
        efficiency_trend = "stable"
        if len(running_durations) >= 3:
            recent_avg = running_durations[-1]
            oldest_avg = running_durations[0]
            
            if recent_avg < oldest_avg * 0.8:
                efficiency_trend = "improving"
            elif recent_avg > oldest_avg * 1.2:
                efficiency_trend = "degrading"
        
        payload = {
            "total_analyzed": len(sorted_tasks),
            "running_duration_averages": running_durations[-10:] if len(running_durations) > 10 else running_durations,
            "efficiency_trend": efficiency_trend,
            "has_sufficient_data": len(sorted_tasks) >= 5
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _identify_performance_bottlenecks(self, ctx: RunContextWrapper) -> JSONResult:
        """
        Identify performance bottlenecks in the system
        
        Returns:
            Dictionary with identified bottlenecks
        """
        # Get completed tasks with duration info
        completed_tasks = [
            t for t in self.task_registry.values() 
            if t.completed and t.start_time and t.end_time
        ]
        
        if not completed_tasks:
            payload = {
                "has_bottlenecks": False,
                "message": "Insufficient data - no completed tasks"
            }
            return JSONResult(json=json.dumps(payload))
        
        # Calculate average duration
        durations = [(t.end_time - t.start_time).total_seconds() for t in completed_tasks]
        avg_duration = sum(durations) / len(durations)
        
        # Identify slow tasks (significantly above average)
        slow_tasks = []
        for task in completed_tasks:
            duration = (task.end_time - task.start_time).total_seconds()
            if duration > avg_duration * 1.5:  # 50% slower than average
                slow_tasks.append({
                    "task_id": task.task_id,
                    "subsystem": task.subsystem_name,
                    "duration": duration,
                    "vs_average": duration / avg_duration
                })
        
        # Sort by relative slowness
        slow_tasks.sort(key=lambda x: x["vs_average"], reverse=True)
        
        # Check for subsystem patterns
        subsystem_counts = {}
        for task in slow_tasks:
            subsystem = task["subsystem"]
            subsystem_counts[subsystem] = subsystem_counts.get(subsystem, 0) + 1
        
        # Identify bottleneck subsystems
        bottleneck_subsystems = [
            {
                "subsystem": subsystem,
                "slow_task_count": count,
                "percentage": count / len(slow_tasks) * 100 if slow_tasks else 0
            }
            for subsystem, count in subsystem_counts.items()
            if count >= 2  # At least 2 slow tasks
        ]
        
        payload = {
            "has_bottlenecks": len(slow_tasks) > 0,
            "average_duration": avg_duration,
            "slow_tasks": slow_tasks[:5],  # Top 5 slowest
            "bottleneck_subsystems": bottleneck_subsystems,
            "recommendation": "Investigate subsystem performance" if bottleneck_subsystems else "No clear bottlenecks"
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _get_registered_tasks(self, ctx: RunContextWrapper) -> JSONResult:
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
                "started": task.started,
                "completed": task.completed,
                "error": task.error,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "end_time": task.end_time.isoformat() if task.end_time else None
            }
        
        payload = {
            "tasks": task_info,
            "total_count": len(task_info),
            "started_count": sum(1 for t in self.task_registry.values() if t.started),
            "completed_count": sum(1 for t in self.task_registry.values() if t.completed)
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _schedule_task(self, ctx: RunContextWrapper, task_id: str, execution_time: int) -> JSONResult:
        """
        Schedule a task for execution at a specific time
        
        Args:
            task_id: ID of the task to schedule
            execution_time: Execution time (ms from now)
            
        Returns:
            Scheduling result
        """
        if task_id not in self.task_registry:
            payload = {
                "success": False,
                "error": f"Task {task_id} not found in registry"
            }
            return JSONResult(json=json.dumps(payload))
        
        # Create a delayed execution task
        async def delayed_execution():
            await asyncio.sleep(execution_time / 1000.0)  # Convert ms to seconds
            result = await self._execute_single_task(ctx, task_id)
            return json.loads(result.json)
        
        # Schedule the task
        self.active_tasks[task_id] = asyncio.create_task(delayed_execution())
        
        payload = {
            "success": True,
            "task_id": task_id,
            "scheduled_time": datetime.datetime.now().isoformat(),
            "execution_delay_ms": execution_time
        }
        return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _execute_single_task(self, ctx: RunContextWrapper, task_id: str) -> JSONResult:
        """
        Execute a single task directly
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Task execution result
        """
        if task_id not in self.task_registry:
            payload = {
                "success": False,
                "error": f"Task {task_id} not found in registry"
            }
            return JSONResult(json=json.dumps(payload))
        
        task = self.task_registry[task_id]
        
        try:
            # Record start time
            task.start_time = datetime.datetime.now()
            task.started = True
            
            # Get the coroutine from task
            coroutine = task.coroutine
            
            # Execute the coroutine
            result = await coroutine
            
            # Record completion
            task.result = result
            task.completed = True
            task.end_time = datetime.datetime.now()
            
            # Update performance metrics
            self.performance_metrics["tasks_completed"] += 1
            
            duration = (task.end_time - task.start_time).total_seconds()
            payload = {
                "success": True,
                "task_id": task_id,
                "result": result,
                "duration": duration
            }
            return JSONResult(json=json.dumps(payload))
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            
            # Record failure
            task.error = str(e)
            task.end_time = datetime.datetime.now()
            
            payload = {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "duration": (task.end_time - task.start_time).total_seconds() if task.start_time else None
            }
            return JSONResult(json=json.dumps(payload))

    @function_tool
    async def _handle_task_failure(self, ctx: RunContextWrapper, task_id: str, error: str, retry: bool = False) -> JSONResult:
        """
        Handle a task failure
        
        Args:
            task_id: ID of the failed task
            error: Error message
            retry: Whether to retry the task
            
        Returns:
            Handling result
        """
        if task_id not in self.task_registry:
            payload = {
                "success": False,
                "error": f"Task {task_id} not found in registry"
            }
            return JSONResult(json=json.dumps(payload))
        
        task = self.task_registry[task_id]
        task.error = error
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            # Cancel the task if it's still running
            self.active_tasks[task_id].cancel()
            del self.active_tasks[task_id]
        
        if retry:
            # Create a new execution task
            result = await self._execute_single_task(ctx, task_id)
            retry_result = json.loads(result.json)
            payload = {
                "success": retry_result.get("success", False),
                "task_id": task_id,
                "action": "retry",
                "retry_result": retry_result
            }
            return JSONResult(json=json.dumps(payload))
        else:
            # Just record the failure
            payload = {
                "success": True,
                "task_id": task_id,
                "action": "recorded_failure",
                "error": error
            }
            return JSONResult(json=json.dumps(payload))
    
    async def register_task(self, 
                          task_id: str,
                          subsystem_name: str, 
                          coroutine: Coroutine, 
                          dependencies: List[str] = None,
                          priority: int = 1,
                          group: str = None) -> Dict[str, Any]:
        """
        Register a task for parallel execution
        
        Args:
            task_id: Unique task identifier
            subsystem_name: Name of the subsystem responsible for the task
            coroutine: Async coroutine to execute
            dependencies: List of task IDs that must complete before this task
            priority: Task priority (higher = more important)
            group: Task group for resource allocation
            
        Returns:
            Registration result
        """
        with trace(workflow_name="RegisterTask", group_id=self.trace_group_id):
            # Create task object
            task = SubsystemTask(
                task_id=task_id,
                subsystem_name=subsystem_name,
                dependencies=dependencies or [],
                priority=priority
            )
            
            # Store the coroutine (not part of the model)
            task.coroutine = coroutine
            
            # Validate the task using the validation agent
            validation_result = await Runner.run(
                self.validation_agent,
                json.dumps({
                    "task_id": task_id,
                    "subsystem_name": subsystem_name,
                    "dependencies": dependencies or [],
                    "priority": priority
                }),
                run_config={
                    "workflow_name": "TaskValidation",
                    "trace_metadata": {
                        "task_id": task_id,
                        "subsystem": subsystem_name
                    }
                }
            )
            
            validation_output = validation_result.final_output
            
            # If task is not valid, return error
            if not validation_output.is_valid:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": validation_output.reason,
                    "recommended_priority": validation_output.recommended_priority
                }
            
            # Register the task
            self.task_registry[task_id] = task
            
            # Add to task group if specified
            if group and group in self.task_groups:
                self.task_groups[group].append(task_id)
            
            logger.info(f"Registered task {task_id} for subsystem {subsystem_name}")
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "Task registered successfully"
            }
    
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
                }),
                run_config={
                    "workflow_name": "TaskExecution",
                    "trace_metadata": {
                        "total_tasks": len(self.task_registry)
                    }
                }
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
                        # Execute the task
                        tasks.append(self._execute_single_task(None, task_id))
                
                # Wait for all tasks in this group to complete
                if tasks:
                    group_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Record results
                    for i, task_id in enumerate(task_group):
                        if i < len(group_results):
                            if isinstance(group_results[i], JSONResult):
                                result_data = json.loads(group_results[i].json)
                                results[task_id] = result_data
                                
                                # Update metrics
                                if result_data.get("success", False):
                                    execution_metrics["completed_tasks"] += 1
                                else:
                                    execution_metrics["failed_tasks"] += 1
                            else:
                                # Handle exception case
                                results[task_id] = {"success": False, "error": str(group_results[i])}
                                execution_metrics["failed_tasks"] += 1
                
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
                if task.completed and task.start_time and task.end_time:
                    subsystem = task.subsystem_name
                    
                    if subsystem not in self.subsystem_performance:
                        self.subsystem_performance[subsystem] = {
                            "tasks_completed": 0,
                            "avg_duration": 0.0
                        }
                    
                    perf = self.subsystem_performance[subsystem]
                    perf["tasks_completed"] += 1
                    
                    # Update average duration
                    duration = (task.end_time - task.start_time).total_seconds()
                    perf["avg_duration"] = (
                        (perf["tasks_completed"] - 1) * perf["avg_duration"] + duration
                    ) / perf["tasks_completed"]
            
            # Identify bottlenecks
            bottlenecks = {}
            avg_duration = sum(
                (t.end_time - t.start_time).total_seconds() 
                for t in self.task_registry.values() 
                if t.completed and t.start_time and t.end_time
            ) / max(1, len([t for t in self.task_registry.values() if t.completed and t.start_time and t.end_time]))
            
            for task_id, task in self.task_registry.items():
                if task.completed and task.start_time and task.end_time:
                    duration = (task.end_time - task.start_time).total_seconds()
                    if duration > avg_duration * 1.5:
                        bottlenecks[task_id] = {
                            "subsystem": task.subsystem_name,
                            "duration": duration,
                            "relative_slowdown": duration / avg_duration
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
