# nyx/core/a2a/context_aware_parallel.py

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareParallelExecutor(ContextAwareModule):
    """
    Advanced ParallelToolExecutor with full context distribution capabilities
    """
    
    def __init__(self, original_parallel_executor):
        super().__init__("parallel_executor")
        self.original_executor = original_parallel_executor
        self.context_subscriptions = [
            "system_load_update", "priority_task_requested", 
            "resource_allocation_update", "task_completion",
            "urgent_execution_needed", "batch_processing_request",
            "performance_metrics_request", "optimization_suggestion"
        ]
        
        # Track execution metrics with context
        self.context_aware_metrics = {
            "module_execution_times": {},
            "priority_executions": 0,
            "context_driven_optimizations": 0,
            "resource_conflicts_resolved": 0,
            "batch_optimizations": 0
        }
        
        # Priority queue for context-aware execution
        self.priority_queue = asyncio.PriorityQueue()
        self.executing_tasks = {}
        self.execution_history = []
    
    async def on_context_received(self, context: SharedContext):
        """Process incoming context for execution optimization"""
        logger.debug(f"ParallelExecutor received context for user: {context.user_id}")
        
        # Analyze system state for optimization
        system_analysis = await self._analyze_system_state(context)
        
        # Check for urgent execution needs
        urgent_tasks = await self._identify_urgent_tasks(context)
        
        # Optimize execution strategy based on context
        execution_strategy = await self._optimize_execution_strategy(context, system_analysis)
        
        # Send execution context
        await self.send_context_update(
            update_type="execution_context_available",
            data={
                "current_capacity": self._get_current_capacity(),
                "system_load": system_analysis.get("load_level", "normal"),
                "urgent_tasks": len(urgent_tasks),
                "execution_strategy": execution_strategy,
                "active_executions": len(self.executing_tasks)
            },
            priority=ContextPriority.LOW
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules for execution coordination"""
        
        if update.update_type == "priority_task_requested":
            # Handle high-priority task execution
            task_data = update.data
            task_info = task_data.get("task_info")
            priority = task_data.get("priority", 0.5)
            
            if task_info:
                await self._queue_priority_task(task_info, priority)
                self.context_aware_metrics["priority_executions"] += 1
        
        elif update.update_type == "system_load_update":
            # Adjust execution parameters based on system load
            load_data = update.data
            load_level = load_data.get("load_level", "normal")
            
            await self._adjust_execution_parameters(load_level)
        
        elif update.update_type == "resource_allocation_update":
            # Handle resource allocation changes
            resource_data = update.data
            available_resources = resource_data.get("available_resources", {})
            
            await self._update_resource_allocation(available_resources)
        
        elif update.update_type == "batch_processing_request":
            # Optimize batch execution
            batch_data = update.data
            tasks = batch_data.get("tasks", [])
            
            if tasks:
                optimized_batch = await self._optimize_batch_execution(tasks)
                await self._execute_optimized_batch(optimized_batch)
                self.context_aware_metrics["batch_optimizations"] += 1
        
        elif update.update_type == "urgent_execution_needed":
            # Handle urgent execution requests
            urgent_data = update.data
            task = urgent_data.get("task")
            reason = urgent_data.get("reason", "unspecified")
            
            if task:
                await self._execute_urgent_task(task, reason)
        
        elif update.update_type == "optimization_suggestion":
            # Apply optimization suggestions
            optimization_data = update.data
            suggestion_type = optimization_data.get("type")
            parameters = optimization_data.get("parameters", {})
            
            await self._apply_optimization(suggestion_type, parameters)
            self.context_aware_metrics["context_driven_optimizations"] += 1
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context-aware parallel execution"""
        # Extract tasks from context
        tasks_to_execute = await self._extract_tasks_from_context(context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Determine execution priority and strategy
        execution_plan = await self._create_execution_plan(tasks_to_execute, context, messages)
        
        # Execute tasks with context awareness
        execution_results = []
        if execution_plan["tasks"]:
            # Group tasks by priority
            priority_groups = self._group_tasks_by_priority(execution_plan["tasks"])
            
            # Execute each priority group
            for priority, tasks in sorted(priority_groups.items(), reverse=True):
                group_results = await self._execute_task_group(tasks, priority, context)
                execution_results.extend(group_results)
        
        # Update metrics
        await self._update_execution_metrics(execution_results, context)
        
        # Send execution summary
        if execution_results:
            await self.send_context_update(
                update_type="parallel_execution_complete",
                data={
                    "tasks_executed": len(execution_results),
                    "average_execution_time": self._calculate_average_execution_time(execution_results),
                    "success_rate": self._calculate_success_rate(execution_results),
                    "execution_strategy": execution_plan.get("strategy", "standard")
                },
                priority=ContextPriority.LOW
            )
        
        return {
            "parallel_processing": True,
            "tasks_executed": len(execution_results),
            "execution_plan": execution_plan,
            "results": execution_results
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze parallel execution patterns and performance"""
        # Get current execution state
        current_state = self._get_execution_state()
        
        # Analyze execution patterns
        execution_patterns = await self._analyze_execution_patterns()
        
        # Identify bottlenecks
        bottlenecks = await self._identify_bottlenecks(context)
        
        # Calculate resource utilization
        resource_utilization = await self._calculate_resource_utilization()
        
        # Get cross-module dependencies
        messages = await self.get_cross_module_messages()
        dependencies = await self._analyze_cross_module_dependencies(messages)
        
        # Generate optimization recommendations
        optimizations = await self._generate_optimization_recommendations(
            execution_patterns, bottlenecks, resource_utilization
        )
        
        return {
            "current_state": current_state,
            "execution_patterns": execution_patterns,
            "bottlenecks": bottlenecks,
            "resource_utilization": resource_utilization,
            "cross_module_dependencies": dependencies,
            "optimization_recommendations": optimizations,
            "context_aware_metrics": self.context_aware_metrics
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize execution insights for system optimization"""
        # Get recent execution history
        recent_executions = self.execution_history[-20:] if self.execution_history else []
        
        # Analyze performance trends
        performance_trends = await self._analyze_performance_trends(recent_executions)
        
        # Generate execution summary
        execution_summary = {
            "total_executions": len(self.execution_history),
            "recent_performance": performance_trends,
            "optimization_applied": self.context_aware_metrics["context_driven_optimizations"],
            "resource_efficiency": await self._calculate_resource_efficiency(),
            "recommendations": []
        }
        
        # Add specific recommendations based on patterns
        if performance_trends.get("degrading_performance"):
            execution_summary["recommendations"].append(
                "Consider reducing parallel execution limit due to performance degradation"
            )
        
        if self.context_aware_metrics["priority_executions"] > 10:
            execution_summary["recommendations"].append(
                "High number of priority executions - consider rebalancing task priorities"
            )
        
        # Send synthesis update
        await self.send_context_update(
            update_type="execution_synthesis_complete",
            data={
                "summary": execution_summary,
                "key_insights": self._extract_key_insights(performance_trends),
                "system_health": self._assess_execution_health()
            },
            priority=ContextPriority.LOW
        )
        
        return execution_summary
    
    # Advanced parallel execution methods
    
    async def execute_tools(self, tools_info: List[Dict[str, Any]], 
                          priority: float = 0.5,
                          context: Optional[SharedContext] = None) -> List[Any]:
        """Execute tools with context-aware prioritization"""
        # Track execution context
        execution_id = f"exec_{datetime.now().timestamp()}"
        self.executing_tasks[execution_id] = {
            "tools": tools_info,
            "priority": priority,
            "start_time": time.time(),
            "context_summary": self._summarize_context(context) if context else None
        }
        
        try:
            # Apply context-aware optimizations
            if context:
                tools_info = await self._optimize_tool_execution_order(tools_info, context)
            
            # Use original executor with our optimizations
            results = await self.original_executor.execute_tools(tools_info)
            
            # Update execution history
            execution_time = time.time() - self.executing_tasks[execution_id]["start_time"]
            self.execution_history.append({
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "tool_count": len(tools_info),
                "execution_time": execution_time,
                "priority": priority,
                "success_rate": self._calculate_success_rate(results),
                "context_driven": context is not None
            })
            
            # Trim history if needed
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            return results
            
        finally:
            # Clean up tracking
            if execution_id in self.executing_tasks:
                del self.executing_tasks[execution_id]
    
    async def _analyze_system_state(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze current system state from context"""
        state_analysis = {
            "load_level": "normal",
            "resource_availability": 1.0,
            "optimization_potential": 0.5
        }
        
        # Check number of active executions
        active_count = len(self.executing_tasks)
        if active_count > self.original_executor.max_concurrent * 0.8:
            state_analysis["load_level"] = "high"
            state_analysis["resource_availability"] = 0.2
        elif active_count > self.original_executor.max_concurrent * 0.5:
            state_analysis["load_level"] = "medium"
            state_analysis["resource_availability"] = 0.5
        
        # Check execution history for patterns
        if self.execution_history:
            recent_times = [e["execution_time"] for e in self.execution_history[-10:]]
            avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
            
            # High average execution time suggests optimization potential
            if avg_time > 1.0:  # More than 1 second average
                state_analysis["optimization_potential"] = 0.8
        
        return state_analysis
    
    async def _identify_urgent_tasks(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify tasks that need urgent execution"""
        urgent_tasks = []
        
        # Check for urgent markers in context
        if context.action_context:
            pending_actions = context.action_context.get("pending_actions", [])
            for action in pending_actions:
                if action.get("urgency", 0) > 0.8:
                    urgent_tasks.append({
                        "task": action,
                        "urgency": action.get("urgency"),
                        "source": "action_context"
                    })
        
        # Check for time-sensitive tasks
        if context.temporal_context:
            if context.temporal_context.get("time_pressure", False):
                urgent_tasks.append({
                    "task": {"type": "time_sensitive", "description": "Time pressure detected"},
                    "urgency": 0.9,
                    "source": "temporal_context"
                })
        
        return urgent_tasks
    
    async def _optimize_execution_strategy(self, context: SharedContext, 
                                         system_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Determine optimal execution strategy based on context"""
        strategy = {
            "parallelism_level": "normal",
            "prioritization": "balanced",
            "batching": "disabled",
            "resource_allocation": "standard"
        }
        
        load_level = system_analysis.get("load_level", "normal")
        
        # Adjust based on load
        if load_level == "high":
            strategy["parallelism_level"] = "reduced"
            strategy["prioritization"] = "strict"
            strategy["batching"] = "aggressive"
        elif load_level == "low":
            strategy["parallelism_level"] = "increased"
            strategy["batching"] = "opportunistic"
        
        # Adjust based on context
        if context.emotional_state:
            # High stress/anxiety - be more conservative
            primary_emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            if primary_emotion in ["Anxiety", "Stress", "Overwhelm"]:
                strategy["parallelism_level"] = "reduced"
                strategy["resource_allocation"] = "conservative"
        
        return strategy
    
    async def _queue_priority_task(self, task_info: Dict[str, Any], priority: float):
        """Queue a task with priority"""
        # Higher priority = lower number (for PriorityQueue)
        queue_priority = 1.0 - priority
        
        await self.priority_queue.put((queue_priority, datetime.now(), task_info))
        
        # Trigger execution if capacity available
        if len(self.executing_tasks) < self.original_executor.max_concurrent:
            asyncio.create_task(self._process_priority_queue())
    
    async def _process_priority_queue(self):
        """Process tasks from priority queue"""
        while not self.priority_queue.empty():
            if len(self.executing_tasks) >= self.original_executor.max_concurrent:
                # At capacity, wait
                await asyncio.sleep(0.1)
                continue
            
            try:
                queue_priority, timestamp, task_info = await asyncio.wait_for(
                    self.priority_queue.get(), timeout=0.1
                )
                
                # Convert back to normal priority
                priority = 1.0 - queue_priority
                
                # Execute the task
                await self.execute_tools([task_info], priority=priority)
                
            except asyncio.TimeoutError:
                break
    
    async def _adjust_execution_parameters(self, load_level: str):
        """Adjust execution parameters based on load"""
        if load_level == "high":
            # Reduce parallelism
            self.original_executor.max_concurrent = max(2, self.original_executor.max_concurrent - 2)
        elif load_level == "low":
            # Increase parallelism (up to original limit)
            self.original_executor.max_concurrent = min(10, self.original_executor.max_concurrent + 1)
        
        logger.info(f"Adjusted max concurrent to {self.original_executor.max_concurrent} based on {load_level} load")
    
    async def _update_resource_allocation(self, available_resources: Dict[str, Any]):
        """Update resource allocation based on availability"""
        # Could implement more sophisticated resource management
        cpu_available = available_resources.get("cpu", 1.0)
        memory_available = available_resources.get("memory", 1.0)
        
        # Simple scaling based on resources
        resource_factor = min(cpu_available, memory_available)
        optimal_concurrent = int(self.original_executor.max_concurrent * resource_factor)
        
        self.original_executor.max_concurrent = max(1, optimal_concurrent)
    
    async def _optimize_batch_execution(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Optimize task batching for efficient execution"""
        # Group tasks by estimated execution time
        quick_tasks = []
        normal_tasks = []
        heavy_tasks = []
        
        for task in tasks:
            # Estimate based on task properties
            estimated_time = task.get("estimated_time", 0.5)
            
            if estimated_time < 0.1:
                quick_tasks.append(task)
            elif estimated_time < 1.0:
                normal_tasks.append(task)
            else:
                heavy_tasks.append(task)
        
        # Create optimized batches
        batches = []
        
        # Quick tasks can be batched together
        if quick_tasks:
            batch_size = min(self.original_executor.max_concurrent, len(quick_tasks))
            for i in range(0, len(quick_tasks), batch_size):
                batches.append(quick_tasks[i:i+batch_size])
        
        # Normal tasks in smaller batches
        if normal_tasks:
            batch_size = max(2, self.original_executor.max_concurrent // 2)
            for i in range(0, len(normal_tasks), batch_size):
                batches.append(normal_tasks[i:i+batch_size])
        
        # Heavy tasks individually or in very small batches
        if heavy_tasks:
            batch_size = max(1, self.original_executor.max_concurrent // 4)
            for i in range(0, len(heavy_tasks), batch_size):
                batches.append(heavy_tasks[i:i+batch_size])
        
        return batches
    
    async def _execute_optimized_batch(self, batches: List[List[Dict[str, Any]]]):
        """Execute optimized batches"""
        for batch in batches:
            # Execute batch with appropriate priority
            avg_priority = sum(t.get("priority", 0.5) for t in batch) / len(batch) if batch else 0.5
            
            await self.execute_tools(batch, priority=avg_priority)
            
            # Small delay between batches to prevent overload
            await asyncio.sleep(0.05)
    
    async def _execute_urgent_task(self, task: Dict[str, Any], reason: str):
        """Execute an urgent task immediately"""
        logger.info(f"Executing urgent task due to: {reason}")
        
        # Execute with highest priority
        await self.execute_tools([task], priority=1.0)
    
    async def _apply_optimization(self, suggestion_type: str, parameters: Dict[str, Any]):
        """Apply optimization suggestions"""
        if suggestion_type == "increase_parallelism":
            new_limit = parameters.get("new_limit", self.original_executor.max_concurrent + 2)
            self.original_executor.max_concurrent = min(20, new_limit)  # Cap at 20
            
        elif suggestion_type == "reduce_parallelism":
            new_limit = parameters.get("new_limit", self.original_executor.max_concurrent - 2)
            self.original_executor.max_concurrent = max(1, new_limit)  # At least 1
            
        elif suggestion_type == "enable_batching":
            # Enable batch optimizations
            logger.info("Batch optimizations enabled via context suggestion")
    
    async def _extract_tasks_from_context(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract executable tasks from context"""
        tasks = []
        
        # Extract from action context
        if context.action_context:
            pending_actions = context.action_context.get("pending_actions", [])
            for action in pending_actions:
                if action.get("requires_execution"):
                    tasks.append({
                        "tool": action.get("tool"),
                        "args": action.get("args", {}),
                        "priority": action.get("priority", 0.5),
                        "metadata": {"source": "action_context", "action_id": action.get("id")}
                    })
        
        return tasks
    
    async def _create_execution_plan(self, tasks: List[Dict[str, Any]], 
                                   context: SharedContext, 
                                   messages: List[Any]) -> Dict[str, Any]:
        """Create an execution plan based on context and dependencies"""
        plan = {
            "tasks": tasks,
            "strategy": "standard",
            "optimizations": []
        }
        
        # Check for high-priority modules in messages
        high_priority_modules = ["goal_manager", "needs_system", "emotional_core"]
        has_high_priority = any(m.get("module_name") in high_priority_modules for m in messages)
        
        if has_high_priority:
            plan["strategy"] = "priority_focused"
            plan["optimizations"].append("prioritize_core_system_tasks")
        
        # Check system load
        if len(self.executing_tasks) > self.original_executor.max_concurrent * 0.7:
            plan["strategy"] = "load_balanced"
            plan["optimizations"].append("distribute_load")
        
        return plan
    
    def _group_tasks_by_priority(self, tasks: List[Dict[str, Any]]) -> Dict[float, List[Dict[str, Any]]]:
        """Group tasks by priority level"""
        groups = {}
        
        for task in tasks:
            priority = task.get("priority", 0.5)
            # Round to nearest 0.1 for grouping
            priority_group = round(priority, 1)
            
            if priority_group not in groups:
                groups[priority_group] = []
            
            groups[priority_group].append(task)
        
        return groups
    
    async def _execute_task_group(self, tasks: List[Dict[str, Any]], 
                                priority: float, 
                                context: SharedContext) -> List[Any]:
        """Execute a group of tasks with same priority"""
        # Add module tracking to tasks
        for task in tasks:
            if "metadata" not in task:
                task["metadata"] = {}
            task["metadata"]["execution_priority"] = priority
            task["metadata"]["context_id"] = context.conversation_id if context else None
        
        # Execute using parent method with context
        return await self.execute_tools(tasks, priority=priority, context=context)
    
    async def _update_execution_metrics(self, results: List[Any], context: SharedContext):
        """Update execution metrics with context information"""
        # Track module-specific execution times
        for result in results:
            if hasattr(result, "metadata") and "source" in result.metadata:
                source = result.metadata["source"]
                if source not in self.context_aware_metrics["module_execution_times"]:
                    self.context_aware_metrics["module_execution_times"][source] = []
                
                self.context_aware_metrics["module_execution_times"][source].append(
                    result.execution_time
                )
    
    def _calculate_average_execution_time(self, results: List[Any]) -> float:
        """Calculate average execution time from results"""
        if not results:
            return 0.0
        
        times = [r.execution_time for r in results if hasattr(r, "execution_time")]
        return sum(times) / len(times) if times else 0.0
    
    def _calculate_success_rate(self, results: List[Any]) -> float:
        """Calculate success rate from results"""
        if not results:
            return 1.0
        
        successes = sum(1 for r in results if hasattr(r, "success") and r.success)
        return successes / len(results)
    
    def _get_current_capacity(self) -> float:
        """Get current execution capacity"""
        used = len(self.executing_tasks)
        total = self.original_executor.max_concurrent
        return 1.0 - (used / total) if total > 0 else 0.0
    
    def _get_execution_state(self) -> Dict[str, Any]:
        """Get current execution state"""
        return {
            "active_executions": len(self.executing_tasks),
            "max_concurrent": self.original_executor.max_concurrent,
            "capacity_percentage": self._get_current_capacity() * 100,
            "priority_queue_size": self.priority_queue.qsize(),
            "total_executions": len(self.execution_history)
        }
    
    async def _analyze_execution_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in execution history"""
        patterns = {
            "peak_times": [],
            "average_batch_size": 0,
            "priority_distribution": {},
            "common_failure_patterns": []
        }
        
        if not self.execution_history:
            return patterns
        
        # Analyze timing patterns
        hour_counts = {}
        for execution in self.execution_history:
            timestamp = datetime.fromisoformat(execution["timestamp"])
            hour = timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours
        if hour_counts:
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            patterns["peak_times"] = [h for h, _ in sorted_hours[:3]]
        
        # Analyze batch sizes
        batch_sizes = [e["tool_count"] for e in self.execution_history]
        if batch_sizes:
            patterns["average_batch_size"] = sum(batch_sizes) / len(batch_sizes)
        
        # Analyze priority distribution
        for execution in self.execution_history:
            priority = round(execution.get("priority", 0.5), 1)
            patterns["priority_distribution"][priority] = patterns["priority_distribution"].get(priority, 0) + 1
        
        return patterns
    
    async def _identify_bottlenecks(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify execution bottlenecks"""
        bottlenecks = []
        
        # Check for long execution times
        if self.execution_history:
            recent_times = [e["execution_time"] for e in self.execution_history[-20:]]
            avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
            
            if avg_time > 2.0:
                bottlenecks.append({
                    "type": "slow_execution",
                    "severity": "high",
                    "average_time": avg_time,
                    "recommendation": "Consider optimizing tool implementations"
                })
        
        # Check for queue buildup
        if self.priority_queue.qsize() > 10:
            bottlenecks.append({
                "type": "queue_buildup",
                "severity": "medium",
                "queue_size": self.priority_queue.qsize(),
                "recommendation": "Increase parallel execution capacity"
            })
        
        # Check for resource conflicts
        if self.context_aware_metrics["resource_conflicts_resolved"] > 5:
            bottlenecks.append({
                "type": "resource_conflicts",
                "severity": "medium",
                "conflict_count": self.context_aware_metrics["resource_conflicts_resolved"],
                "recommendation": "Review resource allocation strategy"
            })
        
        return bottlenecks
    
    async def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization metrics"""
        utilization = {
            "execution_capacity": 1.0 - self._get_current_capacity(),
            "queue_utilization": min(1.0, self.priority_queue.qsize() / 20),  # Assume 20 is high
            "efficiency_score": 0.0
        }
        
        # Calculate efficiency based on success rate and execution time
        if self.execution_history:
            recent = self.execution_history[-20:]
            avg_success = sum(e.get("success_rate", 1.0) for e in recent) / len(recent)
            avg_time = sum(e["execution_time"] for e in recent) / len(recent)
            
            # Efficiency is high when success is high and time is low
            time_factor = max(0, 1.0 - (avg_time / 5.0))  # 5 seconds as baseline
            utilization["efficiency_score"] = avg_success * time_factor
        
        return utilization
    
    async def _analyze_cross_module_dependencies(self, messages: List[Any]) -> Dict[str, List[str]]:
        """Analyze dependencies between modules based on messages"""
        dependencies = {}
        
        # Track which modules send messages to which
        for message in messages:
            source = message.get("module_name", "unknown")
            targets = message.get("target_modules", [])
            
            if source not in dependencies:
                dependencies[source] = []
            
            dependencies[source].extend(targets)
        
        # Remove duplicates
        for source in dependencies:
            dependencies[source] = list(set(dependencies[source]))
        
        return dependencies
    
    async def _generate_optimization_recommendations(self, patterns: Dict[str, Any],
                                                   bottlenecks: List[Dict[str, Any]],
                                                   utilization: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Based on patterns
        if patterns.get("average_batch_size", 0) > self.original_executor.max_concurrent:
            recommendations.append(
                f"Batch size ({patterns['average_batch_size']:.1f}) exceeds capacity - consider smaller batches"
            )
        
        # Based on bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck["severity"] == "high":
                recommendations.append(bottleneck["recommendation"])
        
        # Based on utilization
        if utilization["execution_capacity"] > 0.9:
            recommendations.append("System at high capacity - consider scaling or optimization")
        elif utilization["execution_capacity"] < 0.3:
            recommendations.append("Low utilization - could handle more parallel tasks")
        
        if utilization["efficiency_score"] < 0.5:
            recommendations.append("Low efficiency score - review task implementations")
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def _analyze_performance_trends(self, recent_executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends from recent executions"""
        trends = {
            "execution_time_trend": "stable",
            "success_rate_trend": "stable",
            "degrading_performance": False,
            "improving_performance": False
        }
        
        if len(recent_executions) < 5:
            return trends
        
        # Analyze execution time trends
        times = [e["execution_time"] for e in recent_executions]
        first_half_avg = sum(times[:len(times)//2]) / (len(times)//2)
        second_half_avg = sum(times[len(times)//2:]) / (len(times) - len(times)//2)
        
        if second_half_avg > first_half_avg * 1.2:
            trends["execution_time_trend"] = "increasing"
            trends["degrading_performance"] = True
        elif second_half_avg < first_half_avg * 0.8:
            trends["execution_time_trend"] = "decreasing"
            trends["improving_performance"] = True
        
        # Analyze success rate trends
        success_rates = [e.get("success_rate", 1.0) for e in recent_executions]
        first_half_success = sum(success_rates[:len(success_rates)//2]) / (len(success_rates)//2)
        second_half_success = sum(success_rates[len(success_rates)//2:]) / (len(success_rates) - len(success_rates)//2)
        
        if second_half_success < first_half_success * 0.9:
            trends["success_rate_trend"] = "decreasing"
            trends["degrading_performance"] = True
        elif second_half_success > first_half_success * 1.1:
            trends["success_rate_trend"] = "increasing"
            trends["improving_performance"] = True
        
        return trends
    
    async def _calculate_resource_efficiency(self) -> float:
        """Calculate overall resource efficiency"""
        if not self.execution_history:
            return 1.0
        
        # Consider multiple factors
        recent = self.execution_history[-20:]
        
        # Success rate component
        avg_success = sum(e.get("success_rate", 1.0) for e in recent) / len(recent)
        
        # Time efficiency component
        avg_time = sum(e["execution_time"] for e in recent) / len(recent)
        time_efficiency = max(0, 1.0 - (avg_time / 3.0))  # 3 seconds as baseline
        
        # Capacity utilization component
        capacity_efficiency = 0.7  # Optimal is around 70% utilization
        current_capacity = self._get_current_capacity()
        if current_capacity > 0.3:
            capacity_efficiency = 1.0 - abs(0.7 - (1.0 - current_capacity))
        
        # Combine factors
        efficiency = (avg_success * 0.4 + time_efficiency * 0.4 + capacity_efficiency * 0.2)
        
        return min(1.0, max(0.0, efficiency))
    
    def _extract_key_insights(self, performance_trends: Dict[str, Any]) -> List[str]:
        """Extract key insights from performance trends"""
        insights = []
        
        if performance_trends.get("degrading_performance"):
            insights.append("Performance is degrading - investigation recommended")
        
        if performance_trends.get("improving_performance"):
            insights.append("Performance improvements detected - optimizations working")
        
        if performance_trends["execution_time_trend"] == "increasing":
            insights.append("Execution times are increasing - possible system load issue")
        
        if performance_trends["success_rate_trend"] == "decreasing":
            insights.append("Success rates declining - check error patterns")
        
        return insights
    
    def _assess_execution_health(self) -> str:
        """Assess overall execution system health"""
        health_score = 1.0
        
        # Factor in capacity
        capacity = self._get_current_capacity()
        if capacity < 0.2:
            health_score -= 0.3  # Overloaded
        elif capacity > 0.9:
            health_score -= 0.1  # Underutilized
        
        # Factor in queue size
        if self.priority_queue.qsize() > 10:
            health_score -= 0.2
        
        # Factor in recent performance
        if self.execution_history:
            recent_success = sum(e.get("success_rate", 1.0) for e in self.execution_history[-10:]) / 10
            health_score = health_score * 0.7 + recent_success * 0.3
        
        if health_score > 0.8:
            return "healthy"
        elif health_score > 0.6:
            return "moderate"
        else:
            return "needs_attention"
    
    def _summarize_context(self, context: Optional[SharedContext]) -> str:
        """Create summary of context for tracking"""
        if not context:
            return "no_context"
        
        elements = []
        
        if context.user_id:
            elements.append(f"user:{context.user_id}")
        
        if context.conversation_id:
            elements.append(f"conv:{context.conversation_id[:8]}")
        
        if context.emotional_state:
            emotion = context.emotional_state.get("primary_emotion", {}).get("name")
            if emotion:
                elements.append(f"emotion:{emotion}")
        
        return "_".join(elements) if elements else "minimal_context"
    
    async def _optimize_tool_execution_order(self, tools_info: List[Dict[str, Any]], 
                                           context: SharedContext) -> List[Dict[str, Any]]:
        """Optimize tool execution order based on context"""
        # Sort by priority if available
        tools_with_priority = []
        tools_without_priority = []
        
        for tool in tools_info:
            if "priority" in tool or (tool.get("metadata", {}).get("priority")):
                tools_with_priority.append(tool)
            else:
                tools_without_priority.append(tool)
        
        # Sort prioritized tools
        tools_with_priority.sort(key=lambda x: x.get("priority", x.get("metadata", {}).get("priority", 0.5)), reverse=True)
        
        # Combine - high priority first
        return tools_with_priority + tools_without_priority
    
    # Delegate all other methods to the original executor
    def __getattr__(self, name):
        """Delegate any missing methods to the original executor"""
        return getattr(self.original_executor, name)
