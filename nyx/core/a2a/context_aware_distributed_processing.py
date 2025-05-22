# nyx/core/a2a/context_aware_distributed_processing.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareDistributedProcessing(ContextAwareModule):
    """
    Enhanced DistributedProcessingManager with context distribution capabilities
    """
    
    def __init__(self, original_distributed_manager):
        super().__init__("distributed_processing")
        self.original_manager = original_distributed_manager
        self.context_subscriptions = [
            "task_request", "resource_availability", "subsystem_status",
            "priority_change", "task_completion", "performance_update"
        ]
        
        # Track context-aware task scheduling
        self.context_task_mapping = {}  # context_id -> task_ids
        self.task_context_requirements = {}  # task_id -> required_context
    
    async def on_context_received(self, context: SharedContext):
        """Initialize distributed processing for this context"""
        logger.debug(f"DistributedProcessing received context with {len(context.active_modules)} active modules")
        
        # Analyze processing requirements
        processing_requirements = await self._analyze_processing_requirements(context)
        
        # Create initial task distribution plan
        distribution_plan = await self._create_distribution_plan(context, processing_requirements)
        
        # Send distributed processing context
        await self.send_context_update(
            update_type="distributed_processing_initialized",
            data={
                "active_modules": list(context.active_modules),
                "processing_requirements": processing_requirements,
                "distribution_plan": distribution_plan,
                "available_resources": self._get_available_resources(),
                "estimated_completion_time": self._estimate_completion_time(distribution_plan)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates affecting distributed processing"""
        
        if update.update_type == "task_request":
            # New task requested by a module
            task_data = update.data
            await self._handle_task_request(
                source_module=update.source_module,
                task_data=task_data,
                priority=update.priority
            )
        
        elif update.update_type == "resource_availability":
            # Resource availability changed
            resource_data = update.data
            await self._adjust_task_allocation(resource_data)
        
        elif update.update_type == "subsystem_status":
            # Subsystem status change affects task routing
            status_data = update.data
            subsystem = status_data.get("subsystem")
            status = status_data.get("status")
            
            if status == "unavailable":
                await self._reroute_subsystem_tasks(subsystem)
            elif status == "available":
                await self._rebalance_task_distribution()
        
        elif update.update_type == "priority_change":
            # Task priority changed
            task_id = update.data.get("task_id")
            new_priority = update.data.get("new_priority")
            
            if task_id and new_priority:
                await self._update_task_priority(task_id, new_priority)
        
        elif update.update_type == "task_completion":
            # Task completed, update tracking
            task_id = update.data.get("task_id")
            result = update.data.get("result")
            
            await self._handle_task_completion(task_id, result, update.source_module)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for distributed execution needs"""
        # Analyze input for parallelizable components
        parallel_analysis = await self._analyze_parallelization_opportunities(context)
        
        # Create tasks for parallel execution
        created_tasks = []
        if parallel_analysis["parallelizable"]:
            for component in parallel_analysis["components"]:
                task_id = await self._create_context_aware_task(
                    component=component,
                    context=context,
                    dependencies=component.get("dependencies", [])
                )
                created_tasks.append(task_id)
        
        # Schedule tasks
        if created_tasks:
            scheduling_result = await self._schedule_context_tasks(created_tasks, context)
            
            # Notify about distributed processing
            await self.send_context_update(
                update_type="distributed_tasks_created",
                data={
                    "task_count": len(created_tasks),
                    "task_ids": created_tasks,
                    "scheduling": scheduling_result,
                    "context_id": id(context)
                }
            )
        
        return {
            "parallel_analysis": parallel_analysis,
            "created_tasks": created_tasks,
            "distributed_processing": len(created_tasks) > 0,
            "processing_strategy": parallel_analysis.get("strategy", "sequential")
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze distributed processing state and optimization opportunities"""
        messages = await self.get_cross_module_messages()
        
        # Analyze current task distribution
        distribution_analysis = await self._analyze_task_distribution()
        
        # Identify bottlenecks
        bottleneck_analysis = await self._analyze_processing_bottlenecks(messages)
        
        # Optimization recommendations
        optimization_recs = await self._generate_optimization_recommendations(
            distribution_analysis,
            bottleneck_analysis
        )
        
        # Resource utilization analysis
        resource_utilization = await self._analyze_resource_utilization()
        
        return {
            "distribution_analysis": distribution_analysis,
            "bottlenecks": bottleneck_analysis,
            "optimization_recommendations": optimization_recs,
            "resource_utilization": resource_utilization,
            "active_context_tasks": len(self.context_task_mapping.get(id(context), [])),
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize distributed processing results"""
        messages = await self.get_cross_module_messages()
        
        # Collect results from distributed tasks
        context_tasks = self.context_task_mapping.get(id(context), [])
        task_results = await self._collect_task_results(context_tasks)
        
        # Synthesize results
        synthesis = {
            "distributed_results": task_results,
            "processing_summary": await self._generate_processing_summary(task_results),
            "performance_metrics": await self._calculate_performance_metrics(context_tasks),
            "completion_status": self._get_completion_status(context_tasks)
        }
        
        # Send synthesis complete notification
        if synthesis["completion_status"]["all_complete"]:
            await self.send_context_update(
                update_type="distributed_processing_complete",
                data={
                    "context_id": id(context),
                    "total_tasks": len(context_tasks),
                    "successful_tasks": synthesis["completion_status"]["successful"],
                    "failed_tasks": synthesis["completion_status"]["failed"],
                    "performance_metrics": synthesis["performance_metrics"]
                }
            )
        
        return synthesis
    
    # Helper methods
    
    async def _analyze_processing_requirements(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze what processing is required for this context"""
        requirements = {
            "module_requirements": {},
            "total_estimated_load": 0.0,
            "parallelization_potential": 0.0,
            "critical_path": []
        }
        
        # Analyze each active module's requirements
        for module_name in context.active_modules:
            module_req = {
                "estimated_load": self._estimate_module_load(module_name, context),
                "can_parallelize": self._can_module_parallelize(module_name),
                "dependencies": self._get_module_dependencies(module_name)
            }
            requirements["module_requirements"][module_name] = module_req
            requirements["total_estimated_load"] += module_req["estimated_load"]
        
        # Calculate parallelization potential
        parallelizable_modules = sum(1 for m in requirements["module_requirements"].values() 
                                   if m["can_parallelize"])
        requirements["parallelization_potential"] = parallelizable_modules / max(1, len(context.active_modules))
        
        # Identify critical path
        requirements["critical_path"] = self._identify_critical_path(requirements["module_requirements"])
        
        return requirements
    
    async def _create_distribution_plan(self, context: SharedContext, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan for distributing processing tasks"""
        plan = {
            "parallel_groups": [],
            "sequential_tasks": [],
            "resource_allocation": {},
            "estimated_duration": 0.0
        }
        
        # Group modules by dependencies
        dependency_levels = self._group_by_dependencies(requirements["module_requirements"])
        
        # Create parallel groups
        for level, modules in enumerate(dependency_levels):
            if len(modules) > 1:
                plan["parallel_groups"].append({
                    "level": level,
                    "modules": modules,
                    "can_execute_parallel": True
                })
            else:
                plan["sequential_tasks"].extend(modules)
        
        # Allocate resources
        total_resources = 1.0
        for module, req in requirements["module_requirements"].items():
            allocation = req["estimated_load"] / requirements["total_estimated_load"]
            plan["resource_allocation"][module] = min(0.5, allocation)  # Cap at 50% per module
        
        # Estimate duration
        plan["estimated_duration"] = self._estimate_plan_duration(plan, requirements)
        
        return plan
    
    def _get_available_resources(self) -> Dict[str, float]:
        """Get current available resources"""
        return {
            "cpu": 1.0 - (len(self.original_manager.active_tasks) / self.original_manager.max_parallel_tasks),
            "memory": 0.8,  # Placeholder
            "task_slots": self.original_manager.max_parallel_tasks - len(self.original_manager.active_tasks)
        }
    
    def _estimate_completion_time(self, plan: Dict[str, Any]) -> float:
        """Estimate completion time for a distribution plan"""
        # Simplified estimation
        parallel_time = sum(group.get("estimated_duration", 0.1) 
                          for group in plan.get("parallel_groups", []))
        sequential_time = len(plan.get("sequential_tasks", [])) * 0.1
        
        return parallel_time + sequential_time
    
    async def _handle_task_request(self, source_module: str, task_data: Dict[str, Any], priority: ContextPriority):
        """Handle a task request from a module"""
        # Create task ID
        task_id = f"{source_module}_{datetime.now().timestamp()}"
        
        # Map priority
        numeric_priority = {
            ContextPriority.CRITICAL: 5,
            ContextPriority.HIGH: 4,
            ContextPriority.NORMAL: 3,
            ContextPriority.LOW: 2
        }.get(priority, 3)
        
        # Register task with original manager
        async def module_task():
            # Execute the task
            result = await self._execute_module_task(source_module, task_data)
            return result
        
        registration_result = await self.original_manager.register_task(
            task_id=task_id,
            subsystem_name=source_module,
            coroutine=module_task(),
            dependencies=task_data.get("dependencies", []),
            priority=numeric_priority,
            group=task_data.get("group", "general")
        )
        
        if registration_result["success"]:
            # Track context requirements
            self.task_context_requirements[task_id] = {
                "source_module": source_module,
                "requires_context": task_data.get("requires_context", []),
                "context_data": task_data.get("context_data", {})
            }
            
            # Send confirmation
            await self.send_context_update(
                update_type="task_registered",
                data={
                    "task_id": task_id,
                    "source_module": source_module,
                    "priority": numeric_priority
                },
                target_modules=[source_module],
                scope=ContextScope.TARGETED
            )
    
    async def _execute_module_task(self, module_name: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task for a module"""
        try:
            # Get the context system reference
            if not hasattr(self, '_context_system') or not self._context_system:
                return {"success": False, "error": "No context system available"}
            
            # Get NyxBrain reference through context system
            nyx_brain = getattr(self._context_system, 'nyx_brain', None)
            if not nyx_brain:
                return {"success": False, "error": "No NyxBrain reference available"}
            
            # Get module reference
            module = getattr(nyx_brain, module_name, None)
            if not module:
                return {"success": False, "error": f"Module {module_name} not found"}
            
            # Determine execution method based on module capabilities
            execution_result = None
            
            # Try context-aware execution first
            if isinstance(module, ContextAwareModule):
                # Module is context-aware, use context-based execution
                context = task_data.get("context")
                if context and isinstance(context, SharedContext):
                    # Execute through context-aware interface
                    if task_data.get("stage") == "input":
                        execution_result = await module.process_input(context)
                    elif task_data.get("stage") == "analysis":
                        execution_result = await module.process_analysis(context)
                    elif task_data.get("stage") == "synthesis":
                        execution_result = await module.process_synthesis(context)
                    else:
                        # General execution
                        if hasattr(module, 'execute_task'):
                            execution_result = await module.execute_task(task_data)
            
            # Try standard execution methods
            if execution_result is None:
                task_type = task_data.get("task_type", "process")
                
                if task_type == "process" and hasattr(module, 'process_input'):
                    execution_result = await module.process_input(
                        task_data.get("input", ""),
                        task_data.get("context", {})
                    )
                elif task_type == "analyze" and hasattr(module, 'analyze'):
                    execution_result = await module.analyze(task_data.get("data", {}))
                elif task_type == "execute" and hasattr(module, 'execute'):
                    execution_result = await module.execute(task_data.get("command", {}))
                elif hasattr(module, 'execute_task'):
                    execution_result = await module.execute_task(task_data)
                else:
                    return {
                        "success": False, 
                        "error": f"Module {module_name} has no compatible execution method for task type: {task_type}"
                    }
            
            # Process result
            if execution_result is not None:
                return {
                    "success": True,
                    "result": execution_result,
                    "module": module_name,
                    "task_type": task_data.get("task_type", "unknown"),
                    "execution_time": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Execution returned None",
                    "module": module_name
                }
            
        except Exception as e:
            logger.error(f"Error executing task for {module_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "module": module_name,
                "traceback": True
            }

    
    async def _adjust_task_allocation(self, resource_data: Dict[str, Any]):
        """Adjust task allocation based on resource availability"""
        new_resources = resource_data.get("available_resources", {})
        
        # Update max parallel tasks if needed
        if "task_slots" in new_resources:
            self.original_manager.max_parallel_tasks = int(new_resources["task_slots"])
        
        # Rebalance if resources increased
        if new_resources.get("cpu", 0) > 0.5:
            await self._rebalance_task_distribution()
    
    async def _reroute_subsystem_tasks(self, unavailable_subsystem: str):
        """Reroute tasks from unavailable subsystem"""
        # Find tasks assigned to unavailable subsystem
        affected_tasks = []
        for task_id, task in self.original_manager.task_registry.items():
            if task.subsystem_name == unavailable_subsystem and not task.completed:
                affected_tasks.append(task_id)
        
        # Cancel affected tasks
        for task_id in affected_tasks:
            if task_id in self.original_manager.active_tasks:
                self.original_manager.active_tasks[task_id].cancel()
        
        # Notify modules about rerouting
        await self.send_context_update(
            update_type="tasks_rerouted",
            data={
                "unavailable_subsystem": unavailable_subsystem,
                "affected_tasks": affected_tasks,
                "action": "cancelled"
            }
        )
    
    async def _rebalance_task_distribution(self):
        """Rebalance task distribution across available subsystems"""
        try:
            # Get current distribution
            distribution = defaultdict(list)
            pending_tasks = []
            
            for task_id, task in self.original_manager.task_registry.items():
                if not task.completed:
                    if task.started:
                        distribution[task.subsystem_name].append(task_id)
                    else:
                        pending_tasks.append((task_id, task))
            
            if not distribution and not pending_tasks:
                return
            
            # Calculate ideal distribution
            total_active = sum(len(tasks) for tasks in distribution.values())
            total_pending = len(pending_tasks)
            total_tasks = total_active + total_pending
            
            if total_tasks == 0:
                return
            
            # Get available subsystems
            available_subsystems = set()
            if hasattr(self, '_context_system') and self._context_system:
                nyx_brain = getattr(self._context_system, 'nyx_brain', None)
                if nyx_brain:
                    # Get all active modules that can process tasks
                    for module_name in getattr(nyx_brain, 'active_modules', []):
                        if hasattr(nyx_brain, module_name):
                            available_subsystems.add(module_name)
            
            if not available_subsystems:
                logger.warning("No available subsystems for task redistribution")
                return
            
            ideal_load = total_tasks / len(available_subsystems)
            
            # Identify overloaded and underloaded subsystems
            overloaded = []
            underloaded = []
            
            for subsystem in available_subsystems:
                current_load = len(distribution.get(subsystem, []))
                if current_load > ideal_load * 1.5:
                    overloaded.append((subsystem, current_load))
                elif current_load < ideal_load * 0.5:
                    underloaded.append((subsystem, current_load))
            
            # Redistribute tasks
            redistributed = 0
            
            # First, handle pending tasks
            for task_id, task in pending_tasks:
                if underloaded:
                    # Assign to most underloaded subsystem
                    target_subsystem = min(underloaded, key=lambda x: x[1])[0]
                    
                    # Update task assignment
                    task.subsystem_name = target_subsystem
                    redistributed += 1
                    
                    # Update load tracking
                    for i, (subsys, load) in enumerate(underloaded):
                        if subsys == target_subsystem:
                            underloaded[i] = (subsys, load + 1)
                            if load + 1 >= ideal_load * 0.5:
                                underloaded.pop(i)
                            break
            
            # Then, move tasks from overloaded to underloaded
            for overloaded_subsys, load in overloaded:
                if not underloaded:
                    break
                
                tasks_to_move = int(load - ideal_load * 1.2)  # Keep some buffer
                moved = 0
                
                for task_id in distribution[overloaded_subsys][:tasks_to_move]:
                    if task_id in self.original_manager.active_tasks:
                        # Skip actively running tasks
                        continue
                    
                    if underloaded:
                        target_subsystem = min(underloaded, key=lambda x: x[1])[0]
                        
                        # Update task
                        if task_id in self.original_manager.task_registry:
                            self.original_manager.task_registry[task_id].subsystem_name = target_subsystem
                            moved += 1
                            redistributed += 1
                            
                            # Update load tracking
                            for i, (subsys, load) in enumerate(underloaded):
                                if subsys == target_subsystem:
                                    underloaded[i] = (subsys, load + 1)
                                    if load + 1 >= ideal_load * 0.5:
                                        underloaded.pop(i)
                                    break
                
                logger.info(f"Moved {moved} tasks from {overloaded_subsys} to other subsystems")
            
            if redistributed > 0:
                logger.info(f"Rebalanced task distribution: redistributed {redistributed} tasks")
                
                # Notify about rebalancing
                await self.send_context_update(
                    update_type="distribution_rebalanced",
                    data={
                        "redistributed_count": redistributed,
                        "current_distribution": {
                            subsys: len(tasks) for subsys, tasks in distribution.items()
                        },
                        "ideal_load": ideal_load
                    }
                )
            
        except Exception as e:
            logger.error(f"Error rebalancing task distribution: {e}")
    
    async def _update_task_priority(self, task_id: str, new_priority: int):
        """Update task priority"""
        if task_id in self.original_manager.task_registry:
            task = self.original_manager.task_registry[task_id]
            old_priority = task.priority
            task.priority = new_priority
            
            logger.info(f"Updated task {task_id} priority: {old_priority} -> {new_priority}")
            
            # Notify about priority change
            await self.send_context_update(
                update_type="task_priority_updated",
                data={
                    "task_id": task_id,
                    "old_priority": old_priority,
                    "new_priority": new_priority
                }
            )
    
    async def _handle_task_completion(self, task_id: str, result: Any, source_module: str):
        """Handle task completion"""
        # Update tracking
        for context_id, task_ids in self.context_task_mapping.items():
            if task_id in task_ids:
                # Check if all context tasks complete
                all_complete = all(
                    self.original_manager.task_registry.get(tid, {}).get("completed", False)
                    for tid in task_ids
                )
                
                if all_complete:
                    await self.send_context_update(
                        update_type="context_tasks_complete",
                        data={
                            "context_id": context_id,
                            "total_tasks": len(task_ids),
                            "completion_time": datetime.now().isoformat()
                        }
                    )
                break
        
        # Send completion notification to source module
        await self.send_context_update(
            update_type="task_completed",
            data={
                "task_id": task_id,
                "result": result,
                "success": result.get("success", False) if isinstance(result, dict) else True
            },
            target_modules=[source_module],
            scope=ContextScope.TARGETED
        )
    
    async def _analyze_parallelization_opportunities(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze opportunities for parallel execution"""
        analysis = {
            "parallelizable": False,
            "components": [],
            "strategy": "sequential",
            "estimated_speedup": 1.0
        }
        
        # Check if multiple modules are active
        if len(context.active_modules) < 2:
            return analysis
        
        # Identify independent modules
        independent_modules = []
        dependent_modules = []
        
        for module in context.active_modules:
            deps = self._get_module_dependencies(module)
            if not any(dep in context.active_modules for dep in deps):
                independent_modules.append(module)
            else:
                dependent_modules.append(module)
        
        if len(independent_modules) >= 2:
            analysis["parallelizable"] = True
            analysis["strategy"] = "parallel_independent"
            
            # Create components for parallel execution
            for module in independent_modules:
                analysis["components"].append({
                    "module": module,
                    "type": "independent",
                    "dependencies": [],
                    "estimated_duration": 0.1  # Placeholder
                })
            
            # Estimate speedup
            analysis["estimated_speedup"] = min(len(independent_modules), 
                                               self.original_manager.max_parallel_tasks)
        
        return analysis
    
    async def _create_context_aware_task(self, component: Dict[str, Any], 
                                       context: SharedContext, 
                                       dependencies: List[str]) -> str:
        """Create a context-aware task"""
        module_name = component["module"]
        task_id = f"context_{id(context)}_{module_name}_{datetime.now().timestamp()}"
        
        # Create task coroutine
        async def context_task():
            # Execute module processing with context
            if hasattr(self, '_context_system') and hasattr(self._context_system, 'nyx_brain'):
                module = getattr(self._context_system.nyx_brain, module_name, None)
                
                if module and hasattr(module, 'process_input'):
                    result = await module.process_input(context)
                    return result
            
            return {"error": f"Module {module_name} not found"}
        
        # Register with original manager
        await self.original_manager.register_task(
            task_id=task_id,
            subsystem_name=module_name,
            coroutine=context_task(),
            dependencies=dependencies,
            priority=3,  # Normal priority
            group="context_processing"
        )
        
        # Track context mapping
        context_id = id(context)
        if context_id not in self.context_task_mapping:
            self.context_task_mapping[context_id] = []
        self.context_task_mapping[context_id].append(task_id)
        
        return task_id
    
    async def _schedule_context_tasks(self, task_ids: List[str], context: SharedContext) -> Dict[str, Any]:
        """Schedule context-aware tasks"""
        # Use original manager's execution
        scheduling_result = {
            "scheduled_tasks": task_ids,
            "execution_strategy": "parallel" if len(task_ids) > 1 else "sequential",
            "estimated_completion": self._estimate_task_completion(task_ids)
        }
        
        # Start execution
        asyncio.create_task(self.original_manager.execute_tasks())
        
        return scheduling_result
    
    async def _analyze_task_distribution(self) -> Dict[str, Any]:
        """Analyze current task distribution"""
        distribution = {
            "by_subsystem": {},
            "by_priority": {},
            "by_status": {
                "pending": 0,
                "active": 0,
                "completed": 0,
                "failed": 0
            }
        }
        
        for task in self.original_manager.task_registry.values():
            # By subsystem
            subsystem = task.subsystem_name
            if subsystem not in distribution["by_subsystem"]:
                distribution["by_subsystem"][subsystem] = {"count": 0, "active": 0}
            distribution["by_subsystem"][subsystem]["count"] += 1
            
            # By priority
            priority = task.priority
            distribution["by_priority"][priority] = distribution["by_priority"].get(priority, 0) + 1
            
            # By status
            if task.completed:
                if task.error:
                    distribution["by_status"]["failed"] += 1
                else:
                    distribution["by_status"]["completed"] += 1
            elif task.started:
                distribution["by_status"]["active"] += 1
                distribution["by_subsystem"][subsystem]["active"] += 1
            else:
                distribution["by_status"]["pending"] += 1
        
        return distribution
    
    async def _analyze_processing_bottlenecks(self, messages: Dict) -> Dict[str, Any]:
        """Analyze processing bottlenecks"""
        bottlenecks = {
            "overloaded_subsystems": [],
            "blocking_tasks": [],
            "resource_constraints": []
        }
        
        # Check subsystem loads
        distribution = await self._analyze_task_distribution()
        avg_load = sum(s["count"] for s in distribution["by_subsystem"].values()) / max(1, len(distribution["by_subsystem"]))
        
        for subsystem, stats in distribution["by_subsystem"].items():
            if stats["count"] > avg_load * 1.5:
                bottlenecks["overloaded_subsystems"].append({
                    "subsystem": subsystem,
                    "load": stats["count"],
                    "active": stats["active"]
                })
        
        # Check for long-running tasks
        for task_id, task in self.original_manager.task_registry.items():
            if task.started and not task.completed and task.start_time:
                duration = (datetime.now() - task.start_time).total_seconds()
                if duration > 5.0:  # Tasks running > 5 seconds
                    bottlenecks["blocking_tasks"].append({
                        "task_id": task_id,
                        "subsystem": task.subsystem_name,
                        "duration": duration
                    })
        
        # Check resource constraints
        available = self._get_available_resources()
        if available["cpu"] < 0.2:
            bottlenecks["resource_constraints"].append("cpu_limited")
        if available["task_slots"] < 2:
            bottlenecks["resource_constraints"].append("task_slots_limited")
        
        return bottlenecks
    
    async def _generate_optimization_recommendations(self, distribution: Dict[str, Any], 
                                                   bottlenecks: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check for overloaded subsystems
        if bottlenecks["overloaded_subsystems"]:
            recommendations.append(
                f"Consider load balancing for subsystems: {[s['subsystem'] for s in bottlenecks['overloaded_subsystems']]}"
            )
        
        # Check for blocking tasks
        if bottlenecks["blocking_tasks"]:
            recommendations.append(
                f"Long-running tasks detected: {len(bottlenecks['blocking_tasks'])} tasks exceeding 5s"
            )
        
        # Check priority distribution
        high_priority_count = distribution["by_priority"].get(4, 0) + distribution["by_priority"].get(5, 0)
        total_tasks = sum(distribution["by_priority"].values())
        
        if total_tasks > 0 and high_priority_count / total_tasks > 0.5:
            recommendations.append(
                "High proportion of high-priority tasks - consider priority adjustment"
            )
        
        # Check parallelization
        if distribution["by_status"]["active"] < 2 and distribution["by_status"]["pending"] > 5:
            recommendations.append(
                "Low parallelization - increase concurrent task execution"
            )
        
        return recommendations
    
    async def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization"""
        available = self._get_available_resources()
        active_tasks = len(self.original_manager.active_tasks)
        max_tasks = self.original_manager.max_parallel_tasks
        
        return {
            "cpu_utilization": 1.0 - available["cpu"],
            "task_slot_utilization": active_tasks / max_tasks if max_tasks > 0 else 0,
            "memory_utilization": 1.0 - available["memory"],
            "efficiency_score": self._calculate_efficiency_score(available, active_tasks)
        }
    
    def _calculate_efficiency_score(self, available: Dict[str, float], active_tasks: int) -> float:
        """Calculate efficiency score"""
        # Simple efficiency calculation
        cpu_efficiency = 1.0 - available["cpu"]
        slot_efficiency = active_tasks / self.original_manager.max_parallel_tasks if self.original_manager.max_parallel_tasks > 0 else 0
        
        # Penalize if resources are available but not used
        if available["cpu"] > 0.5 and slot_efficiency < 0.5:
            return slot_efficiency * 0.5
        
        return (cpu_efficiency + slot_efficiency) / 2
    
    async def _collect_task_results(self, task_ids: List[str]) -> Dict[str, Any]:
        """Collect results from tasks"""
        results = {}
        
        for task_id in task_ids:
            if task_id in self.original_manager.task_registry:
                task = self.original_manager.task_registry[task_id]
                if task.completed:
                    results[task_id] = {
                        "success": not bool(task.error),
                        "result": task.result,
                        "error": task.error,
                        "duration": (task.end_time - task.start_time).total_seconds() if task.start_time and task.end_time else None
                    }
                else:
                    results[task_id] = {
                        "success": False,
                        "result": None,
                        "error": "Task not completed",
                        "started": task.started
                    }
        
        return results
    
    async def _generate_processing_summary(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of processing results"""
        total_tasks = len(task_results)
        successful_tasks = sum(1 for r in task_results.values() if r["success"])
        failed_tasks = total_tasks - successful_tasks
        
        durations = [r["duration"] for r in task_results.values() if r.get("duration")]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_tasks": total_tasks,
            "successful": successful_tasks,
            "failed": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_duration": avg_duration,
            "total_duration": sum(durations)
        }
    
    async def _calculate_performance_metrics(self, task_ids: List[str]) -> Dict[str, Any]:
        """Calculate performance metrics for tasks"""
        metrics = {
            "parallelization_achieved": 0.0,
            "resource_efficiency": 0.0,
            "throughput": 0.0
        }
        
        if not task_ids:
            return metrics
        
        # Calculate parallelization
        max_concurrent = 0
        for task_id in task_ids:
            if task_id in self.original_manager.active_tasks:
                max_concurrent += 1
        
        metrics["parallelization_achieved"] = max_concurrent / len(task_ids) if task_ids else 0
        
        # Resource efficiency from original manager
        metrics["resource_efficiency"] = await self._analyze_resource_utilization()
        
        # Throughput
        completed_tasks = sum(1 for tid in task_ids 
                            if tid in self.original_manager.task_registry 
                            and self.original_manager.task_registry[tid].completed)
        
        # Simple throughput calculation
        metrics["throughput"] = completed_tasks / max(1, len(task_ids))
        
        return metrics
    
    def _get_completion_status(self, task_ids: List[str]) -> Dict[str, Any]:
        """Get completion status for tasks"""
        completed = 0
        successful = 0
        failed = 0
        pending = 0
        
        for task_id in task_ids:
            if task_id in self.original_manager.task_registry:
                task = self.original_manager.task_registry[task_id]
                if task.completed:
                    completed += 1
                    if task.error:
                        failed += 1
                    else:
                        successful += 1
                else:
                    pending += 1
        
        return {
            "all_complete": pending == 0,
            "completed": completed,
            "successful": successful,
            "failed": failed,
            "pending": pending,
            "completion_rate": completed / len(task_ids) if task_ids else 0
        }
    
    def _estimate_module_load(self, module_name: str, context: SharedContext) -> float:
        """Estimate processing load for a module"""
        # Base load estimates by module type
        base_loads = {
            "emotional_core": 0.2,
            "memory_core": 0.3,
            "reasoning_core": 0.4,
            "goal_manager": 0.2,
            "relationship_manager": 0.2,
            "knowledge_core": 0.3,
            "identity_evolution": 0.1,
            "autobiographical_narrative": 0.3,
            "cross_user_experience": 0.4,
            "dominance": 0.3
        }
        
        base_load = base_loads.get(module_name, 0.2)
        
        # Adjust based on context complexity
        input_length = len(context.user_input)
        if input_length > 500:
            base_load *= 1.5
        elif input_length > 200:
            base_load *= 1.2
        
        return min(1.0, base_load)
    
    def _can_module_parallelize(self, module_name: str) -> bool:
        """Check if module can run in parallel"""
        # Modules that typically need sequential processing
        sequential_modules = ["mode_integration", "attentional_controller"]
        
        return module_name not in sequential_modules
    
    def _get_module_dependencies(self, module_name: str) -> List[str]:
        """Get dependencies for a module"""
        # Define module dependencies
        dependencies = {
            "emotional_core": [],
            "memory_core": [],
            "reasoning_core": ["memory_core"],
            "goal_manager": ["memory_core", "emotional_core"],
            "relationship_manager": ["memory_core", "emotional_core"],
            "autobiographical_narrative": ["memory_core", "identity_evolution"],
            "cross_user_experience": ["memory_core", "relationship_manager"],
            "dominance": ["relationship_manager", "emotional_core"]
        }
        
        return dependencies.get(module_name, [])
    
    def _group_by_dependencies(self, module_requirements: Dict[str, Dict[str, Any]]) -> List[List[str]]:
        """Group modules by dependency levels"""
        levels = []
        processed = set()
        
        while len(processed) < len(module_requirements):
            current_level = []
            
            for module, req in module_requirements.items():
                if module in processed:
                    continue
                
                # Check if all dependencies are processed
                deps = req["dependencies"]
                if all(dep in processed or dep not in module_requirements for dep in deps):
                    current_level.append(module)
            
            if not current_level:
                # Circular dependency or error
                break
            
            levels.append(current_level)
            processed.update(current_level)
        
        return levels
    
    def _identify_critical_path(self, module_requirements: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify critical path through module dependencies"""
        # Simplified critical path - find longest dependency chain
        def find_longest_path(module: str, visited: Set[str]) -> List[str]:
            if module in visited:
                return []
            
            visited.add(module)
            req = module_requirements.get(module, {})
            deps = req.get("dependencies", [])
            
            if not deps:
                return [module]
            
            longest = []
            for dep in deps:
                if dep in module_requirements:
                    path = find_longest_path(dep, visited.copy())
                    if len(path) > len(longest):
                        longest = path
            
            return longest + [module]
        
        critical_path = []
        for module in module_requirements:
            path = find_longest_path(module, set())
            if len(path) > len(critical_path):
                critical_path = path
        
        return critical_path
    
    def _estimate_plan_duration(self, plan: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """Estimate duration for execution plan"""
        total_duration = 0.0
        
        # Time for parallel groups (max time in each group)
        for group in plan["parallel_groups"]:
            group_duration = max(
                requirements["module_requirements"].get(m, {}).get("estimated_load", 0.1)
                for m in group["modules"]
            )
            total_duration += group_duration
        
        # Time for sequential tasks
        for task in plan["sequential_tasks"]:
            total_duration += requirements["module_requirements"].get(task, {}).get("estimated_load", 0.1)
        
        return total_duration
    
    def _estimate_task_completion(self, task_ids: List[str]) -> float:
        """Estimate completion time for tasks"""
        # Simple estimation based on task count
        return len(task_ids) * 0.1  # 100ms per task estimate
    
    # Delegate missing methods to original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original manager"""
        return getattr(self.original_manager, name)
