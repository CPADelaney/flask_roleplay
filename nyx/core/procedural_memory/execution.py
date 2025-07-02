# nyx/core/procedural_memory/execution.py

import datetime
import random
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from pydantic import BaseModel, Field
from threading import Lock
import traceback
import uuid

# OpenAI Agents SDK imports
from agents import (
    function_tool, custom_span, trace, RunContextWrapper, RunConfig
)
from agents.tracing import Span, Trace
from agents.exceptions import UserError


from .models import Procedure, StepResult

logger = logging.getLogger(__name__)

class ExecutionStrategy(BaseModel):
    """Strategy for executing a procedure"""
    id: str
    name: str
    description: str
    selection_criteria: Dict[str, Any] = Field(default_factory=dict)
    # Added fields for enhanced execution
    timeout_seconds: float = 60.0  # Default timeout
    max_retries: int = 3  # Default retries
    retry_delay: float = 1.0  # Initial retry delay
    priority: int = 5  # Default priority (1-10, 10 is highest)
    resource_limits: Dict[str, float] = Field(default_factory=dict)  # Resource usage limits
    requires_exclusive_lock: bool = False  # Whether this strategy needs exclusive access
    trace_id: Optional[str] = None  # Trace ID for execution tracing
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the procedure according to this strategy"""
        # Base implementation - must be overridden
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def should_select(self, context: Dict[str, Any], procedure: Procedure) -> float:
        """Calculate how well this strategy matches the current context"""
        score = 0.5  # Default score
        
        # Create trace span for strategy selection
        with custom_span(
            name=f"strategy_selection_{self.id}",
            data={
                "strategy_id": self.id,
                "procedure_id": procedure.id,
                "procedure_name": procedure.name
            }
        ):
            # Check each selection criterion
            for key, value in self.selection_criteria.items():
                if key in context:
                    if isinstance(value, (list, tuple, set)):
                        # Check if context value is in list
                        if context[key] in value:
                            score += 0.1
                    elif isinstance(value, dict) and "min" in value and "max" in value:
                        # Range check
                        if value["min"] <= context[key] <= value["max"]:
                            score += 0.1
                    elif context[key] == value:
                        # Exact match
                        score += 0.2
            
            # Check execution history if available
            if hasattr(procedure, "execution_history") and procedure.execution_history:
                recent_history = procedure.execution_history[-5:]  # Last 5 executions
                
                # Count how many used this strategy
                strategy_count = sum(1 for h in recent_history 
                                if h.get("strategy_id") == self.id)
                
                # If strategy was successful recently, increase score
                strategy_success = sum(1 for h in recent_history 
                                if h.get("strategy_id") == self.id and h.get("success", False))
                
                if strategy_count > 0:
                    success_rate = strategy_success / strategy_count
                    score += success_rate * 0.2
        
        return min(1.0, score)
    
    async def _execute_with_timeout(
        self,
        execution_func: Callable,
        procedure: Procedure,
        context: Dict[str, Any],
        timeout: float = None
    ) -> Dict[str, Any]:
        """Execute with timeout protection"""
        # Use provided timeout or default
        exec_timeout = timeout or self.timeout_seconds
        
        # Create trace span for execution with timeout
        span_id = f"span_{uuid.uuid4().hex}"
        with custom_span(
            name=f"execute_with_timeout_{procedure.name}",
            data={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "strategy_id": self.id,
                "timeout": exec_timeout
            },
            span_id=span_id
        ) as span:
            try:
                # Execute with timeout
                return await asyncio.wait_for(
                    execution_func(procedure, context),
                    timeout=exec_timeout
                )
            except asyncio.TimeoutError:
                # Handle timeout
                logger.warning(f"Execution of {procedure.name} timed out after {exec_timeout} seconds")
                
                # Log timeout in trace
                with custom_span(
                    name=f"execution_timeout_{procedure.name}",
                    data={
                        "procedure_id": procedure.id,
                        "procedure_name": procedure.name,
                        "strategy_id": self.id,
                        "timeout": exec_timeout
                    },
                    parent=span
                ):
                    pass
                
                return {
                    "success": False,
                    "error": f"Execution timed out after {exec_timeout} seconds",
                    "execution_time": exec_timeout,
                    "timeout": True,
                    "strategy": self.id
                }
            except Exception as e:
                # Handle other errors
                logger.error(f"Error executing {procedure.name}: {str(e)}")
                
                # Log error in trace
                with custom_span(
                    name=f"execution_error_{procedure.name}",
                    data={
                        "procedure_id": procedure.id,
                        "procedure_name": procedure.name,
                        "strategy_id": self.id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    parent=span
                ):
                    pass
                
                return {
                    "success": False,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "execution_time": 0.0,
                    "strategy": self.id
                }
    
    async def _execute_with_retry(
        self,
        execution_func: Callable,
        procedure: Procedure,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute with retry mechanism"""
        retry_count = 0
        current_delay = self.retry_delay
        last_error = None
        
        # Create trace span for execution with retry
        with custom_span(
            name=f"execute_with_retry_{procedure.name}",
            data={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "strategy_id": self.id,
                "max_retries": self.max_retries
            }
        ) as parent_span:
            while retry_count <= self.max_retries:
                # Create trace span for each attempt
                with custom_span(
                    name=f"retry_attempt_{retry_count}",
                    data={
                        "procedure_id": procedure.id,
                        "procedure_name": procedure.name,
                        "attempt": retry_count + 1,
                        "current_delay": current_delay
                    },
                    parent=parent_span
                ):
                    try:
                        # Execute the function
                        result = await execution_func(procedure, context)
                        
                        # If successful, return result
                        if result.get("success", False):
                            # Add retry information if retries were used
                            if retry_count > 0:
                                result["retries_used"] = retry_count
                            return result
                        
                        # Failed but no error raised - break with the failed result
                        if "error" in result:
                            last_error = result["error"]
                        
                        # Handle retry for specific errors differently
                        if "error" in result and any(skip in result["error"].lower() for skip in ["permission", "unauthorized"]):
                            # No point retrying permission errors
                            return result
                        
                    except Exception as e:
                        # Track error for possible retry
                        last_error = str(e)
                        logger.warning(f"Execution attempt {retry_count + 1} failed: {last_error}")
                
                # Increment retry count
                retry_count += 1
                
                # Stop if max retries reached
                if retry_count > self.max_retries:
                    break
                    
                # Exponential backoff
                await asyncio.sleep(current_delay)
                current_delay *= 2  # Double the delay for each retry
            
            # All retries failed - log in trace
            with custom_span(
                name="retries_exhausted",
                data={
                    "procedure_id": procedure.id,
                    "procedure_name": procedure.name,
                    "retry_count": retry_count,
                    "last_error": last_error
                },
                parent=parent_span
            ):
                pass
            
            # All retries failed
            return {
                "success": False,
                "error": f"Failed after {retry_count} retries. Last error: {last_error}",
                "retries_exhausted": True,
                "execution_time": 0.0,
                "strategy": self.id
            }
    
    def _validate_resources(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate if execution can proceed based on resource limits"""
        # Check each resource limit
        for resource, limit in self.resource_limits.items():
            # Get current usage from context
            current_usage = context.get(f"resource_{resource}", 0.0)
            
            # Check if usage exceeds limit
            if current_usage > limit:
                return False, f"Resource limit exceeded: {resource} ({current_usage:.2f} > {limit:.2f})"
        
        # All resources available
        return True, None
    
    def _log_execution_metrics(
        self, 
        procedure: Procedure, 
        result: Dict[str, Any], 
        start_time: datetime.datetime
    ) -> None:
        """Log metrics about procedure execution"""
        # Calculate execution time if not already provided
        if "execution_time" not in result:
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            result["execution_time"] = execution_time
        
        # Log basic metrics
        logger.info(
            f"Executed procedure {procedure.name} with strategy {self.id}: "
            f"success={result.get('success', False)}, "
            f"time={result['execution_time']:.4f}s"
        )
        
        # Log detailed metrics if available
        if "results" in result:
            step_times = [r.get("execution_time", 0.0) for r in result["results"]]
            if step_times:
                avg_step_time = sum(step_times) / len(step_times)
                max_step_time = max(step_times)
                logger.debug(
                    f"Step metrics for {procedure.name}: "
                    f"avg_time={avg_step_time:.4f}s, max_time={max_step_time:.4f}s, "
                    f"steps={len(step_times)}"
                )

class DeliberateExecutionStrategy(ExecutionStrategy):
    """Executes procedure carefully with validation between steps"""
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set specific defaults for deliberate execution
        if "timeout_seconds" not in data:
            self.timeout_seconds = 120.0  # Longer timeout for careful execution
        if "max_retries" not in data:
            self.max_retries = 2  # Fewer retries, since we validate carefully
        if "retry_delay" not in data:
            self.retry_delay = 2.0  # Longer initial delay
        # Initialize locks for thread safety
        self._validation_locks = {}
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure deliberately with checks between steps"""
        # Generate trace ID for this execution
        trace_id = f"trace_{uuid.uuid4().hex}"
        self.trace_id = trace_id
        
        # Use trace for entire execution
        with trace(
            workflow_name=f"deliberate_execution_{procedure.name}",
            trace_id=trace_id,
            metadata={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "strategy": "deliberate",
                "timestamp": datetime.datetime.now().isoformat()
            }
        ):
            # Use timeout wrapper
            return await self._execute_with_timeout(self._execute_impl, procedure, context)
    
    async def _execute_impl(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implementation of deliberate execution"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Create trace span for this execution
        with custom_span(
            name=f"deliberate_execution_impl_{procedure.name}",
            data={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "steps_count": len(procedure.steps)
            }
        ) as execution_span:
            # Initialize execution state
            execution_state = context.copy()
            execution_state["strategy"] = "deliberate"
            execution_state["execution_history"] = []
            
            # Check resources
            resources_ok, resource_error = self._validate_resources(execution_state)
            if not resources_ok:
                with custom_span(
                    name="resource_validation_failed",
                    data={
                        "procedure_id": procedure.id,
                        "procedure_name": procedure.name,
                        "error": resource_error
                    },
                    parent=execution_span
                ):
                    pass
                
                return {
                    "success": False,
                    "error": resource_error,
                    "execution_time": 0.0,
                    "strategy": "deliberate"
                }
            
            # Collect preconditions for all steps
            step_preconditions = {}
            all_preconditions_met = True
            precondition_failures = []
            
            # Check all preconditions first
            with custom_span(
                name="precondition_validation",
                data={
                    "procedure_id": procedure.id,
                    "procedure_name": procedure.name
                },
                parent=execution_span
            ) as precondition_span:
                for step in procedure.steps:
                    step_id = step["id"]
                    preconditions = step.get("preconditions", {})
                    step_preconditions[step_id] = preconditions
                    
                    # Validate preconditions
                    if not self._validate_preconditions(step, execution_state):
                        all_preconditions_met = False
                        failed_preconditions = {}
                        for key, value in preconditions.items():
                            if key not in execution_state or execution_state[key] != value:
                                failed_preconditions[key] = {
                                    "expected": value,
                                    "actual": execution_state.get(key, "missing")
                                }
                        
                        precondition_failures.append({
                            "step_id": step_id,
                            "failed_preconditions": failed_preconditions
                        })
                        
                        # Log failure in trace
                        with custom_span(
                            name=f"precondition_failure_{step_id}",
                            data={
                                "step_id": step_id,
                                "failed_preconditions": failed_preconditions
                            },
                            parent=precondition_span
                        ):
                            pass
            
            # If any preconditions failed, abort early
            if not all_preconditions_met:
                return {
                    "success": False,
                    "error": "Procedure preconditions not met",
                    "precondition_failures": precondition_failures,
                    "execution_time": (datetime.datetime.now() - start_time).total_seconds(),
                    "strategy": "deliberate"
                }
            
            # Execute steps sequentially with validation
            for i, step in enumerate(procedure.steps):
                step_id = step["id"]
                
                # Create trace span for this step
                with custom_span(
                    name=f"execute_step_{step_id}",
                    data={
                        "step_id": step_id,
                        "step_index": i,
                        "function": step.get("function")
                    },
                    parent=execution_span
                ) as step_span:
                    # Validate preconditions
                    if not self._validate_preconditions(step, execution_state):
                        with custom_span(
                            name=f"precondition_failure_{step_id}",
                            data={"step_id": step_id},
                            parent=step_span
                        ):
                            pass
                        
                        results.append({
                            "step_id": step_id,
                            "success": False,
                            "error": "Preconditions not met",
                            "execution_time": 0.0
                        })
                        success = False
                        break
                    
                    # Execute the step (with retries if enabled)
                    step_result = await self._execute_step_with_retry(step, execution_state)
                    results.append(step_result)
                    
                    # Update execution state
                    execution_state[f"step_{step_id}_result"] = step_result
                    execution_state["execution_history"].append({
                        "step_id": step_id,
                        "function": step.get("function"),
                        "success": step_result["success"],
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    # Check for failure
                    if not step_result["success"]:
                        with custom_span(
                            name=f"step_failure_{step_id}",
                            data={
                                "step_id": step_id,
                                "error": step_result.get("error", "Unknown error")
                            },
                            parent=step_span
                        ):
                            pass
                        
                        success = False
                        break
                    
                    # Validate postconditions
                    if not self._validate_postconditions(step, execution_state):
                        with custom_span(
                            name=f"postcondition_failure_{step_id}",
                            data={"step_id": step_id},
                            parent=step_span
                        ):
                            pass
                        
                        results.append({
                            "step_id": step_id,
                            "success": False,
                            "error": "Postconditions not met",
                            "execution_time": 0.0
                        })
                        success = False
                        break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Log execution metrics
            self._log_execution_metrics(procedure, {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "strategy": "deliberate"
            }, start_time)
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "strategy": "deliberate"
            }
    
    async def _execute_step_with_retry(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a step with retry logic"""
        if self.max_retries > 0:
            # Create a lambda for the retry logic
            execution_func = lambda s, c: self._execute_step(s, c)
            
            # Create trace span for step retry
            with custom_span(
                name=f"step_with_retry_{step['id']}",
                data={
                    "step_id": step["id"],
                    "function": step.get("function"),
                    "max_retries": self.max_retries
                }
            ):
                return await self._execute_with_retry(execution_func, step, context)
        else:
            # No retries, just execute directly
            return await self._execute_step(step, context)
    
    def _validate_preconditions(self, step: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Validate preconditions for a step"""
        preconditions = step.get("preconditions", {})
        
        for key, value in preconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    def _validate_postconditions(self, step: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Validate postconditions for a step"""
        postconditions = step.get("postconditions", {})
        
        for key, value in postconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        # Create trace span for step execution
        with custom_span(
            name=f"execute_step_{step.get('id', 'unknown')}",
            data={
                "step_id": step.get("id", "unknown"),
                "function": step.get("function")
            }
        ) as step_span:
            # Get the function to call
            function_name = step.get("function")
            if not function_name:
                return {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": "No function specified",
                    "execution_time": 0.0
                }
            
            # Try to get the actual function
            function = context.get("function_registry", {}).get(function_name)
            
            if not function:
                # Try to get from global context
                function = context.get("global_functions", {}).get(function_name)
                
            if not function:
                # No function found
                with custom_span(
                    name=f"function_not_found_{function_name}",
                    data={
                        "function_name": function_name,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    pass
                
                return {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": f"Function '{function_name}' not found in registry",
                    "execution_time": 0.0
                }
            
            # Execute with timing
            step_start = datetime.datetime.now()
            
            try:
                # Prepare parameters
                parameters = step.get("parameters", {}).copy()
                
                # Add context to parameters if function accepts it
                if hasattr(function, "__code__") and "context" in function.__code__.co_varnames:
                    parameters["context"] = context
                
                # Execute the function
                with custom_span(
                    name=f"function_call_{function_name}",
                    data={
                        "function": function_name,
                        "parameters": parameters,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    result = await function(**parameters)
                
                # Process result
                if isinstance(result, dict):
                    success = "error" not in result
                    step_result = {
                        "step_id": step.get("id", "unknown"),
                        "success": success,
                        "data": result,
                        "execution_time": 0.0
                    }
                    
                    if not success:
                        step_result["error"] = result.get("error")
                else:
                    step_result = {
                        "step_id": step.get("id", "unknown"),
                        "success": True,
                        "data": {"result": result},
                        "execution_time": 0.0
                    }
            except Exception as e:
                # Log and create error result
                logger.error(f"Error executing step {step.get('id', 'unknown')}: {str(e)}")
                
                # Log error in trace
                with custom_span(
                    name=f"step_error_{step.get('id', 'unknown')}",
                    data={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    pass
                
                step_result = {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "execution_time": 0.0
                }
            
            # Calculate execution time
            step_time = (datetime.datetime.now() - step_start).total_seconds()
            step_result["execution_time"] = step_time
            
            return step_result

class AutomaticExecutionStrategy(ExecutionStrategy):
    """Fast execution without validation between steps"""
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set specific defaults for automatic execution
        if "timeout_seconds" not in data:
            self.timeout_seconds = 30.0  # Shorter timeout for fast execution
        if "max_retries" not in data:
            self.max_retries = 0  # No retries by default for automatic execution
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure automatically without validation"""
        # Generate trace ID for this execution
        trace_id = f"trace_{uuid.uuid4().hex}"
        self.trace_id = trace_id
        
        # Use trace for entire execution
        with trace(
            workflow_name=f"automatic_execution_{procedure.name}",
            trace_id=trace_id,
            metadata={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "strategy": "automatic",
                "timestamp": datetime.datetime.now().isoformat()
            }
        ):
            # Use timeout wrapper
            return await self._execute_with_timeout(self._execute_impl, procedure, context)
    
    async def _execute_impl(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implementation of automatic execution"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Create trace span for this execution
        with custom_span(
            name=f"automatic_execution_impl_{procedure.name}",
            data={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "steps_count": len(procedure.steps)
            }
        ) as execution_span:
            # Initialize execution state
            execution_state = context.copy()
            execution_state["strategy"] = "automatic"
            execution_state["execution_history"] = []
            
            # Check if procedure is chunked
            if procedure.is_chunked:
                # Execute chunks
                with custom_span(
                    name="execute_chunked_procedure",
                    data={
                        "procedure_id": procedure.id,
                        "procedure_name": procedure.name,
                        "chunks_count": len(procedure.chunked_steps)
                    },
                    parent=execution_span
                ) as chunks_span:
                    chunks = self._get_chunks(procedure)
                    
                    # Check resources once for the entire procedure
                    resources_ok, resource_error = self._validate_resources(execution_state)
                    if not resources_ok:
                        return {
                            "success": False,
                            "error": resource_error,
                            "execution_time": 0.0,
                            "strategy": "automatic"
                        }
                    
                    for chunk_id, chunk_steps in chunks.items():
                        # Execute chunk as a unit
                        with custom_span(
                            name=f"execute_chunk_{chunk_id}",
                            data={
                                "chunk_id": chunk_id,
                                "steps_count": len(chunk_steps)
                            },
                            parent=chunks_span
                        ) as chunk_span:
                            chunk_result = await self._execute_chunk(
                                chunk_steps, 
                                execution_state, 
                                chunk_id,
                                procedure
                            )
                            
                            results.extend(chunk_result["results"])
                            
                            # Update execution state
                            execution_state[f"chunk_{chunk_id}_result"] = chunk_result
                            for step_result in chunk_result["results"]:
                                step_id = step_result["step_id"]
                                execution_state[f"step_{step_id}_result"] = step_result
                                execution_state["execution_history"].append({
                                    "step_id": step_id,
                                    "chunk_id": chunk_id,
                                    "success": step_result["success"],
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                            
                            # Check for failure
                            if not chunk_result["success"]:
                                success = False
                                break
            else:
                # Execute steps sequentially without validation
                with custom_span(
                    name="execute_sequential_steps",
                    data={
                        "procedure_id": procedure.id,
                        "procedure_name": procedure.name,
                        "steps_count": len(procedure.steps)
                    },
                    parent=execution_span
                ) as steps_span:
                    for step in procedure.steps:
                        # Execute the step
                        with custom_span(
                            name=f"execute_step_{step['id']}",
                            data={
                                "step_id": step["id"],
                                "function": step.get("function")
                            },
                            parent=steps_span
                        ) as step_span:
                            step_result = await self._execute_step(step, execution_state)
                            results.append(step_result)
                            
                            # Update execution state
                            execution_state[f"step_{step['id']}_result"] = step_result
                            execution_state["execution_history"].append({
                                "step_id": step["id"],
                                "function": step.get("function"),
                                "success": step_result["success"],
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                            
                            # Check for failure
                            if not step_result["success"]:
                                success = False
                                break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Log execution metrics
            self._log_execution_metrics(procedure, {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "strategy": "automatic"
            }, start_time)
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "strategy": "automatic"
            }
    
    def _get_chunks(self, procedure: Procedure) -> Dict[str, List[Dict[str, Any]]]:
        """Get chunks from a procedure"""
        chunks = {}
        
        for chunk_id, step_ids in procedure.chunked_steps.items():
            # Convert step IDs to actual step dictionaries
            steps = [next((s for s in procedure.steps if s["id"] == step_id), None) for step_id in step_ids]
            steps = [s for s in steps if s is not None]  # Remove None values
            
            chunks[chunk_id] = steps
            
        return chunks
    
    async def _execute_chunk(
        self, 
        steps: List[Dict[str, Any]], 
        context: Dict[str, Any], 
        chunk_id: str,
        procedure: Procedure
    ) -> Dict[str, Any]:
        """Execute a chunk of steps as a unit"""
        results = []
        success = True
        start_time = datetime.datetime.now()
        
        # Create trace span for chunk execution
        with custom_span(
            name=f"execute_chunk_{chunk_id}",
            data={
                "chunk_id": chunk_id,
                "steps_count": len(steps)
            }
        ) as chunk_span:
            # Set chunk context
            chunk_context = context.copy()
            chunk_context["current_chunk"] = chunk_id
            
            # Track resource usage for the chunk
            chunk_resources = {
                "cpu": 0.0,
                "memory": 0.0,
                "io": 0.0
            }
            
            for step in steps:
                # Execute the step
                with custom_span(
                    name=f"chunk_step_{step['id']}",
                    data={
                        "step_id": step["id"],
                        "function": step.get("function"),
                        "chunk_id": chunk_id
                    },
                    parent=chunk_span
                ) as step_span:
                    step_result = await self._execute_step(step, chunk_context)
                    results.append(step_result)
                    
                    # Update resource tracking (rough estimate)
                    chunk_resources["cpu"] += step_result.get("execution_time", 0.0) * 0.1
                    chunk_resources["memory"] += 1.0  # Arbitrary unit
                    
                    # Update chunk context with result
                    chunk_context[f"step_{step['id']}_result"] = step_result
                    
                    # Check for failure
                    if not step_result["success"]:
                        success = False
                        break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update template statistics if applicable
            if hasattr(procedure, "chunk_templates") and chunk_id in procedure.chunk_templates:
                template_id = procedure.chunk_templates[chunk_id]
                if hasattr(procedure, "update_template_success"):
                    procedure.update_template_success(template_id, success)
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "chunk_id": chunk_id,
                "resources_used": chunk_resources
            }
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        # Create trace span for step execution
        with custom_span(
            name=f"execute_step_{step.get('id', 'unknown')}",
            data={
                "step_id": step.get("id", "unknown"),
                "function": step.get("function")
            }
        ) as step_span:
            # Get the function to call
            function_name = step.get("function")
            if not function_name:
                return {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": "No function specified",
                    "execution_time": 0.0
                }
            
            # Try to get the actual function
            function = context.get("function_registry", {}).get(function_name)
            
            if not function:
                # Try to get from global context
                function = context.get("global_functions", {}).get(function_name)
                
            if not function:
                # No function found
                with custom_span(
                    name=f"function_not_found_{function_name}",
                    data={
                        "function_name": function_name,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    pass
                
                return {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": f"Function '{function_name}' not found in registry",
                    "execution_time": 0.0
                }
            
            # Execute with timing
            step_start = datetime.datetime.now()
            
            try:
                # Prepare parameters
                parameters = step.get("parameters", {}).copy()
                
                # Add context to parameters if function accepts it
                if hasattr(function, "__code__") and "context" in function.__code__.co_varnames:
                    parameters["context"] = context
                
                # Execute the function
                with custom_span(
                    name=f"function_call_{function_name}",
                    data={
                        "function": function_name,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    result = await function(**parameters)
                
                # Process result
                if isinstance(result, dict):
                    success = "error" not in result
                    step_result = {
                        "step_id": step.get("id", "unknown"),
                        "success": success,
                        "data": result,
                        "execution_time": 0.0
                    }
                    
                    if not success:
                        step_result["error"] = result.get("error")
                else:
                    step_result = {
                        "step_id": step.get("id", "unknown"),
                        "success": True,
                        "data": {"result": result},
                        "execution_time": 0.0
                    }
            except Exception as e:
                # Log and create error result
                logger.error(f"Error executing step {step.get('id', 'unknown')}: {str(e)}")
                
                # Log error in trace
                with custom_span(
                    name=f"step_error_{step.get('id', 'unknown')}",
                    data={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    pass
                
                step_result = {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "execution_time": 0.0
                }
            
            # Calculate execution time
            step_time = (datetime.datetime.now() - step_start).total_seconds()
            step_result["execution_time"] = step_time
            
            return step_result

class AdaptiveExecutionStrategy(ExecutionStrategy):
    """Adapts execution based on context and feedback"""
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set specific defaults for adaptive execution
        if "timeout_seconds" not in data:
            self.timeout_seconds = 90.0  # Medium timeout
        if "max_retries" not in data:
            self.max_retries = 2  # Some retries
        if "retry_delay" not in data:
            self.retry_delay = 1.5  # Medium initial delay
        
        # Initialize strategy-specific fields
        self.adaptive_thresholds = {
            "proficiency": 0.8,  # Threshold for switching to automatic
            "risk": 0.7,  # Threshold for switching to deliberate
            "recovery_attempts": 2  # Max recovery attempts per execution
        }
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure with adaptive strategy selection"""
        # Generate trace ID for this execution
        trace_id = f"trace_{uuid.uuid4().hex}"
        self.trace_id = trace_id
        
        # Use trace for entire execution
        with trace(
            workflow_name=f"adaptive_execution_{procedure.name}",
            trace_id=trace_id,
            metadata={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "strategy": "adaptive",
                "timestamp": datetime.datetime.now().isoformat()
            }
        ):
            # Use timeout wrapper
            return await self._execute_with_timeout(self._execute_impl, procedure, context)
    
    async def _execute_impl(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implementation of adaptive execution"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Create trace span for this execution
        with custom_span(
            name=f"adaptive_execution_impl_{procedure.name}",
            data={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "steps_count": len(procedure.steps)
            }
        ) as execution_span:
            # Initialize execution state
            execution_state = context.copy()
            execution_state["strategy"] = "adaptive"
            execution_state["execution_history"] = []
            execution_state["adaptations"] = []
            
            # Check resources
            resources_ok, resource_error = self._validate_resources(execution_state)
            if not resources_ok:
                return {
                    "success": False,
                    "error": resource_error,
                    "execution_time": 0.0,
                    "strategy": "adaptive"
                }
            
            # Determine initial execution mode based on proficiency and risk
            risk_level = context.get("risk_level", 0.5)
            deliberate_execution = (
                procedure.proficiency < self.adaptive_thresholds["proficiency"] or
                risk_level >= self.adaptive_thresholds["risk"]
            )
            
            # Create trace span for initial strategy selection
            with custom_span(
                name=f"initial_strategy_selection",
                data={
                    "deliberate_execution": deliberate_execution,
                    "proficiency": procedure.proficiency,
                    "risk_level": risk_level,
                    "proficiency_threshold": self.adaptive_thresholds["proficiency"],
                    "risk_threshold": self.adaptive_thresholds["risk"]
                },
                parent=execution_span
            ):
                pass
            
            # Track adaptations
            adaptations = []
            recovery_attempts = 0
            
            # Execute steps with adaptive strategy
            for i, step in enumerate(procedure.steps):
                step_id = step["id"]
                
                # Create trace span for this step
                with custom_span(
                    name=f"execute_step_{step_id}",
                    data={
                        "step_id": step_id,
                        "step_index": i,
                        "function": step.get("function")
                    },
                    parent=execution_span
                ) as step_span:
                    # Decide execution strategy for this step
                    step_strategy = self._select_step_strategy(step, execution_state, deliberate_execution)
                    
                    # Log strategy in trace
                    with custom_span(
                        name=f"step_strategy_{step_id}",
                        data={
                            "strategy": step_strategy,
                            "deliberate_execution": deliberate_execution
                        },
                        parent=step_span
                    ):
                        pass
                    
                    # Log adaptation if strategy changed
                    if i > 0 and step_strategy != execution_state.get("last_step_strategy"):
                        adaptation = {
                            "type": "strategy_change",
                            "step_id": step_id,
                            "from_strategy": execution_state.get("last_step_strategy"),
                            "to_strategy": step_strategy,
                            "reason": "Dynamic adjustment based on context",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        adaptations.append(adaptation)
                        
                        # Log adaptation in trace
                        with custom_span(
                            name=f"strategy_adaptation_{step_id}",
                            data=adaptation,
                            parent=step_span
                        ):
                            pass
                    
                    # Update last strategy
                    execution_state["last_step_strategy"] = step_strategy
                    
                    # Execute step with selected strategy
                    if step_strategy == "deliberate":
                        # Careful execution with validation
                        if not self._validate_preconditions(step, execution_state):
                            # Log failure in trace
                            with custom_span(
                                name=f"precondition_failure_{step_id}",
                                data={"step_id": step_id},
                                parent=step_span
                            ):
                                pass
                            
                            results.append({
                                "step_id": step_id,
                                "success": False,
                                "error": "Preconditions not met",
                                "execution_time": 0.0,
                                "strategy": "deliberate"
                            })
                            success = False
                            
                            # Record adaptation
                            adaptation = {
                                "type": "execution_failure",
                                "step_id": step_id,
                                "reason": "Preconditions not met",
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            adaptations.append(adaptation)
                            
                            # Log adaptation in trace
                            with custom_span(
                                name=f"execution_failure_{step_id}",
                                data=adaptation,
                                parent=step_span
                            ):
                                pass
                            
                            break
                        
                        step_result = await self._execute_step(step, execution_state)
                        step_result["strategy"] = "deliberate"
                        
                        if not self._validate_postconditions(step, execution_state, step_result):
                            # Log failure in trace
                            with custom_span(
                                name=f"postcondition_failure_{step_id}",
                                data={"step_id": step_id},
                                parent=step_span
                            ):
                                pass
                            
                            step_result["success"] = False
                            step_result["error"] = "Postconditions not met"
                            success = False
                            results.append(step_result)
                            
                            # Record adaptation
                            adaptation = {
                                "type": "execution_failure",
                                "step_id": step_id,
                                "reason": "Postconditions not met",
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            adaptations.append(adaptation)
                            
                            # Log adaptation in trace
                            with custom_span(
                                name=f"execution_failure_{step_id}",
                                data=adaptation,
                                parent=step_span
                            ):
                                pass
                            
                            break
                    else:
                        # Fast execution without validation
                        step_result = await self._execute_step(step, execution_state)
                        step_result["strategy"] = "automatic"
                    
                    # Add to results
                    results.append(step_result)
                    
                    # Update execution state
                    execution_state[f"step_{step_id}_result"] = step_result
                    execution_state["execution_history"].append({
                        "step_id": step_id,
                        "function": step.get("function"),
                        "success": step_result["success"],
                        "strategy": step_strategy,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    # Check for failure and adapt
                    if not step_result["success"]:
                        # Adapt strategy on failure
                        deliberate_execution = True
                        
                        # Record adaptation
                        adaptation = {
                            "type": "strategy_adaptation",
                            "trigger": "step_failure",
                            "step_id": step_id,
                            "new_strategy": "deliberate",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        adaptations.append(adaptation)
                        
                        # Log adaptation in trace
                        with custom_span(
                            name=f"strategy_adaptation_on_failure_{step_id}",
                            data=adaptation,
                            parent=step_span
                        ):
                            pass
                        
                        # Try to recover if within limits
                        if recovery_attempts < self.adaptive_thresholds["recovery_attempts"]:
                            # Create trace span for recovery attempt
                            with custom_span(
                                name=f"recovery_attempt_{step_id}",
                                data={
                                    "step_id": step_id,
                                    "attempt": recovery_attempts + 1,
                                    "max_attempts": self.adaptive_thresholds["recovery_attempts"]
                                },
                                parent=step_span
                            ) as recovery_span:
                                # Attempt recovery
                                recovery_result = await self._attempt_recovery(
                                    step, 
                                    procedure.steps[i+1:] if i < len(procedure.steps) - 1 else [],
                                    execution_state
                                )
                                
                                recovery_attempts += 1
                                
                                if recovery_result["success"]:
                                    # Recovery worked, continue execution
                                    adaptation = {
                                        "type": "recovery",
                                        "step_id": step_id,
                                        "success": True,
                                        "method": recovery_result["method"],
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                    adaptations.append(adaptation)
                                    
                                    # Log adaptation in trace
                                    with custom_span(
                                        name=f"recovery_success_{step_id}",
                                        data=adaptation,
                                        parent=recovery_span
                                    ):
                                        pass
                                    
                                    # Add recovery steps to results
                                    results.extend(recovery_result.get("recovery_steps", []))
                                    
                                    # Update execution state with recovery results
                                    for recovery_step in recovery_result.get("recovery_steps", []):
                                        recovery_step_id = recovery_step["step_id"]
                                        execution_state[f"step_{recovery_step_id}_result"] = recovery_step
                                        execution_state["execution_history"].append({
                                            "step_id": recovery_step_id,
                                            "function": recovery_step.get("function", "recovery"),
                                            "success": recovery_step["success"],
                                            "strategy": "recovery",
                                            "timestamp": datetime.datetime.now().isoformat()
                                        })
                                    
                                    continue
                                else:
                                    # Recovery failed
                                    adaptation = {
                                        "type": "recovery",
                                        "step_id": step_id,
                                        "success": False,
                                        "reason": recovery_result.get("reason", "Unknown failure"),
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                    adaptations.append(adaptation)
                                    
                                    # Log adaptation in trace
                                    with custom_span(
                                        name=f"recovery_failure_{step_id}",
                                        data=adaptation,
                                        parent=recovery_span
                                    ):
                                        pass
                        
                        # If we got here, either no recovery was attempted or it failed
                        success = False
                        break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Log execution metrics
            self._log_execution_metrics(procedure, {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "strategy": "adaptive",
                "adaptations": adaptations
            }, start_time)
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "strategy": "adaptive",
                "adaptations": adaptations
            }
    
    def _select_step_strategy(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any], 
        default_deliberate: bool
    ) -> str:
        """Select execution strategy for a step"""
        # Create trace span for strategy selection
        with custom_span(
            name=f"select_step_strategy_{step['id']}",
            data={
                "step_id": step["id"],
                "default_deliberate": default_deliberate
            }
        ) as selection_span:
            # Check if step has explicit strategy preference
            if "preferred_strategy" in step:
                return step["preferred_strategy"]
            
            # Check if step is high-risk
            if "risk_level" in step and step["risk_level"] > 0.7:
                return "deliberate"
            
            # Check execution history for this step
            history = state.get("execution_history", [])
            step_history = [h for h in history if h.get("step_id") == step["id"]]
            
            if step_history:
                # Check success rate
                success_rate = sum(1 for h in step_history if h.get("success", False)) / len(step_history)
                
                if success_rate < 0.8:
                    # Low success rate, use deliberate execution
                    return "deliberate"
                    
                # Check if the deliberate strategy was more successful
                deliberate_history = [h for h in step_history if h.get("strategy") == "deliberate"]
                automatic_history = [h for h in step_history if h.get("strategy") == "automatic"]
                
                if deliberate_history and automatic_history:
                    deliberate_success_rate = sum(1 for h in deliberate_history if h.get("success", False)) / len(deliberate_history)
                    automatic_success_rate = sum(1 for h in automatic_history if h.get("success", False)) / len(automatic_history)
                    
                    # Choose the more successful strategy
                    if deliberate_success_rate > automatic_success_rate + 0.1:  # Deliberate needs to be noticeably better
                        return "deliberate"
                    elif automatic_success_rate > deliberate_success_rate:
                        return "automatic"
            
            # Check if this step has dependencies
            if "dependencies" in step:
                # Check if all dependencies succeeded
                dependencies_ok = True
                for dep_id in step["dependencies"]:
                    # Check if we have a result for this dependency
                    if f"step_{dep_id}_result" in state:
                        result = state[f"step_{dep_id}_result"]
                        if not result.get("success", False):
                            dependencies_ok = False
                            break
                    else:
                        # No result for dependency, be cautious
                        dependencies_ok = False
                        break
                        
                if not dependencies_ok:
                    # Dependencies had issues, use deliberate execution
                    return "deliberate"
            
            # Use default (based on overall procedure proficiency)
            return "deliberate" if default_deliberate else "automatic"
    
    def _validate_preconditions(self, step: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Validate preconditions for a step"""
        preconditions = step.get("preconditions", {})
        
        for key, value in preconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    def _validate_postconditions(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> bool:
        """Validate postconditions for a step"""
        postconditions = step.get("postconditions", {})
        
        # First check result success
        if not result.get("success", False):
            return False
        
        # Check explicit postconditions
        for key, value in postconditions.items():
            if key not in state:
                return False
            
            # Compare values
            if isinstance(value, (list, tuple, set)):
                if state[key] not in value:
                    return False
            elif isinstance(value, dict) and "min" in value and "max" in value:
                if not (value["min"] <= state[key] <= value["max"]):
                    return False
            elif state[key] != value:
                return False
        
        return True
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        # Create trace span for step execution
        with custom_span(
            name=f"execute_step_{step.get('id', 'unknown')}",
            data={
                "step_id": step.get("id", "unknown"),
                "function": step.get("function")
            }
        ) as step_span:
            # Get the function to call
            function_name = step.get("function")
            if not function_name:
                return {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": "No function specified",
                    "execution_time": 0.0
                }
            
            # Try to get the actual function
            function = context.get("function_registry", {}).get(function_name)
            
            if not function:
                # Try to get from global context
                function = context.get("global_functions", {}).get(function_name)
                
            if not function:
                # No function found
                with custom_span(
                    name=f"function_not_found_{function_name}",
                    data={
                        "function_name": function_name,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    pass
                
                return {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": f"Function '{function_name}' not found in registry",
                    "execution_time": 0.0
                }
            
            # Execute with timing
            step_start = datetime.datetime.now()
            
            try:
                # Prepare parameters
                parameters = step.get("parameters", {}).copy()
                
                # Add context to parameters if function accepts it
                if hasattr(function, "__code__") and "context" in function.__code__.co_varnames:
                    parameters["context"] = context
                
                # Execute the function
                with custom_span(
                    name=f"function_call_{function_name}",
                    data={
                        "function": function_name,
                        "parameters": parameters,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    result = await function(**parameters)
                
                # Process result
                if isinstance(result, dict):
                    success = "error" not in result
                    step_result = {
                        "step_id": step.get("id", "unknown"),
                        "success": success,
                        "data": result,
                        "execution_time": 0.0
                    }
                    
                    if not success:
                        step_result["error"] = result.get("error")
                else:
                    step_result = {
                        "step_id": step.get("id", "unknown"),
                        "success": True,
                        "data": {"result": result},
                        "execution_time": 0.0
                    }
            except Exception as e:
                # Log and create error result
                logger.error(f"Error executing step {step.get('id', 'unknown')}: {str(e)}")
                
                # Log error in trace
                with custom_span(
                    name=f"step_error_{step.get('id', 'unknown')}",
                    data={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "step_id": step.get("id", "unknown")
                    },
                    parent=step_span
                ):
                    pass
                
                step_result = {
                    "step_id": step.get("id", "unknown"),
                    "success": False,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "execution_time": 0.0
                }
            
            # Calculate execution time
            step_time = (datetime.datetime.now() - step_start).total_seconds()
            step_result["execution_time"] = step_time
            
            return step_result
    
    async def _attempt_recovery(
        self, 
        failed_step: Dict[str, Any], 
        remaining_steps: List[Dict[str, Any]], 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to recover from a step failure"""
        # Create trace span for recovery attempt
        with custom_span(
            name=f"attempt_recovery_{failed_step['id']}",
            data={
                "step_id": failed_step["id"],
                "function": failed_step.get("function")
            }
        ) as recovery_span:
            # Get error information
            step_result = state.get(f"step_{failed_step['id']}_result", {})
            error = step_result.get("error", "Unknown error")
            
            # Determine recovery method based on error
            if "timeout" in error.lower():
                # Timeout error - try with extended timeout
                return await self._recover_from_timeout(failed_step, state)
            elif "precondition" in error.lower():
                # Precondition failure - try to establish preconditions
                return await self._recover_from_precondition_failure(failed_step, state)
            elif "permission" in error.lower() or "access" in error.lower():
                # Permission error - likely unrecoverable
                return {
                    "success": False,
                    "reason": "Unrecoverable permission error",
                    "error": error
                }
            else:
                # Generic error - try parameter modification
                return await self._recover_with_parameter_modification(failed_step, state)
    
    async def _recover_from_timeout(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from timeout by retrying with longer timeout"""
        # Create trace span for timeout recovery
        with custom_span(
            name=f"recover_from_timeout_{step['id']}",
            data={
                "step_id": step["id"],
                "function": step.get("function")
            }
        ) as timeout_span:
            # Create modified step with longer timeout
            recovery_step = step.copy()
            recovery_step["id"] = f"{step['id']}_recovery"
            
            # Add timeout parameter or increase existing one
            if "parameters" not in recovery_step:
                recovery_step["parameters"] = {}
                
            if "timeout" in recovery_step["parameters"]:
                # Double the timeout
                recovery_step["parameters"]["timeout"] = recovery_step["parameters"]["timeout"] * 2
            else:
                # Add a timeout parameter (30 seconds)
                recovery_step["parameters"]["timeout"] = 30.0
                
            # Execute recovery step with increased timeout
            recovery_result = await self._execute_step(recovery_step, state)
            
            # Return recovery information
            if recovery_result["success"]:
                return {
                    "success": True,
                    "method": "extended_timeout",
                    "recovery_steps": [recovery_result]
                }
            else:
                return {
                    "success": False,
                    "reason": "Timeout recovery failed",
                    "error": recovery_result.get("error")
                }
    
    async def _recover_from_precondition_failure(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from precondition failure by trying to establish preconditions"""
        # Create trace span for precondition recovery
        with custom_span(
            name=f"recover_from_precondition_failure_{step['id']}",
            data={
                "step_id": step["id"],
                "function": step.get("function")
            }
        ) as precondition_span:
            preconditions = step.get("preconditions", {})
            recovery_steps = []
            
            # Try to establish each missing precondition
            for key, expected in preconditions.items():
                # Check if precondition is missing or has wrong value
                if key not in state or state[key] != expected:
                    # Try to find a way to establish this precondition
                    recovery_step = self._create_recovery_step_for_precondition(key, expected, step["id"])
                    
                    if recovery_step:
                        # Execute recovery step
                        recovery_result = await self._execute_step(recovery_step, state)
                        recovery_steps.append(recovery_result)
                        
                        # Update state with result
                        if recovery_result["success"]:
                            # Update the precondition in state
                            state[key] = expected
                        else:
                            # Recovery step failed
                            return {
                                "success": False,
                                "reason": f"Failed to establish precondition {key}",
                                "error": recovery_result.get("error")
                            }
            
            # If we got here, all recovery steps succeeded
            if recovery_steps:
                # Now retry the original step
                retry_result = await self._execute_step(step, state)
                recovery_steps.append(retry_result)
                
                if retry_result["success"]:
                    return {
                        "success": True,
                        "method": "establish_preconditions",
                        "recovery_steps": recovery_steps
                    }
                else:
                    return {
                        "success": False,
                        "reason": "Step still failed after establishing preconditions",
                        "error": retry_result.get("error")
                    }
            else:
                # No recovery steps were created
                return {
                    "success": False,
                    "reason": "Could not create recovery steps for preconditions",
                    "error": "No applicable recovery method"
                }
    
    def _create_recovery_step_for_precondition(
        self, 
        key: str, 
        value: Any, 
        step_id: str
    ) -> Optional[Dict[str, Any]]:
        """Create a recovery step to establish a precondition"""
        # This would be more sophisticated in a real implementation
        # This is a simplified example
        
        # Create basic function name based on key
        function_name = f"set_{key}"
        
        # Create recovery step
        return {
            "id": f"{step_id}_recovery_{key}",
            "description": f"Establish precondition {key}={value}",
            "function": function_name,
            "parameters": {key: value}
        }
    
    async def _recover_with_parameter_modification(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover by modifying parameters and retrying"""
        # Create trace span for parameter modification recovery
        with custom_span(
            name=f"recover_with_parameter_modification_{step['id']}",
            data={
                "step_id": step["id"],
                "function": step.get("function")
            }
        ) as param_span:
            # Create modified step
            modified_step = step.copy()
            modified_step["id"] = f"{step['id']}_recovery"
            
            # Modify parameters
            if "parameters" in modified_step:
                modified_params = self._modify_parameters(modified_step["parameters"])
                modified_step["parameters"] = modified_params
                
                # Log parameter modification in trace
                with custom_span(
                    name=f"parameter_modification_{step['id']}",
                    data={
                        "original_parameters": step.get("parameters", {}),
                        "modified_parameters": modified_params
                    },
                    parent=param_span
                ):
                    pass
            
            # Execute with modified parameters
            recovery_result = await self._execute_step(modified_step, state)
            
            # Return recovery information
            if recovery_result["success"]:
                return {
                    "success": True,
                    "method": "parameter_modification",
                    "recovery_steps": [recovery_result]
                }
            else:
                return {
                    "success": False,
                    "reason": "Parameter modification recovery failed",
                    "error": recovery_result.get("error")
                }
    
    def _modify_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Modify parameters for recovery attempt"""
        modified = parameters.copy()
        
        # Modify numeric parameters slightly
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Adjust by small percentage
                if random.random() < 0.5:
                    modified[key] = value * 1.1  # Increase by 10%
                else:
                    modified[key] = value * 0.9  # Decrease by 10%
            elif isinstance(value, bool):
                # Flip boolean values
                modified[key] = not value
            elif isinstance(value, str) and value.lower() in ["true", "false"]:
                # Flip string boolean values
                modified[key] = "true" if value.lower() == "false" else "false"
        
        return modified

class StrategySelector:
    """Selects appropriate execution strategy based on context"""
    
    def __init__(self, config=None):
        self.strategies = {}  # id -> ExecutionStrategy
        self.execution_history = []
        self.max_history = 50
        # Added fields for enhanced functionality
        self.selector_lock = Lock()
        self.strategy_weights = {
            "context_match": 0.5,
            "historical_success": 0.3,
            "resource_efficiency": 0.2
        }
        self.config = config or {}
        self.default_strategy_id = None  # ID of fallback strategy
        self.performance_stats = {}  # Strategy ID -> performance metrics
        self.trace_id = None  # Current trace ID
    
    def register_strategy(self, strategy: ExecutionStrategy) -> None:
        """Register an execution strategy"""
        with self.selector_lock:
            self.strategies[strategy.id] = strategy
            
            # Set as default if first strategy or explicitly default
            if self.default_strategy_id is None or strategy.id == "default":
                self.default_strategy_id = strategy.id
                
            # Initialize performance stats
            self.performance_stats[strategy.id] = {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "total_executions": 0,
                "successful_executions": 0
            }
            
        # Log strategy registration
        logger.info(f"Registered execution strategy: {strategy.id} ({strategy.name})")
    
    def select_strategy(
        self, 
        context: Dict[str, Any], 
        procedure: Procedure,
        prefer_strategy_id: Optional[str] = None
    ) -> ExecutionStrategy:
        """Select the most appropriate execution strategy"""
        # Generate trace ID for strategy selection
        self.trace_id = f"trace_{uuid.uuid4().hex}"
        
        # Create trace for strategy selection
        with trace(
            workflow_name=f"strategy_selection_{procedure.name}",
            trace_id=self.trace_id,
            metadata={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name,
                "context_summary": str({k: v for k, v in context.items() if not isinstance(v, (dict, list))})[:500]
            }
        ):
            with self.selector_lock:
                if not self.strategies:
                    # No strategies registered, return a default
                    return ExecutionStrategy(
                        id="default", 
                        name="Default Strategy", 
                        description="Default execution strategy"
                    )
                
                # If a specific strategy is preferred, use it if available
                if prefer_strategy_id and prefer_strategy_id in self.strategies:
                    # Create span for selection by preference
                    with custom_span(
                        name="strategy_selection_by_preference",
                        data={
                            "prefer_strategy_id": prefer_strategy_id,
                            "procedure_id": procedure.id
                        }
                    ):
                        pass
                    
                    return self.strategies[prefer_strategy_id]
                    
                # Calculate scores for each strategy
                scores = []
                
                # Create span for strategy scoring
                with custom_span(
                    name="strategy_scoring",
                    data={
                        "procedure_id": procedure.id,
                        "procedure_name": procedure.name,
                        "strategies_count": len(self.strategies)
                    }
                ) as scoring_span:
                    for strategy_id, strategy in self.strategies.items():
                        # Get context match score
                        context_score = strategy.should_select(context, procedure)
                        
                        # Get historical performance score
                        history_score = self._calculate_historical_score(strategy_id, procedure.id)
                        
                        # Get resource efficiency score
                        resource_score = self._calculate_resource_score(strategy_id, context)
                        
                        # Calculate weighted total score
                        total_score = (
                            context_score * self.strategy_weights["context_match"] +
                            history_score * self.strategy_weights["historical_success"] +
                            resource_score * self.strategy_weights["resource_efficiency"]
                        )
                        
                        scores.append((strategy_id, total_score))
                        
                        # Create span for individual strategy score
                        with custom_span(
                            name=f"strategy_score_{strategy_id}",
                            data={
                                "strategy_id": strategy_id,
                                "context_score": context_score,
                                "history_score": history_score,
                                "resource_score": resource_score,
                                "total_score": total_score
                            },
                            parent=scoring_span
                        ):
                            pass
                
                # Get highest scoring strategy
                best_strategy_id, best_score = max(scores, key=lambda x: x[1])
                
                # Record selection
                self.execution_history.append({
                    "strategy_id": best_strategy_id,
                    "score": best_score,
                    "context": {k: v for k, v in context.items() if not isinstance(v, (dict, list))},
                    "procedure_id": procedure.id,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Create span for selection result
                with custom_span(
                    name="strategy_selection_result",
                    data={
                        "selected_strategy": best_strategy_id,
                        "score": best_score,
                        "procedure_id": procedure.id
                    }
                ):
                    pass
                
                # Trim history
                if len(self.execution_history) > self.max_history:
                    self.execution_history = self.execution_history[-self.max_history:]
                
                return self.strategies[best_strategy_id]
    
    def _calculate_historical_score(self, strategy_id: str, procedure_id: str) -> float:
        """Calculate score based on historical performance"""
        # Create trace span for historical score calculation
        with custom_span(
            name=f"historical_score_{strategy_id}",
            data={
                "strategy_id": strategy_id,
                "procedure_id": procedure_id
            }
        ) as history_span:
            # Get relevant history
            relevant_history = [h for h in self.execution_history 
                            if h["strategy_id"] == strategy_id and h["procedure_id"] == procedure_id]
            
            if not relevant_history:
                # No history for this strategy/procedure combination
                return 0.5  # Neutral score
                
            # Calculate success rate
            strategy_executions = [h for h in relevant_history]
            
            successful_executions = sum(1 for h in strategy_executions 
                                    if h.get("success", False))
            
            if not strategy_executions:
                return 0.5  # Neutral score
                
            success_rate = successful_executions / len(strategy_executions)
            
            # Get performance stats
            stats = self.performance_stats.get(strategy_id, {})
            
            # Consider both recent performance and overall stats
            recent_score = success_rate
            overall_score = stats.get("success_rate", 0.5)
            
            # Weight recent performance higher
            score = recent_score * 0.7 + overall_score * 0.3
            
            # Log score in trace
            with custom_span(
                name=f"historical_score_calculation_{strategy_id}",
                data={
                    "recent_score": recent_score,
                    "overall_score": overall_score,
                    "final_score": score
                },
                parent=history_span
            ):
                pass
            
            return score
    
    def _calculate_resource_score(self, strategy_id: str, context: Dict[str, Any]) -> float:
        """Calculate score based on resource efficiency"""
        # Create trace span for resource score calculation
        with custom_span(
            name=f"resource_score_{strategy_id}",
            data={"strategy_id": strategy_id}
        ) as resource_span:
            # Get strategy resource limits
            strategy = self.strategies.get(strategy_id)
            if not strategy:
                return 0.5  # Neutral score
                
            # Check current resource usage against limits
            resource_scores = []
            
            for resource, limit in strategy.resource_limits.items():
                # Get current usage from context
                current_usage = context.get(f"resource_{resource}", 0.0)
                
                # Calculate score for this resource (1.0 = no usage, 0.0 = at limit)
                if limit > 0:
                    resource_score = max(0.0, 1.0 - (current_usage / limit))
                    resource_scores.append(resource_score)
                    
                    # Log resource score in trace
                    with custom_span(
                        name=f"resource_{resource}_score",
                        data={
                            "resource": resource,
                            "current_usage": current_usage,
                            "limit": limit,
                            "score": resource_score
                        },
                        parent=resource_span
                    ):
                        pass
            
            # Calculate overall resource score
            if resource_scores:
                score = sum(resource_scores) / len(resource_scores)
            else:
                score = 0.8  # Good score if no resource constraints
                
            return score
    
    def update_performance_stats(
        self, 
        strategy_id: str, 
        success: bool, 
        execution_time: float
    ) -> None:
        """Update performance statistics for a strategy"""
        with self.selector_lock:
            stats = self.performance_stats.get(strategy_id)
            if not stats:
                return
                
            # Update statistics
            stats["total_executions"] += 1
            if success:
                stats["successful_executions"] += 1
                
            # Update success rate
            stats["success_rate"] = (
                stats["successful_executions"] / stats["total_executions"]
            )
            
            # Update average execution time with exponential smoothing
            if stats["avg_execution_time"] == 0.0:
                stats["avg_execution_time"] = execution_time
            else:
                stats["avg_execution_time"] = (
                    stats["avg_execution_time"] * 0.9 + execution_time * 0.1
                )
        
        # Create trace span for performance stats update
        with custom_span(
            name=f"update_performance_stats_{strategy_id}",
            data={
                "strategy_id": strategy_id,
                "success": success,
                "execution_time": execution_time,
                "new_success_rate": stats["success_rate"],
                "new_avg_execution_time": stats["avg_execution_time"],
                "total_executions": stats["total_executions"]
            }
        ):
            pass
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all strategies"""
        with self.selector_lock:
            return {
                strategy_id: {
                    "success_rate": stats["success_rate"],
                    "avg_execution_time": stats["avg_execution_time"],
                    "total_executions": stats["total_executions"]
                }
                for strategy_id, stats in self.performance_stats.items()
            }
    
    def get_recommended_strategy(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get a recommendation for the best strategy with explanation"""
        # Generate trace ID for recommendation
        self.trace_id = f"trace_{uuid.uuid4().hex}"
        
        # Create trace for strategy recommendation
        with trace(
            workflow_name=f"strategy_recommendation_{procedure.name}",
            trace_id=self.trace_id,
            metadata={
                "procedure_id": procedure.id,
                "procedure_name": procedure.name
            }
        ):
            with self.selector_lock:
                # Calculate scores with explanations
                strategy_scores = []
                
                for strategy_id, strategy in self.strategies.items():
                    # Calculate detailed scores
                    context_score = strategy.should_select(context, procedure)
                    history_score = self._calculate_historical_score(strategy_id, procedure.id)
                    resource_score = self._calculate_resource_score(strategy_id, context)
                    
                    # Calculate weighted total
                    total_score = (
                        context_score * self.strategy_weights["context_match"] +
                        history_score * self.strategy_weights["historical_success"] +
                        resource_score * self.strategy_weights["resource_efficiency"]
                    )
                    
                    # Create explanation
                    strategy_scores.append({
                        "strategy_id": strategy_id,
                        "strategy_name": strategy.name,
                        "total_score": total_score,
                        "scores": {
                            "context_match": context_score,
                            "historical_success": history_score,
                            "resource_efficiency": resource_score
                        },
                        "weights": self.strategy_weights,
                        "description": strategy.description
                    })
                
                # Sort by total score
                strategy_scores.sort(key=lambda x: x["total_score"], reverse=True)
                
                # Create span for recommendation result
                with custom_span(
                    name="strategy_recommendation_result",
                    data={
                        "recommended_strategy": strategy_scores[0]["strategy_id"] if strategy_scores else None,
                        "procedure_id": procedure.id
                    }
                ):
                    pass
                
                return {
                    "recommended_strategy": strategy_scores[0]["strategy_id"] if strategy_scores else None,
                    "all_scores": strategy_scores,
                    "procedure_id": procedure.id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
