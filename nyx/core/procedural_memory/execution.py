# nyx/core/procedural_memory/execution.py

import datetime
import random
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from .models import Procedure

class ExecutionStrategy(BaseModel):
    """Strategy for executing a procedure"""
    id: str
    name: str
    description: str
    selection_criteria: Dict[str, Any] = Field(default_factory=dict)
    
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
        
        return min(1.0, score)

class DeliberateExecutionStrategy(ExecutionStrategy):
    """Executes procedure carefully with validation between steps"""
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure deliberately with checks between steps"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Initialize execution state
        execution_state = context.copy()
        execution_state["strategy"] = "deliberate"
        execution_state["execution_history"] = []
        
        # Execute steps sequentially with validation
        for i, step in enumerate(procedure.steps):
            # Validate preconditions
            if not self._validate_preconditions(step, execution_state):
                results.append({
                    "step_id": step["id"],
                    "success": False,
                    "error": "Preconditions not met",
                    "execution_time": 0.0
                })
                success = False
                break
            
            # Execute the step
            step_result = await self._execute_step(step, execution_state)
            results.append(step_result)
            
            # Update execution state
            execution_state[f"step_{step['id']}_result"] = step_result
            execution_state["execution_history"].append({
                "step_id": step["id"],
                "function": step["function"],
                "success": step_result["success"],
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Check for failure
            if not step_result["success"]:
                success = False
                break
            
            # Validate postconditions
            if not self._validate_postconditions(step, execution_state):
                results.append({
                    "step_id": step["id"],
                    "success": False,
                    "error": "Postconditions not met",
                    "execution_time": 0.0
                })
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "strategy": "deliberate"
        }
    
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
        # This would normally call the actual function
        # For now, just return a success result
        return {
            "step_id": step["id"],
            "success": True,
            "execution_time": 0.1,
            "data": {}
        }

class AutomaticExecutionStrategy(ExecutionStrategy):
    """Fast execution without validation between steps"""
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure automatically without validation"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Initialize execution state
        execution_state = context.copy()
        execution_state["strategy"] = "automatic"
        execution_state["execution_history"] = []
        
        # Check if procedure is chunked
        if procedure.is_chunked:
            # Execute chunks
            chunks = self._get_chunks(procedure)
            
            for chunk_id, chunk_steps in chunks.items():
                # Execute chunk as a unit
                chunk_result = await self._execute_chunk(
                    chunk_steps, 
                    execution_state, 
                    chunk_id
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
            for step in procedure.steps:
                # Execute the step
                step_result = await self._execute_step(step, execution_state)
                results.append(step_result)
                
                # Update execution state
                execution_state[f"step_{step['id']}_result"] = step_result
                execution_state["execution_history"].append({
                    "step_id": step["id"],
                    "function": step["function"],
                    "success": step_result["success"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Check for failure
                if not step_result["success"]:
                    success = False
                    break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
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
            # Get steps for this chunk
            chunk_steps = [step for step in procedure.steps if step["id"] in step_ids]
            chunks[chunk_id] = chunk_steps
        
        return chunks
    
    async def _execute_chunk(
        self, 
        steps: List[Dict[str, Any]], 
        context: Dict[str, Any], 
        chunk_id: str
    ) -> Dict[str, Any]:
        """Execute a chunk of steps as a unit"""
        results = []
        success = True
        start_time = datetime.datetime.now()
        
        for step in steps:
            # Execute the step
            step_result = await self._execute_step(step, context)
            results.append(step_result)
            
            # Check for failure
            if not step_result["success"]:
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "chunk_id": chunk_id
        }
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step"""
        # This would normally call the actual function
        # For now, just return a success result
        return {
            "step_id": step["id"],
            "success": True,
            "execution_time": 0.05,  # Faster than deliberate execution
            "data": {}
        }

class AdaptiveExecutionStrategy(ExecutionStrategy):
    """Adapts execution based on context and feedback"""
    
    async def execute(
        self, 
        procedure: Procedure, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute procedure with adaptive strategy selection"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Initialize execution state
        execution_state = context.copy()
        execution_state["strategy"] = "adaptive"
        execution_state["execution_history"] = []
        
        # Determine initial execution mode based on proficiency
        deliberate_execution = procedure.proficiency < 0.8
        
        # Execute steps with adaptive strategy
        for i, step in enumerate(procedure.steps):
            # Decide execution strategy for this step
            step_strategy = self._select_step_strategy(step, execution_state, deliberate_execution)
            
            # Execute step with selected strategy
            if step_strategy == "deliberate":
                # Careful execution with validation
                if not self._validate_preconditions(step, execution_state):
                    results.append({
                        "step_id": step["id"],
                        "success": False,
                        "error": "Preconditions not met",
                        "execution_time": 0.0,
                        "strategy": "deliberate"
                    })
                    success = False
                    break
                
                step_result = await self._execute_step(step, execution_state)
                step_result["strategy"] = "deliberate"
                
                if not self._validate_postconditions(step, execution_state, step_result):
                    step_result["success"] = False
                    step_result["error"] = "Postconditions not met"
                    success = False
                    results.append(step_result)
                    break
            else:
                # Fast execution without validation
                step_result = await self._execute_step(step, execution_state)
                step_result["strategy"] = "automatic"
            
            # Add to results
            results.append(step_result)
            
            # Update execution state
            execution_state[f"step_{step['id']}_result"] = step_result
            execution_state["execution_history"].append({
                "step_id": step["id"],
                "function": step["function"],
                "success": step_result["success"],
                "strategy": step_strategy,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Check for failure and adapt
            if not step_result["success"]:
                # Adapt strategy on failure
                deliberate_execution = True
                
                # Try to recover if possible
                if i < len(procedure.steps) - 1:
                    recovery_successful = await self._attempt_recovery(
                        step, 
                        procedure.steps[i+1:], 
                        execution_state
                    )
                    
                    if recovery_successful:
                        # Continue execution
                        continue
                
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "strategy": "adaptive",
            "adaptations": execution_state.get("adaptations", [])
        }
    
    def _select_step_strategy(
        self, 
        step: Dict[str, Any], 
        state: Dict[str, Any], 
        default_deliberate: bool
    ) -> str:
        """Select execution strategy for a step"""
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
        # This would normally call the actual function
        # For now, just return a success result
        return {
            "step_id": step["id"],
            "success": True,
            "execution_time": 0.07,  # Between deliberate and automatic
            "data": {}
        }
    
    async def _attempt_recovery(
        self, 
        failed_step: Dict[str, Any], 
        remaining_steps: List[Dict[str, Any]], 
        state: Dict[str, Any]
    ) -> bool:
        """Attempt to recover from a step failure"""
        # Track adaptation
        if "adaptations" not in state:
            state["adaptations"] = []
        
        state["adaptations"].append({
            "type": "recovery_attempt",
            "step_id": failed_step["id"],
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Try a retry with modified parameters
        modified_params = self._modify_parameters(failed_step.get("parameters", {}))
        
        retry_step = failed_step.copy()
        retry_step["parameters"] = modified_params
        retry_step["is_recovery"] = True
        
        # Execute with modified parameters
        retry_result = await self._execute_step(retry_step, state)
        
        if retry_result.get("success", False):
            # Recovery successful
            state["adaptations"][-1]["result"] = "success"
            state["adaptations"][-1]["method"] = "parameter_modification"
            
            # Update execution state
            state[f"step_{failed_step['id']}_result"] = retry_result
            state["execution_history"].append({
                "step_id": failed_step["id"],
                "function": failed_step["function"],
                "success": True,
                "strategy": "recovery",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return True
        
        # Recovery failed
        state["adaptations"][-1]["result"] = "failure"
        return False
    
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
        
        return modified

class StrategySelector:
    """Selects appropriate execution strategy based on context"""
    
    def __init__(self):
        self.strategies = {}  # id -> ExecutionStrategy
        self.execution_history = []
        self.max_history = 50
    
    def register_strategy(self, strategy: ExecutionStrategy) -> None:
        """Register an execution strategy"""
        self.strategies[strategy.id] = strategy
    
    def select_strategy(self, context: Dict[str, Any], procedure: Procedure) -> ExecutionStrategy:
        """Select the most appropriate execution strategy"""
        if not self.strategies:
            # No strategies registered, return a default
            return ExecutionStrategy(
                id="default", 
                name="Default Strategy", 
                description="Default execution strategy"
            )
        
        # Calculate scores for each strategy
        scores = []
        for strategy_id, strategy in self.strategies.items():
            score = strategy.should_select(context, procedure)
            scores.append((strategy_id, score))
        
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
        
        # Trim history
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
        
        return self.strategies[best_strategy_id]
