# nyx/core/procedural_memory/manager.py

import asyncio
import datetime
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from collections import Counter, defaultdict

# OpenAI Agents SDK imports
from agents import Agent, Runner, trace, function_tool, handoff, RunContextWrapper
from agents.exceptions import ModelBehaviorError, UserError

# Import core components 
from .models import (
    ActionTemplate, ChunkTemplate, ContextPattern, ChunkPrediction,
    ControlMapping, ProcedureTransferRecord, Procedure, StepResult,
    ProcedureStats, TransferStats, HierarchicalProcedure
)
from .chunk_selection import ContextAwareChunkSelector
from .generalization import ProceduralChunkLibrary

# Import enhanced components
from .models import (
    CausalModel, TemporalNode, TemporalProcedureGraph, 
    ProcedureGraph, WorkingMemoryController,
    ParameterOptimizer, TransferLearningOptimizer
)
from .learning import ObservationLearner, ProceduralMemoryConsolidator
from .execution import (
    ExecutionStrategy, DeliberateExecutionStrategy, 
    AutomaticExecutionStrategy, AdaptiveExecutionStrategy,
    StrategySelector
)
from .temporal import TemporalProcedureGraph, ProcedureGraph

# Set up logging
logger = logging.getLogger(__name__)


class ProceduralMemoryManager:
    """
    Procedural memory system integrated with Agents SDK
    
    Manages procedural knowledge including learning, execution,
    chunking, and cross-domain transfer through agent-based architecture.
    """
    
    def __init__(self, memory_core=None, knowledge_core=None):
        self.procedures = {}  # name -> Procedure
        self.memory_core = memory_core
        self.knowledge_core = knowledge_core
        
        # Context awareness
        self.chunk_selector = ContextAwareChunkSelector()
        
        # Generalization
        self.chunk_library = ProceduralChunkLibrary()
        
        # Function registry
        self.function_registry = {}  # Global function registry
        
        # Transfer stats
        self.transfer_stats = {
            "total_transfers": 0,
            "successful_transfers": 0,
            "avg_success_level": 0.0,
            "avg_practice_needed": 0
        }
        
        # Initialize agents
        self._proc_manager_agent = self._create_manager_agent()
        self._proc_execution_agent = self._create_execution_agent()
        self._proc_analysis_agent = self._create_analysis_agent()
        
        # Initialize common control mappings
        self._initialize_control_mappings()
        
        # Initialization flag
        self.initialized = False
    
    async def initialize(self):
        """Initialize the procedural memory manager"""
        if self.initialized:
            return
            
        # Initialize control mappings
        self._initialize_control_mappings()
        
        # Set up default functions
        self._register_default_functions()
        
        self.initialized = True
        logger.info("Procedural memory manager initialized")
    
    def _create_manager_agent(self) -> Agent:
        """Create the main procedural memory manager agent"""
        return Agent(
            name="Procedural Memory Manager",
            instructions="""
            You are a procedural memory manager agent that handles the storage, retrieval,
            and execution of procedural knowledge. You help the system learn, optimize, and
            transfer procedural skills across domains.
            
            Your responsibilities include:
            - Managing procedural memory entries
            - Facilitating procedural skill transfer between domains
            - Tracking procedural memory statistics
            - Optimizing procedures through chunking and refinement
            
            You have enhanced capabilities for cross-domain transfer:
            - You can identify and transfer chunks of procedural knowledge
            - You can generalize procedures across different domains
            - You can find similar patterns across diverse procedural skills
            - You can optimize transfer through specialized chunk mapping
            """,
            tools=[]  # Tools will be added during initialization
        )
    
    def _create_execution_agent(self) -> Agent:
        """Create the agent responsible for procedure execution"""
        return Agent(
            name="Procedure Execution Agent",
            instructions="""
            You are a procedure execution agent that carries out procedural skills.
            Your job is to execute procedures efficiently, adapting to context and
            making appropriate decisions during execution.
            
            Your responsibilities include:
            - Executing procedure steps in the correct order
            - Adapting to different execution contexts
            - Monitoring execution success and performance
            - Providing feedback on execution quality
            """,
            tools=[]  # Tools will be added during initialization
        )
    
    def _create_analysis_agent(self) -> Agent:
        """Create the agent responsible for procedure analysis"""
        return Agent(
            name="Procedure Analysis Agent",
            instructions="""
            You are a procedure analysis agent that examines procedural knowledge.
            Your job is to identify patterns, optimization opportunities, and
            potential transfers between domains.
            
            Your responsibilities include:
            - Identifying chunking opportunities
            - Finding similarities between procedures
            - Recommending procedure refinements
            - Analyzing procedure performance
            - Identifying generalizable chunks across domains
            - Evaluating transfer potential between domains
            - Finding optimal chunking strategies for transfer
            """,
            tools=[]  # Tools will be added during initialization
        )
    
    def _register_default_functions(self):
        """Register default functions that can be used in procedures"""
        # Example of registering a simple function
        async def noop(*args, **kwargs):
            """No-operation function"""
            return {"success": True}
            
        self.register_function("noop", noop)
    
    def _initialize_control_mappings(self):
        """Initialize common control mappings between domains"""
        # PlayStation to Xbox mappings
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="primary_action",
            source_control="R1",
            target_control="RB"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="secondary_action",
            source_control="L1",
            target_control="LB"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="aim",
            source_control="L2",
            target_control="LT"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="playstation",
            target_domain="xbox",
            action_type="shoot",
            source_control="R2",
            target_control="RT"
        ))
        
        # Input method mappings (touch to mouse/keyboard)
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="touch_interface",
            target_domain="mouse_interface",
            action_type="select",
            source_control="tap",
            target_control="click"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="voice_interface",
            target_domain="touch_interface",
            action_type="activate",
            source_control="speak_command",
            target_control="tap"
        ))
        
        # Cross-domain action mappings (driving to flying)
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="driving",
            target_domain="flying",
            action_type="accelerate",
            source_control="pedal_press",
            target_control="throttle_forward"
        ))
        
        # Game-specific mappings
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="dbd",  # Dead by Daylight
            target_domain="dbd",  # Default same-game mapping
            action_type="sprint",
            source_control="L1",
            target_control="L1"
        ))
        
        self.chunk_library.add_control_mapping(ControlMapping(
            source_domain="dbd",
            target_domain="dbd",
            action_type="interaction",
            source_control="R1",
            target_control="R1"
        ))
    
    def register_function(self, name: str, func: Callable):
        """Register a function for use in procedures"""
        self.function_registry[name] = func
    
    async def execute_procedure_steps(self, 
                                    procedure: Procedure, 
                                    context: Dict[str, Any], 
                                    conscious_execution: bool) -> Dict[str, Any]:
        """Execute the steps of a procedure"""
        start_time = datetime.datetime.now()
        results = []
        success = True
        
        # Record execution context
        execution_context = context.copy()
        execution_context["timestamp"] = start_time.isoformat()
        execution_context["conscious_execution"] = conscious_execution
        
        if hasattr(procedure, "context_history"):
            if len(procedure.context_history) >= procedure.max_history:
                procedure.context_history = procedure.context_history[-(procedure.max_history-1):]
            procedure.context_history.append(execution_context)
        
        # Execute in different modes based on proficiency and settings
        if conscious_execution or procedure.proficiency < 0.8:
            # Deliberate step-by-step execution
            for step in procedure.steps:
                step_result = await self.execute_step(step, context)
                results.append(step_result)
                
                # Update context with step result
                context[f"step_{step['id']}_result"] = step_result
                
                # Add to action history
                if "action_history" not in context:
                    context["action_history"] = []
                context["action_history"].append({
                    "step_id": step["id"],
                    "function": step["function"],
                    "success": step_result["success"]
                })
                
                # Stop execution if a step fails and we're in conscious mode
                if not step_result["success"] and conscious_execution:
                    success = False
                    break
        else:
            # Automatic chunked execution if available
            if procedure.is_chunked:
                # Get available chunks
                chunks = self._get_chunks(procedure)
                
                if hasattr(self, "chunk_selector") and self.chunk_selector:
                    # Context-aware chunk selection
                    prediction = self.chunk_selector.select_chunk(
                        available_chunks=chunks,
                        context=context,
                        procedure_domain=procedure.domain
                    )
                    
                    # Execute chunks based on prediction
                    executed_chunks = []
                    
                    # First execute most likely chunk
                    main_chunk_id = prediction.chunk_id
                    
                    if main_chunk_id in chunks:
                        chunk_steps = chunks[main_chunk_id]
                        chunk_result = await self._execute_chunk(
                            chunk_steps=chunk_steps, 
                            context=context, 
                            minimal_monitoring=True,
                            chunk_id=main_chunk_id,
                            procedure=procedure
                        )
                        results.extend(chunk_result["results"])
                        executed_chunks.append(main_chunk_id)
                        
                        if not chunk_result["success"]:
                            success = False
                    
                    # Execute remaining steps that weren't in chunks
                    remaining_steps = self._get_steps_not_in_chunks(procedure, executed_chunks)
                    
                    for step in remaining_steps:
                        step_result = await self.execute_step(step, context, minimal_monitoring=True)
                        results.append(step_result)
                        
                        if not step_result["success"]:
                            success = False
                            break
                else:
                    # Simple chunk execution without context awareness
                    for chunk_id, chunk_steps in chunks.items():
                        chunk_result = await self._execute_chunk(
                            chunk_steps=chunk_steps, 
                            context=context, 
                            minimal_monitoring=True,
                            chunk_id=chunk_id,
                            procedure=procedure
                        )
                        results.extend(chunk_result["results"])
                        
                        if not chunk_result["success"]:
                            success = False
                            break
            else:
                # No chunks yet, but still in automatic mode - execute with minimal monitoring
                for step in procedure.steps:
                    step_result = await self.execute_step(step, context, minimal_monitoring=True)
                    results.append(step_result)
                    
                    if not step_result["success"]:
                        success = False
                        break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Update overall statistics
        self.update_procedure_stats(procedure, execution_time, success)
        
        # Check for opportunities to improve
        if procedure.execution_count % 5 == 0:  # Every 5 executions
            self._identify_refinement_opportunities(procedure, results)
        
        # Check for chunking opportunities
        if not procedure.is_chunked and procedure.proficiency > 0.7 and procedure.execution_count >= 10:
            self._identify_chunking_opportunities(procedure, results)
            
            # After chunking, try to generalize chunks
            if procedure.is_chunked and hasattr(self, "chunk_library"):
                self._generalize_chunks(procedure)
        
        # Return execution results
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time,
            "proficiency": procedure.proficiency,
            "automatic": not conscious_execution and procedure.proficiency >= 0.8,
            "chunked": procedure.is_chunked
        }
    
    async def execute_step(self, 
                         step: Dict[str, Any], 
                         context: Dict[str, Any], 
                         minimal_monitoring: bool = False) -> Dict[str, Any]:
        """Execute a single step of a procedure"""
        # Get the actual function to call
        func_name = step["function"]
        func = self.function_registry.get(func_name)
        
        if not func:
            return StepResult(
                success=False,
                error=f"Function {func_name} not registered",
                execution_time=0.0
            ).dict()
        
        # Execute with timing
        step_start = datetime.datetime.now()
        try:
            # Prepare parameters with context
            params = step.get("parameters", {}).copy()
            
            # Check if function accepts context parameter
            if callable(func) and hasattr(func, "__code__") and "context" in func.__code__.co_varnames:
                params["context"] = context
                
            # Execute the function
            result = await func(**params)
            
            # Check result format and standardize
            if isinstance(result, dict):
                success = "error" not in result
                step_result = {
                    "success": success,
                    "data": result,
                    "execution_time": 0.0
                }
                
                if not success:
                    step_result["error"] = result.get("error")
            else:
                step_result = {
                    "success": True,
                    "data": {"result": result},
                    "execution_time": 0.0
                }
        except Exception as e:
            logger.error(f"Error executing step {step['id']}: {str(e)}")
            step_result = {
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }
        
        # Calculate execution time
        step_time = (datetime.datetime.now() - step_start).total_seconds()
        step_result["execution_time"] = step_time
        
        return step_result
    
    async def _execute_chunk(self, 
                           chunk_steps: List[Dict[str, Any]], 
                           context: Dict[str, Any], 
                           minimal_monitoring: bool = False,
                           chunk_id: str = None,
                           procedure: Procedure = None) -> Dict[str, Any]:
        """Execute a chunk of steps as a unit"""
        results = []
        success = True
        start_time = datetime.datetime.now()
        
        # Create chunk-specific context
        chunk_context = context.copy()
        if chunk_id:
            chunk_context["current_chunk"] = chunk_id
        
        # Execute steps
        for step in chunk_steps:
            step_result = await self.execute_step(step, chunk_context, minimal_monitoring)
            results.append(step_result)
            
            if not step_result["success"]:
                success = False
                break
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Update chunk template if using library
        if hasattr(self, "chunk_library") and procedure and hasattr(procedure, "generalized_chunks") and chunk_id in procedure.generalized_chunks:
            template_id = procedure.generalized_chunks[chunk_id]
            self.chunk_library.update_template_success(
                template_id=template_id,
                domain=procedure.domain,
                success=success
            )
        
        return {
            "success": success,
            "results": results,
            "execution_time": execution_time
        }
    
    def update_procedure_stats(self, procedure: Procedure, execution_time: float, success: bool):
        """Update statistics for a procedure after execution"""
        # Update average time
        if procedure.execution_count == 0:
            procedure.average_execution_time = execution_time
        else:
            procedure.average_execution_time = (procedure.average_execution_time * 0.8) + (execution_time * 0.2)
        
        # Update counts
        procedure.execution_count += 1
        if success:
            procedure.successful_executions += 1
        
        # Update proficiency based on multiple factors
        count_factor = min(procedure.execution_count / 100, 1.0)
        success_rate = procedure.successful_executions / max(1, procedure.execution_count)
        
        # Calculate time factor (simplified)
        time_factor = 0.5
        
        # Combine factors with weights
        procedure.proficiency = (count_factor * 0.3) + (success_rate * 0.5) + (time_factor * 0.2)
        
        # Update last execution timestamp
        procedure.last_execution = datetime.datetime.now().isoformat()
        procedure.last_updated = datetime.datetime.now().isoformat()
    
    def _get_chunks(self, procedure: Procedure) -> Dict[str, List[Dict[str, Any]]]:
        """Get the current chunks as step dictionaries"""
        chunks = {}
        
        for chunk_id, step_ids in procedure.chunked_steps.items():
            # Convert step IDs to actual step dictionaries
            steps = [next((s for s in procedure.steps if s["id"] == step_id), None) for step_id in step_ids]
            steps = [s for s in steps if s is not None]  # Remove None values
            
            chunks[chunk_id] = steps
            
        return chunks
    
    def _get_steps_not_in_chunks(self, procedure: Procedure, executed_chunks: List[str]) -> List[Dict[str, Any]]:
        """Get steps that aren't in the specified chunks"""
        # Get all step IDs in executed chunks
        chunked_step_ids = set()
        for chunk_id in executed_chunks:
            if chunk_id in procedure.chunked_steps:
                chunked_step_ids.update(procedure.chunked_steps[chunk_id])
        
        # Return steps not in chunks
        return [step for step in procedure.steps if step["id"] not in chunked_step_ids]
    
    def _identify_chunking_opportunities(self, procedure: Procedure, recent_results: List[Dict[str, Any]]):
        """Look for opportunities to chunk steps together"""
        # Need at least 3 steps to consider chunking
        if len(procedure.steps) < 3:
            return
        
        # Find sequences of steps that always succeed together
        chunks = []
        current_chunk = []
        
        for i in range(len(procedure.steps) - 1):
            # Start a new potential chunk
            if not current_chunk:
                current_chunk = [procedure.steps[i]["id"]]
            
            # Check if next step is consistently executed after this one
            co_occurrence = self.calculate_step_co_occurrence(
                procedure,
                procedure.steps[i]["id"], 
                procedure.steps[i+1]["id"]
            )
            
            if co_occurrence > 0.9:  # High co-occurrence threshold
                # Add to current chunk
                current_chunk.append(procedure.steps[i+1]["id"])
            else:
                # End current chunk if it has multiple steps
                if len(current_chunk) > 1:
                    chunks.append(current_chunk)
                current_chunk = []
        
        # Add the last chunk if it exists
        if len(current_chunk) > 1:
            chunks.append(current_chunk)
        
        # Apply chunking if we found opportunities
        if chunks:
            self._apply_chunking(procedure, chunks)
    
    def calculate_step_co_occurrence(self, procedure: Procedure, step1_id: str, step2_id: str) -> float:
        """Calculate how often step2 follows step1 in successful executions"""
        # Check if we have sufficient context history
        if hasattr(procedure, "context_history") and len(procedure.context_history) >= 5:
            # Check historical co-occurrence in context history
            actual_co_occurrences = 0
            possible_co_occurrences = 0
            
            for context in procedure.context_history:
                action_history = context.get("action_history", [])
                
                # Look for sequential occurrences
                for i in range(len(action_history) - 1):
                    if action_history[i].get("step_id") == step1_id:
                        possible_co_occurrences += 1
                        
                        if i+1 < len(action_history) and action_history[i+1].get("step_id") == step2_id:
                            actual_co_occurrences += 1
            
            if possible_co_occurrences > 0:
                return actual_co_occurrences / possible_co_occurrences
        
        # Fallback: use execution count as a proxy for co-occurrence
        # Higher execution count = more likely the steps have been executed together
        if procedure.execution_count > 5:
            return 0.8
        
        return 0.5  # Default moderate co-occurrence
    
    def _apply_chunking(self, procedure: Procedure, chunks: List[List[str]]):
        """Apply identified chunks to the procedure"""
        # Create chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1}"
            procedure.chunked_steps[chunk_id] = chunk
            
            # Look for context patterns in history
            if hasattr(self, "chunk_selector") and self.chunk_selector:
                context_pattern = self.chunk_selector.create_context_pattern_from_history(
                    chunk_id=chunk_id,
                    domain=procedure.domain
                )
                
                if context_pattern:
                    # Store reference to context pattern
                    procedure.chunk_contexts[chunk_id] = context_pattern.id
        
        # Mark as chunked
        procedure.is_chunked = True
        
        logger.info(f"Applied chunking to procedure {procedure.name}: {procedure.chunked_steps}")
    
    def _identify_refinement_opportunities(self, procedure: Procedure, recent_results: List[Dict[str, Any]]):
        """Look for opportunities to refine the procedure"""
        # Skip if too few executions
        if procedure.execution_count < 5:
            return
        
        # Check for steps with low success rates
        step_success_rates = {}
        for i, step in enumerate(procedure.steps):
            # Try to find success rate in recent results
            if i < len(recent_results):
                success = recent_results[i].get("success", False)
                
                # Initialize if not exists
                if step["id"] not in step_success_rates:
                    step_success_rates[step["id"]] = {"successes": 0, "total": 0}
                
                # Update counts
                step_success_rates[step["id"]]["total"] += 1
                if success:
                    step_success_rates[step["id"]]["successes"] += 1
        
        # Check for low success rates
        for step_id, stats in step_success_rates.items():
            if stats["total"] >= 3:  # Only consider steps executed at least 3 times
                success_rate = stats["successes"] / stats["total"]
                
                if success_rate < 0.8:
                    # This step needs improvement
                    step = next((s for s in procedure.steps if s["id"] == step_id), None)
                    if step:
                        # Create refinement opportunity
                        new_opportunity = {
                            "step_id": step_id,
                            "type": "improve_reliability",
                            "current_success_rate": success_rate,
                            "identified_at": datetime.datetime.now().isoformat(),
                            "description": f"Step '{step.get('description', step_id)}' has a low success rate of {success_rate:.2f}"
                        }
                        
                        # Add to opportunities if not already present
                        if not any(r.get("step_id") == step_id and r.get("type") == "improve_reliability" 
                                for r in procedure.refinement_opportunities):
                            procedure.refinement_opportunities.append(new_opportunity)
    
    def _generalize_chunks(self, procedure: Procedure):
        """Try to create generalizable templates from chunks"""
        if not hasattr(self, "chunk_library") or not self.chunk_library:
            return
            
        # Skip if not chunked
        if not procedure.is_chunked:
            return
            
        # Get chunks as steps
        chunks = self._get_chunks(procedure)
        
        for chunk_id, chunk_steps in chunks.items():
            # Skip if already generalized
            if hasattr(procedure, "generalized_chunks") and chunk_id in procedure.generalized_chunks:
                continue
                
            # Try to create a template
            template = self.chunk_library.create_chunk_template_from_steps(
                chunk_id=f"template_{chunk_id}_{procedure.name}",
                name=f"{procedure.name} - {chunk_id}",
                steps=chunk_steps,
                domain=procedure.domain,
                success_rate=0.9  # High initial success rate in source domain
            )
            
            if template:
                # Store reference to template
                procedure.generalized_chunks[chunk_id] = template.id
                logger.info(f"Created generalized template {template.id} from chunk {chunk_id}")
    
    def calculate_procedure_similarity(self, proc1: Procedure, proc2: Procedure) -> float:
        """Calculate similarity between two procedures"""
        # If either doesn't have steps, return 0
        if not proc1.steps or not proc2.steps:
            return 0.0
        
        # If they have the same domain, higher base similarity
        domain_similarity = 0.3 if proc1.domain == proc2.domain else 0.0
        
        # Compare steps
        steps1 = [(s["function"], s.get("description", "")) for s in proc1.steps]
        steps2 = [(s["function"], s.get("description", "")) for s in proc2.steps]
        
        # Calculate Jaccard similarity on functions
        funcs1 = set(f for f, _ in steps1)
        funcs2 = set(f for f, _ in steps2)
        
        if not funcs1 or not funcs2:
            func_similarity = 0.0
        else:
            intersection = len(funcs1.intersection(funcs2))
            union = len(funcs1.union(funcs2))
            func_similarity = intersection / union
        
        # Calculate approximate sequence similarity
        step_similarity = 0.0
        if len(steps1) > 0 and len(steps2) > 0:
            # Simplified sequence comparison
            matched_steps = 0
            for i in range(min(len(steps1), len(steps2))):
                if steps1[i][0] == steps2[i][0]:
                    matched_steps += 1
            
            step_similarity = matched_steps / min(len(steps1), len(steps2))
        
        # Calculate final similarity
        return 0.3 * domain_similarity + 0.4 * func_similarity + 0.3 * step_similarity
    
    def map_step_to_domain(self, step: Dict[str, Any], source_domain: str, target_domain: str) -> Optional[Dict[str, Any]]:
        """Map a procedure step from one domain to another"""
        # Get original function and parameters
        function = step.get("function")
        params = step.get("parameters", {})
        
        if not function:
            return None
        
        # Try to find a control mapping
        mapped_params = params.copy()
        
        # Check for control-like parameters that might need mapping
        param_keys = ["button", "control", "input_method", "key"]
        
        for param_key in param_keys:
            if param_key in params:
                control_value = params[param_key]
                
                # Look for control mappings for this action type
                for mapping in self.chunk_library.control_mappings:
                    if (mapping.source_domain == source_domain and 
                        mapping.target_domain == target_domain and 
                        mapping.source_control == control_value):
                        # Found a mapping, apply it
                        mapped_params[param_key] = mapping.target_control
                        break
        
        # Create mapped step
        mapped_step = {
            "id": step["id"],
            "description": step["description"],
            "function": function,
            "parameters": mapped_params,
            "original_id": step["id"]
        }
        
        return mapped_step
    
    async def analyze_execution_history(self, procedure_name: str) -> Dict[str, Any]:
        """Analyze execution history of a procedure for patterns"""
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Skip if insufficient execution history
        if procedure.execution_count < 3:
            return {
                "procedure_name": procedure_name,
                "executions": procedure.execution_count,
                "analysis": "Insufficient execution history for analysis"
            }
        
        # Analyze context history if available
        context_patterns = []
        if hasattr(procedure, "context_history") and len(procedure.context_history) >= 3:
            # Look for common context indicators
            context_keys = set()
            for context in procedure.context_history:
                context_keys.update(context.keys())
            
            # Filter out standard keys
            standard_keys = {"timestamp", "conscious_execution", "result", "execution_time", "action_history"}
            context_keys = context_keys - standard_keys
            
            # Analyze values for each key
            for key in context_keys:
                values = [context.get(key) for context in procedure.context_history if key in context]
                if len(values) >= 3:  # Need at least 3 occurrences
                    # Check consistency
                    unique_values = set(str(v) for v in values)
                    if len(unique_values) == 1:
                        # Consistent value
                        context_patterns.append({
                            "key": key,
                            "value": values[0],
                            "occurrences": len(values),
                            "pattern_type": "consistent_value"
                        })
                    elif len(unique_values) <= len(values) / 2:
                        # Semi-consistent values
                        value_counts = {}
                        for v in values:
                            v_str = str(v)
                            if v_str not in value_counts:
                                value_counts[v_str] = 0
                            value_counts[v_str] += 1
                        
                        # Find most common value
                        most_common = max(value_counts.items(), key=lambda x: x[1])
                        
                        context_patterns.append({
                            "key": key,
                            "most_common_value": most_common[0],
                            "occurrence_rate": most_common[1] / len(values),
                            "pattern_type": "common_value"
                        })
        
        # Analyze successful vs. unsuccessful executions
        success_patterns = []
        if hasattr(procedure, "context_history") and len(procedure.context_history) >= 3:
            successful_contexts = [ctx for ctx in procedure.context_history if ctx.get("result", False)]
            unsuccessful_contexts = [ctx for ctx in procedure.context_history if not ctx.get("result", True)]
            
            if successful_contexts and unsuccessful_contexts:
                # Find keys that differ between successful and unsuccessful executions
                for key in context_keys:
                    # Get values for successful executions
                    success_values = [context.get(key) for context in successful_contexts if key in context]
                    if not success_values:
                        continue
                        
                    # Get values for unsuccessful executions
                    failure_values = [context.get(key) for context in unsuccessful_contexts if key in context]
                    if not failure_values:
                        continue
                    
                    # Check if values are consistently different
                    success_unique = set(str(v) for v in success_values)
                    failure_unique = set(str(v) for v in failure_values)
                    
                    # If no overlap, this might be a discriminating factor
                    if not success_unique.intersection(failure_unique):
                        success_patterns.append({
                            "key": key,
                            "success_values": list(success_unique),
                            "failure_values": list(failure_unique),
                            "pattern_type": "success_factor"
                        })
        
        # Analyze chunks if available
        chunk_patterns = []
        if procedure.is_chunked:
            for chunk_id, step_ids in procedure.chunked_steps.items():
                chunk_patterns.append({
                    "chunk_id": chunk_id,
                    "step_count": len(step_ids),
                    "has_template": chunk_id in procedure.generalized_chunks if hasattr(procedure, "generalized_chunks") else False,
                    "has_context_pattern": chunk_id in procedure.chunk_contexts if hasattr(procedure, "chunk_contexts") else False
                })
        
        return {
            "procedure_name": procedure_name,
            "executions": procedure.execution_count,
            "success_rate": procedure.successful_executions / max(1, procedure.execution_count),
            "avg_execution_time": procedure.average_execution_time,
            "proficiency": procedure.proficiency,
            "is_chunked": procedure.is_chunked,
            "chunks_count": len(procedure.chunked_steps) if procedure.is_chunked else 0,
            "context_patterns": context_patterns,
            "success_patterns": success_patterns,
            "chunk_patterns": chunk_patterns,
            "refinement_opportunities": len(procedure.refinement_opportunities) if hasattr(procedure, "refinement_opportunities") else 0
        }
    
    async def get_manager_agent(self) -> Agent:
        """Get the procedural memory manager agent"""
        return self._proc_manager_agent
    
    async def get_execution_agent(self) -> Agent:
        """Get the procedure execution agent"""
        return self._proc_execution_agent
    
    async def get_analysis_agent(self) -> Agent:
        """Get the procedure analysis agent"""
        return self._proc_analysis_agent


class EnhancedProceduralMemoryManager(ProceduralMemoryManager):
    """Enhanced version of ProceduralMemoryManager with advanced capabilities"""
    
    def __init__(self, memory_core=None, knowledge_core=None):
        # Initialize base class
        super().__init__(memory_core, knowledge_core)
        
        # Initialize new components
        self.observation_learner = ObservationLearner()
        self.causal_model = CausalModel()
        self.working_memory = WorkingMemoryController()
        self.parameter_optimizer = ParameterOptimizer()
        self.strategy_selector = StrategySelector()
        self.memory_consolidator = ProceduralMemoryConsolidator(memory_core)
        self.transfer_optimizer = TransferLearningOptimizer()
        
        # Initialize execution strategies
        self._init_execution_strategies()
        
        # Add hierarchical procedure storage
        self.hierarchical_procedures = {}  # name -> HierarchicalProcedure
        
        # Add temporal procedure graph storage
        self.temporal_graphs = {}  # id -> TemporalProcedureGraph
        
        # Add procedure graph storage
        self.procedure_graphs = {}  # id -> ProcedureGraph
        
        # Initialization flag
        self.enhanced_initialized = False
    
    async def initialize_enhanced_components(self):
        """Initialize enhanced components and integrations"""
        if self.enhanced_initialized:
            return
        
        # Initialize base components first
        if not self.initialized:
            await self.initialize()
        
        # Set up error recovery patterns
        self._initialize_causal_model()
        
        # Integrate with memory core if available
        if self.memory_core:
            await self.integrate_with_memory_core()
        
        # Integrate with knowledge core if available
        if self.knowledge_core:
            await self.integrate_with_knowledge_core()
        
        # Initialize pre-built templates for common patterns
        self._initialize_common_templates()
        
        self.enhanced_initialized = True
        logger.info("Enhanced procedural memory components initialized")
    
    def _initialize_causal_model(self):
        """Initialize causal model with common error patterns"""
        # Define common error causes for different error types
        self.causal_model.causes = {
            "execution_failure": [
                {
                    "cause": "invalid_parameters",
                    "description": "Invalid parameters provided to function",
                    "probability": 0.6,
                    "context_factors": {}
                },
                {
                    "cause": "missing_precondition",
                    "description": "Required precondition not met",
                    "probability": 0.4,
                    "context_factors": {}
                }
            ],
            "timeout": [
                {
                    "cause": "slow_execution",
                    "description": "Operation taking too long to complete",
                    "probability": 0.5,
                    "context_factors": {}
                },
                {
                    "cause": "resource_contention",
                    "description": "Resources needed are being used by another process",
                    "probability": 0.3,
                    "context_factors": {}
                }
            ],
            "parameter_error": [
                {
                    "cause": "type_mismatch",
                    "description": "Parameter type does not match expected type",
                    "probability": 0.7,
                    "context_factors": {}
                },
                {
                    "cause": "out_of_range",
                    "description": "Parameter value outside of valid range",
                    "probability": 0.5,
                    "context_factors": {}
                }
            ]
        }
        
        # Define common interventions for each cause
        self.causal_model.interventions = {
            "invalid_parameters": [
                {
                    "type": "modify_parameters",
                    "description": "Modify parameters to valid values",
                    "effectiveness": 0.8
                },
                {
                    "type": "check_documentation",
                    "description": "Check documentation for correct parameter format",
                    "effectiveness": 0.6
                }
            ],
            "missing_precondition": [
                {
                    "type": "establish_precondition",
                    "description": "Ensure required precondition is met before execution",
                    "effectiveness": 0.9
                },
                {
                    "type": "alternative_approach",
                    "description": "Use an alternative approach that doesn't require this precondition",
                    "effectiveness": 0.5
                }
            ],
            "slow_execution": [
                {
                    "type": "optimization",
                    "description": "Optimize the operation for faster execution",
                    "effectiveness": 0.7
                },
                {
                    "type": "incremental_execution",
                    "description": "Break operation into smaller steps",
                    "effectiveness": 0.6
                }
            ],
            "resource_contention": [
                {
                    "type": "retry_later",
                    "description": "Retry operation after a delay",
                    "effectiveness": 0.8
                },
                {
                    "type": "release_resources",
                    "description": "Release unused resources before execution",
                    "effectiveness": 0.7
                }
            ],
            "type_mismatch": [
                {
                    "type": "convert_type",
                    "description": "Convert parameter to required type",
                    "effectiveness": 0.9
                }
            ],
            "out_of_range": [
                {
                    "type": "clamp_value",
                    "description": "Clamp parameter value to valid range",
                    "effectiveness": 0.8
                }
            ]
        }
    
    def _initialize_common_templates(self):
        """Initialize common procedure templates"""
        # Define common templates for navigation
        navigation_template = ChunkTemplate(
            id="template_navigation",
            name="Navigation Template",
            description="Template for navigation operations",
            actions=[
                ActionTemplate(
                    action_type="move",
                    intent="navigation",
                    parameters={"destination": "target_location"},
                    domain_mappings={
                        "gaming": {
                            "function": "move_character",
                            "parameters": {"location": "target_location"},
                            "description": "Move character to location"
                        },
                        "ui": {
                            "function": "navigate_to",
                            "parameters": {"page": "target_location"},
                            "description": "Navigate to page"
                        }
                    }
                )
            ],
            domains=["gaming", "ui"],
            success_rate={"gaming": 0.9, "ui": 0.9},
            execution_count={"gaming": 10, "ui": 10}
        )
        
        # Define template for interaction
        interaction_template = ChunkTemplate(
            id="template_interaction",
            name="Interaction Template",
            description="Template for interaction operations",
            actions=[
                ActionTemplate(
                    action_type="select",
                    intent="interaction",
                    parameters={"target": "interaction_target"},
                    domain_mappings={
                        "gaming": {
                            "function": "select_object",
                            "parameters": {"object": "interaction_target"},
                            "description": "Select object in game"
                        },
                        "ui": {
                            "function": "click_element",
                            "parameters": {"element": "interaction_target"},
                            "description": "Click UI element"
                        }
                    }
                ),
                ActionTemplate(
                    action_type="activate",
                    intent="interaction",
                    parameters={"action": "interaction_action"},
                    domain_mappings={
                        "gaming": {
                            "function": "use_object",
                            "parameters": {"action": "interaction_action"},
                            "description": "Use selected object"
                        },
                        "ui": {
                            "function": "submit_form",
                            "parameters": {"action": "interaction_action"},
                            "description": "Submit form with action"
                        }
                    }
                )
            ],
            domains=["gaming", "ui"],
            success_rate={"gaming": 0.85, "ui": 0.9},
            execution_count={"gaming": 8, "ui": 12}
        )
        
        # Add templates to library
        self.chunk_library.add_chunk_template(navigation_template)
        self.chunk_library.add_chunk_template(interaction_template)
    
    def _init_execution_strategies(self):
        """Initialize execution strategies"""
        # Create deliberate execution strategy
        deliberate = DeliberateExecutionStrategy(
            id="deliberate",
            name="Deliberate Execution",
            description="Careful step-by-step execution with validation",
            selection_criteria={
                "proficiency": {"min": 0.0, "max": 0.7},
                "importance": {"min": 0.7, "max": 1.0},
                "risk_level": {"min": 0.7, "max": 1.0}
            }
        )
        
        # Create automatic execution strategy
        automatic = AutomaticExecutionStrategy(
            id="automatic",
            name="Automatic Execution",
            description="Fast execution with minimal monitoring",
            selection_criteria={
                "proficiency": {"min": 0.8, "max": 1.0},
                "importance": {"min": 0.0, "max": 0.6},
                "risk_level": {"min": 0.0, "max": 0.3}
            }
        )
        
        # Create adaptive execution strategy
        adaptive = AdaptiveExecutionStrategy(
            id="adaptive",
            name="Adaptive Execution",
            description="Execution that adapts based on context and feedback",
            selection_criteria={
                "proficiency": {"min": 0.4, "max": 0.9},
                "importance": {"min": 0.3, "max": 0.8},
                "risk_level": {"min": 0.3, "max": 0.7},
                "adaptivity_required": True
            }
        )
        
        # Register strategies
        self.strategy_selector.register_strategy(deliberate)
        self.strategy_selector.register_strategy(automatic)
        self.strategy_selector.register_strategy(adaptive)
    
    # -------------------------------------------------------------------------
    # New Function Tools for Enhanced Manager
    # -------------------------------------------------------------------------
    
    async def learn_from_demonstration(
        self, 
        observation_sequence: List[Dict[str, Any]], 
        domain: str,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Learn a procedure from a sequence of observed actions
        
        Args:
            observation_sequence: Sequence of observed actions with state
            domain: Domain for the new procedure
            name: Optional name for the new procedure
            
        Returns:
            Information about the learned procedure
        """
        # Learn from observations
        procedure_data = await self.observation_learner.learn_from_demonstration(
            observation_sequence=observation_sequence,
            domain=domain
        )
        
        # Use provided name if available
        if name:
            procedure_data["name"] = name
        
        # Create the procedure
        ctx = RunContextWrapper(context=self)
        from .function_tools import add_procedure
        procedure_result = await add_procedure(
            ctx,
            name=procedure_data["name"],
            steps=procedure_data["steps"],
            description=procedure_data["description"],
            domain=domain
        )
        
        # Add confidence information
        procedure_result["confidence"] = procedure_data["confidence"]
        procedure_result["learned_from_observations"] = True
        procedure_result["observation_count"] = procedure_data["observation_count"]
        
        return procedure_result
    
    async def create_hierarchical_procedure(
        self,
        name: str,
        description: str,
        domain: str,
        steps: List[Dict[str, Any]],
        goal_state: Dict[str, Any] = None,
        preconditions: Dict[str, Any] = None,
        postconditions: Dict[str, Any] = None,
        parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a hierarchical procedure
        
        Args:
            name: Name of the procedure
            description: Description of what the procedure does
            domain: Domain for the procedure
            steps: List of step definitions
            goal_state: Optional goal state for the procedure
            preconditions: Optional preconditions
            postconditions: Optional postconditions
            parent_id: Optional parent procedure ID
            
        Returns:
            Information about the created procedure
        """
        # Generate ID
        proc_id = f"hierproc_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        
        # Create the hierarchical procedure
        procedure = HierarchicalProcedure(
            id=proc_id,
            name=name,
            description=description,
            domain=domain,
            steps=steps,
            goal_state=goal_state or {},
            preconditions=preconditions or {},
            postconditions=postconditions or {},
            parent_id=parent_id
        )
        
        # Store the procedure
        self.hierarchical_procedures[name] = procedure
        
        # Create standard procedure as well
        ctx = RunContextWrapper(context=self)
        from .function_tools import add_procedure
        standard_proc = await add_procedure(
            ctx,
            name=name,
            steps=steps,
            description=description,
            domain=domain
        )
        
        # If has parent, update parent's children list
        if parent_id:
            for parent in self.hierarchical_procedures.values():
                if parent.id == parent_id:
                    parent.add_child(proc_id)
                    break
        
        # Return information
        return {
            "id": proc_id,
            "name": name,
            "domain": domain,
            "steps_count": len(steps),
            "standard_procedure_id": standard_proc["procedure_id"],
            "hierarchical": True,
            "parent_id": parent_id
        }
    
    async def execute_hierarchical_procedure(
        self,
        name: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a hierarchical procedure
        
        Args:
            name: Name of the procedure
            context: Execution context
            
        Returns:
            Execution results
        """
        if name not in self.hierarchical_procedures:
            return {"error": f"Hierarchical procedure '{name}' not found"}
        
        procedure = self.hierarchical_procedures[name]
        
        # Create trace for execution
        with trace(workflow_name="execute_hierarchical_procedure"):
            # Check preconditions
            if not procedure.meets_preconditions(context or {}):
                return {
                    "success": False,
                    "error": "Preconditions not met",
                    "procedure_name": name
                }
            
            # Initialize context if needed
            execution_context = context.copy() if context else {}
            
            # Set procedure context
            execution_context["current_procedure"] = name
            execution_context["hierarchical"] = True
            
            # Update working memory
            self.working_memory.update(execution_context, procedure)
            
            # Select execution strategy
            strategy = self.strategy_selector.select_strategy(execution_context, procedure)
            
            # Execute with selected strategy
            start_time = datetime.datetime.now()
            execution_result = await strategy.execute(procedure, execution_context)
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update procedure statistics
            self._update_hierarchical_stats(procedure, execution_time, execution_result["success"])
            
            # Verify goal state was achieved
            goal_achieved = True
            if procedure.goal_state:
                for key, value in procedure.goal_state.items():
                    if key not in execution_context or execution_context[key] != value:
                        goal_achieved = False
                        break
            
            # Add information to result
            execution_result["procedure_name"] = name
            execution_result["hierarchical"] = True
            execution_result["goal_achieved"] = goal_achieved
            execution_result["strategy_id"] = strategy.id
            execution_result["working_memory"] = self.working_memory.get_attention_focus()
            
            return execution_result
    
    def _update_hierarchical_stats(self, procedure: HierarchicalProcedure, execution_time: float, success: bool):
        """Update statistics for a hierarchical procedure"""
        # Update count
        procedure.execution_count += 1
        if success:
            procedure.successful_executions += 1
        
        # Update average time
        if procedure.execution_count == 1:
            procedure.average_execution_time = execution_time
        else:
            procedure.average_execution_time = (
                (procedure.average_execution_time * (procedure.execution_count - 1) + execution_time) / 
                procedure.execution_count
            )
        
        # Update proficiency based on multiple factors
        count_factor = min(procedure.execution_count / 50, 1.0)
        success_rate = procedure.successful_executions / max(1, procedure.execution_count)
        
        # Calculate time factor (lower times = higher proficiency)
        if procedure.execution_count < 2:
            time_factor = 0.5  # Default for first execution
        else:
            # Normalize time - lower is better
            time_factor = max(0.0, 1.0 - (procedure.average_execution_time / 10.0))
            time_factor = min(1.0, time_factor)
        
        # Combine factors with weights
        procedure.proficiency = (count_factor * 0.3) + (success_rate * 0.5) + (time_factor * 0.2)
        
        # Update timestamps
        procedure.last_execution = datetime.datetime.now().isoformat()
        procedure.last_updated = datetime.datetime.now().isoformat()
    
    async def optimize_procedure_parameters(
        self,
        procedure_name: str,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a procedure using Bayesian optimization
        
        Args:
            procedure_name: Name of the procedure to optimize
            iterations: Number of optimization iterations
            
        Returns:
            Optimization results
        """
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Define objective function (success rate and execution time)
        async def objective_function(test_procedure: Procedure) -> float:
            # Create simulated context
            test_context = {"optimization_run": True}
            
            # Execute procedure
            ctx = RunContextWrapper(context=self)
            from .function_tools import execute_procedure
            result = await execute_procedure(ctx, test_procedure.name, test_context)
            
            # Calculate objective score (combination of success and speed)
            success_score = 1.0 if result["success"] else 0.0
            time_score = max(0.0, 1.0 - (result["execution_time"] / 10.0))  # Lower time is better
            
            # Combined score (success is more important)
            return success_score * 0.7 + time_score * 0.3
        
        # Run optimization
        optimization_result = await self.parameter_optimizer.optimize_parameters(
            procedure=procedure,
            objective_function=objective_function,
            iterations=iterations
        )
        
        # Apply best parameters if optimization succeeded
        if optimization_result["status"] == "success" and optimization_result["best_parameters"]:
            # Create modified procedure
            modified_procedure = procedure.model_copy(deep=True)
            self.parameter_optimizer._apply_parameters(modified_procedure, optimization_result["best_parameters"])
            
            # Update original procedure
            for step in procedure.steps:
                for modified_step in modified_procedure.steps:
                    if step["id"] == modified_step["id"]:
                        step["parameters"] = modified_step["parameters"]
            
            # Update timestamp
            procedure.last_updated = datetime.datetime.now().isoformat()
            
            # Add update information
            optimization_result["procedure_updated"] = True
        else:
            optimization_result["procedure_updated"] = False
        
        return optimization_result
    
    async def handle_execution_error(
        self,
        error: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle an execution error using the causal model
        
        Args:
            error: Error details
            context: Execution context
            
        Returns:
            Recovery suggestions
        """
        # Identify likely causes
        likely_causes = self.causal_model.identify_likely_causes(error)
        
        # Get recovery suggestions
        interventions = self.causal_model.suggest_interventions(likely_causes)
        
        # Return results
        return {
            "likely_causes": likely_causes,
            "interventions": interventions,
            "context": context
        }
    
    async def create_temporal_procedure(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        temporal_constraints: List[Dict[str, Any]],
        domain: str,
        description: str = None
    ) -> Dict[str, Any]:
        """
        Create a procedure with temporal constraints
        
        Args:
            name: Name of the procedure
            steps: List of step definitions
            temporal_constraints: List of temporal constraints between steps
            domain: Domain for the procedure
            description: Optional description
            
        Returns:
            Information about the created procedure
        """
        # Create normal procedure first
        ctx = RunContextWrapper(context=self)
        from .function_tools import add_procedure
        normal_proc = await add_procedure(
            ctx,
            name=name,
            steps=steps,
            description=description or f"Temporal procedure: {name}",
            domain=domain
        )
        
        # Create temporal graph
        procedure = self.procedures[name]
        graph = TemporalProcedureGraph.from_procedure(procedure)
        
        # Add temporal constraints
        for constraint in temporal_constraints:
            from_id = constraint.get("from_step")
            to_id = constraint.get("to_step")
            constraint_type = constraint.get("type")
            
            if from_id and to_id and constraint_type:
                # Find nodes
                from_node_id = f"node_{from_id}"
                to_node_id = f"node_{to_id}"
                
                if from_node_id in graph.nodes and to_node_id in graph.nodes:
                    # Add constraint based on type
                    if constraint_type == "min_delay":
                        # Minimum delay between steps
                        min_delay = constraint.get("delay", 0)
                        
                        # Add to edge
                        for i, edge in enumerate(graph.edges):
                            if edge[0] == from_node_id and edge[1] == to_node_id:
                                if not edge[2]:
                                    edge[2] = {}
                                edge[2]["min_duration"] = min_delay
                                break
                    elif constraint_type == "must_follow":
                        # Must follow constraint
                        if to_node_id in graph.nodes:
                            graph.nodes[to_node_id].add_constraint({
                                "type": "after",
                                "action": graph.nodes[from_node_id].action["function"]
                            })
        
        # Validate constraints
        if not graph.validate_temporal_constraints():
            return {
                "error": "Invalid temporal constraints - contains negative cycles",
                "procedure_id": normal_proc["procedure_id"]
            }
        
        # Store the temporal graph
        self.temporal_graphs[graph.id] = graph
        
        # Link procedure to graph
        procedure.temporal_graph_id = graph.id
        
        return {
            "procedure_id": normal_proc["procedure_id"],
            "temporal_graph_id": graph.id,
            "name": name,
            "domain": domain,
            "steps_count": len(steps),
            "constraints_count": len(temporal_constraints),
            "is_temporal": True
        }
    
    async def execute_temporal_procedure(
        self,
        name: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure with temporal constraints
        
        Args:
            name: Name of the procedure
            context: Execution context
            
        Returns:
            Execution results
        """
        if name not in self.procedures:
            return {"error": f"Procedure '{name}' not found"}
        
        procedure = self.procedures[name]
        
        # Check if procedure has temporal graph
        if not hasattr(procedure, "temporal_graph_id") or procedure.temporal_graph_id not in self.temporal_graphs:
            # Fall back to normal execution
            ctx = RunContextWrapper(context=self)
            from .function_tools import execute_procedure
            return await execute_procedure(ctx, name, context)
        
        # Get temporal graph
        graph = self.temporal_graphs[procedure.temporal_graph_id]
        
        # Create execution trace
        with trace(workflow_name="execute_temporal_procedure"):
            start_time = datetime.datetime.now()
            results = []
            success = True
            
            # Initialize execution context
            execution_context = context.copy() if context else {}
            execution_context["temporal_execution"] = True
            execution_context["execution_history"] = []
            
            # Execute in temporal order
            while True:
                # Get next executable nodes
                next_nodes = graph.get_next_executable_nodes(execution_context["execution_history"])
                
                if not next_nodes:
                    # Check if we've executed all exit nodes
                    executed_nodes = set(hist["node_id"] for hist in execution_context["execution_history"])
                    if all(exit_node in executed_nodes for exit_node in graph.exit_points):
                        # Successfully completed all nodes
                        break
                    else:
                        # No executable nodes but haven't reached all exits
                        success = False
                        break
                
                # Execute first valid node
                node_id = next_nodes[0]
                node = graph.nodes[node_id]
                
                # Extract step information
                step = {
                    "id": node.action.get("step_id", node_id),
                    "function": node.action["function"],
                    "parameters": node.action.get("parameters", {}),
                    "description": node.action.get("description", f"Step {node_id}")
                }
                
                # Execute the step
                step_result = await self.execute_step(step, execution_context)
                results.append(step_result)
                
                # Update execution history
                execution_context["execution_history"].append({
                    "node_id": node_id,
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
            
            # Update procedure statistics
            self.update_procedure_stats(procedure, execution_time, success)
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "is_temporal": True,
                "nodes_executed": len(execution_context["execution_history"])
            }
    
    async def create_procedure_graph(
        self,
        procedure_name: str
    ) -> Dict[str, Any]:
        """
        Create a graph representation of a procedure for flexible execution
        
        Args:
            procedure_name: Name of the existing procedure
            
        Returns:
            Information about the created graph
        """
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Create graph representation
        graph = ProcedureGraph.from_procedure(procedure)
        
        # Generate graph ID
        graph_id = f"graph_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        
        # Store graph
        self.procedure_graphs[graph_id] = graph
        
        # Link procedure to graph
        procedure.graph_id = graph_id
        
        return {
            "graph_id": graph_id,
            "procedure_name": procedure_name,
            "nodes_count": len(graph.nodes),
            "edges_count": len(graph.edges),
            "entry_points": len(graph.entry_points),
            "exit_points": len(graph.exit_points)
        }
    
    async def execute_graph_procedure(
        self,
        procedure_name: str,
        context: Dict[str, Any] = None,
        goal: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure using its graph representation
        
        Args:
            procedure_name: Name of the procedure
            context: Execution context
            goal: Optional goal state to achieve
            
        Returns:
            Execution results
        """
        if procedure_name not in self.procedures:
            return {"error": f"Procedure '{procedure_name}' not found"}
        
        procedure = self.procedures[procedure_name]
        
        # Check if procedure has graph
        if not hasattr(procedure, "graph_id") or procedure.graph_id not in self.procedure_graphs:
            # Fall back to normal execution
            ctx = RunContextWrapper(context=self)
            from .function_tools import execute_procedure
            return await execute_procedure(ctx, procedure_name, context)
        
        # Get graph
        graph = self.procedure_graphs[procedure.graph_id]
        
        # Create execution trace
        with trace(workflow_name="execute_graph_procedure"):
            start_time = datetime.datetime.now()
            results = []
            success = True
            
            # Initialize execution context
            execution_context = context.copy() if context else {}
            execution_context["graph_execution"] = True
            
            # Find execution path to goal
            path = graph.find_execution_path(execution_context, goal or {})
            
            if not path:
                return {
                    "success": False,
                    "error": "Could not find a valid execution path",
                    "procedure_name": procedure_name
                }
            
            # Execute nodes in path
            for node_id in path:
                # Get node data
                node_data = graph.nodes[node_id]
                
                # Create step from node data
                step = {
                    "id": node_data.get("step_id", node_id),
                    "function": node_data["function"],
                    "parameters": node_data.get("parameters", {}),
                    "description": node_data.get("description", f"Step {node_id}")
                }
                
                # Execute the step
                step_result = await self.execute_step(step, execution_context)
                results.append(step_result)
                
                # Update execution context
                execution_context[f"step_{step['id']}_result"] = step_result
                
                # Check for failure
                if not step_result["success"]:
                    success = False
                    break
            
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update procedure statistics
            self.update_procedure_stats(procedure, execution_time, success)
            
            # Check if goal achieved
            goal_achieved = True
            if goal:
                for key, value in goal.items():
                    if key not in execution_context or execution_context[key] != value:
                        goal_achieved = False
                        break
            
            return {
                "success": success,
                "results": results,
                "execution_time": execution_time,
                "is_graph": True,
                "path_length": len(path),
                "goal_achieved": goal_achieved
            }
    
    async def consolidate_procedural_memory(self) -> Dict[str, Any]:
        """
        Consolidate procedural memory to optimize storage and execution
        
        Returns:
            Consolidation results
        """
        # Run memory consolidation
        return await self.memory_consolidator.consolidate_procedural_memory()
    
    async def optimize_procedure_transfer(
        self,
        source_procedure: str,
        target_domain: str
    ) -> Dict[str, Any]:
        """
        Optimize transfer of a procedure to a new domain
        
        Args:
            source_procedure: Name of the source procedure
            target_domain: Target domain
            
        Returns:
            Transfer optimization plan
        """
        if source_procedure not in self.procedures:
            return {"error": f"Procedure '{source_procedure}' not found"}
        
        # Get source procedure
        procedure = self.procedures[source_procedure]
        
        # Optimize transfer
        transfer_plan = await self.transfer_optimizer.optimize_transfer(
            source_procedure=procedure,
            target_domain=target_domain
        )
        
        return transfer_plan
        
    async def execute_transfer_plan(
        self,
        transfer_plan: Dict[str, Any],
        target_name: str
    ) -> Dict[str, Any]:
        """
        Execute a transfer plan to create a new procedure
        
        Args:
            transfer_plan: Transfer plan from optimize_procedure_transfer
            target_name: Name for the new procedure
            
        Returns:
            Results of transfer execution
        """
        source_domain = transfer_plan.get("source_domain")
        target_domain = transfer_plan.get("target_domain")
        mappings = transfer_plan.get("mappings", [])
        
        if not source_domain or not target_domain or not mappings:
            return {
                "success": False,
                "error": "Invalid transfer plan"
            }
        
        # Find source procedure
        source_procedure = None
        for name, proc in self.procedures.items():
            if proc.domain == source_domain:
                source_procedure = proc
                break
        
        if not source_procedure:
            return {
                "success": False,
                "error": f"Could not find procedure in domain {source_domain}"
            }
        
        # Create new steps based on mappings
        new_steps = []
        
        for i, mapping in enumerate(mappings):
            source_func = mapping.get("source_function")
            target_func = mapping.get("target_function")
            target_params = mapping.get("target_parameters", {})
            
            if not source_func or not target_func:
                continue
            
            # Find corresponding step in source procedure
            source_step = None
            for step in source_procedure.steps:
                if step["function"] == source_func:
                    source_step = step
                    break
            
            if not source_step:
                continue
            
            # Create new step
            new_step = {
                "id": f"step_{i+1}",
                "function": target_func,
                "parameters": target_params,
                "description": f"Transferred from {source_step.get('description', source_func)}"
            }
            
            new_steps.append(new_step)
        
        if not new_steps:
            return {
                "success": False,
                "error": "No steps could be transferred"
            }
        
        # Create new procedure
        ctx = RunContextWrapper(context=self)
        from .function_tools import add_procedure
        new_procedure = await add_procedure(
            ctx,
            name=target_name,
            steps=new_steps,
            description=f"Transferred from {source_procedure.name} ({source_domain} to {target_domain})",
            domain=target_domain
        )
        
        # Update transfer history
        self.transfer_optimizer.update_from_transfer_result(
            source_domain=source_domain,
            target_domain=target_domain,
            success_rate=0.8,  # Initial estimate
            mappings=mappings
        )
        
        return {
            "success": True,
            "procedure_id": new_procedure["procedure_id"],
            "name": target_name,
            "domain": target_domain,
            "steps_count": len(new_steps),
            "transfer_strategy": transfer_plan.get("transfer_strategy")
        }
    
    # -------------------------------------------------------------------------
    # Integration with Memory Core
    # -------------------------------------------------------------------------
    
    async def integrate_with_memory_core(self) -> bool:
        """Integrate procedural memory with main memory core"""
        if not self.memory_core:
            logger.warning("No memory core available for integration")
            return False
        
        try:
            # Register handlers for procedural memory operations
            self.memory_core.register_procedural_handler(self)
            
            # Set up memory event listeners
            self._setup_memory_listeners()
            
            logger.info("Procedural memory integrated with memory core")
            return True
        except Exception as e:
            logger.error(f"Error integrating with memory core: {e}")
            return False
    
    def _setup_memory_listeners(self):
        """Set up listeners for memory core events"""
        if not self.memory_core:
            return
        
        # Listen for new procedural observations
        self.memory_core.add_event_listener(
            "new_procedural_observation",
            self._handle_procedural_observation
        )
        
        # Listen for memory decay events
        self.memory_core.add_event_listener(
            "memory_decay",
            self._handle_memory_decay
        )
    
    async def _handle_procedural_observation(self, data: Dict[str, Any]):
        """Handle new procedural observation from memory core"""
        # Check if observation has steps
        if "steps" not in data:
            return
        
        # Create sequence of observations
        observation_sequence = [{
            "action": step.get("action"),
            "state": step.get("state", {}),
            "timestamp": step.get("timestamp", datetime.datetime.now().isoformat())
        } for step in data["steps"]]
        
        # Learn from demonstration
        if len(observation_sequence) >= 3:  # Need at least 3 steps
            await self.learn_from_demonstration(
                observation_sequence=observation_sequence,
                domain=data.get("domain", "general"),
                name=data.get("name")
            )
    
    async def _handle_memory_decay(self, data: Dict[str, Any]):
        """Handle memory decay events"""
        # Check if affecting procedural memory
        if data.get("memory_type") != "procedural":
            return
        
        # Run consolidation to optimize storage
        await self.consolidate_procedural_memory()
    
    # -------------------------------------------------------------------------
    # Integration with Knowledge Core
    # -------------------------------------------------------------------------
    
    async def integrate_with_knowledge_core(self) -> bool:
        """Integrate procedural memory with knowledge core"""
        if not self.knowledge_core:
            logger.warning("No knowledge core available for integration")
            return False
        
        try:
            # Register handlers for knowledge queries
            self.knowledge_core.register_procedural_handler(self)
            
            # Set up knowledge listeners
            self._setup_knowledge_listeners()
            
            logger.info("Procedural memory integrated with knowledge core")
            return True
        except Exception as e:
            logger.error(f"Error integrating with knowledge core: {e}")
            return False
    
    def _setup_knowledge_listeners(self):
        """Set up listeners for knowledge core events"""
        if not self.knowledge_core:
            return
        
        # Listen for new domain knowledge
        self.knowledge_core.add_event_listener(
            "new_domain_knowledge",
            self._handle_domain_knowledge
        )
    
    async def _handle_domain_knowledge(self, data: Dict[str, Any]):
        """Handle new domain knowledge from knowledge core"""
        domain = data.get("domain")
        if not domain:
            return
        
        # Update domain similarities in transfer optimizer
        if hasattr(self, "transfer_optimizer"):
            # Create or update domain embedding
            await self.transfer_optimizer._get_domain_embedding(domain)
    
    async def share_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """Share procedural knowledge about a domain with knowledge core"""
        if not self.knowledge_core:
            return {"error": "No knowledge core available"}
        
        # Find procedures in this domain
        domain_procedures = [p for p in self.procedures.values() if p.domain == domain]
        
        if not domain_procedures:
            return {"error": f"No procedures found for domain {domain}"}
        
        # Extract knowledge from procedures
        knowledge_items = []
        
        for procedure in domain_procedures:
            # Create knowledge about procedure purpose
            knowledge_items.append({
                "content": f"In the {domain} domain, '{procedure.name}' is a procedure for {procedure.description}",
                "confidence": procedure.proficiency,
                "source": "procedural_memory"
            })
            
            # Create knowledge about specific steps
            for i, step in enumerate(procedure.steps):
                knowledge_items.append({
                    "content": f"In the {domain} domain, the '{step['function']}' function is used for {step.get('description', f'step {i+1}')}",
                    "confidence": procedure.proficiency * 0.9,  # Slightly lower confidence
                    "source": "procedural_memory"
                })
        
        # Add knowledge to knowledge core
        added_count = 0
        for item in knowledge_items:
            try:
                await self.knowledge_core.add_knowledge_item(
                    domain=domain,
                    content=item["content"],
                    source=item["source"],
                    confidence=item["confidence"]
                )
                added_count += 1
            except Exception as e:
                logger.error(f"Error adding knowledge item: {e}")
        
        return {
            "domain": domain,
            "knowledge_items_added": added_count,
            "procedures_analyzed": len(domain_procedures)
        }


# Demonstration functions

async def demonstrate_cross_game_transfer():
    """Demonstrate procedural memory with cross-game transfer"""
    
    # Create an enhanced procedural memory manager
    manager = EnhancedProceduralMemoryManager()
    
    # Define step functions for our Dead by Daylight example
    async def press_button(button: str, context: Dict[str, Any] = None):
        print(f"Pressing {button}")
        # Update context
        if context and button == "L1":
            context["sprinting"] = True
        return {"button": button, "pressed": True}
        
    async def approach_object(object_type: str, context: Dict[str, Any] = None):
        print(f"Approaching {object_type}")
        # Update context
        if context:
            context[f"near_{object_type}"] = True
        return {"object": object_type, "approached": True}
        
    async def check_surroundings(context: Dict[str, Any] = None):
        print(f"Checking surroundings")
        return {"surroundings_checked": True, "clear": True}
        
    async def vault_window(context: Dict[str, Any] = None):
        print(f"Vaulting through window")
        # Use context to see if we're sprinting
        sprinting = context.get("sprinting", False) if context else False
        return {"vaulted": True, "fast_vault": sprinting}
        
    async def work_on_generator(context: Dict[str, Any] = None):
        print(f"Working on generator")
        # Simulate a skill check
        skill_check_success = random.random() > 0.3  # 70% success rate
        return {"working_on_gen": True, "skill_check": skill_check_success}
    
    # Register functions
    manager.register_function("press_button", press_button)
    manager.register_function("approach_object", approach_object)
    manager.register_function("check_surroundings", check_surroundings)
    manager.register_function("vault_window", vault_window)
    manager.register_function("work_on_generator", work_on_generator)
    
    # Define steps for DBD window-generator procedure
    window_gen_steps = [
        {
            "id": "start_sprint",
            "description": "Start sprinting",
            "function": "press_button",
            "parameters": {"button": "L1"}
        },
        {
            "id": "approach_window",
            "description": "Approach the window",
            "function": "approach_object",
            "parameters": {"object_type": "window"}
        },
        {
            "id": "vault",
            "description": "Vault through the window",
            "function": "vault_window",
            "parameters": {}
        },
        {
            "id": "resume_sprint",
            "description": "Resume sprinting",
            "function": "press_button", 
            "parameters": {"button": "L1"}
        },
        {
            "id": "approach_gen",
            "description": "Approach the generator",
            "function": "approach_object",
            "parameters": {"object_type": "generator"}
        },
        {
            "id": "repair_gen",
            "description": "Work on the generator",
            "function": "work_on_generator",
            "parameters": {}
        }
    ]
    
    # Create RunContextWrapper for agent tools
    ctx = RunContextWrapper(context=manager)
    
    # Learn the procedure
    print("\nLearning procedure:")
    from .function_tools import add_procedure, execute_procedure, identify_chunking_opportunities, apply_chunking, generalize_chunk_from_steps, transfer_procedure, get_procedure_proficiency, find_similar_procedures
    
    dbd_result = await add_procedure(
        ctx,
        name="window_to_generator",
        steps=window_gen_steps,
        description="Navigate through a window and start working on a generator",
        domain="dbd"  # Dead by Daylight
    )
    
    print(f"Created procedure: {dbd_result}")
    
    # Execute procedure multiple times
    print("\nPracticing procedure...")
    for i in range(10):
        print(f"\nExecution {i+1}:")
        context = {"sprinting": False}
        result = await execute_procedure(ctx, "window_to_generator", context)
        
        dbd_procedure = manager.procedures["window_to_generator"]
        print(f"Success: {result['success']}, " 
              f"Time: {result['execution_time']:.4f}s, "
              f"Proficiency: {dbd_procedure.proficiency:.2f}")
    
    # Check for chunking opportunities
    print("\nIdentifying chunking opportunities:")
    chunking_result = await identify_chunking_opportunities(ctx, "window_to_generator")
    print(f"Chunking analysis: {chunking_result}")
    
    # Apply chunking
    if chunking_result.get("can_chunk", False):
        print("\nApplying chunking:")
        chunk_result = await apply_chunking(ctx, "window_to_generator")
        print(f"Chunking result: {chunk_result}")
    
    # Create a template from the main chunk
    if manager.procedures["window_to_generator"].is_chunked:
        print("\nGeneralizing chunk template:")
        template_result = await generalize_chunk_from_steps(
            ctx,
            chunk_name="window_to_generator_combo",
            procedure_name="window_to_generator",
            step_ids=["start_sprint", "approach_window", "vault"],
            domain="dbd"
        )
        print(f"Template created: {template_result}")
    
    # Transfer to another domain
    print("\nTransferring procedure to new domain:")
    transfer_result = await transfer_procedure(
        ctx,
        source_name="window_to_generator",
        target_name="xbox_window_to_generator",
        target_domain="xbox"
    )
    print(f"Transfer result: {transfer_result}")
    
    # Execute the transferred procedure
    print("\nExecuting transferred procedure:")
    xbox_result = await execute_procedure(ctx, "xbox_window_to_generator")
    print(f"Xbox execution result: {xbox_result}")
    
    # Get procedure statistics
    print("\nProcedure statistics:")
    stats = await get_procedure_proficiency(ctx, "window_to_generator")
    print(f"Original procedure: {stats}")
    
    # Find similar procedures
    print("\nFinding similar procedures:")
    similar = await find_similar_procedures(ctx, "window_to_generator")
    print(f"Similar procedures: {similar}")
    
    return manager

async def demonstrate_procedural_memory():
    """Demonstrate the procedural memory system with Agents SDK"""
    
    # Create the procedural memory manager
    manager = EnhancedProceduralMemoryManager()
    
    # Register example functions
    async def perform_action(action_name: str, action_target: str, context: Dict[str, Any] = None):
        print(f"Performing action: {action_name} on {action_target}")
        return {"action": action_name, "target": action_target, "performed": True}
    
    async def check_condition(condition_name: str, context: Dict[str, Any] = None):
        print(f"Checking condition: {condition_name}")
        # Simulate condition check
        result = True  # In real usage, this would evaluate something
        return {"condition": condition_name, "result": result}
        
    async def select_item(item_id: str, control: str, context: Dict[str, Any] = None):
        print(f"Selecting item {item_id} using {control}")
        return {"item": item_id, "selected": True}
        
    async def navigate_to(location: str, method: str, context: Dict[str, Any] = None):
        print(f"Navigating to {location} using {method}")
        return {"location": location, "arrived": True}
    
    # Register functions
    manager.register_function("perform_action", perform_action)
    manager.register_function("check_condition", check_condition)
    manager.register_function("select_item", select_item)
    manager.register_function("navigate_to", navigate_to)
    
    # Create RunContextWrapper for agent tools
    ctx = RunContextWrapper(context=manager)
    
    # Create a procedure in the "touch_interface" domain
    touch_procedure_steps = [
        {
            "id": "step_1",
            "description": "Navigate to menu",
            "function": "navigate_to",
            "parameters": {"location": "main_menu", "method": "swipe"}
        },
        {
            "id": "step_2",
            "description": "Select item from menu",
            "function": "select_item",
            "parameters": {"item_id": "settings", "control": "tap"}
        },
        {
            "id": "step_3",
            "description": "Adjust settings",
            "function": "perform_action",
            "parameters": {"action_name": "adjust", "action_target": "brightness"}
        }
    ]
    
    print("Creating touch interface procedure...")
    
    from .function_tools import add_procedure, apply_chunking, generalize_chunk_from_steps, transfer_with_chunking, execute_procedure, find_similar_procedures, list_procedures, get_transfer_statistics
    
    touch_result = await add_procedure(
        ctx,
        name="touch_settings_procedure",
        steps=touch_procedure_steps,
        description="Adjust settings using touch interface",
        domain="touch_interface"
    )
    
    print(f"Created procedure: {touch_result}")
    
    # Execute the procedure multiple times to develop proficiency
    print("\nPracticing touch procedure...")
    for i in range(5):
        await execute_procedure(ctx, "touch_settings_procedure")
    
    # Apply chunking to identify patterns
    print("\nApplying chunking to the procedure...")
    
    chunking_result = await apply_chunking(ctx, "touch_settings_procedure")
    print(f"Chunking result: {chunking_result}")
    
    # Generalize a chunk for navigation and selection
    print("\nGeneralizing a chunk for navigation and selection...")
    
    chunk_result = await generalize_chunk_from_steps(
        ctx,
        chunk_name="navigate_and_select",
        procedure_name="touch_settings_procedure",
        step_ids=["step_1", "step_2"]
    )
    
    print(f"Chunk generalization result: {chunk_result}")
    
    # Transfer the procedure to mouse_interface domain
    print("\nTransferring procedure to mouse interface domain...")
    
    transfer_result = await transfer_with_chunking(
        ctx,
        source_name="touch_settings_procedure",
        target_name="mouse_settings_procedure",
        target_domain="mouse_interface"
    )
    
    print(f"Transfer result: {transfer_result}")
    
    # Execute the transferred procedure
    print("\nExecuting transferred procedure...")
    
    transfer_execution = await execute_procedure(
        ctx,
        name="mouse_settings_procedure"
    )
    
    print(f"Transferred procedure execution: {transfer_execution}")
    
    # Compare the two procedures
    print("\nFinding similar procedures to our touch procedure...")
    
    similar = await find_similar_procedures(ctx, "touch_settings_procedure")
    
    print(f"Similar procedures: {similar}")
    
    # Get transfer statistics
    print("\nGetting transfer statistics...")
    
    stats = await get_transfer_statistics(ctx)
    
    print(f"Transfer statistics: {stats}")
    
    # List all procedures
    print("\nListing all procedures:")
    
    procedures = await list_procedures(ctx)
    
    for proc in procedures:
        print(f"- {proc['name']} ({proc['domain']}) - Proficiency: {proc['proficiency']:.2f}")
    
    return manager
