# nyx/core/procedural_memory/manager.py

# Standard library
import asyncio
import datetime
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from typing_extensions import TypedDict
from collections import Counter, defaultdict

# OpenAI Agents SDK imports
from agents import Agent, Runner, trace, function_tool, handoff, RunContextWrapper, ModelSettings
from agents.exceptions import ModelBehaviorError, UserError
from agents.tracing import agent_span, function_span, generation_span, trace as agents_trace

# Core procedural memory components
from nyx.core.procedural_memory.models import (
    ActionTemplate, ChunkTemplate, ContextPattern, ChunkPrediction,
    ControlMapping, ProcedureTransferRecord, Procedure, StepResult,
    ProcedureStats, TransferStats, HierarchicalProcedure, CausalModel,
    WorkingMemoryController, ParameterOptimizer, TransferLearningOptimizer
)
from nyx.core.procedural_memory.chunk_selection import ContextAwareChunkSelector
from nyx.core.procedural_memory.generalization import ProceduralChunkLibrary
from nyx.core.procedural_memory.learning import ObservationLearner, ProceduralMemoryConsolidator
from nyx.core.procedural_memory.execution import (
    ExecutionStrategy, DeliberateExecutionStrategy, 
    AutomaticExecutionStrategy, AdaptiveExecutionStrategy,
    StrategySelector
)
from nyx.core.procedural_memory.temporal import TemporalNode, TemporalProcedureGraph, ProcedureGraph


# Set up logging
logger = logging.getLogger(__name__)

class StepParameters(TypedDict, total=False):
    """Flexible parameters for procedure steps"""
    button: Optional[str]
    control: Optional[str]
    input_method: Optional[str]
    key: Optional[str]
    # Add other common parameters as needed

class ProcedureStep(TypedDict):
    """Schema for a procedure step"""
    id: str
    description: str
    function: str
    parameters: StepParameters

class ExecutionContext(TypedDict, total=False):
    """Schema for execution context"""
    timestamp: str
    conscious_execution: bool
    result: bool
    execution_time: float
    action_history: List[Dict[str, str]]  # Keep this simple
    current_chunk: str
    current_procedure: str
    hierarchical: bool
    temporal_execution: bool
    graph_execution: bool
    optimization_run: bool
    execution_history: List[Dict[str, Any]]

class ObservationStep(TypedDict):
    """Schema for observation sequence step"""
    action: str
    state: Dict[str, Any]
    timestamp: str

class ErrorInfo(TypedDict):
    """Schema for error information"""
    error_type: str
    message: str
    context: Dict[str, Any]

class MappingInfo(TypedDict):
    """Schema for transfer mapping"""
    source_function: str
    target_function: str
    target_parameters: StepParameters

class TransferPlan(TypedDict):
    """Schema for transfer plan"""
    source_domain: str
    target_domain: str
    mappings: List[MappingInfo]
    transfer_strategy: str

class GoalState(TypedDict, total=False):
    """Schema for goal state - flexible"""
    pass

class TemporalConstraint(TypedDict):
    """Schema for temporal constraints"""
    from_step: str
    to_step: str
    type: str
    delay: Optional[float]

class PreconditionSet(TypedDict, total=False):
    """Schema for preconditions - flexible"""
    pass

class PostconditionSet(TypedDict, total=False):
    """Schema for postconditions - flexible"""
    pass

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
        
        # Initialization flag
        self.initialized = False
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize the agents for procedural memory management"""
        # Main manager agent
        self._proc_manager_agent = Agent(
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
        
        # Execution agent
        self._proc_execution_agent = Agent(
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
        
        # Analysis agent
        self._proc_analysis_agent = Agent(
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
    
    async def initialize(self):
        """Initialize the procedural memory manager"""
        if self.initialized:
            return
        
        # Register function tools for agents
        self._register_function_tools()
        
        # Set up default functions
        self._register_default_functions()
        
        # Set up agent handoffs
        self._setup_agent_handoffs()
        
        # Initialize control mappings
        self._initialize_control_mappings()
        
        self.initialized = True
        logger.info("Procedural memory manager initialized")
    
    def _register_function_tools(self):
        """Register function tools for agents"""
        # Register common function tools for all agents
        common_tools = [
            function_tool(self.list_procedures),
            function_tool(self.get_procedure),
            function_tool(self.get_procedure_proficiency),
        ]
        
        # Manager agent specific tools
        manager_tools = common_tools + [
            function_tool(self.add_procedure),
            function_tool(self.update_procedure),
            function_tool(self.delete_procedure),
            function_tool(self.find_similar_procedures),
            function_tool(self.transfer_procedure),
            function_tool(self.get_transfer_statistics),
        ]
        
        # Execution agent specific tools
        execution_tools = common_tools + [
            function_tool(self.execute_procedure),
            function_tool(self.execute_step),
            function_tool(self.execute_hierarchical_procedure),
            function_tool(self.execute_temporal_procedure),
            function_tool(self.execute_graph_procedure),
        ]
        
        # Analysis agent specific tools
        analysis_tools = common_tools + [
            function_tool(self.analyze_execution_history),
            function_tool(self.identify_chunking_opportunities),
            function_tool(self.apply_chunking),
            function_tool(self.generalize_chunk_from_steps),
            function_tool(self.optimize_procedure_parameters),
            function_tool(self.optimize_procedural_memory),
        ]
        
        # Update agents with tools
        self._proc_manager_agent = self._proc_manager_agent.clone(tools=manager_tools)
        self._proc_execution_agent = self._proc_execution_agent.clone(tools=execution_tools)
        self._proc_analysis_agent = self._proc_analysis_agent.clone(tools=analysis_tools)
    
    def _setup_agent_handoffs(self):
        """Set up handoffs between agents"""
        # Manager can hand off to execution and analysis
        manager_handoffs = [
            handoff(self._proc_execution_agent),
            handoff(self._proc_analysis_agent),
        ]
        
        # Execution can hand off to analysis
        execution_handoffs = [
            handoff(self._proc_analysis_agent),
        ]
        
        # Update agents with handoffs
        self._proc_manager_agent = self._proc_manager_agent.clone(handoffs=manager_handoffs)
        self._proc_execution_agent = self._proc_execution_agent.clone(handoffs=execution_handoffs)
    
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
        
        # Additional control mappings as in original implementation...
        # (Code condensed for brevity)
    
    def register_function(self, name: str, func: Callable):
        """Register a function for use in procedures"""
        self.function_registry[name] = func
    
    # Function tools for the agents
    @staticmethod
    @function_tool
    async def list_procedures(ctx: RunContextWrapper) -> Dict[str, Any]:
        """List all available procedures"""
        with agents_trace("list_procedures"):
            procedures_list = []
            for name, proc in self.procedures.items():
                procedures_list.append({
                    "name": name,
                    "domain": proc.domain,
                    "steps_count": len(proc.steps),
                    "proficiency": proc.proficiency,
                    "execution_count": proc.execution_count,
                    "is_chunked": proc.is_chunked,
                })
                
            return {
                "count": len(procedures_list),
                "procedures": procedures_list
            }

    @staticmethod
    @function_tool
    async def add_procedure(
        ctx: RunContextWrapper,
        name: str,
        steps: List[ProcedureStep],
        description: str,
        domain: str
    ) -> Dict[str, Any]:
        """
        Add a new procedure to procedural memory
        
        Args:
            name: Name of the procedure
            steps: List of step definitions
            description: Description of what the procedure does
            domain: Domain for the procedure
            
        Returns:
            Information about the created procedure
        """
        with agents_trace("add_procedure"):
            # Generate ID
            proc_id = f"proc_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
            
            # Create procedure
            procedure = Procedure(
                id=proc_id,
                name=name,
                description=description,
                domain=domain,
                steps=steps
            )
            
            # Store procedure
            self.procedures[name] = procedure
            
            return {
                "procedure_id": proc_id,
                "name": name,
                "domain": domain,
                "steps_count": len(steps),
                "status": "created"
            }
            
    @staticmethod
    @function_tool
    async def get_procedure(ctx: RunContextWrapper,
        name: str
    ) -> Dict[str, Any]:
        """
        Get details of a procedure
        
        Args:
            name: Name of the procedure
            
        Returns:
            Details of the procedure
        """
        with agents_trace("get_procedure"):
            if name not in self.procedures:
                return {"error": f"Procedure '{name}' not found"}
            
            procedure = self.procedures[name]
            
            return {
                "id": procedure.id,
                "name": procedure.name,
                "description": procedure.description,
                "domain": procedure.domain,
                "steps": procedure.steps,
                "proficiency": procedure.proficiency,
                "execution_count": procedure.execution_count,
                "is_chunked": procedure.is_chunked,
                "chunked_steps": procedure.chunked_steps if procedure.is_chunked else {}
            }

    @staticmethod
    @function_tool
    async def update_procedure(
        ctx: RunContextWrapper,
        name: str,
        steps: Optional[List[ProcedureStep]] = None,
        description: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an existing procedure
        
        Args:
            name: Name of the procedure
            steps: Optional updated steps
            description: Optional updated description
            domain: Optional updated domain
            
        Returns:
            Status of the update
        """
        with agents_trace("update_procedure"):
            if name not in self.procedures:
                return {"error": f"Procedure '{name}' not found"}
            
            procedure = self.procedures[name]
            
            # Update fields if provided
            if steps is not None:
                procedure.steps = steps
                
            if description is not None:
                procedure.description = description
                
            if domain is not None:
                procedure.domain = domain
                
            # Update timestamp
            procedure.last_updated = datetime.datetime.now().isoformat()
            
            return {
                "procedure_id": procedure.id,
                "name": name,
                "status": "updated",
                "steps_count": len(procedure.steps)
            }
            
    @staticmethod
    @function_tool
    async def delete_procedure(ctx: RunContextWrapper,
        name: str
    ) -> Dict[str, Any]:
        """
        Delete a procedure
        
        Args:
            name: Name of the procedure
            
        Returns:
            Status of the deletion
        """
        with agents_trace("delete_procedure"):
            if name not in self.procedures:
                return {"error": f"Procedure '{name}' not found"}
            
            # Store ID for response
            proc_id = self.procedures[name].id
            
            # Delete procedure
            del self.procedures[name]
            
            return {
                "procedure_id": proc_id,
                "name": name,
                "status": "deleted"
            }
            
    @staticmethod
    @function_tool
    async def execute_procedure(
        ctx: RunContextWrapper,
        name: str,
        context: Optional[ExecutionContext] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure
        
        Args:
            name: Name of the procedure
            context: Optional execution context
            
        Returns:
            Execution results
        """
        with agents_trace(workflow_name="execute_procedure"):
            if name not in self.procedures:
                return {"error": f"Procedure '{name}' not found"}
            
            procedure = self.procedures[name]
            
            # Determine execution mode based on proficiency
            conscious_execution = procedure.proficiency < 0.8
            
            # Execute procedure
            with agent_span("procedure_execution", 
                          {"procedure_name": name, "conscious_execution": conscious_execution}):
                result = await self.execute_procedure_steps(
                    procedure=procedure,
                    context=context or {},
                    conscious_execution=conscious_execution
                )
                
            return result
    
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

    @staticmethod
    @function_tool
    async def execute_step(
        ctx: RunContextWrapper,
        step: ProcedureStep,
        context: ExecutionContext,
        minimal_monitoring: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a single step of a procedure
        
        Args:
            step: Step definition
            context: Execution context
            minimal_monitoring: Whether to use minimal monitoring
            
        Returns:
            Step execution result
        """
        with function_span("execute_step", input=str(step)):
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
        with function_span("execute_chunk", 
                         {"chunk_id": chunk_id, "steps_count": len(chunk_steps)}):
            results = []
            success = True
            start_time = datetime.datetime.now()
            
            # Create chunk-specific context
            chunk_context = context.copy()
            if chunk_id:
                chunk_context["current_chunk"] = chunk_id
            
            # Execute steps
            for step in chunk_steps:
                step_result = await self.execute_step(RunContextWrapper(context=None), step, chunk_context, minimal_monitoring)
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

    @staticmethod
    @function_tool
    async def get_procedure_proficiency(ctx: RunContextWrapper,
        name: str
    ) -> Dict[str, Any]:
        """
        Get proficiency statistics for a procedure
        
        Args:
            name: Name of the procedure
            
        Returns:
            Proficiency statistics
        """
        with agents_trace("get_procedure_proficiency"):
            if name not in self.procedures:
                return {"error": f"Procedure '{name}' not found"}
            
            procedure = self.procedures[name]
            
            # Calculate success rate
            success_rate = 0.0
            if procedure.execution_count > 0:
                success_rate = procedure.successful_executions / procedure.execution_count
                
            # Determine level based on proficiency
            level = "novice"
            if procedure.proficiency >= 0.9:
                level = "expert"
            elif procedure.proficiency >= 0.7:
                level = "proficient"
            elif procedure.proficiency >= 0.4:
                level = "intermediate"
                
            return {
                "procedure_name": name,
                "procedure_id": procedure.id,
                "proficiency": procedure.proficiency,
                "level": level,
                "execution_count": procedure.execution_count,
                "success_rate": success_rate,
                "average_execution_time": procedure.average_execution_time,
                "is_chunked": procedure.is_chunked,
                "chunks_count": len(procedure.chunked_steps) if procedure.is_chunked else 0,
                "domain": procedure.domain,
                "last_execution": procedure.last_execution
            }

    @staticmethod
    @function_tool
    async def find_similar_procedures(ctx: RunContextWrapper,
        name: str,
        similarity_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Find procedures similar to the specified one
        
        Args:
            name: Name of the reference procedure
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar procedures with similarity scores
        """
        with agents_trace("find_similar_procedures"):
            if name not in self.procedures:
                return {"error": f"Procedure '{name}' not found"}
            
            reference = self.procedures[name]
            similar_procedures = []
            
            for other_name, other_proc in self.procedures.items():
                # Skip self comparison
                if other_name == name:
                    continue
                    
                # Calculate similarity
                similarity = self.calculate_procedure_similarity(reference, other_proc)
                
                if similarity >= similarity_threshold:
                    similar_procedures.append({
                        "name": other_name,
                        "similarity": similarity,
                        "domain": other_proc.domain,
                        "steps_count": len(other_proc.steps)
                    })
            
            # Sort by similarity (highest first)
            similar_procedures.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "reference_procedure": name,
                "similar_procedures": similar_procedures,
                "count": len(similar_procedures)
            }
    
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

    @staticmethod
    @function_tool
    async def identify_chunking_opportunities(ctx: RunContextWrapper,
        procedure_name: str
    ) -> Dict[str, Any]:
        """
        Identify opportunities for chunking in a procedure
        
        Args:
            procedure_name: Name of the procedure
            
        Returns:
            Chunking opportunities
        """
        with agents_trace("identify_chunking_opportunities"):
            if procedure_name not in self.procedures:
                return {"error": f"Procedure '{procedure_name}' not found"}
            
            procedure = self.procedures[procedure_name]
            
            # Need at least 3 steps to consider chunking
            if len(procedure.steps) < 3:
                return {
                    "procedure_name": procedure_name,
                    "can_chunk": False,
                    "reason": "Not enough steps for chunking (need at least 3)"
                }
            
            # Already chunked
            if procedure.is_chunked:
                return {
                    "procedure_name": procedure_name,
                    "can_chunk": False,
                    "reason": "Procedure is already chunked",
                    "existing_chunks": len(procedure.chunked_steps)
                }
            
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
            
            return {
                "procedure_name": procedure_name,
                "can_chunk": len(chunks) > 0,
                "potential_chunks": len(chunks),
                "chunks": chunks
            }
    
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

    @staticmethod
    @function_tool
    async def apply_chunking(ctx: RunContextWrapper,
        procedure_name: str,
        chunks: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Apply chunking to a procedure
        
        Args:
            procedure_name: Name of the procedure
            chunks: Optional list of chunks (lists of step IDs)
            
        Returns:
            Status of chunking application
        """
        with agents_trace("apply_chunking"):
            if procedure_name not in self.procedures:
                return {"error": f"Procedure '{procedure_name}' not found"}
            
            procedure = self.procedures[procedure_name]
            
            # If chunks not provided, identify them
            if chunks is None:
                chunking_result = await self.identify_chunking_opportunities(ctx, procedure_name)
                
                if not chunking_result.get("can_chunk", False):
                    return {
                        "procedure_name": procedure_name,
                        "chunking_applied": False,
                        "reason": chunking_result.get("reason", "No chunking opportunities found")
                    }
                    
                chunks = chunking_result.get("chunks", [])
            
            # Apply identified chunks
            self._apply_chunking(procedure, chunks)
            
            return {
                "procedure_name": procedure_name,
                "chunking_applied": True,
                "chunks_count": len(procedure.chunked_steps),
                "chunks": procedure.chunked_steps
            }
    
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

    @staticmethod
    @function_tool
    async def generalize_chunk_from_steps(ctx: RunContextWrapper,
        chunk_name: str,
        procedure_name: str,
        step_ids: List[str],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a generalized chunk template from procedure steps
        
        Args:
            chunk_name: Name for the new chunk template
            procedure_name: Name of the source procedure
            step_ids: List of step IDs to include in the chunk
            domain: Optional domain override
            
        Returns:
            Status of template creation
        """
        with agents_trace("generalize_chunk_from_steps"):
            if procedure_name not in self.procedures:
                return {"error": f"Procedure '{procedure_name}' not found"}
            
            procedure = self.procedures[procedure_name]
            
            # Get steps
            steps = []
            for step_id in step_ids:
                step = next((s for s in procedure.steps if s["id"] == step_id), None)
                if step:
                    steps.append(step)
            
            if not steps:
                return {
                    "error": "No valid steps found",
                    "procedure_name": procedure_name
                }
            
            # Use procedure domain if not specified
            if domain is None:
                domain = procedure.domain
            
            # Create template
            template = self.chunk_library.create_chunk_template_from_steps(
                chunk_id=f"template_{chunk_name}",
                name=chunk_name,
                steps=steps,
                domain=domain,
                success_rate=0.9  # Start with high success rate
            )
            
            if not template:
                return {
                    "error": "Failed to create template",
                    "procedure_name": procedure_name,
                    "step_count": len(steps)
                }
            
            # If procedure is chunked, try to link template to existing chunks
            if procedure.is_chunked:
                # Find a chunk that matches the steps
                for chunk_id, chunk_step_ids in procedure.chunked_steps.items():
                    if set(step_ids).issubset(set(chunk_step_ids)):
                        # This chunk includes all the requested steps
                        procedure.generalized_chunks[chunk_id] = template.id
                        break
            
            return {
                "template_id": template.id,
                "name": chunk_name,
                "domain": domain,
                "steps_count": len(steps),
                "template_created": True
            }
            
    @staticmethod
    @function_tool
    async def transfer_procedure(ctx: RunContextWrapper,
        source_name: str,
        target_name: str,
        target_domain: str
    ) -> Dict[str, Any]:
        """
        Transfer a procedure to a different domain
        
        Args:
            source_name: Name of the source procedure
            target_name: Name for the new procedure
            target_domain: Target domain
            
        Returns:
            Status of the transfer
        """
        with agents_trace("transfer_procedure"):
            if source_name not in self.procedures:
                return {"error": f"Source procedure '{source_name}' not found"}
            
            source_procedure = self.procedures[source_name]
            source_domain = source_procedure.domain
            
            # Map procedure steps to target domain
            mapped_steps = []
            
            for step in source_procedure.steps:
                mapped_step = self.map_step_to_domain(
                    step, source_domain, target_domain
                )
                
                if mapped_step:
                    mapped_steps.append(mapped_step)
                else:
                    # Fallback: use original step if mapping fails
                    mapped_steps.append(step.copy())
            
            # Create new procedure in target domain
            target_procedure = await self.add_procedure(
                ctx,
                name=target_name,
                steps=mapped_steps,
                description=f"Transferred from {source_name} ({source_domain} to {target_domain})",
                domain=target_domain
            )
            
            # Record transfer
            self.transfer_stats["total_transfers"] += 1
            
            # Update chunk mappings if needed
            if source_procedure.is_chunked and hasattr(source_procedure, "generalized_chunks"):
                # Create corresponding chunks in target procedure
                if target_name in self.procedures:
                    target_proc = self.procedures[target_name]
                    
                    # Copy chunk structure
                    target_proc.chunked_steps = {}
                    for chunk_id, step_ids in source_procedure.chunked_steps.items():
                        # Map step IDs
                        target_step_ids = []
                        for step_id in step_ids:
                            # Find corresponding step in target procedure
                            for mapped_step in mapped_steps:
                                if mapped_step.get("original_id") == step_id:
                                    target_step_ids.append(mapped_step["id"])
                                    break
                        
                        if target_step_ids:
                            target_proc.chunked_steps[chunk_id] = target_step_ids
                    
                    # Mark as chunked if chunks were created
                    target_proc.is_chunked = len(target_proc.chunked_steps) > 0
            
            return {
                "source_procedure": source_name,
                "target_procedure": target_name,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "steps_transferred": len(mapped_steps),
                "procedure_id": target_procedure.get("procedure_id")
            }
    
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
        
    @staticmethod 
    @function_tool
    async def analyze_execution_history(ctx: RunContextWrapper,
        procedure_name: str
    ) -> Dict[str, Any]:
        """
        Analyze execution history of a procedure for patterns
        
        Args:
            procedure_name: Name of the procedure
            
        Returns:
            Analysis of execution history
        """
        with agents_trace("analyze_execution_history"):
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

    @staticmethod
    @function_tool
    async def get_transfer_statistics(ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """
        Get statistics about procedure transfers
        
        Returns:
            Transfer statistics
        """
        with agents_trace("get_transfer_statistics"):
            # Calculate average statistics
            avg_success = 0.0
            avg_practice = 0
            
            if self.transfer_stats["total_transfers"] > 0:
                avg_success = self.transfer_stats["avg_success_level"]
                avg_practice = self.transfer_stats["avg_practice_needed"]
            
            # Get domain information
            domain_mappings = {}
            for mapping in self.chunk_library.control_mappings:
                source = mapping.source_domain
                target = mapping.target_domain
                
                key = f"{source}→{target}"
                if key not in domain_mappings:
                    domain_mappings[key] = 0
                    
                domain_mappings[key] += 1
            
            return {
                "total_transfers": self.transfer_stats["total_transfers"],
                "successful_transfers": self.transfer_stats["successful_transfers"],
                "success_rate": self.transfer_stats["successful_transfers"] / max(1, self.transfer_stats["total_transfers"]),
                "avg_success_level": avg_success,
                "avg_practice_needed": avg_practice,
                "domain_mappings": domain_mappings,
                "template_count": len(self.chunk_library.templates) if hasattr(self.chunk_library, "templates") else 0
            }

    @staticmethod
    @function_tool
    async def optimize_procedure_parameters(ctx: RunContextWrapper,
        procedure_name: str,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a procedure
        
        Args:
            procedure_name: Name of the procedure to optimize
            iterations: Number of optimization iterations
            
        Returns:
            Optimization results
        """
        with agents_trace("optimize_procedure_parameters"):
            if procedure_name not in self.procedures:
                return {"error": f"Procedure '{procedure_name}' not found"}
            
            procedure = self.procedures[procedure_name]
            
            # Check if parameter optimizer exists
            if not hasattr(self, "parameter_optimizer"):
                return {
                    "error": "Parameter optimizer not available",
                    "procedure_name": procedure_name
                }
            
            # Define objective function
            async def objective_function(test_procedure: Procedure) -> float:
                # Create simulated context
                test_context = {"optimization_run": True}
                
                # Execute procedure
                result = await self.execute_procedure_steps(
                    procedure=test_procedure,
                    context=test_context,
                    conscious_execution=True
                )
                
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
            
            return optimization_result

    @staticmethod
    @function_tool
    async def optimize_procedural_memory(ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """
        Optimize procedural memory by consolidating and cleaning up
        
        Returns:
            Optimization results
        """
        with agents_trace("optimize_procedural_memory"):
            # Check if memory consolidator exists
            if hasattr(self, "memory_consolidator"):
                return await self.memory_consolidator.consolidate_procedural_memory()
            else:
                # Basic implementation
                start_time = datetime.datetime.now()
                
                # Clean up old procedures
                old_procedures = []
                for name, procedure in self.procedures.items():
                    # Check if procedure hasn't been used in a long time
                    if hasattr(procedure, "last_execution") and procedure.last_execution:
                        last_exec = datetime.datetime.fromisoformat(procedure.last_execution)
                        days_since_last_exec = (datetime.datetime.now() - last_exec).days
                        
                        if days_since_last_exec > 90:  # More than 90 days
                            old_procedures.append(name)
                
                # Remove old procedures
                procedures_cleaned = 0
                for name in old_procedures:
                    # Only remove if low proficiency
                    if self.procedures[name].proficiency < 0.7:
                        del self.procedures[name]
                        procedures_cleaned += 1
                
                return {
                    "procedures_cleaned": procedures_cleaned,
                    "execution_time": (datetime.datetime.now() - start_time).total_seconds(),
                    "status": "basic_optimization_completed"
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

    @staticmethod
    @function_tool
    async def execute_hierarchical_procedure(
        ctx: RunContextWrapper,
        name: str,
        context: Optional[ExecutionContext] = None
    ) -> Dict[str, Any]:
        """
        Execute a hierarchical procedure
        
        Args:
            name: Name of the procedure
            context: Execution context
            
        Returns:
            Execution results
        """
        # Placeholder implementation since the EnhancedProceduralMemoryManager handles this
        return {"error": "Hierarchical procedure execution requires the EnhancedProceduralMemoryManager"}

    @staticmethod
    @function_tool
    async def execute_temporal_procedure(
        ctx: RunContextWrapper,
        name: str,
        context: Optional[ExecutionContext] = None
    ) -> Dict[str, Any]:
        """
        Execute a procedure with temporal constraints
        
        Args:
            name: Name of the procedure
            context: Execution context
            
        Returns:
            Execution results
        """
        # Placeholder implementation since the EnhancedProceduralMemoryManager handles this
        return {"error": "Temporal procedure execution requires the EnhancedProceduralMemoryManager"}

    @staticmethod
    @function_tool
    async def execute_graph_procedure(
        ctx: RunContextWrapper,
        procedure_name: str,
        context: Optional[ExecutionContext] = None,
        goal: Optional[GoalState] = None
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
        # Placeholder implementation since the EnhancedProceduralMemoryManager handles this
        return {"error": "Graph-based procedure execution requires the EnhancedProceduralMemoryManager"}


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
        
        # Register additional function tools
        self._register_enhanced_function_tools()
        
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
    
    def _register_enhanced_function_tools(self):
        """Register enhanced function tools for agents"""
        # Define enhanced tools
        enhanced_tools = [
            function_tool(self.learn_from_demonstration),
            function_tool(self.create_hierarchical_procedure),
            function_tool(self.execute_hierarchical_procedure),
            function_tool(self.create_temporal_procedure),
            function_tool(self.execute_temporal_procedure),
            function_tool(self.create_procedure_graph),
            function_tool(self.execute_graph_procedure),
            function_tool(self.optimize_procedure_transfer),
            function_tool(self.execute_transfer_plan),
            function_tool(self.share_domain_knowledge),
            function_tool(self.handle_execution_error),
        ]
        
        # Add enhanced tools to all agents
        self._proc_manager_agent = self._proc_manager_agent.clone(
            tools=self._proc_manager_agent.tools + enhanced_tools
        )
        
        self._proc_execution_agent = self._proc_execution_agent.clone(
            tools=self._proc_execution_agent.tools + [
                function_tool(self.execute_hierarchical_procedure),
                function_tool(self.execute_temporal_procedure),
                function_tool(self.execute_graph_procedure),
                function_tool(self.handle_execution_error),
            ]
        )
        
        self._proc_analysis_agent = self._proc_analysis_agent.clone(
            tools=self._proc_analysis_agent.tools + [
                function_tool(self.optimize_procedure_transfer),
                function_tool(self.share_domain_knowledge),
            ]
        )
    
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
            # Additional error causes...
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
            # Additional interventions...
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
        
        # Add templates to library
        self.chunk_library.add_chunk_template(navigation_template)
        # Additional templates could be added here...
    
    # Implement enhanced agent function tools
    @staticmethod
    @function_tool
    async def learn_from_demonstration(
        ctx: RunContextWrapper,
        observation_sequence: List[Dict[str, Any]],  # This one might be okay as is
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
        with agents_trace("learn_from_demonstration"):
            # Learn from observations
            procedure_data = await self.observation_learner.learn_from_demonstration(
                observation_sequence=observation_sequence,
                domain=domain
            )
            
            # Use provided name if available
            if name:
                procedure_data["name"] = name
            
            # Create the procedure
            procedure_result = await self.add_procedure(
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

    @staticmethod
    @function_tool
    async def create_hierarchical_procedure(
        ctx: RunContextWrapper,
        name: str,
        description: str,
        domain: str,
        steps: List[ProcedureStep],
        goal_state: Optional[GoalState] = None,
        preconditions: Optional[Dict[str, Any]] = None,
        postconditions: Optional[Dict[str, Any]] = None,
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
        with agents_trace("create_hierarchical_procedure"):
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
            standard_proc = await self.add_procedure(
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

    @staticmethod
    @function_tool
    async def execute_hierarchical_procedure(ctx: RunContextWrapper,
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
        with agents_trace(workflow_name="execute_hierarchical_procedure"):
            if name not in self.hierarchical_procedures:
                return {"error": f"Hierarchical procedure '{name}' not found"}
            
            procedure = self.hierarchical_procedures[name]
            
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
        
        # Update proficiency
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

    @staticmethod
    @function_tool
    async def create_temporal_procedure(
        ctx: RunContextWrapper,
        name: str,
        steps: List[ProcedureStep],
        temporal_constraints: List[TemporalConstraint],
        domain: str,
        description: Optional[str] = None
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
        with agents_trace("create_temporal_procedure"):
            # Create normal procedure first
            normal_proc = await self.add_procedure(
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
            
    @staticmethod
    @function_tool
    async def execute_temporal_procedure(ctx: RunContextWrapper,
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
        with agents_trace(workflow_name="execute_temporal_procedure"):
            if name not in self.procedures:
                return {"error": f"Procedure '{name}' not found"}
            
            procedure = self.procedures[name]
            
            # Check if procedure has temporal graph
            if not hasattr(procedure, "temporal_graph_id") or procedure.temporal_graph_id not in self.temporal_graphs:
                # Fall back to normal execution
                return await self.execute_procedure(ctx, name, context)
            
            # Get temporal graph
            graph = self.temporal_graphs[procedure.temporal_graph_id]
            
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
                step_result = await self.execute_step(ctx, step, execution_context)
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

    @staticmethod
    @function_tool
    async def create_procedure_graph(ctx: RunContextWrapper,
        procedure_name: str
    ) -> Dict[str, Any]:
        """
        Create a graph representation of a procedure for flexible execution
        
        Args:
            procedure_name: Name of the existing procedure
            
        Returns:
            Information about the created graph
        """
        with agents_trace("create_procedure_graph"):
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

    @staticmethod
    @function_tool
    async def execute_graph_procedure(ctx: RunContextWrapper,
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
        with agents_trace(workflow_name="execute_graph_procedure"):
            if procedure_name not in self.procedures:
                return {"error": f"Procedure '{procedure_name}' not found"}
            
            procedure = self.procedures[procedure_name]
            
            # Check if procedure has graph
            if not hasattr(procedure, "graph_id") or procedure.graph_id not in self.procedure_graphs:
                # Fall back to normal execution
                return await self.execute_procedure(ctx, procedure_name, context)
            
            # Get graph
            graph = self.procedure_graphs[procedure.graph_id]
            
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
                step_result = await self.execute_step(ctx, step, execution_context)
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

    @staticmethod
    @function_tool
    async def optimize_procedure_transfer(ctx: RunContextWrapper,
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
        with agents_trace("optimize_procedure_transfer"):
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
    
# Continuing EnhancedProceduralMemoryManager class implementation

    @staticmethod
    @function_tool
    async def execute_transfer_plan(
        ctx: RunContextWrapper,
        transfer_plan: TransferPlan,
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
        with agents_trace("execute_transfer_plan"):
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
            new_procedure = await self.add_procedure(
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

    @staticmethod
    @function_tool
    async def handle_execution_error(
        ctx: RunContextWrapper,
        error: ErrorInfo,
        context: Optional[ExecutionContext] = None
    ) -> Dict[str, Any]:
        """
        Handle an execution error using the causal model
        
        Args:
            error: Error details
            context: Execution context
            
        Returns:
            Recovery suggestions
        """
        with agents_trace("handle_execution_error"):
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

    @staticmethod
    @function_tool
    async def share_domain_knowledge(ctx: RunContextWrapper,
        domain: str
    ) -> Dict[str, Any]:
        """
        Share procedural knowledge about a domain with knowledge core
        
        Args:
            domain: Domain to share knowledge about
            
        Returns:
            Status of knowledge sharing
        """
        with agents_trace("share_domain_knowledge"):
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
                RunContextWrapper(context=None),
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
        await self.memory_consolidator.consolidate_procedural_memory()
    
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
