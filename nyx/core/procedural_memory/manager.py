# nyx/core/procedural_memory/manager.py

# Standard library
import asyncio
import datetime
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, Literal
from typing_extensions import TypedDict, NotRequired
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

# ============================================================================
# Base Types and Common Structures
# ============================================================================

class StepParameters(TypedDict, total=False):
    """Flexible parameters for procedure steps"""
    button: Optional[str]
    control: Optional[str]
    input_method: Optional[str]
    key: Optional[str]
    target: Optional[str]
    value: Optional[Union[str, int, float, bool]]
    direction: Optional[str]
    speed: Optional[float]
    duration: Optional[float]
    coordinates: Optional[Dict[str, float]]  # {"x": 0.0, "y": 0.0, "z": 0.0}
    options: Optional[Dict[str, Union[str, int, float, bool]]]

class ProcedureStep(TypedDict):
    """Schema for a procedure step"""
    id: str
    description: str
    function: str
    parameters: StepParameters
    original_id: NotRequired[str]  # Used in transfer operations

class ActionHistoryItem(TypedDict):
    """Single action history entry"""
    step_id: str
    function: str
    success: bool
    timestamp: Optional[str]
    execution_time: Optional[float]
    error: Optional[str]

class ExecutionContext(TypedDict, total=False):
    """Schema for execution context"""
    timestamp: str
    conscious_execution: bool
    result: bool
    execution_time: float
    action_history: List[ActionHistoryItem]
    current_chunk: str
    current_procedure: str
    hierarchical: bool
    temporal_execution: bool
    graph_execution: bool
    optimization_run: bool
    execution_history: List[ActionHistoryItem]
    step_results: Dict[str, 'StepExecutionResult']  # step_id -> result mapping
    user_context: Optional[Dict[str, Union[str, int, float, bool]]]
    environment: Optional[Dict[str, str]]

# ============================================================================
# Observation and Learning Types
# ============================================================================

class ObservationState(TypedDict, total=False):
    """State information for observations"""
    position: Optional[Dict[str, float]]  # {"x": 0.0, "y": 0.0, "z": 0.0}
    rotation: Optional[Dict[str, float]]  # {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    status: Optional[str]
    health: Optional[float]
    energy: Optional[float]
    inventory: Optional[List[str]]
    active_item: Optional[str]
    mode: Optional[str]
    flags: Optional[Dict[str, bool]]
    custom_values: Optional[Dict[str, Union[str, int, float, bool]]]

class ObservationStep(TypedDict):
    """Schema for observation sequence step"""
    action: str
    state: ObservationState
    timestamp: str
    duration: NotRequired[float]
    success: NotRequired[bool]

# ============================================================================
# Error Handling Types
# ============================================================================

class ErrorContext(TypedDict, total=False):
    """Context information for errors"""
    procedure_name: Optional[str]
    procedure_id: Optional[str]
    step_id: Optional[str]
    step_index: Optional[int]
    function_name: Optional[str]
    parameters: Optional[StepParameters]
    execution_time: Optional[float]
    attempt_number: Optional[int]
    stack_trace: Optional[str]

class ErrorInfo(TypedDict):
    """Schema for error information"""
    error_type: str
    message: str
    context: ErrorContext
    timestamp: NotRequired[str]
    severity: NotRequired[Literal["low", "medium", "high", "critical"]]

# ============================================================================
# Transfer and Mapping Types
# ============================================================================

class MappingInfo(TypedDict):
    """Schema for transfer mapping"""
    source_function: str
    target_function: str
    target_parameters: StepParameters
    confidence: NotRequired[float]
    mapping_type: NotRequired[Literal["direct", "adapted", "synthesized"]]

class TransferPlan(TypedDict):
    """Schema for transfer plan"""
    source_domain: str
    target_domain: str
    mappings: List[MappingInfo]
    transfer_strategy: str
    estimated_success_rate: NotRequired[float]
    required_adaptations: NotRequired[List[str]]

# ============================================================================
# Goal and Constraint Types
# ============================================================================

class GoalCondition(TypedDict):
    """Single goal condition"""
    attribute: str
    operator: Literal["equals", "greater_than", "less_than", "contains", "exists"]
    value: Union[str, int, float, bool, List[str]]

class GoalState(TypedDict, total=False):
    """Schema for goal state"""
    conditions: List[GoalCondition]
    description: str
    priority: Literal["low", "medium", "high"]
    timeout: Optional[float]

class TemporalConstraint(TypedDict):
    """Schema for temporal constraints"""
    from_step: str
    to_step: str
    type: Literal["before", "after", "concurrent", "min_delay", "max_delay", "exact_delay"]
    delay: Optional[float]
    tolerance: NotRequired[float]

class PreconditionCheck(TypedDict):
    """Single precondition check"""
    attribute: str
    check_type: Literal["exists", "equals", "not_equals", "in_range"]
    expected_value: Optional[Union[str, int, float, bool]]
    min_value: Optional[Union[int, float]]
    max_value: Optional[Union[int, float]]

class PreconditionSet(TypedDict, total=False):
    """Schema for preconditions"""
    checks: List[PreconditionCheck]
    require_all: bool
    description: str

class PostconditionSet(TypedDict, total=False):
    """Schema for postconditions"""
    checks: List[PreconditionCheck]
    verify_all: bool
    description: str

# ============================================================================
# Procedure Information Types
# ============================================================================

class ProcedureSummary(TypedDict):
    """Summary information for a procedure in list results"""
    name: str
    domain: str
    steps_count: int
    proficiency: float
    execution_count: int
    is_chunked: bool
    last_execution: Optional[str]
    success_rate: Optional[float]
    average_execution_time: Optional[float]
    description: Optional[str]

class ChunkInfo(TypedDict):
    """Information about a procedure chunk"""
    chunk_id: str
    step_ids: List[str]
    step_count: int
    has_template: bool
    has_context_pattern: bool
    execution_count: NotRequired[int]
    success_rate: NotRequired[float]

class RefinementOpportunity(TypedDict):
    """Refinement opportunity for a procedure"""
    step_id: str
    type: Literal["improve_reliability", "optimize_speed", "reduce_complexity"]
    current_success_rate: float
    identified_at: str
    description: str
    priority: NotRequired[Literal["low", "medium", "high"]]

# ============================================================================
# Return Types for Function Tools
# ============================================================================

class ProcedureListResult(TypedDict):
    """Return type for list_procedures"""
    count: int
    procedures: List[ProcedureSummary]

class ProcedureResult(TypedDict):
    """Return type for add/update/delete procedure operations"""
    procedure_id: str
    name: str
    status: Literal["created", "updated", "deleted"]
    domain: Optional[str]
    steps_count: Optional[int]
    error: NotRequired[str]

class ProcedureDetails(TypedDict, total=False):
    """Return type for get_procedure"""
    id: str
    name: str
    description: str
    domain: str
    steps: List[ProcedureStep]
    proficiency: float
    execution_count: int
    successful_executions: int
    is_chunked: bool
    chunked_steps: Dict[str, List[str]]
    average_execution_time: float
    last_execution: Optional[str]
    last_updated: Optional[str]
    refinement_opportunities: List[RefinementOpportunity]
    generalized_chunks: Dict[str, str]  # chunk_id -> template_id
    chunk_contexts: Dict[str, str]  # chunk_id -> context_pattern_id
    error: NotRequired[str]

class StepData(TypedDict, total=False):
    """Data returned from step execution"""
    result: Optional[Union[str, int, float, bool]]
    output: Optional[str]
    state_change: Optional[Dict[str, Union[str, int, float, bool]]]
    metrics: Optional[Dict[str, float]]

class StepExecutionResult(TypedDict):
    """Return type for execute_step"""
    success: bool
    data: Optional[StepData]
    execution_time: float
    error: Optional[str]

class ExecutionResult(TypedDict):
    """Return type for execute procedures"""
    success: bool
    results: List[StepExecutionResult]
    execution_time: float
    proficiency: Optional[float]
    automatic: Optional[bool]
    chunked: Optional[bool]
    error: Optional[str]
    procedure_name: NotRequired[str]
    procedure_id: NotRequired[str]
    strategy_used: NotRequired[str]

class ProficiencyInfo(TypedDict):
    """Return type for get_procedure_proficiency"""
    procedure_name: str
    procedure_id: str
    proficiency: float
    level: Literal["novice", "intermediate", "proficient", "expert"]
    execution_count: int
    success_rate: float
    average_execution_time: float
    is_chunked: bool
    chunks_count: int
    domain: str
    last_execution: Optional[str]

class SimilarProcedure(TypedDict):
    """Information about a similar procedure"""
    name: str
    similarity: float
    domain: str
    steps_count: int
    proficiency: NotRequired[float]
    shared_functions: NotRequired[List[str]]

class SimilarityResult(TypedDict):
    """Return type for find_similar_procedures"""
    reference_procedure: str
    similar_procedures: List[SimilarProcedure]
    count: int

class ChunkingOpportunity(TypedDict):
    """Information about potential chunks"""
    step_ids: List[str]
    co_occurrence_rate: float
    estimated_benefit: NotRequired[float]

class ChunkingAnalysis(TypedDict):
    """Return type for identify_chunking_opportunities"""
    procedure_name: str
    can_chunk: bool
    reason: Optional[str]
    potential_chunks: int
    chunks: List[List[str]]
    existing_chunks: NotRequired[int]
    opportunities: NotRequired[List[ChunkingOpportunity]]

class ChunkingResult(TypedDict):
    """Return type for apply_chunking"""
    procedure_name: str
    chunking_applied: bool
    reason: Optional[str]
    chunks_count: int
    chunks: Dict[str, List[str]]

class ContextPattern(TypedDict):
    """Pattern found in execution context"""
    key: str
    pattern_type: Literal["consistent_value", "common_value", "success_factor"]
    value: Optional[Union[str, int, float, bool]]
    most_common_value: Optional[Union[str, int, float, bool]]
    occurrence_rate: Optional[float]
    occurrences: Optional[int]

class SuccessPattern(TypedDict):
    """Pattern differentiating successful/failed executions"""
    key: str
    pattern_type: Literal["success_factor"]
    success_values: List[str]
    failure_values: List[str]

class ExecutionAnalysis(TypedDict):
    """Return type for analyze_execution_history"""
    procedure_name: str
    executions: int
    success_rate: float
    avg_execution_time: float
    proficiency: float
    is_chunked: bool
    chunks_count: int
    context_patterns: List[ContextPattern]
    success_patterns: List[SuccessPattern]
    chunk_patterns: List[ChunkInfo]
    refinement_opportunities: int
    analysis: NotRequired[str]

class TransferStatistics(TypedDict):
    """Return type for get_transfer_statistics"""
    total_transfers: int
    successful_transfers: int
    success_rate: float
    avg_success_level: float
    avg_practice_needed: int
    domain_mappings: Dict[str, int]  # "source→target" -> count
    template_count: int

class OptimizationResult(TypedDict):
    """Return type for optimization operations"""
    procedure_name: Optional[str]
    optimization_type: str
    iterations: Optional[int]
    initial_score: Optional[float]
    final_score: Optional[float]
    improvement: Optional[float]
    execution_time: float
    parameters_optimized: Optional[Dict[str, Union[str, int, float]]]
    status: Literal["completed", "failed", "partial"]
    error: Optional[str]

class MemoryOptimizationResult(TypedDict):
    """Return type for optimize_procedural_memory"""
    procedures_cleaned: int
    procedures_consolidated: NotRequired[int]
    chunks_generalized: NotRequired[int]
    templates_created: NotRequired[int]
    execution_time: float
    status: str

class GeneralizationResult(TypedDict):
    """Return type for generalize_chunk_from_steps"""
    template_id: str
    name: str
    domain: str
    steps_count: int
    template_created: bool
    error: Optional[str]

class TransferResult(TypedDict):
    """Return type for transfer operations"""
    source_procedure: str
    target_procedure: str
    source_domain: str
    target_domain: str
    steps_transferred: int
    procedure_id: Optional[str]
    success: bool
    confidence: NotRequired[float]
    adaptations_made: NotRequired[List[str]]
    error: Optional[str]

class LearningResult(TypedDict):
    """Return type for learn_from_demonstration"""
    procedure_id: str
    name: str
    status: str
    domain: str
    steps_count: int
    confidence: float
    learned_from_observations: bool
    observation_count: int

class HierarchicalProcedureResult(TypedDict):
    """Return type for create_hierarchical_procedure"""
    id: str
    name: str
    domain: str
    steps_count: int
    standard_procedure_id: str
    hierarchical: bool
    parent_id: Optional[str]

class ErrorHandlingResult(TypedDict):
    """Return type for handle_execution_error"""
    likely_causes: List[Dict[str, Union[str, float]]]
    interventions: List[Dict[str, Union[str, float]]]
    context: Optional[ExecutionContext]

class KnowledgeSharingResult(TypedDict):
    """Return type for share_domain_knowledge"""
    domain: str
    knowledge_items_added: int
    procedures_analyzed: int
    error: Optional[str]

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

        self._instance = self
        
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
        # Create the instance reference for closures
        manager = self
        
        # ============================================================================
        # Base Function Tools
        # ============================================================================
        
        @function_tool
        async def list_procedures(ctx: RunContextWrapper) -> ProcedureListResult:
            """List all available procedures"""
            with agents_trace("list_procedures"):
                procedures_list: List[ProcedureSummary] = []
                for name, proc in manager.procedures.items():
                    success_rate = 0.0
                    if proc.execution_count > 0:
                        success_rate = proc.successful_executions / proc.execution_count
                        
                    procedures_list.append(ProcedureSummary(
                        name=name,
                        domain=proc.domain,
                        steps_count=len(proc.steps),
                        proficiency=proc.proficiency,
                        execution_count=proc.execution_count,
                        is_chunked=proc.is_chunked,
                        last_execution=proc.last_execution,
                        success_rate=success_rate,
                        average_execution_time=proc.average_execution_time,
                        description=proc.description
                    ))
                    
                return ProcedureListResult(
                    count=len(procedures_list),
                    procedures=procedures_list
                )
    
        @function_tool
        async def add_procedure(
            ctx: RunContextWrapper,
            name: str,
            steps: List[ProcedureStep],
            description: str,
            domain: str
        ) -> ProcedureResult:
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
                manager.procedures[name] = procedure
                
                return ProcedureResult(
                    procedure_id=proc_id,
                    name=name,
                    domain=domain,
                    steps_count=len(steps),
                    status="created"
                )
                
        @function_tool
        async def get_procedure(
            ctx: RunContextWrapper,
            name: str
        ) -> ProcedureDetails:
            """
            Get details of a procedure
            
            Args:
                name: Name of the procedure
                
            Returns:
                Details of the procedure
            """
            with agents_trace("get_procedure"):
                if name not in manager.procedures:
                    return ProcedureDetails(error=f"Procedure '{name}' not found")
                
                procedure = manager.procedures[name]
                
                # Build refinement opportunities list
                refinement_opps: List[RefinementOpportunity] = []
                if hasattr(procedure, "refinement_opportunities"):
                    for opp in procedure.refinement_opportunities:
                        refinement_opps.append(RefinementOpportunity(
                            step_id=opp["step_id"],
                            type=opp["type"],
                            current_success_rate=opp["current_success_rate"],
                            identified_at=opp["identified_at"],
                            description=opp["description"]
                        ))
                
                return ProcedureDetails(
                    id=procedure.id,
                    name=procedure.name,
                    description=procedure.description,
                    domain=procedure.domain,
                    steps=procedure.steps,
                    proficiency=procedure.proficiency,
                    execution_count=procedure.execution_count,
                    successful_executions=procedure.successful_executions,
                    is_chunked=procedure.is_chunked,
                    chunked_steps=procedure.chunked_steps if procedure.is_chunked else {},
                    average_execution_time=procedure.average_execution_time,
                    last_execution=procedure.last_execution,
                    last_updated=procedure.last_updated,
                    refinement_opportunities=refinement_opps,
                    generalized_chunks=getattr(procedure, "generalized_chunks", {}),
                    chunk_contexts=getattr(procedure, "chunk_contexts", {})
                )
    
        @function_tool
        async def update_procedure(
            ctx: RunContextWrapper,
            name: str,
            steps: Optional[List[ProcedureStep]] = None,
            description: Optional[str] = None,
            domain: Optional[str] = None
        ) -> ProcedureResult:
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
                if name not in manager.procedures:
                    return ProcedureResult(
                        procedure_id="",
                        name=name,
                        status="updated",
                        error=f"Procedure '{name}' not found"
                    )
                
                procedure = manager.procedures[name]
                
                # Update fields if provided
                if steps is not None:
                    procedure.steps = steps
                    
                if description is not None:
                    procedure.description = description
                    
                if domain is not None:
                    procedure.domain = domain
                    
                # Update timestamp
                procedure.last_updated = datetime.datetime.now().isoformat()
                
                return ProcedureResult(
                    procedure_id=procedure.id,
                    name=name,
                    status="updated",
                    domain=procedure.domain,
                    steps_count=len(procedure.steps)
                )
                
        @function_tool
        async def delete_procedure(
            ctx: RunContextWrapper,
            name: str
        ) -> ProcedureResult:
            """
            Delete a procedure
            
            Args:
                name: Name of the procedure
                
            Returns:
                Status of the deletion
            """
            with agents_trace("delete_procedure"):
                if name not in manager.procedures:
                    return ProcedureResult(
                        procedure_id="",
                        name=name,
                        status="deleted",
                        error=f"Procedure '{name}' not found"
                    )
                
                # Store ID for response
                proc_id = manager.procedures[name].id
                
                # Delete procedure
                del manager.procedures[name]
                
                return ProcedureResult(
                    procedure_id=proc_id,
                    name=name,
                    status="deleted",
                    domain=None,
                    steps_count=None
                )
                
        @function_tool
        async def execute_procedure(
            ctx: RunContextWrapper,
            name: str,
            context: Optional[ExecutionContext] = None
        ) -> ExecutionResult:
            """
            Execute a procedure
            
            Args:
                name: Name of the procedure
                context: Optional execution context
                
            Returns:
                Execution results
            """
            with agents_trace(workflow_name="execute_procedure"):
                if name not in manager.procedures:
                    return ExecutionResult(
                        success=False,
                        results=[],
                        execution_time=0.0,
                        error=f"Procedure '{name}' not found"
                    )
                
                procedure = manager.procedures[name]
                
                # Determine execution mode based on proficiency
                conscious_execution = procedure.proficiency < 0.8
                
                # Execute procedure
                with agent_span("procedure_execution", 
                              {"procedure_name": name, "conscious_execution": conscious_execution}):
                    result = await manager.execute_procedure_steps(
                        procedure=procedure,
                        context=context or {},
                        conscious_execution=conscious_execution
                    )
                    
                # Add procedure information
                result["procedure_name"] = name
                result["procedure_id"] = procedure.id
                
                return ExecutionResult(**result)
    
        @function_tool
        async def execute_step(
            ctx: RunContextWrapper,
            step: ProcedureStep,
            context: ExecutionContext,
            minimal_monitoring: bool = False
        ) -> StepExecutionResult:
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
                func = manager.function_registry.get(func_name)
                
                if not func:
                    return StepExecutionResult(
                        success=False,
                        data=None,
                        execution_time=0.0,
                        error=f"Function {func_name} not registered"
                    )
                
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
                    
                    # Process result
                    if isinstance(result, dict):
                        success = "error" not in result
                        step_data = StepData(
                            result=result.get("result"),
                            output=result.get("output"),
                            state_change=result.get("state_change"),
                            metrics=result.get("metrics")
                        )
                        error = result.get("error")
                    else:
                        success = True
                        step_data = StepData(result=result)
                        error = None
                        
                except Exception as e:
                    logger.error(f"Error executing step {step['id']}: {str(e)}")
                    success = False
                    step_data = None
                    error = str(e)
                
                # Calculate execution time
                execution_time = (datetime.datetime.now() - step_start).total_seconds()
                
                return StepExecutionResult(
                    success=success,
                    data=step_data,
                    execution_time=execution_time,
                    error=error
                )
    
        @function_tool
        async def get_procedure_proficiency(
            ctx: RunContextWrapper,
            name: str
        ) -> ProficiencyInfo:
            """
            Get proficiency statistics for a procedure
            
            Args:
                name: Name of the procedure
                
            Returns:
                Proficiency statistics
            """
            with agents_trace("get_procedure_proficiency"):
                if name not in manager.procedures:
                    # Return default values with error indication
                    return ProficiencyInfo(
                        procedure_name=name,
                        procedure_id="",
                        proficiency=0.0,
                        level="novice",
                        execution_count=0,
                        success_rate=0.0,
                        average_execution_time=0.0,
                        is_chunked=False,
                        chunks_count=0,
                        domain="",
                        last_execution=None
                    )
                
                procedure = manager.procedures[name]
                
                # Calculate success rate
                success_rate = 0.0
                if procedure.execution_count > 0:
                    success_rate = procedure.successful_executions / procedure.execution_count
                    
                # Determine level based on proficiency
                level: Literal["novice", "intermediate", "proficient", "expert"] = "novice"
                if procedure.proficiency >= 0.9:
                    level = "expert"
                elif procedure.proficiency >= 0.7:
                    level = "proficient"
                elif procedure.proficiency >= 0.4:
                    level = "intermediate"
                    
                return ProficiencyInfo(
                    procedure_name=name,
                    procedure_id=procedure.id,
                    proficiency=procedure.proficiency,
                    level=level,
                    execution_count=procedure.execution_count,
                    success_rate=success_rate,
                    average_execution_time=procedure.average_execution_time,
                    is_chunked=procedure.is_chunked,
                    chunks_count=len(procedure.chunked_steps) if procedure.is_chunked else 0,
                    domain=procedure.domain,
                    last_execution=procedure.last_execution
                )
    
        @function_tool
        async def find_similar_procedures(
            ctx: RunContextWrapper,
            name: str,
            similarity_threshold: float = 0.6
        ) -> SimilarityResult:
            """
            Find procedures similar to the specified one
            
            Args:
                name: Name of the reference procedure
                similarity_threshold: Minimum similarity score (0-1)
                
            Returns:
                List of similar procedures with similarity scores
            """
            with agents_trace("find_similar_procedures"):
                if name not in manager.procedures:
                    return SimilarityResult(
                        reference_procedure=name,
                        similar_procedures=[],
                        count=0
                    )
                
                reference = manager.procedures[name]
                similar_procedures: List[SimilarProcedure] = []
                
                for other_name, other_proc in manager.procedures.items():
                    # Skip self comparison
                    if other_name == name:
                        continue
                        
                    # Calculate similarity
                    similarity = manager.calculate_procedure_similarity(reference, other_proc)
                    
                    if similarity >= similarity_threshold:
                        similar_procedures.append(SimilarProcedure(
                            name=other_name,
                            similarity=similarity,
                            domain=other_proc.domain,
                            steps_count=len(other_proc.steps),
                            proficiency=other_proc.proficiency
                        ))
                
                # Sort by similarity (highest first)
                similar_procedures.sort(key=lambda x: x["similarity"], reverse=True)
                
                return SimilarityResult(
                    reference_procedure=name,
                    similar_procedures=similar_procedures,
                    count=len(similar_procedures)
                )
    
        @function_tool
        async def transfer_procedure(
            ctx: RunContextWrapper,
            source_name: str,
            target_name: str,
            target_domain: str
        ) -> TransferResult:
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
                if source_name not in manager.procedures:
                    return TransferResult(
                        source_procedure=source_name,
                        target_procedure=target_name,
                        source_domain="",
                        target_domain=target_domain,
                        steps_transferred=0,
                        procedure_id=None,
                        success=False,
                        error=f"Source procedure '{source_name}' not found"
                    )
                
                source_procedure = manager.procedures[source_name]
                source_domain = source_procedure.domain
                
                # Map procedure steps to target domain
                mapped_steps = []
                
                for step in source_procedure.steps:
                    mapped_step = manager.map_step_to_domain(
                        step, source_domain, target_domain
                    )
                    
                    if mapped_step:
                        mapped_steps.append(mapped_step)
                    else:
                        # Fallback: use original step if mapping fails
                        mapped_steps.append(step.copy())
                
                # Create new procedure in target domain
                target_procedure = await add_procedure(
                    ctx,
                    name=target_name,
                    steps=mapped_steps,
                    description=f"Transferred from {source_name} ({source_domain} to {target_domain})",
                    domain=target_domain
                )
                
                # Record transfer
                manager.transfer_stats["total_transfers"] += 1
                
                # Update chunk mappings if needed
                if source_procedure.is_chunked and hasattr(source_procedure, "generalized_chunks"):
                    # Create corresponding chunks in target procedure
                    if target_name in manager.procedures:
                        target_proc = manager.procedures[target_name]
                        
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
                
                return TransferResult(
                    source_procedure=source_name,
                    target_procedure=target_name,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    steps_transferred=len(mapped_steps),
                    procedure_id=target_procedure.get("procedure_id"),
                    success=True,
                    error=None
                )
    
        @function_tool
        async def get_transfer_statistics(ctx: RunContextWrapper) -> TransferStatistics:
            """
            Get statistics about procedure transfers
            
            Returns:
                Transfer statistics
            """
            with agents_trace("get_transfer_statistics"):
                # Calculate average statistics
                avg_success = 0.0
                avg_practice = 0
                
                if manager.transfer_stats["total_transfers"] > 0:
                    avg_success = manager.transfer_stats["avg_success_level"]
                    avg_practice = manager.transfer_stats["avg_practice_needed"]
                
                # Get domain information
                domain_mappings = {}
                for mapping in manager.chunk_library.control_mappings:
                    source = mapping.source_domain
                    target = mapping.target_domain
                    
                    key = f"{source}→{target}"
                    if key not in domain_mappings:
                        domain_mappings[key] = 0
                        
                    domain_mappings[key] += 1
                
                return TransferStatistics(
                    total_transfers=manager.transfer_stats["total_transfers"],
                    successful_transfers=manager.transfer_stats["successful_transfers"],
                    success_rate=manager.transfer_stats["successful_transfers"] / max(1, manager.transfer_stats["total_transfers"]),
                    avg_success_level=avg_success,
                    avg_practice_needed=avg_practice,
                    domain_mappings=domain_mappings,
                    template_count=len(manager.chunk_library.templates) if hasattr(manager.chunk_library, "templates") else 0
                )
    
        @function_tool
        async def identify_chunking_opportunities(
            ctx: RunContextWrapper,
            procedure_name: str
        ) -> ChunkingAnalysis:
            """
            Identify opportunities for chunking in a procedure
            
            Args:
                procedure_name: Name of the procedure
                
            Returns:
                Chunking opportunities
            """
            with agents_trace("identify_chunking_opportunities"):
                if procedure_name not in manager.procedures:
                    return ChunkingAnalysis(
                        procedure_name=procedure_name,
                        can_chunk=False,
                        reason="Procedure not found",
                        potential_chunks=0,
                        chunks=[]
                    )
                
                procedure = manager.procedures[procedure_name]
                
                # Need at least 3 steps to consider chunking
                if len(procedure.steps) < 3:
                    return ChunkingAnalysis(
                        procedure_name=procedure_name,
                        can_chunk=False,
                        reason="Not enough steps for chunking (need at least 3)",
                        potential_chunks=0,
                        chunks=[]
                    )
                
                # Already chunked
                if procedure.is_chunked:
                    return ChunkingAnalysis(
                        procedure_name=procedure_name,
                        can_chunk=False,
                        reason="Procedure is already chunked",
                        potential_chunks=0,
                        chunks=[],
                        existing_chunks=len(procedure.chunked_steps)
                    )
                
                # Find sequences of steps that always succeed together
                chunks: List[List[str]] = []
                current_chunk = []
                opportunities: List[ChunkingOpportunity] = []
                
                for i in range(len(procedure.steps) - 1):
                    # Start a new potential chunk
                    if not current_chunk:
                        current_chunk = [procedure.steps[i]["id"]]
                    
                    # Check if next step is consistently executed after this one
                    co_occurrence = manager.calculate_step_co_occurrence(
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
                            opportunities.append(ChunkingOpportunity(
                                step_ids=current_chunk,
                                co_occurrence_rate=0.95  # Average for the chunk
                            ))
                        current_chunk = []
                
                # Add the last chunk if it exists
                if len(current_chunk) > 1:
                    chunks.append(current_chunk)
                    opportunities.append(ChunkingOpportunity(
                        step_ids=current_chunk,
                        co_occurrence_rate=0.95
                    ))
                
                return ChunkingAnalysis(
                    procedure_name=procedure_name,
                    can_chunk=len(chunks) > 0,
                    reason=None if len(chunks) > 0 else "No high co-occurrence sequences found",
                    potential_chunks=len(chunks),
                    chunks=chunks,
                    opportunities=opportunities
                )
    
        @function_tool
        async def apply_chunking(
            ctx: RunContextWrapper,
            procedure_name: str,
            chunks: Optional[List[List[str]]] = None
        ) -> ChunkingResult:
            """
            Apply chunking to a procedure
            
            Args:
                procedure_name: Name of the procedure
                chunks: Optional list of chunks (lists of step IDs)
                
            Returns:
                Status of chunking application
            """
            with agents_trace("apply_chunking"):
                if procedure_name not in manager.procedures:
                    return ChunkingResult(
                        procedure_name=procedure_name,
                        chunking_applied=False,
                        reason=f"Procedure '{procedure_name}' not found",
                        chunks_count=0,
                        chunks={}
                    )
                
                procedure = manager.procedures[procedure_name]
                
                # If chunks not provided, identify them
                if chunks is None:
                    chunking_result = await identify_chunking_opportunities(ctx, procedure_name)
                    
                    if not chunking_result["can_chunk"]:
                        return ChunkingResult(
                            procedure_name=procedure_name,
                            chunking_applied=False,
                            reason=chunking_result.get("reason", "No chunking opportunities found"),
                            chunks_count=0,
                            chunks={}
                        )
                        
                    chunks = chunking_result["chunks"]
                
                # Apply identified chunks
                manager._apply_chunking(procedure, chunks)
                
                return ChunkingResult(
                    procedure_name=procedure_name,
                    chunking_applied=True,
                    reason=None,
                    chunks_count=len(procedure.chunked_steps),
                    chunks=procedure.chunked_steps
                )
    
        @function_tool
        async def generalize_chunk_from_steps(
            ctx: RunContextWrapper,
            chunk_name: str,
            procedure_name: str,
            step_ids: List[str],
            domain: Optional[str] = None
        ) -> GeneralizationResult:
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
                if procedure_name not in manager.procedures:
                    return GeneralizationResult(
                        template_id="",
                        name=chunk_name,
                        domain=domain or "",
                        steps_count=0,
                        template_created=False,
                        error=f"Procedure '{procedure_name}' not found"
                    )
                
                procedure = manager.procedures[procedure_name]
                
                # Get steps
                steps = []
                for step_id in step_ids:
                    step = next((s for s in procedure.steps if s["id"] == step_id), None)
                    if step:
                        steps.append(step)
                
                if not steps:
                    return GeneralizationResult(
                        template_id="",
                        name=chunk_name,
                        domain=domain or procedure.domain,
                        steps_count=0,
                        template_created=False,
                        error="No valid steps found"
                    )
                
                # Use procedure domain if not specified
                if domain is None:
                    domain = procedure.domain
                
                # Create template
                template = manager.chunk_library.create_chunk_template_from_steps(
                    chunk_id=f"template_{chunk_name}",
                    name=chunk_name,
                    steps=steps,
                    domain=domain,
                    success_rate=0.9  # Start with high success rate
                )
                
                if not template:
                    return GeneralizationResult(
                        template_id="",
                        name=chunk_name,
                        domain=domain,
                        steps_count=len(steps),
                        template_created=False,
                        error="Failed to create template"
                    )
                
                # If procedure is chunked, try to link template to existing chunks
                if procedure.is_chunked:
                    # Find a chunk that matches the steps
                    for chunk_id, chunk_step_ids in procedure.chunked_steps.items():
                        if set(step_ids).issubset(set(chunk_step_ids)):
                            # This chunk includes all the requested steps
                            if not hasattr(procedure, "generalized_chunks"):
                                procedure.generalized_chunks = {}
                            procedure.generalized_chunks[chunk_id] = template.id
                            break
                
                return GeneralizationResult(
                    template_id=template.id,
                    name=chunk_name,
                    domain=domain,
                    steps_count=len(steps),
                    template_created=True,
                    error=None
                )
    
        @function_tool
        async def analyze_execution_history(
            ctx: RunContextWrapper,
            procedure_name: str
        ) -> ExecutionAnalysis:
            """
            Analyze execution history of a procedure for patterns
            
            Args:
                procedure_name: Name of the procedure
                
            Returns:
                Analysis of execution history
            """
            with agents_trace("analyze_execution_history"):
                if procedure_name not in manager.procedures:
                    return ExecutionAnalysis(
                        procedure_name=procedure_name,
                        executions=0,
                        success_rate=0.0,
                        avg_execution_time=0.0,
                        proficiency=0.0,
                        is_chunked=False,
                        chunks_count=0,
                        context_patterns=[],
                        success_patterns=[],
                        chunk_patterns=[],
                        refinement_opportunities=0,
                        analysis="Procedure not found"
                    )
                
                procedure = manager.procedures[procedure_name]
                
                # Skip if insufficient execution history
                if procedure.execution_count < 3:
                    return ExecutionAnalysis(
                        procedure_name=procedure_name,
                        executions=procedure.execution_count,
                        success_rate=procedure.successful_executions / max(1, procedure.execution_count),
                        avg_execution_time=procedure.average_execution_time,
                        proficiency=procedure.proficiency,
                        is_chunked=procedure.is_chunked,
                        chunks_count=len(procedure.chunked_steps) if procedure.is_chunked else 0,
                        context_patterns=[],
                        success_patterns=[],
                        chunk_patterns=[],
                        refinement_opportunities=0,
                        analysis="Insufficient execution history for analysis"
                    )
                
                # Analyze context history if available
                context_patterns: List[ContextPattern] = []
                success_patterns: List[SuccessPattern] = []
                
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
                                context_patterns.append(ContextPattern(
                                    key=key,
                                    value=values[0],
                                    occurrences=len(values),
                                    pattern_type="consistent_value"
                                ))
                            elif len(unique_values) <= len(values) / 2:
                                # Semi-consistent values
                                value_counts = Counter(str(v) for v in values)
                                most_common = value_counts.most_common(1)[0]
                                
                                context_patterns.append(ContextPattern(
                                    key=key,
                                    most_common_value=most_common[0],
                                    occurrence_rate=most_common[1] / len(values),
                                    pattern_type="common_value"
                                ))
                    
                    # Analyze successful vs. unsuccessful executions
                    successful_contexts = [ctx for ctx in procedure.context_history if ctx.get("result", False)]
                    unsuccessful_contexts = [ctx for ctx in procedure.context_history if not ctx.get("result", True)]
                    
                    if successful_contexts and unsuccessful_contexts:
                        # Find keys that differ between successful and unsuccessful executions
                        for key in context_keys:
                            # Get values for successful executions
                            success_values = [ctx.get(key) for ctx in successful_contexts if key in ctx]
                            if not success_values:
                                continue
                                
                            # Get values for unsuccessful executions
                            failure_values = [ctx.get(key) for ctx in unsuccessful_contexts if key in ctx]
                            if not failure_values:
                                continue
                            
                            # Check if values are consistently different
                            success_unique = set(str(v) for v in success_values)
                            failure_unique = set(str(v) for v in failure_values)
                            
                            # If no overlap, this might be a discriminating factor
                            if not success_unique.intersection(failure_unique):
                                success_patterns.append(SuccessPattern(
                                    key=key,
                                    success_values=list(success_unique),
                                    failure_values=list(failure_unique),
                                    pattern_type="success_factor"
                                ))
                
                # Analyze chunks if available
                chunk_patterns: List[ChunkInfo] = []
                if procedure.is_chunked:
                    for chunk_id, step_ids in procedure.chunked_steps.items():
                        chunk_patterns.append(ChunkInfo(
                            chunk_id=chunk_id,
                            step_count=len(step_ids),
                            has_template=chunk_id in getattr(procedure, "generalized_chunks", {}),
                            has_context_pattern=chunk_id in getattr(procedure, "chunk_contexts", {})
                        ))
                
                return ExecutionAnalysis(
                    procedure_name=procedure_name,
                    executions=procedure.execution_count,
                    success_rate=procedure.successful_executions / max(1, procedure.execution_count),
                    avg_execution_time=procedure.average_execution_time,
                    proficiency=procedure.proficiency,
                    is_chunked=procedure.is_chunked,
                    chunks_count=len(procedure.chunked_steps) if procedure.is_chunked else 0,
                    context_patterns=context_patterns,
                    success_patterns=success_patterns,
                    chunk_patterns=chunk_patterns,
                    refinement_opportunities=len(getattr(procedure, "refinement_opportunities", []))
                )
    
        @function_tool
        async def optimize_procedure_parameters(
            ctx: RunContextWrapper,
            procedure_name: str,
            iterations: int = 10
        ) -> OptimizationResult:
            """
            Optimize parameters for a procedure
            
            Args:
                procedure_name: Name of the procedure to optimize
                iterations: Number of optimization iterations
                
            Returns:
                Optimization results
            """
            with agents_trace("optimize_procedure_parameters"):
                if procedure_name not in manager.procedures:
                    return OptimizationResult(
                        procedure_name=procedure_name,
                        optimization_type="parameter_optimization",
                        iterations=iterations,
                        initial_score=None,
                        final_score=None,
                        improvement=None,
                        execution_time=0.0,
                        parameters_optimized=None,
                        status="failed",
                        error=f"Procedure '{procedure_name}' not found"
                    )
                
                procedure = manager.procedures[procedure_name]
                
                # Check if parameter optimizer exists
                if not hasattr(manager, "parameter_optimizer"):
                    return OptimizationResult(
                        procedure_name=procedure_name,
                        optimization_type="parameter_optimization",
                        iterations=iterations,
                        initial_score=None,
                        final_score=None,
                        improvement=None,
                        execution_time=0.0,
                        parameters_optimized=None,
                        status="failed",
                        error="Parameter optimizer not available"
                    )
                
                start_time = datetime.datetime.now()
                
                # Define objective function
                async def objective_function(test_procedure: Procedure) -> float:
                    # Create simulated context
                    test_context = ExecutionContext(
                        optimization_run=True,
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    
                    # Execute procedure
                    result = await manager.execute_procedure_steps(
                        procedure=test_procedure,
                        context=test_context,
                        conscious_execution=True
                    )
                    
                    # Calculate objective score (combination of success and speed)
                    success_score = 1.0 if result["success"] else 0.0
                    time_score = max(0.0, 1.0 - (result["execution_time"] / 10.0))  # Lower time is better
                    
                    # Combined score (success is more important)
                    return success_score * 0.7 + time_score * 0.3
                
                # Get initial score
                initial_score = await objective_function(procedure)
                
                # Run optimization
                optimization_result = await manager.parameter_optimizer.optimize_parameters(
                    procedure=procedure,
                    objective_function=objective_function,
                    iterations=iterations
                )
                
                # Get final score
                final_score = await objective_function(procedure)
                
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                return OptimizationResult(
                    procedure_name=procedure_name,
                    optimization_type="parameter_optimization",
                    iterations=iterations,
                    initial_score=initial_score,
                    final_score=final_score,
                    improvement=final_score - initial_score,
                    execution_time=execution_time,
                    parameters_optimized=optimization_result.get("optimized_parameters", {}),
                    status="completed",
                    error=None
                )
    
        @function_tool
        async def optimize_procedural_memory(ctx: RunContextWrapper) -> MemoryOptimizationResult:
            """
            Optimize procedural memory by consolidating and cleaning up
            
            Returns:
                Optimization results
            """
            with agents_trace("optimize_procedural_memory"):
                start_time = datetime.datetime.now()
                
                # Check if memory consolidator exists
                if hasattr(manager, "memory_consolidator"):
                    result = await manager.memory_consolidator.consolidate_procedural_memory()
                    return MemoryOptimizationResult(
                        procedures_cleaned=result.get("procedures_cleaned", 0),
                        procedures_consolidated=result.get("procedures_consolidated", 0),
                        chunks_generalized=result.get("chunks_generalized", 0),
                        templates_created=result.get("templates_created", 0),
                        execution_time=(datetime.datetime.now() - start_time).total_seconds(),
                        status="advanced_optimization_completed"
                    )
                else:
                    # Basic implementation
                    # Clean up old procedures
                    old_procedures = []
                    for name, procedure in manager.procedures.items():
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
                        if manager.procedures[name].proficiency < 0.7:
                            del manager.procedures[name]
                            procedures_cleaned += 1
                    
                    return MemoryOptimizationResult(
                        procedures_cleaned=procedures_cleaned,
                        execution_time=(datetime.datetime.now() - start_time).total_seconds(),
                        status="basic_optimization_completed"
                    )
    
        @function_tool
        async def execute_hierarchical_procedure(
            ctx: RunContextWrapper,
            name: str,
            context: Optional[ExecutionContext] = None
        ) -> ExecutionResult:
            """
            Execute a hierarchical procedure
            
            Args:
                name: Name of the procedure
                context: Execution context
                
            Returns:
                Execution results
            """
            # Placeholder implementation since the EnhancedProceduralMemoryManager handles this
            return ExecutionResult(
                success=False,
                results=[],
                execution_time=0.0,
                error="Hierarchical procedure execution requires the EnhancedProceduralMemoryManager"
            )
    
        @function_tool
        async def execute_temporal_procedure(
            ctx: RunContextWrapper,
            name: str,
            context: Optional[ExecutionContext] = None
        ) -> ExecutionResult:
            """
            Execute a procedure with temporal constraints
            
            Args:
                name: Name of the procedure
                context: Execution context
                
            Returns:
                Execution results
            """
            # Placeholder implementation since the EnhancedProceduralMemoryManager handles this
            return ExecutionResult(
                success=False,
                results=[],
                execution_time=0.0,
                error="Temporal procedure execution requires the EnhancedProceduralMemoryManager"
            )
    
        @function_tool
        async def execute_graph_procedure(
            ctx: RunContextWrapper,
            procedure_name: str,
            context: Optional[ExecutionContext] = None,
            goal: Optional[GoalState] = None
        ) -> ExecutionResult:
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
            return ExecutionResult(
                success=False,
                results=[],
                execution_time=0.0,
                error="Graph-based procedure execution requires the EnhancedProceduralMemoryManager"
            )
        
        # ============================================================================
        # Store references to tools for use in agents
        # ============================================================================
        
        self.list_procedures = list_procedures
        self.add_procedure = add_procedure
        self.get_procedure = get_procedure
        self.update_procedure = update_procedure
        self.delete_procedure = delete_procedure
        self.execute_procedure = execute_procedure
        self.execute_step = execute_step
        self.get_procedure_proficiency = get_procedure_proficiency
        self.find_similar_procedures = find_similar_procedures
        self.transfer_procedure = transfer_procedure
        self.get_transfer_statistics = get_transfer_statistics
        self.identify_chunking_opportunities = identify_chunking_opportunities
        self.apply_chunking = apply_chunking
        self.generalize_chunk_from_steps = generalize_chunk_from_steps
        self.analyze_execution_history = analyze_execution_history
        self.optimize_procedure_parameters = optimize_procedure_parameters
        self.optimize_procedural_memory = optimize_procedural_memory
        self.execute_hierarchical_procedure = execute_hierarchical_procedure
        self.execute_temporal_procedure = execute_temporal_procedure
        self.execute_graph_procedure = execute_graph_procedure
        
        # Register common function tools for all agents
        common_tools = [
            list_procedures,
            get_procedure,
            get_procedure_proficiency,
        ]
        
        # Manager agent specific tools
        manager_tools = common_tools + [
            add_procedure,
            update_procedure,
            delete_procedure,
            find_similar_procedures,
            transfer_procedure,
            get_transfer_statistics,
        ]
        
        # Execution agent specific tools
        execution_tools = common_tools + [
            execute_procedure,
            execute_step,
            execute_hierarchical_procedure,
            execute_temporal_procedure,
            execute_graph_procedure,
        ]
        
        # Analysis agent specific tools
        analysis_tools = common_tools + [
            analyze_execution_history,
            identify_chunking_opportunities,
            apply_chunking,
            generalize_chunk_from_steps,
            optimize_procedure_parameters,
            optimize_procedural_memory,
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
    
def _initialize_causal_model(self):
    """Initialize causal model with common error patterns"""
    # Define common error causes for different error types
    self.causal_model.causes = {
        "execution_failure": [
            {
                "cause": "invalid_parameters",
                "description": "Invalid parameters provided to function",
                "probability": 0.6,
                "context_factors": {
                    "parameter_type_mismatch": 0.4,
                    "missing_required_parameter": 0.3,
                    "out_of_range_value": 0.3
                }
            },
            {
                "cause": "missing_precondition",
                "description": "Required precondition not met",
                "probability": 0.4,
                "context_factors": {
                    "state_not_initialized": 0.5,
                    "dependency_not_available": 0.3,
                    "permission_denied": 0.2
                }
            },
            {
                "cause": "function_not_found",
                "description": "Function not registered or available",
                "probability": 0.3,
                "context_factors": {
                    "typo_in_function_name": 0.4,
                    "function_not_imported": 0.6
                }
            },
            {
                "cause": "state_corruption",
                "description": "System state is corrupted or inconsistent",
                "probability": 0.2,
                "context_factors": {
                    "concurrent_modification": 0.3,
                    "incomplete_previous_operation": 0.7
                }
            }
        ],
        "timeout": [
            {
                "cause": "slow_execution",
                "description": "Operation taking too long to complete",
                "probability": 0.5,
                "context_factors": {
                    "large_data_volume": 0.4,
                    "complex_computation": 0.3,
                    "inefficient_algorithm": 0.3
                }
            },
            {
                "cause": "resource_contention",
                "description": "Resources needed are being used by another process",
                "probability": 0.3,
                "context_factors": {
                    "cpu_overload": 0.3,
                    "memory_pressure": 0.3,
                    "io_bottleneck": 0.4
                }
            },
            {
                "cause": "network_delay",
                "description": "Network latency or connectivity issues",
                "probability": 0.2,
                "context_factors": {
                    "high_latency": 0.5,
                    "packet_loss": 0.3,
                    "bandwidth_limitation": 0.2
                }
            },
            {
                "cause": "deadlock",
                "description": "Circular dependency causing infinite wait",
                "probability": 0.1,
                "context_factors": {
                    "resource_ordering": 0.6,
                    "synchronization_issue": 0.4
                }
            }
        ],
        "memory_error": [
            {
                "cause": "out_of_memory",
                "description": "Insufficient memory available",
                "probability": 0.6,
                "context_factors": {
                    "memory_leak": 0.3,
                    "large_allocation": 0.4,
                    "fragmentation": 0.3
                }
            },
            {
                "cause": "null_reference",
                "description": "Attempting to access null/undefined reference",
                "probability": 0.3,
                "context_factors": {
                    "uninitialized_variable": 0.5,
                    "premature_cleanup": 0.3,
                    "race_condition": 0.2
                }
            },
            {
                "cause": "buffer_overflow",
                "description": "Writing beyond allocated memory",
                "probability": 0.1,
                "context_factors": {
                    "incorrect_size_calculation": 0.6,
                    "unbounded_input": 0.4
                }
            }
        ],
        "permission_error": [
            {
                "cause": "insufficient_privileges",
                "description": "User lacks required permissions",
                "probability": 0.7,
                "context_factors": {
                    "missing_role": 0.4,
                    "expired_credentials": 0.3,
                    "wrong_context": 0.3
                }
            },
            {
                "cause": "resource_locked",
                "description": "Resource is locked by another process",
                "probability": 0.2,
                "context_factors": {
                    "exclusive_access": 0.6,
                    "file_in_use": 0.4
                }
            },
            {
                "cause": "security_policy",
                "description": "Security policy prevents operation",
                "probability": 0.1,
                "context_factors": {
                    "firewall_rule": 0.3,
                    "sandbox_restriction": 0.7
                }
            }
        ],
        "data_error": [
            {
                "cause": "corrupt_data",
                "description": "Data is corrupted or malformed",
                "probability": 0.4,
                "context_factors": {
                    "encoding_issue": 0.3,
                    "partial_write": 0.4,
                    "format_mismatch": 0.3
                }
            },
            {
                "cause": "missing_data",
                "description": "Required data is not available",
                "probability": 0.3,
                "context_factors": {
                    "deleted_file": 0.4,
                    "network_failure": 0.3,
                    "cache_miss": 0.3
                }
            },
            {
                "cause": "schema_mismatch",
                "description": "Data doesn't match expected schema",
                "probability": 0.2,
                "context_factors": {
                    "version_mismatch": 0.5,
                    "missing_fields": 0.3,
                    "type_mismatch": 0.2
                }
            },
            {
                "cause": "data_race",
                "description": "Concurrent access causing inconsistency",
                "probability": 0.1,
                "context_factors": {
                    "missing_synchronization": 0.7,
                    "incorrect_locking": 0.3
                }
            }
        ],
        "configuration_error": [
            {
                "cause": "missing_configuration",
                "description": "Required configuration not set",
                "probability": 0.5,
                "context_factors": {
                    "environment_variable": 0.4,
                    "config_file": 0.4,
                    "default_value": 0.2
                }
            },
            {
                "cause": "invalid_configuration",
                "description": "Configuration values are invalid",
                "probability": 0.3,
                "context_factors": {
                    "syntax_error": 0.3,
                    "value_out_of_range": 0.4,
                    "incompatible_settings": 0.3
                }
            },
            {
                "cause": "configuration_conflict",
                "description": "Multiple configurations conflict",
                "probability": 0.2,
                "context_factors": {
                    "override_conflict": 0.5,
                    "dependency_mismatch": 0.5
                }
            }
        ]
    }
    
    # Define common interventions for each cause
    self.causal_model.interventions = {
        # Execution failure interventions
        "invalid_parameters": [
            {
                "type": "modify_parameters",
                "description": "Modify parameters to valid values",
                "effectiveness": 0.8,
                "steps": [
                    "Identify parameter constraints",
                    "Validate current values",
                    "Apply corrections",
                    "Retry operation"
                ]
            },
            {
                "type": "check_documentation",
                "description": "Check documentation for correct parameter format",
                "effectiveness": 0.6,
                "steps": [
                    "Locate function documentation",
                    "Review parameter specifications",
                    "Compare with current usage"
                ]
            },
            {
                "type": "use_defaults",
                "description": "Use default parameter values",
                "effectiveness": 0.7,
                "steps": [
                    "Identify optional parameters",
                    "Remove or replace with defaults",
                    "Retry operation"
                ]
            }
        ],
        "missing_precondition": [
            {
                "type": "initialize_state",
                "description": "Initialize required state before execution",
                "effectiveness": 0.9,
                "steps": [
                    "Identify missing preconditions",
                    "Execute initialization sequence",
                    "Verify state",
                    "Retry operation"
                ]
            },
            {
                "type": "wait_for_condition",
                "description": "Wait for precondition to be met",
                "effectiveness": 0.7,
                "steps": [
                    "Set up condition monitoring",
                    "Implement timeout",
                    "Poll for condition",
                    "Proceed when ready"
                ]
            },
            {
                "type": "force_precondition",
                "description": "Force precondition to be true",
                "effectiveness": 0.5,
                "steps": [
                    "Override safety checks",
                    "Set required state",
                    "Accept risks"
                ]
            }
        ],
        "function_not_found": [
            {
                "type": "search_registry",
                "description": "Search function registry for similar names",
                "effectiveness": 0.8,
                "steps": [
                    "Get all registered functions",
                    "Find similar names",
                    "Suggest corrections"
                ]
            },
            {
                "type": "lazy_import",
                "description": "Attempt to import missing function",
                "effectiveness": 0.7,
                "steps": [
                    "Identify function source",
                    "Dynamic import",
                    "Register function"
                ]
            },
            {
                "type": "use_alternative",
                "description": "Use alternative function with similar behavior",
                "effectiveness": 0.6,
                "steps": [
                    "Find similar functions",
                    "Adapt parameters",
                    "Execute alternative"
                ]
            }
        ],
        "state_corruption": [
            {
                "type": "reset_state",
                "description": "Reset to known good state",
                "effectiveness": 0.9,
                "steps": [
                    "Save current state",
                    "Load default state",
                    "Reinitialize components"
                ]
            },
            {
                "type": "repair_state",
                "description": "Attempt to repair corrupted state",
                "effectiveness": 0.6,
                "steps": [
                    "Identify corruption",
                    "Apply fixes",
                    "Validate state"
                ]
            },
            {
                "type": "isolate_corruption",
                "description": "Isolate and work around corruption",
                "effectiveness": 0.7,
                "steps": [
                    "Identify affected components",
                    "Quarantine corrupted data",
                    "Use alternative paths"
                ]
            }
        ],
        
        # Timeout interventions
        "slow_execution": [
            {
                "type": "optimize_algorithm",
                "description": "Use more efficient algorithm",
                "effectiveness": 0.9,
                "steps": [
                    "Profile current execution",
                    "Identify bottlenecks",
                    "Apply optimizations"
                ]
            },
            {
                "type": "reduce_data",
                "description": "Process smaller data chunks",
                "effectiveness": 0.8,
                "steps": [
                    "Split data into chunks",
                    "Process incrementally",
                    "Aggregate results"
                ]
            },
            {
                "type": "increase_timeout",
                "description": "Increase timeout threshold",
                "effectiveness": 0.5,
                "steps": [
                    "Calculate required time",
                    "Set new timeout",
                    "Monitor progress"
                ]
            }
        ],
        "resource_contention": [
            {
                "type": "wait_and_retry",
                "description": "Wait for resources to become available",
                "effectiveness": 0.7,
                "steps": [
                    "Implement backoff strategy",
                    "Monitor resource availability",
                    "Retry when available"
                ]
            },
            {
                "type": "request_priority",
                "description": "Request higher priority access",
                "effectiveness": 0.6,
                "steps": [
                    "Elevate process priority",
                    "Request resource reservation",
                    "Execute with priority"
                ]
            },
            {
                "type": "use_alternative_resource",
                "description": "Use alternative resources",
                "effectiveness": 0.8,
                "steps": [
                    "Identify alternatives",
                    "Adapt to different resource",
                    "Execute with alternative"
                ]
            }
        ],
        "network_delay": [
            {
                "type": "retry_with_backoff",
                "description": "Retry with exponential backoff",
                "effectiveness": 0.8,
                "steps": [
                    "Implement backoff algorithm",
                    "Track retry attempts",
                    "Increase delay between retries"
                ]
            },
            {
                "type": "use_cache",
                "description": "Use cached data if available",
                "effectiveness": 0.9,
                "steps": [
                    "Check cache validity",
                    "Return cached data",
                    "Update cache asynchronously"
                ]
            },
            {
                "type": "switch_endpoint",
                "description": "Switch to different network endpoint",
                "effectiveness": 0.7,
                "steps": [
                    "Identify alternative endpoints",
                    "Test connectivity",
                    "Use fastest endpoint"
                ]
            }
        ],
        "deadlock": [
            {
                "type": "timeout_and_restart",
                "description": "Timeout and restart operation",
                "effectiveness": 0.8,
                "steps": [
                    "Detect deadlock condition",
                    "Force timeout",
                    "Clean up resources",
                    "Restart with ordering"
                ]
            },
            {
                "type": "resource_ordering",
                "description": "Enforce consistent resource ordering",
                "effectiveness": 0.9,
                "steps": [
                    "Define resource hierarchy",
                    "Always acquire in order",
                    "Release in reverse order"
                ]
            }
        ],
        
        # Memory error interventions
        "out_of_memory": [
            {
                "type": "free_memory",
                "description": "Free unused memory",
                "effectiveness": 0.8,
                "steps": [
                    "Run garbage collection",
                    "Clear caches",
                    "Release unused resources"
                ]
            },
            {
                "type": "reduce_memory_usage",
                "description": "Reduce memory footprint",
                "effectiveness": 0.7,
                "steps": [
                    "Use smaller data structures",
                    "Process in smaller batches",
                    "Stream instead of loading all"
                ]
            },
            {
                "type": "increase_memory_limit",
                "description": "Request more memory",
                "effectiveness": 0.6,
                "steps": [
                    "Check system limits",
                    "Request increase",
                    "Monitor usage"
                ]
            }
        ],
        "null_reference": [
            {
                "type": "null_check",
                "description": "Add null checks before access",
                "effectiveness": 0.9,
                "steps": [
                    "Identify access points",
                    "Add validation",
                    "Handle null cases"
                ]
            },
            {
                "type": "initialize_reference",
                "description": "Initialize reference before use",
                "effectiveness": 0.8,
                "steps": [
                    "Find initialization point",
                    "Create default instance",
                    "Ensure proper lifecycle"
                ]
            }
        ],
        
        # Permission error interventions
        "insufficient_privileges": [
            {
                "type": "request_permission",
                "description": "Request required permissions",
                "effectiveness": 0.8,
                "steps": [
                    "Identify required permissions",
                    "Request from user/admin",
                    "Retry with permissions"
                ]
            },
            {
                "type": "elevate_privileges",
                "description": "Temporarily elevate privileges",
                "effectiveness": 0.7,
                "steps": [
                    "Request elevation",
                    "Execute with privileges",
                    "Drop privileges after"
                ]
            },
            {
                "type": "use_service_account",
                "description": "Use service account with permissions",
                "effectiveness": 0.9,
                "steps": [
                    "Switch to service context",
                    "Execute operation",
                    "Return to user context"
                ]
            }
        ],
        
        # Data error interventions
        "corrupt_data": [
            {
                "type": "data_recovery",
                "description": "Attempt to recover corrupted data",
                "effectiveness": 0.6,
                "steps": [
                    "Identify corruption pattern",
                    "Apply recovery algorithms",
                    "Validate recovered data"
                ]
            },
            {
                "type": "use_backup",
                "description": "Restore from backup",
                "effectiveness": 0.9,
                "steps": [
                    "Locate recent backup",
                    "Verify backup integrity",
                    "Restore data"
                ]
            },
            {
                "type": "regenerate_data",
                "description": "Regenerate data from source",
                "effectiveness": 0.8,
                "steps": [
                    "Identify data source",
                    "Reprocess from source",
                    "Validate output"
                ]
            }
        ],
        "schema_mismatch": [
            {
                "type": "schema_migration",
                "description": "Migrate data to new schema",
                "effectiveness": 0.9,
                "steps": [
                    "Detect schema version",
                    "Apply migrations",
                    "Validate result"
                ]
            },
            {
                "type": "schema_adaptation",
                "description": "Adapt to handle multiple schemas",
                "effectiveness": 0.8,
                "steps": [
                    "Implement version detection",
                    "Create adapters",
                    "Process accordingly"
                ]
            }
        ],
        
        # Configuration error interventions
        "missing_configuration": [
            {
                "type": "use_defaults",
                "description": "Use default configuration values",
                "effectiveness": 0.7,
                "steps": [
                    "Load default config",
                    "Merge with partial config",
                    "Validate completeness"
                ]
            },
            {
                "type": "prompt_configuration",
                "description": "Prompt user for configuration",
                "effectiveness": 0.9,
                "steps": [
                    "Identify missing values",
                    "Create configuration wizard",
                    "Save configuration"
                ]
            },
            {
                "type": "auto_configure",
                "description": "Automatically detect configuration",
                "effectiveness": 0.8,
                "steps": [
                    "Scan environment",
                    "Detect settings",
                    "Apply configuration"
                ]
            }
        ],
        "invalid_configuration": [
            {
                "type": "validate_and_fix",
                "description": "Validate and fix configuration",
                "effectiveness": 0.8,
                "steps": [
                    "Run validation rules",
                    "Identify issues",
                    "Apply corrections"
                ]
            },
            {
                "type": "reset_to_defaults",
                "description": "Reset to default configuration",
                "effectiveness": 0.9,
                "steps": [
                    "Backup current config",
                    "Load defaults",
                    "Restart with defaults"
                ]
            }
        ]
    }
    
    # Define recovery strategies that combine multiple interventions
    self.causal_model.recovery_strategies = {
        "execution_failure": [
            {
                "name": "standard_recovery",
                "description": "Standard recovery procedure for execution failures",
                "interventions": ["modify_parameters", "initialize_state", "retry_with_backoff"],
                "max_attempts": 3
            },
            {
                "name": "aggressive_recovery",
                "description": "More aggressive recovery with state reset",
                "interventions": ["reset_state", "use_defaults", "force_precondition"],
                "max_attempts": 2
            }
        ],
        "timeout": [
            {
                "name": "performance_optimization",
                "description": "Optimize performance to avoid timeout",
                "interventions": ["optimize_algorithm", "reduce_data", "use_cache"],
                "max_attempts": 2
            },
            {
                "name": "resource_reallocation",
                "description": "Reallocate resources to avoid contention",
                "interventions": ["wait_and_retry", "use_alternative_resource", "request_priority"],
                "max_attempts": 3
            }
        ]
    }


    def _initialize_control_mappings(self):
        """Initialize common control mappings between domains"""
        
        # ===========================================================================
        # Gaming Console Mappings
        # ===========================================================================
        
        # PlayStation to Xbox mappings
        playstation_xbox_mappings = [
            # Face buttons
            ("X", "A", "primary_action"),
            ("Circle", "B", "secondary_action"),
            ("Square", "X", "tertiary_action"),
            ("Triangle", "Y", "quaternary_action"),
            
            # Shoulder buttons
            ("L1", "LB", "left_bumper"),
            ("R1", "RB", "right_bumper"),
            ("L2", "LT", "left_trigger"),
            ("R2", "RT", "right_trigger"),
            
            # Control buttons
            ("Options", "Menu", "pause_menu"),
            ("Share", "View", "share_view"),
            ("PS", "Xbox", "home_button"),
            ("Touchpad", "View", "special_function"),
            
            # D-Pad
            ("D-Up", "D-Up", "dpad_up"),
            ("D-Down", "D-Down", "dpad_down"),
            ("D-Left", "D-Left", "dpad_left"),
            ("D-Right", "D-Right", "dpad_right"),
            
            # Analog sticks
            ("L3", "LS", "left_stick_press"),
            ("R3", "RS", "right_stick_press"),
            ("Left Stick", "Left Stick", "left_stick_move"),
            ("Right Stick", "Right Stick", "right_stick_move")
        ]
        
        for ps_control, xbox_control, action_type in playstation_xbox_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="playstation",
                target_domain="xbox",
                action_type=action_type,
                source_control=ps_control,
                target_control=xbox_control
            ))
            # Add reverse mapping
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="xbox",
                target_domain="playstation",
                action_type=action_type,
                source_control=xbox_control,
                target_control=ps_control
            ))
        
        # Nintendo Switch mappings to PlayStation
        nintendo_playstation_mappings = [
            # Face buttons (Nintendo has different layout)
            ("A", "Circle", "primary_action"),
            ("B", "X", "secondary_action"),
            ("X", "Triangle", "tertiary_action"),
            ("Y", "Square", "quaternary_action"),
            
            # Shoulder buttons
            ("L", "L1", "left_bumper"),
            ("R", "R1", "right_bumper"),
            ("ZL", "L2", "left_trigger"),
            ("ZR", "R2", "right_trigger"),
            
            # Control buttons
            ("+", "Options", "pause_menu"),
            ("-", "Share", "share_view"),
            ("Home", "PS", "home_button"),
            ("Capture", "Share", "capture_button"),
            
            # Sticks
            ("L-Stick", "L3", "left_stick_press"),
            ("R-Stick", "R3", "right_stick_press")
        ]
        
        for nintendo_control, ps_control, action_type in nintendo_playstation_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="nintendo_switch",
                target_domain="playstation",
                action_type=action_type,
                source_control=nintendo_control,
                target_control=ps_control
            ))
        
        # Nintendo Switch to Xbox mappings
        nintendo_xbox_mappings = [
            ("A", "B", "primary_action"),
            ("B", "A", "secondary_action"),
            ("X", "Y", "tertiary_action"),
            ("Y", "X", "quaternary_action"),
            ("L", "LB", "left_bumper"),
            ("R", "RB", "right_bumper"),
            ("ZL", "LT", "left_trigger"),
            ("ZR", "RT", "right_trigger"),
            ("+", "Menu", "pause_menu"),
            ("-", "View", "share_view"),
            ("Home", "Xbox", "home_button")
        ]
        
        for nintendo_control, xbox_control, action_type in nintendo_xbox_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="nintendo_switch",
                target_domain="xbox",
                action_type=action_type,
                source_control=nintendo_control,
                target_control=xbox_control
            ))
        
        # ===========================================================================
        # PC Gaming Mappings (Keyboard/Mouse to Controller)
        # ===========================================================================
        
        pc_to_controller_mappings = [
            # Movement
            ("W", "Left Stick Up", "move_forward"),
            ("S", "Left Stick Down", "move_backward"),
            ("A", "Left Stick Left", "move_left"),
            ("D", "Left Stick Right", "move_right"),
            
            # Camera
            ("Mouse Move", "Right Stick", "camera_control"),
            ("Mouse Left", "R2", "primary_fire"),
            ("Mouse Right", "L2", "aim_down_sights"),
            
            # Actions
            ("Space", "X", "jump"),
            ("Shift", "Circle", "sprint"),
            ("Ctrl", "Square", "crouch"),
            ("E", "Triangle", "interact"),
            ("R", "Square", "reload"),
            ("Tab", "Touchpad", "inventory"),
            ("Esc", "Options", "pause_menu"),
            
            # Numbers
            ("1", "D-Left", "weapon_1"),
            ("2", "D-Up", "weapon_2"),
            ("3", "D-Right", "weapon_3"),
            ("4", "D-Down", "weapon_4")
        ]
        
        for pc_control, ps_control, action_type in pc_to_controller_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="pc_gaming",
                target_domain="playstation",
                action_type=action_type,
                source_control=pc_control,
                target_control=ps_control
            ))
        
        # ===========================================================================
        # Mobile Platform Mappings
        # ===========================================================================
        
        # iOS to Android mappings
        ios_android_mappings = [
            # Gestures
            ("Tap", "Tap", "select"),
            ("Long Press", "Long Press", "context_menu"),
            ("Swipe Up", "Swipe Up", "scroll_up"),
            ("Swipe Down", "Swipe Down", "scroll_down"),
            ("Swipe Left", "Swipe Left", "navigate_back"),
            ("Swipe Right", "Swipe Right", "navigate_forward"),
            ("Pinch In", "Pinch In", "zoom_out"),
            ("Pinch Out", "Pinch Out", "zoom_in"),
            
            # System buttons
            ("Home Button", "Home Button", "go_home"),
            ("Volume Up", "Volume Up", "increase_volume"),
            ("Volume Down", "Volume Down", "decrease_volume"),
            ("Power Button", "Power Button", "power_control"),
            
            # Navigation
            ("Back Swipe", "Back Button", "navigate_back"),
            ("Control Center", "Quick Settings", "system_settings"),
            ("App Switcher", "Recent Apps", "multitasking")
        ]
        
        for ios_control, android_control, action_type in ios_android_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="ios",
                target_domain="android",
                action_type=action_type,
                source_control=ios_control,
                target_control=android_control
            ))
            # Add reverse mapping
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="android",
                target_domain="ios",
                action_type=action_type,
                source_control=android_control,
                target_control=ios_control
            ))
        
        # ===========================================================================
        # UI Framework Mappings
        # ===========================================================================
        
        # React to Vue mappings
        react_vue_mappings = [
            ("onClick", "@click", "click_handler"),
            ("onChange", "@change", "change_handler"),
            ("onSubmit", "@submit", "submit_handler"),
            ("onFocus", "@focus", "focus_handler"),
            ("onBlur", "@blur", "blur_handler"),
            ("onKeyDown", "@keydown", "keydown_handler"),
            ("onKeyUp", "@keyup", "keyup_handler"),
            ("onMouseEnter", "@mouseenter", "mouse_enter"),
            ("onMouseLeave", "@mouseleave", "mouse_leave"),
            ("className", "class", "css_class"),
            ("htmlFor", "for", "label_for"),
            ("key", "key", "list_key"),
            ("ref", "ref", "element_reference")
        ]
        
        for react_attr, vue_attr, action_type in react_vue_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="react",
                target_domain="vue",
                action_type=action_type,
                source_control=react_attr,
                target_control=vue_attr
            ))
        
        # ===========================================================================
        # Desktop Application Mappings
        # ===========================================================================
        
        # Windows to macOS mappings
        windows_macos_mappings = [
            # Keyboard shortcuts
            ("Ctrl+C", "Cmd+C", "copy"),
            ("Ctrl+V", "Cmd+V", "paste"),
            ("Ctrl+X", "Cmd+X", "cut"),
            ("Ctrl+Z", "Cmd+Z", "undo"),
            ("Ctrl+Y", "Cmd+Shift+Z", "redo"),
            ("Ctrl+S", "Cmd+S", "save"),
            ("Ctrl+O", "Cmd+O", "open"),
            ("Ctrl+N", "Cmd+N", "new"),
            ("Ctrl+P", "Cmd+P", "print"),
            ("Ctrl+F", "Cmd+F", "find"),
            ("Ctrl+A", "Cmd+A", "select_all"),
            ("Alt+Tab", "Cmd+Tab", "switch_app"),
            ("Ctrl+Shift+Esc", "Cmd+Option+Esc", "task_manager"),
            ("F5", "Cmd+R", "refresh"),
            ("Delete", "Delete", "delete_forward"),
            ("Backspace", "Delete", "delete_backward"),
            
            # Mouse actions
            ("Right Click", "Control+Click", "context_menu"),
            ("Middle Click", "Cmd+Click", "open_new_tab"),
            ("Ctrl+Scroll", "Pinch", "zoom_control")
        ]
        
        for windows_control, macos_control, action_type in windows_macos_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="windows",
                target_domain="macos",
                action_type=action_type,
                source_control=windows_control,
                target_control=macos_control
            ))
            # Add reverse mapping
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="macos",
                target_domain="windows",
                action_type=action_type,
                source_control=macos_control,
                target_control=windows_control
            ))
        
        # Linux to Windows mappings
        linux_windows_mappings = [
            ("Super", "Win", "system_key"),
            ("Alt+F2", "Win+R", "run_command"),
            ("Ctrl+Alt+T", "Win+Cmd", "terminal"),
            ("Ctrl+Alt+L", "Win+L", "lock_screen"),
            ("Alt+F4", "Alt+F4", "close_window"),
            ("Ctrl+Alt+Delete", "Ctrl+Alt+Delete", "system_interrupt")
        ]
        
        for linux_control, windows_control, action_type in linux_windows_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="linux",
                target_domain="windows",
                action_type=action_type,
                source_control=linux_control,
                target_control=windows_control
            ))
        
        # ===========================================================================
        # IDE/Editor Mappings
        # ===========================================================================
        
        # VS Code to IntelliJ mappings
        vscode_intellij_mappings = [
            ("Ctrl+Shift+P", "Ctrl+Shift+A", "command_palette"),
            ("Ctrl+P", "Ctrl+Shift+N", "quick_open"),
            ("Ctrl+Shift+F", "Ctrl+Shift+F", "search_in_files"),
            ("F12", "Ctrl+B", "go_to_definition"),
            ("Alt+F12", "Ctrl+Shift+I", "peek_definition"),
            ("Shift+F12", "Alt+F7", "find_usages"),
            ("F2", "Shift+F6", "rename"),
            ("Ctrl+Space", "Ctrl+Space", "autocomplete"),
            ("Ctrl+/", "Ctrl+/", "toggle_comment"),
            ("Alt+Up", "Ctrl+Shift+Up", "move_line_up"),
            ("Alt+Down", "Ctrl+Shift+Down", "move_line_down"),
            ("Ctrl+D", "Ctrl+D", "duplicate_line"),
            ("Ctrl+Shift+K", "Ctrl+Y", "delete_line")
        ]
        
        for vscode_shortcut, intellij_shortcut, action_type in vscode_intellij_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="vscode",
                target_domain="intellij",
                action_type=action_type,
                source_control=vscode_shortcut,
                target_control=intellij_shortcut
            ))
        
        # ===========================================================================
        # 3D Software Mappings
        # ===========================================================================
        
        # Blender to Maya mappings
        blender_maya_mappings = [
            # Navigation
            ("Middle Mouse", "Alt+Middle Mouse", "pan_view"),
            ("Scroll", "Alt+Right Mouse", "zoom_view"),
            ("Shift+Middle Mouse", "Alt+Left Mouse", "rotate_view"),
            
            # Selection
            ("Right Click", "Left Click", "select_object"),
            ("Shift+Right Click", "Shift+Left Click", "add_to_selection"),
            ("A", "Ctrl+A", "select_all"),
            ("Alt+A", "Ctrl+Shift+A", "deselect_all"),
            
            # Transform
            ("G", "W", "move_tool"),
            ("R", "E", "rotate_tool"),
            ("S", "R", "scale_tool"),
            
            # Modes
            ("Tab", "F8", "toggle_mode"),
            ("Ctrl+Tab", "F9", "mode_menu")
        ]
        
        for blender_control, maya_control, action_type in blender_maya_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="blender",
                target_domain="maya",
                action_type=action_type,
                source_control=blender_control,
                target_control=maya_control
            ))
        
        # ===========================================================================
        # Music Software Mappings
        # ===========================================================================
        
        # Ableton to Logic Pro mappings
        ableton_logic_mappings = [
            ("Space", "Space", "play_pause"),
            ("Tab", "Tab", "toggle_view"),
            ("Ctrl+T", "Cmd+T", "new_track"),
            ("Ctrl+Shift+T", "Cmd+Shift+T", "new_audio_track"),
            ("Ctrl+D", "Cmd+D", "duplicate"),
            ("Ctrl+L", "L", "loop_toggle"),
            ("Ctrl+Shift+M", "Cmd+Shift+M", "new_midi_track"),
            ("Ctrl+E", "Cmd+E", "export_audio"),
            ("Ctrl+R", "R", "record"),
            ("Ctrl+Z", "Cmd+Z", "undo"),
            ("Ctrl+Y", "Cmd+Shift+Z", "redo")
        ]
        
        for ableton_control, logic_control, action_type in ableton_logic_mappings:
            self.chunk_library.add_control_mapping(ControlMapping(
                source_domain="ableton",
                target_domain="logic_pro",
                action_type=action_type,
                source_control=ableton_control,
                target_control=logic_control
            ))
        
        # ===========================================================================
        # Domain Similarity Mappings
        # ===========================================================================
        
        # Define which domains are similar to each other
        domain_similarities = [
            ("playstation", "xbox", 0.9),
            ("playstation", "nintendo_switch", 0.8),
            ("xbox", "nintendo_switch", 0.8),
            ("pc_gaming", "playstation", 0.6),
            ("pc_gaming", "xbox", 0.6),
            ("ios", "android", 0.9),
            ("windows", "macos", 0.8),
            ("windows", "linux", 0.8),
            ("macos", "linux", 0.7),
            ("react", "vue", 0.8),
            ("react", "angular", 0.7),
            ("vue", "angular", 0.7),
            ("vscode", "intellij", 0.8),
            ("vscode", "sublime", 0.7),
            ("blender", "maya", 0.8),
            ("blender", "3dsmax", 0.7),
            ("maya", "3dsmax", 0.8),
            ("ableton", "logic_pro", 0.8),
            ("ableton", "fl_studio", 0.7),
            ("logic_pro", "fl_studio", 0.7)
        ]
        
        # Store domain similarities
        if not hasattr(self, "domain_similarities"):
            self.domain_similarities = {}
        
        for domain1, domain2, similarity in domain_similarities:
            if domain1 not in self.domain_similarities:
                self.domain_similarities[domain1] = {}
            if domain2 not in self.domain_similarities:
                self.domain_similarities[domain2] = {}
                
            self.domain_similarities[domain1][domain2] = similarity
            self.domain_similarities[domain2][domain1] = similarity
        
        logger.info(f"Initialized {len(self.chunk_library.control_mappings)} control mappings across domains")
    
    def register_function(self, name: str, func: Callable):
        """Register a function for use in procedures"""
        self.function_registry[name] = func
    
    # Function tools for the agents
    @function_tool
    async def list_procedures(self, ctx: RunContextWrapper) -> ProcedureListResult:
        """List all available procedures"""
        with agents_trace("list_procedures"):
            procedures_list: List[ProcedureSummary] = []
            for name, proc in self.procedures.items():
                success_rate = 0.0
                if proc.execution_count > 0:
                    success_rate = proc.successful_executions / proc.execution_count
                    
                procedures_list.append(ProcedureSummary(
                    name=name,
                    domain=proc.domain,
                    steps_count=len(proc.steps),
                    proficiency=proc.proficiency,
                    execution_count=proc.execution_count,
                    is_chunked=proc.is_chunked,
                    last_execution=proc.last_execution,
                    success_rate=success_rate,
                    average_execution_time=proc.average_execution_time,
                    description=proc.description
                ))
                
            return ProcedureListResult(
                count=len(procedures_list),
                procedures=procedures_list
            )
    
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
        # Get reference to the manager
        manager = self
        
        # Define enhanced tools
        @function_tool
        async def learn_from_demonstration(
            ctx: RunContextWrapper,
            observation_sequence: List[ObservationStep],
            domain: str,
            name: Optional[str] = None
        ) -> LearningResult:
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
                procedure_data = await manager.observation_learner.learn_from_demonstration(
                    observation_sequence=observation_sequence,
                    domain=domain
                )
                
                # Use provided name if available
                if name:
                    procedure_data["name"] = name
                
                # Create the procedure
                procedure_result = await manager.add_procedure(
                    ctx,
                    name=procedure_data["name"],
                    steps=procedure_data["steps"],
                    description=procedure_data["description"],
                    domain=domain
                )
                
                return LearningResult(
                    procedure_id=procedure_result["procedure_id"],
                    name=procedure_data["name"],
                    status="created",
                    domain=domain,
                    steps_count=len(procedure_data["steps"]),
                    confidence=procedure_data["confidence"],
                    learned_from_observations=True,
                    observation_count=procedure_data["observation_count"]
                )
    
        @function_tool
        async def create_hierarchical_procedure(
            ctx: RunContextWrapper,
            name: str,
            description: str,
            domain: str,
            steps: List[ProcedureStep],
            goal_state: Optional[GoalState] = None,
            preconditions: Optional[PreconditionSet] = None,
            postconditions: Optional[PostconditionSet] = None,
            parent_id: Optional[str] = None
        ) -> HierarchicalProcedureResult:
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
                    goal_state=goal_state or GoalState(),
                    preconditions=preconditions or PreconditionSet(),
                    postconditions=postconditions or PostconditionSet(),
                    parent_id=parent_id
                )
                
                # Store the procedure
                manager.hierarchical_procedures[name] = procedure
                
                # Create standard procedure as well
                standard_proc = await manager.add_procedure(
                    ctx,
                    name=name,
                    steps=steps,
                    description=description,
                    domain=domain
                )
                
                # If has parent, update parent's children list
                if parent_id:
                    for parent in manager.hierarchical_procedures.values():
                        if parent.id == parent_id:
                            parent.add_child(proc_id)
                            break
                
                # Return information
                return HierarchicalProcedureResult(
                    id=proc_id,
                    name=name,
                    domain=domain,
                    steps_count=len(steps),
                    standard_procedure_id=standard_proc["procedure_id"],
                    hierarchical=True,
                    parent_id=parent_id
                )
    
        @function_tool
        async def execute_hierarchical_procedure_enhanced(
            ctx: RunContextWrapper,
            name: str,
            context: Optional[ExecutionContext] = None
        ) -> ExecutionResult:
            """
            Execute a hierarchical procedure
            
            Args:
                name: Name of the procedure
                context: Execution context
                
            Returns:
                Execution results
            """
            with agents_trace(workflow_name="execute_hierarchical_procedure"):
                if name not in manager.hierarchical_procedures:
                    return ExecutionResult(
                        success=False,
                        results=[],
                        execution_time=0.0,
                        error=f"Hierarchical procedure '{name}' not found"
                    )
                
                procedure = manager.hierarchical_procedures[name]
                
                # Check preconditions
                if not procedure.meets_preconditions(context or {}):
                    return ExecutionResult(
                        success=False,
                        results=[],
                        execution_time=0.0,
                        error="Preconditions not met",
                        procedure_name=name
                    )
                
                # Initialize context if needed
                execution_context = context.copy() if context else ExecutionContext()
                
                # Set procedure context
                execution_context["current_procedure"] = name
                execution_context["hierarchical"] = True
                
                # Update working memory
                manager.working_memory.update(execution_context, procedure)
                
                # Select execution strategy
                strategy = manager.strategy_selector.select_strategy(execution_context, procedure)
                
                # Execute with selected strategy
                start_time = datetime.datetime.now()
                execution_result = await strategy.execute(procedure, execution_context)
                
                # Calculate execution time
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                # Update procedure statistics
                manager._update_hierarchical_stats(procedure, execution_time, execution_result["success"])
                
                # Verify goal state was achieved
                goal_achieved = True
                if procedure.goal_state and hasattr(procedure.goal_state, "conditions"):
                    for condition in procedure.goal_state.conditions:
                        attr = condition["attribute"]
                        op = condition["operator"]
                        expected = condition["value"]
                        
                        actual = execution_context.get(attr)
                        
                        if op == "equals" and actual != expected:
                            goal_achieved = False
                            break
                        elif op == "greater_than" and not (actual > expected):
                            goal_achieved = False
                            break
                        # ... other operators
                
                # Convert to ExecutionResult
                return ExecutionResult(
                    success=execution_result["success"],
                    results=execution_result.get("results", []),
                    execution_time=execution_time,
                    procedure_name=name,
                    procedure_id=procedure.id,
                    strategy_used=strategy.id,
                    error=execution_result.get("error")
                )
    
        @function_tool
        async def create_temporal_procedure(
            ctx: RunContextWrapper,
            name: str,
            steps: List[ProcedureStep],
            temporal_constraints: List[TemporalConstraint],
            domain: str,
            description: Optional[str] = None
        ) -> HierarchicalProcedureResult:
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
                normal_proc = await manager.add_procedure(
                    ctx,
                    name=name,
                    steps=steps,
                    description=description or f"Temporal procedure: {name}",
                    domain=domain
                )
                
                # Create temporal graph
                procedure = manager.procedures[name]
                graph = TemporalProcedureGraph.from_procedure(procedure)
                
                # Add temporal constraints
                for constraint in temporal_constraints:
                    from_id = constraint["from_step"]
                    to_id = constraint["to_step"]
                    constraint_type = constraint["type"]
                    
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
                                        if len(edge) < 3:
                                            edge.append({})
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
                    return HierarchicalProcedureResult(
                        id=normal_proc["procedure_id"],
                        name=name,
                        domain=domain,
                        steps_count=len(steps),
                        standard_procedure_id=normal_proc["procedure_id"],
                        hierarchical=False,
                        parent_id=None
                    )
                
                # Store the temporal graph
                manager.temporal_graphs[graph.id] = graph
                
                # Link procedure to graph
                procedure.temporal_graph_id = graph.id
                
                return HierarchicalProcedureResult(
                    id=graph.id,
                    name=name,
                    domain=domain,
                    steps_count=len(steps),
                    standard_procedure_id=normal_proc["procedure_id"],
                    hierarchical=True,
                    parent_id=None
                )
    
        @function_tool
        async def execute_temporal_procedure_enhanced(
            ctx: RunContextWrapper,
            name: str,
            context: Optional[ExecutionContext] = None
        ) -> ExecutionResult:
            """
            Execute a procedure with temporal constraints
            
            Args:
                name: Name of the procedure
                context: Execution context
                
            Returns:
                Execution results
            """
            with agents_trace(workflow_name="execute_temporal_procedure"):
                if name not in manager.procedures:
                    return ExecutionResult(
                        success=False,
                        results=[],
                        execution_time=0.0,
                        error=f"Procedure '{name}' not found"
                    )
                
                procedure = manager.procedures[name]
                
                # Check if procedure has temporal graph
                if not hasattr(procedure, "temporal_graph_id") or procedure.temporal_graph_id not in manager.temporal_graphs:
                    # Fall back to normal execution
                    return await manager.execute_procedure(ctx, name, context)
                
                # Get temporal graph
                graph = manager.temporal_graphs[procedure.temporal_graph_id]
                
                start_time = datetime.datetime.now()
                results: List[StepExecutionResult] = []
                success = True
                
                # Initialize execution context
                execution_context = context.copy() if context else ExecutionContext()
                execution_context["temporal_execution"] = True
                if "execution_history" not in execution_context:
                    execution_context["execution_history"] = []
                
                # Execute in temporal order
                while True:
                    # Get next executable nodes
                    next_nodes = graph.get_next_executable_nodes(
                        [hist["node_id"] for hist in execution_context["execution_history"]]
                    )
                    
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
                    step = ProcedureStep(
                        id=node.action.get("step_id", node_id),
                        function=node.action["function"],
                        parameters=node.action.get("parameters", StepParameters()),
                        description=node.action.get("description", f"Step {node_id}")
                    )
                    
                    # Execute the step
                    step_result = await manager.execute_step(ctx, step, execution_context)
                    results.append(step_result)
                    
                    # Update execution history
                    execution_context["execution_history"].append(ActionHistoryItem(
                        step_id=step["id"],
                        function=step["function"],
                        success=step_result["success"],
                        timestamp=datetime.datetime.now().isoformat(),
                        execution_time=step_result["execution_time"]
                    ))
                    
                    # Check for failure
                    if not step_result["success"]:
                        success = False
                        break
                
                # Calculate execution time
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                # Update procedure statistics
                manager.update_procedure_stats(procedure, execution_time, success)
                
                return ExecutionResult(
                    success=success,
                    results=results,
                    execution_time=execution_time,
                    procedure_name=name,
                    procedure_id=procedure.id
                )
    
        @function_tool
        async def create_procedure_graph(
            ctx: RunContextWrapper,
            procedure_name: str
        ) -> GeneralizationResult:
            """
            Create a graph representation of a procedure for flexible execution
            
            Args:
                procedure_name: Name of the existing procedure
                
            Returns:
                Information about the created graph
            """
            with agents_trace("create_procedure_graph"):
                if procedure_name not in manager.procedures:
                    return GeneralizationResult(
                        template_id="",
                        name=procedure_name,
                        domain="",
                        steps_count=0,
                        template_created=False,
                        error=f"Procedure '{procedure_name}' not found"
                    )
                
                procedure = manager.procedures[procedure_name]
                
                # Create graph representation
                graph = ProcedureGraph.from_procedure(procedure)
                
                # Generate graph ID
                graph_id = f"graph_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
                
                # Store graph
                manager.procedure_graphs[graph_id] = graph
                
                # Link procedure to graph
                procedure.graph_id = graph_id
                
                return GeneralizationResult(
                    template_id=graph_id,
                    name=procedure_name,
                    domain=procedure.domain,
                    steps_count=len(graph.nodes),
                    template_created=True,
                    error=None
                )
    
        @function_tool
        async def execute_graph_procedure_enhanced(
            ctx: RunContextWrapper,
            procedure_name: str,
            context: Optional[ExecutionContext] = None,
            goal: Optional[GoalState] = None
        ) -> ExecutionResult:
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
                if procedure_name not in manager.procedures:
                    return ExecutionResult(
                        success=False,
                        results=[],
                        execution_time=0.0,
                        error=f"Procedure '{procedure_name}' not found"
                    )
                
                procedure = manager.procedures[procedure_name]
                
                # Check if procedure has graph
                if not hasattr(procedure, "graph_id") or procedure.graph_id not in manager.procedure_graphs:
                    # Fall back to normal execution
                    return await manager.execute_procedure(ctx, procedure_name, context)
                
                # Get graph
                graph = manager.procedure_graphs[procedure.graph_id]
                
                start_time = datetime.datetime.now()
                results: List[StepExecutionResult] = []
                success = True
                
                # Initialize execution context
                execution_context = context.copy() if context else ExecutionContext()
                execution_context["graph_execution"] = True
                
                # Find execution path to goal
                goal_dict = {}
                if goal and hasattr(goal, "conditions"):
                    for condition in goal.conditions:
                        goal_dict[condition["attribute"]] = condition["value"]
                
                path = graph.find_execution_path(execution_context, goal_dict)
                
                if not path:
                    return ExecutionResult(
                        success=False,
                        results=[],
                        execution_time=0.0,
                        error="Could not find a valid execution path",
                        procedure_name=procedure_name
                    )
                
                # Execute nodes in path
                for node_id in path:
                    # Get node data
                    node_data = graph.nodes[node_id]
                    
                    # Create step from node data
                    step = ProcedureStep(
                        id=node_data.get("step_id", node_id),
                        function=node_data["function"],
                        parameters=node_data.get("parameters", StepParameters()),
                        description=node_data.get("description", f"Step {node_id}")
                    )
                    
                    # Execute the step
                    step_result = await manager.execute_step(ctx, step, execution_context)
                    results.append(step_result)
                    
                    # Update execution context
                    if "step_results" not in execution_context:
                        execution_context["step_results"] = {}
                    execution_context["step_results"][step["id"]] = step_result
                    
                    # Check for failure
                    if not step_result["success"]:
                        success = False
                        break
                
                # Calculate execution time
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                # Update procedure statistics
                manager.update_procedure_stats(procedure, execution_time, success)
                
                # Check if goal achieved
                goal_achieved = True
                if goal_dict:
                    for key, value in goal_dict.items():
                        if key not in execution_context or execution_context[key] != value:
                            goal_achieved = False
                            break
                
                return ExecutionResult(
                    success=success and goal_achieved,
                    results=results,
                    execution_time=execution_time,
                    procedure_name=procedure_name,
                    procedure_id=procedure.id
                )
    
        @function_tool
        async def optimize_procedure_transfer(
            ctx: RunContextWrapper,
            source_procedure: str,
            target_domain: str
        ) -> TransferPlan:
            """
            Optimize transfer of a procedure to a new domain
            
            Args:
                source_procedure: Name of the source procedure
                target_domain: Target domain
                
            Returns:
                Transfer optimization plan
            """
            with agents_trace("optimize_procedure_transfer"):
                if source_procedure not in manager.procedures:
                    return TransferPlan(
                        source_domain="",
                        target_domain=target_domain,
                        mappings=[],
                        transfer_strategy="none"
                    )
                
                # Get source procedure
                procedure = manager.procedures[source_procedure]
                
                # Optimize transfer
                transfer_plan = await manager.transfer_optimizer.optimize_transfer(
                    source_procedure=procedure,
                    target_domain=target_domain
                )
                
                # Convert to TransferPlan TypedDict
                mappings: List[MappingInfo] = []
                for mapping in transfer_plan.get("mappings", []):
                    mappings.append(MappingInfo(
                        source_function=mapping["source_function"],
                        target_function=mapping["target_function"],
                        target_parameters=mapping.get("target_parameters", StepParameters())
                    ))
                
                return TransferPlan(
                    source_domain=procedure.domain,
                    target_domain=target_domain,
                    mappings=mappings,
                    transfer_strategy=transfer_plan.get("strategy", "direct_mapping")
                )
    
        @function_tool
        async def execute_transfer_plan(
            ctx: RunContextWrapper,
            transfer_plan: TransferPlan,
            target_name: str
        ) -> TransferResult:
            """
            Execute a transfer plan to create a new procedure
            
            Args:
                transfer_plan: Transfer plan from optimize_procedure_transfer
                target_name: Name for the new procedure
                
            Returns:
                Results of transfer execution
            """
            with agents_trace("execute_transfer_plan"):
                source_domain = transfer_plan["source_domain"]
                target_domain = transfer_plan["target_domain"]
                mappings = transfer_plan["mappings"]
                
                if not source_domain or not target_domain or not mappings:
                    return TransferResult(
                        source_procedure="",
                        target_procedure=target_name,
                        source_domain=source_domain,
                        target_domain=target_domain,
                        steps_transferred=0,
                        procedure_id=None,
                        success=False,
                        error="Invalid transfer plan"
                    )
                
                # Find source procedure
                source_procedure = None
                source_name = ""
                for name, proc in manager.procedures.items():
                    if proc.domain == source_domain:
                        source_procedure = proc
                        source_name = name
                        break
                
                if not source_procedure:
                    return TransferResult(
                        source_procedure="",
                        target_procedure=target_name,
                        source_domain=source_domain,
                        target_domain=target_domain,
                        steps_transferred=0,
                        procedure_id=None,
                        success=False,
                        error=f"Could not find procedure in domain {source_domain}"
                    )
                
                # Create new steps based on mappings
                new_steps: List[ProcedureStep] = []
                
                for i, mapping in enumerate(mappings):
                    source_func = mapping["source_function"]
                    target_func = mapping["target_function"]
                    target_params = mapping["target_parameters"]
                    
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
                    new_step = ProcedureStep(
                        id=f"step_{i+1}",
                        function=target_func,
                        parameters=target_params,
                        description=f"Transferred from {source_step.get('description', source_func)}"
                    )
                    
                    new_steps.append(new_step)
                
                if not new_steps:
                    return TransferResult(
                        source_procedure=source_name,
                        target_procedure=target_name,
                        source_domain=source_domain,
                        target_domain=target_domain,
                        steps_transferred=0,
                        procedure_id=None,
                        success=False,
                        error="No steps could be transferred"
                    )
                
                # Create new procedure
                new_procedure = await manager.add_procedure(
                    ctx,
                    name=target_name,
                    steps=new_steps,
                    description=f"Transferred from {source_name} ({source_domain} to {target_domain})",
                    domain=target_domain
                )
                
                # Update transfer history
                manager.transfer_optimizer.update_from_transfer_result(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    success_rate=0.8,  # Initial estimate
                    mappings=[{
                        "source_function": m["source_function"],
                        "target_function": m["target_function"],
                        "target_parameters": m["target_parameters"]
                    } for m in mappings]
                )
                
                # Update transfer stats
                manager.transfer_stats["successful_transfers"] += 1
                
                return TransferResult(
                    source_procedure=source_name,
                    target_procedure=target_name,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    steps_transferred=len(new_steps),
                    procedure_id=new_procedure["procedure_id"],
                    success=True,
                    confidence=0.8,
                    error=None
                )
    
        @function_tool
        async def handle_execution_error(
            ctx: RunContextWrapper,
            error: ErrorInfo,
            context: Optional[ExecutionContext] = None
        ) -> ErrorHandlingResult:
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
                likely_causes = manager.causal_model.identify_likely_causes(error)
                
                # Get recovery suggestions
                interventions = manager.causal_model.suggest_interventions(likely_causes)
                
                # Return results
                return ErrorHandlingResult(
                    likely_causes=likely_causes,
                    interventions=interventions,
                    context=context
                )
    
        @function_tool
        async def share_domain_knowledge(
            ctx: RunContextWrapper,
            domain: str
        ) -> KnowledgeSharingResult:
            """
            Share procedural knowledge about a domain with knowledge core
            
            Args:
                domain: Domain to share knowledge about
                
            Returns:
                Status of knowledge sharing
            """
            with agents_trace("share_domain_knowledge"):
                if not manager.knowledge_core:
                    return KnowledgeSharingResult(
                        domain=domain,
                        knowledge_items_added=0,
                        procedures_analyzed=0,
                        error="No knowledge core available"
                    )
                
                # Find procedures in this domain
                domain_procedures = [p for p in manager.procedures.values() if p.domain == domain]
                
                if not domain_procedures:
                    return KnowledgeSharingResult(
                        domain=domain,
                        knowledge_items_added=0,
                        procedures_analyzed=0,
                        error=f"No procedures found for domain {domain}"
                    )
                
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
                        await manager.knowledge_core.add_knowledge_item(
                            domain=domain,
                            content=item["content"],
                            source=item["source"],
                            confidence=item["confidence"]
                        )
                        added_count += 1
                    except Exception as e:
                        logger.error(f"Error adding knowledge item: {e}")
                
                return KnowledgeSharingResult(
                    domain=domain,
                    knowledge_items_added=added_count,
                    procedures_analyzed=len(domain_procedures),
                    error=None
                )
        
        # Store references to enhanced tools
        self.learn_from_demonstration = learn_from_demonstration
        self.create_hierarchical_procedure = create_hierarchical_procedure
        self.execute_hierarchical_procedure_enhanced = execute_hierarchical_procedure_enhanced
        self.create_temporal_procedure = create_temporal_procedure
        self.execute_temporal_procedure_enhanced = execute_temporal_procedure_enhanced
        self.create_procedure_graph = create_procedure_graph
        self.execute_graph_procedure_enhanced = execute_graph_procedure_enhanced
        self.optimize_procedure_transfer = optimize_procedure_transfer
        self.execute_transfer_plan = execute_transfer_plan
        self.handle_execution_error = handle_execution_error
        self.share_domain_knowledge = share_domain_knowledge
        
        # Add enhanced tools to all agents
        enhanced_tools = [
            learn_from_demonstration,
            create_hierarchical_procedure,
            execute_hierarchical_procedure_enhanced,
            create_temporal_procedure,
            execute_temporal_procedure_enhanced,
            create_procedure_graph,
            execute_graph_procedure_enhanced,
            optimize_procedure_transfer,
            execute_transfer_plan,
            handle_execution_error,
            share_domain_knowledge,
        ]
        
        # Update the hierarchical/temporal/graph execution tools to use enhanced versions
        self.execute_hierarchical_procedure = execute_hierarchical_procedure_enhanced
        self.execute_temporal_procedure = execute_temporal_procedure_enhanced
        self.execute_graph_procedure = execute_graph_procedure_enhanced
        
        # Update agents with enhanced tools
        self._proc_manager_agent = self._proc_manager_agent.clone(
            tools=self._proc_manager_agent.tools + enhanced_tools
        )
        
        self._proc_execution_agent = self._proc_execution_agent.clone(
            tools=self._proc_execution_agent.tools + [
                execute_hierarchical_procedure_enhanced,
                execute_temporal_procedure_enhanced,
                execute_graph_procedure_enhanced,
                handle_execution_error,
            ]
        )
        
        self._proc_analysis_agent = self._proc_analysis_agent.clone(
            tools=self._proc_analysis_agent.tools + [
                optimize_procedure_transfer,
                share_domain_knowledge,
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
                    "context_factors": {
                        "parameter_type_mismatch": 0.4,
                        "missing_required_parameter": 0.3,
                        "out_of_range_value": 0.3
                    }
                },
                {
                    "cause": "missing_precondition",
                    "description": "Required precondition not met",
                    "probability": 0.4,
                    "context_factors": {
                        "state_not_initialized": 0.5,
                        "dependency_not_available": 0.3,
                        "permission_denied": 0.2
                    }
                },
                {
                    "cause": "function_not_found",
                    "description": "Function not registered or available",
                    "probability": 0.3,
                    "context_factors": {
                        "typo_in_function_name": 0.4,
                        "function_not_imported": 0.6
                    }
                },
                {
                    "cause": "state_corruption",
                    "description": "System state is corrupted or inconsistent",
                    "probability": 0.2,
                    "context_factors": {
                        "concurrent_modification": 0.3,
                        "incomplete_previous_operation": 0.7
                    }
                }
            ],
            "timeout": [
                {
                    "cause": "slow_execution",
                    "description": "Operation taking too long to complete",
                    "probability": 0.5,
                    "context_factors": {
                        "large_data_volume": 0.4,
                        "complex_computation": 0.3,
                        "inefficient_algorithm": 0.3
                    }
                },
                {
                    "cause": "resource_contention",
                    "description": "Resources needed are being used by another process",
                    "probability": 0.3,
                    "context_factors": {
                        "cpu_overload": 0.3,
                        "memory_pressure": 0.3,
                        "io_bottleneck": 0.4
                    }
                },
                {
                    "cause": "network_delay",
                    "description": "Network latency or connectivity issues",
                    "probability": 0.2,
                    "context_factors": {
                        "high_latency": 0.5,
                        "packet_loss": 0.3,
                        "bandwidth_limitation": 0.2
                    }
                },
                {
                    "cause": "deadlock",
                    "description": "Circular dependency causing infinite wait",
                    "probability": 0.1,
                    "context_factors": {
                        "resource_ordering": 0.6,
                        "synchronization_issue": 0.4
                    }
                }
            ],
            "memory_error": [
                {
                    "cause": "out_of_memory",
                    "description": "Insufficient memory available",
                    "probability": 0.6,
                    "context_factors": {
                        "memory_leak": 0.3,
                        "large_allocation": 0.4,
                        "fragmentation": 0.3
                    }
                },
                {
                    "cause": "null_reference",
                    "description": "Attempting to access null/undefined reference",
                    "probability": 0.3,
                    "context_factors": {
                        "uninitialized_variable": 0.5,
                        "premature_cleanup": 0.3,
                        "race_condition": 0.2
                    }
                },
                {
                    "cause": "buffer_overflow",
                    "description": "Writing beyond allocated memory",
                    "probability": 0.1,
                    "context_factors": {
                        "incorrect_size_calculation": 0.6,
                        "unbounded_input": 0.4
                    }
                }
            ],
            "permission_error": [
                {
                    "cause": "insufficient_privileges",
                    "description": "User lacks required permissions",
                    "probability": 0.7,
                    "context_factors": {
                        "missing_role": 0.4,
                        "expired_credentials": 0.3,
                        "wrong_context": 0.3
                    }
                },
                {
                    "cause": "resource_locked",
                    "description": "Resource is locked by another process",
                    "probability": 0.2,
                    "context_factors": {
                        "exclusive_access": 0.6,
                        "file_in_use": 0.4
                    }
                },
                {
                    "cause": "security_policy",
                    "description": "Security policy prevents operation",
                    "probability": 0.1,
                    "context_factors": {
                        "firewall_rule": 0.3,
                        "sandbox_restriction": 0.7
                    }
                }
            ],
            "data_error": [
                {
                    "cause": "corrupt_data",
                    "description": "Data is corrupted or malformed",
                    "probability": 0.4,
                    "context_factors": {
                        "encoding_issue": 0.3,
                        "partial_write": 0.4,
                        "format_mismatch": 0.3
                    }
                },
                {
                    "cause": "missing_data",
                    "description": "Required data is not available",
                    "probability": 0.3,
                    "context_factors": {
                        "deleted_file": 0.4,
                        "network_failure": 0.3,
                        "cache_miss": 0.3
                    }
                },
                {
                    "cause": "schema_mismatch",
                    "description": "Data doesn't match expected schema",
                    "probability": 0.2,
                    "context_factors": {
                        "version_mismatch": 0.5,
                        "missing_fields": 0.3,
                        "type_mismatch": 0.2
                    }
                },
                {
                    "cause": "data_race",
                    "description": "Concurrent access causing inconsistency",
                    "probability": 0.1,
                    "context_factors": {
                        "missing_synchronization": 0.7,
                        "incorrect_locking": 0.3
                    }
                }
            ],
            "configuration_error": [
                {
                    "cause": "missing_configuration",
                    "description": "Required configuration not set",
                    "probability": 0.5,
                    "context_factors": {
                        "environment_variable": 0.4,
                        "config_file": 0.4,
                        "default_value": 0.2
                    }
                },
                {
                    "cause": "invalid_configuration",
                    "description": "Configuration values are invalid",
                    "probability": 0.3,
                    "context_factors": {
                        "syntax_error": 0.3,
                        "value_out_of_range": 0.4,
                        "incompatible_settings": 0.3
                    }
                },
                {
                    "cause": "configuration_conflict",
                    "description": "Multiple configurations conflict",
                    "probability": 0.2,
                    "context_factors": {
                        "override_conflict": 0.5,
                        "dependency_mismatch": 0.5
                    }
                }
            ]
        }
        
        # Define common interventions for each cause
        self.causal_model.interventions = {
            # Execution failure interventions
            "invalid_parameters": [
                {
                    "type": "modify_parameters",
                    "description": "Modify parameters to valid values",
                    "effectiveness": 0.8,
                    "steps": [
                        "Identify parameter constraints",
                        "Validate current values",
                        "Apply corrections",
                        "Retry operation"
                    ]
                },
                {
                    "type": "check_documentation",
                    "description": "Check documentation for correct parameter format",
                    "effectiveness": 0.6,
                    "steps": [
                        "Locate function documentation",
                        "Review parameter specifications",
                        "Compare with current usage"
                    ]
                },
                {
                    "type": "use_defaults",
                    "description": "Use default parameter values",
                    "effectiveness": 0.7,
                    "steps": [
                        "Identify optional parameters",
                        "Remove or replace with defaults",
                        "Retry operation"
                    ]
                }
            ],
            "missing_precondition": [
                {
                    "type": "initialize_state",
                    "description": "Initialize required state before execution",
                    "effectiveness": 0.9,
                    "steps": [
                        "Identify missing preconditions",
                        "Execute initialization sequence",
                        "Verify state",
                        "Retry operation"
                    ]
                },
                {
                    "type": "wait_for_condition",
                    "description": "Wait for precondition to be met",
                    "effectiveness": 0.7,
                    "steps": [
                        "Set up condition monitoring",
                        "Implement timeout",
                        "Poll for condition",
                        "Proceed when ready"
                    ]
                },
                {
                    "type": "force_precondition",
                    "description": "Force precondition to be true",
                    "effectiveness": 0.5,
                    "steps": [
                        "Override safety checks",
                        "Set required state",
                        "Accept risks"
                    ]
                }
            ],
            "function_not_found": [
                {
                    "type": "search_registry",
                    "description": "Search function registry for similar names",
                    "effectiveness": 0.8,
                    "steps": [
                        "Get all registered functions",
                        "Find similar names",
                        "Suggest corrections"
                    ]
                },
                {
                    "type": "lazy_import",
                    "description": "Attempt to import missing function",
                    "effectiveness": 0.7,
                    "steps": [
                        "Identify function source",
                        "Dynamic import",
                        "Register function"
                    ]
                },
                {
                    "type": "use_alternative",
                    "description": "Use alternative function with similar behavior",
                    "effectiveness": 0.6,
                    "steps": [
                        "Find similar functions",
                        "Adapt parameters",
                        "Execute alternative"
                    ]
                }
            ],
            "state_corruption": [
                {
                    "type": "reset_state",
                    "description": "Reset to known good state",
                    "effectiveness": 0.9,
                    "steps": [
                        "Save current state",
                        "Load default state",
                        "Reinitialize components"
                    ]
                },
                {
                    "type": "repair_state",
                    "description": "Attempt to repair corrupted state",
                    "effectiveness": 0.6,
                    "steps": [
                        "Identify corruption",
                        "Apply fixes",
                        "Validate state"
                    ]
                },
                {
                    "type": "isolate_corruption",
                    "description": "Isolate and work around corruption",
                    "effectiveness": 0.7,
                    "steps": [
                        "Identify affected components",
                        "Quarantine corrupted data",
                        "Use alternative paths"
                    ]
                }
            ],
            
            # Timeout interventions
            "slow_execution": [
                {
                    "type": "optimize_algorithm",
                    "description": "Use more efficient algorithm",
                    "effectiveness": 0.9,
                    "steps": [
                        "Profile current execution",
                        "Identify bottlenecks",
                        "Apply optimizations"
                    ]
                },
                {
                    "type": "reduce_data",
                    "description": "Process smaller data chunks",
                    "effectiveness": 0.8,
                    "steps": [
                        "Split data into chunks",
                        "Process incrementally",
                        "Aggregate results"
                    ]
                },
                {
                    "type": "increase_timeout",
                    "description": "Increase timeout threshold",
                    "effectiveness": 0.5,
                    "steps": [
                        "Calculate required time",
                        "Set new timeout",
                        "Monitor progress"
                    ]
                }
            ],
            "resource_contention": [
                {
                    "type": "wait_and_retry",
                    "description": "Wait for resources to become available",
                    "effectiveness": 0.7,
                    "steps": [
                        "Implement backoff strategy",
                        "Monitor resource availability",
                        "Retry when available"
                    ]
                },
                {
                    "type": "request_priority",
                    "description": "Request higher priority access",
                    "effectiveness": 0.6,
                    "steps": [
                        "Elevate process priority",
                        "Request resource reservation",
                        "Execute with priority"
                    ]
                },
                {
                    "type": "use_alternative_resource",
                    "description": "Use alternative resources",
                    "effectiveness": 0.8,
                    "steps": [
                        "Identify alternatives",
                        "Adapt to different resource",
                        "Execute with alternative"
                    ]
                }
            ],
            "network_delay": [
                {
                    "type": "retry_with_backoff",
                    "description": "Retry with exponential backoff",
                    "effectiveness": 0.8,
                    "steps": [
                        "Implement backoff algorithm",
                        "Track retry attempts",
                        "Increase delay between retries"
                    ]
                },
                {
                    "type": "use_cache",
                    "description": "Use cached data if available",
                    "effectiveness": 0.9,
                    "steps": [
                        "Check cache validity",
                        "Return cached data",
                        "Update cache asynchronously"
                    ]
                },
                {
                    "type": "switch_endpoint",
                    "description": "Switch to different network endpoint",
                    "effectiveness": 0.7,
                    "steps": [
                        "Identify alternative endpoints",
                        "Test connectivity",
                        "Use fastest endpoint"
                    ]
                }
            ],
            "deadlock": [
                {
                    "type": "timeout_and_restart",
                    "description": "Timeout and restart operation",
                    "effectiveness": 0.8,
                    "steps": [
                        "Detect deadlock condition",
                        "Force timeout",
                        "Clean up resources",
                        "Restart with ordering"
                    ]
                },
                {
                    "type": "resource_ordering",
                    "description": "Enforce consistent resource ordering",
                    "effectiveness": 0.9,
                    "steps": [
                        "Define resource hierarchy",
                        "Always acquire in order",
                        "Release in reverse order"
                    ]
                }
            ],
            
            # Memory error interventions
            "out_of_memory": [
                {
                    "type": "free_memory",
                    "description": "Free unused memory",
                    "effectiveness": 0.8,
                    "steps": [
                        "Run garbage collection",
                        "Clear caches",
                        "Release unused resources"
                    ]
                },
                {
                    "type": "reduce_memory_usage",
                    "description": "Reduce memory footprint",
                    "effectiveness": 0.7,
                    "steps": [
                        "Use smaller data structures",
                        "Process in smaller batches",
                        "Stream instead of loading all"
                    ]
                },
                {
                    "type": "increase_memory_limit",
                    "description": "Request more memory",
                    "effectiveness": 0.6,
                    "steps": [
                        "Check system limits",
                        "Request increase",
                        "Monitor usage"
                    ]
                }
            ],
            "null_reference": [
                {
                    "type": "null_check",
                    "description": "Add null checks before access",
                    "effectiveness": 0.9,
                    "steps": [
                        "Identify access points",
                        "Add validation",
                        "Handle null cases"
                    ]
                },
                {
                    "type": "initialize_reference",
                    "description": "Initialize reference before use",
                    "effectiveness": 0.8,
                    "steps": [
                        "Find initialization point",
                        "Create default instance",
                        "Ensure proper lifecycle"
                    ]
                }
            ],
            
            # Permission error interventions
            "insufficient_privileges": [
                {
                    "type": "request_permission",
                    "description": "Request required permissions",
                    "effectiveness": 0.8,
                    "steps": [
                        "Identify required permissions",
                        "Request from user/admin",
                        "Retry with permissions"
                    ]
                },
                {
                    "type": "elevate_privileges",
                    "description": "Temporarily elevate privileges",
                    "effectiveness": 0.7,
                    "steps": [
                        "Request elevation",
                        "Execute with privileges",
                        "Drop privileges after"
                    ]
                },
                {
                    "type": "use_service_account",
                    "description": "Use service account with permissions",
                    "effectiveness": 0.9,
                    "steps": [
                        "Switch to service context",
                        "Execute operation",
                        "Return to user context"
                    ]
                }
            ],
            
            # Data error interventions
            "corrupt_data": [
                {
                    "type": "data_recovery",
                    "description": "Attempt to recover corrupted data",
                    "effectiveness": 0.6,
                    "steps": [
                        "Identify corruption pattern",
                        "Apply recovery algorithms",
                        "Validate recovered data"
                    ]
                },
                {
                    "type": "use_backup",
                    "description": "Restore from backup",
                    "effectiveness": 0.9,
                    "steps": [
                        "Locate recent backup",
                        "Verify backup integrity",
                        "Restore data"
                    ]
                },
                {
                    "type": "regenerate_data",
                    "description": "Regenerate data from source",
                    "effectiveness": 0.8,
                    "steps": [
                        "Identify data source",
                        "Reprocess from source",
                        "Validate output"
                    ]
                }
            ],
            "schema_mismatch": [
                {
                    "type": "schema_migration",
                    "description": "Migrate data to new schema",
                    "effectiveness": 0.9,
                    "steps": [
                        "Detect schema version",
                        "Apply migrations",
                        "Validate result"
                    ]
                },
                {
                    "type": "schema_adaptation",
                    "description": "Adapt to handle multiple schemas",
                    "effectiveness": 0.8,
                    "steps": [
                        "Implement version detection",
                        "Create adapters",
                        "Process accordingly"
                    ]
                }
            ],
            
            # Configuration error interventions
            "missing_configuration": [
                {
                    "type": "use_defaults",
                    "description": "Use default configuration values",
                    "effectiveness": 0.7,
                    "steps": [
                        "Load default config",
                        "Merge with partial config",
                        "Validate completeness"
                    ]
                },
                {
                    "type": "prompt_configuration",
                    "description": "Prompt user for configuration",
                    "effectiveness": 0.9,
                    "steps": [
                        "Identify missing values",
                        "Create configuration wizard",
                        "Save configuration"
                    ]
                },
                {
                    "type": "auto_configure",
                    "description": "Automatically detect configuration",
                    "effectiveness": 0.8,
                    "steps": [
                        "Scan environment",
                        "Detect settings",
                        "Apply configuration"
                    ]
                }
            ],
            "invalid_configuration": [
                {
                    "type": "validate_and_fix",
                    "description": "Validate and fix configuration",
                    "effectiveness": 0.8,
                    "steps": [
                        "Run validation rules",
                        "Identify issues",
                        "Apply corrections"
                    ]
                },
                {
                    "type": "reset_to_defaults",
                    "description": "Reset to default configuration",
                    "effectiveness": 0.9,
                    "steps": [
                        "Backup current config",
                        "Load defaults",
                        "Restart with defaults"
                    ]
                }
            ]
        }
        
        # Define recovery strategies that combine multiple interventions
        self.causal_model.recovery_strategies = {
            "execution_failure": [
                {
                    "name": "standard_recovery",
                    "description": "Standard recovery procedure for execution failures",
                    "interventions": ["modify_parameters", "initialize_state", "retry_with_backoff"],
                    "max_attempts": 3
                },
                {
                    "name": "aggressive_recovery",
                    "description": "More aggressive recovery with state reset",
                    "interventions": ["reset_state", "use_defaults", "force_precondition"],
                    "max_attempts": 2
                }
            ],
            "timeout": [
                {
                    "name": "performance_optimization",
                    "description": "Optimize performance to avoid timeout",
                    "interventions": ["optimize_algorithm", "reduce_data", "use_cache"],
                    "max_attempts": 2
                },
                {
                    "name": "resource_reallocation",
                    "description": "Reallocate resources to avoid contention",
                    "interventions": ["wait_and_retry", "use_alternative_resource", "request_priority"],
                    "max_attempts": 3
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
        
        # Add templates to library
        self.chunk_library.add_chunk_template(navigation_template)

    
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
