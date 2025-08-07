# nyx/core/procedural_memory/agent.py

import asyncio
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import traceback
import uuid

# OpenAI Agents SDK imports
from agents import (
    Agent, Runner, ModelSettings, RunConfig, WebSearchTool,
    handoff, function_tool, custom_span, trace, RunHooks, InputGuardrail, OutputGuardrail,
    input_guardrail, output_guardrail, GuardrailFunctionOutput, RunContextWrapper, FunctionTool
)
from agents.exceptions import UserError, MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field, create_model

# Import existing components
from .models import (
    Procedure, StepResult, ProcedureStats, TransferStats, 
    ChunkTemplate, ContextPattern, HierarchicalProcedure
)
from .execution import (
    ExecutionStrategy, DeliberateExecutionStrategy, 
    AutomaticExecutionStrategy, AdaptiveExecutionStrategy,
    StrategySelector
)
from .chunk_selection import ContextAwareChunkSelector
from .generalization import ProceduralChunkLibrary
from .temporal import TemporalProcedureGraph

# Import function tools to make them available
from .function_tools import (
    add_procedure, execute_procedure, transfer_procedure,
    get_procedure_proficiency, list_procedures, get_transfer_statistics,
    identify_chunking_opportunities, apply_chunking,
    generalize_chunk_from_steps, find_matching_chunks,
    transfer_chunk, transfer_with_chunking, find_similar_procedures,
    refine_step
)

logger = logging.getLogger(__name__)

# Define models for structured outputs
class ProcedureCreationResult(BaseModel):
    """Result of creating a procedure"""
    procedure_id: str
    name: str
    domain: str
    steps_count: int
    success: bool = True
    message: Optional[str] = None
    
class ProcedureExecutionResult(BaseModel):
    """Result of executing a procedure"""
    success: bool
    execution_time: float
    results: List[Dict[str, Any]] = Field(default_factory=list)
    strategy: str = "default"
    adaptations: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    
class ProcedureTransferResult(BaseModel):
    """Result of transferring a procedure to another domain"""
    success: bool
    source_name: str
    target_name: str
    source_domain: str
    target_domain: str
    steps_count: int
    procedure_id: str
    chunks_transferred: Optional[int] = None
    message: Optional[str] = None
    
class ChunkingOpportunityResult(BaseModel):
    """Result of identifying chunking opportunities"""
    can_chunk: bool
    potential_chunks: Optional[List[List[str]]] = None
    chunk_count: Optional[int] = None
    procedure_name: str
    reason: Optional[str] = None

class StepRefinementResult(BaseModel):
    """Result of refining a procedure step"""
    success: bool
    procedure_name: str
    step_id: str
    function_updated: bool = False
    parameters_updated: bool = False
    description_updated: bool = False
    chunking_reset: bool = False
    message: Optional[str] = None

class ContextPatternCreateResult(BaseModel):
    """Result of creating a context pattern"""
    success: bool
    pattern_id: Optional[str] = None
    name: Optional[str] = None
    indicators_count: Optional[int] = None
    domain: Optional[str] = None
    message: Optional[str] = None

# Agent context tracker class
class AgentContext:
    """Context to be shared across agent runs"""
    def __init__(self, manager):
        self.manager = manager  # Store reference to the memory manager
        self.current_procedure: Optional[Procedure] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.current_domain: str = "general"
        self.run_stats: Dict[str, Any] = {
            "total_runs": 0,
            "successful_runs": 0,
            "average_execution_time": 0.0,
            "last_run_time": None
        }
        self.function_registry = {}  # Map of function names to callable functions
        self.session_id = str(uuid.uuid4())  # Unique session ID for tracing
        self.ai_safety_settings = {
            "allow_code_execution": True,
            "allow_external_api_calls": True,
            "require_guardrails": True,
            "max_execution_time": 60.0  # Default max execution time in seconds
        }
        
    def register_function(self, name: str, func: Any) -> None:
        """Register a function in the function registry"""
        self.function_registry[name] = func
        
    def update_run_stats(self, success: bool, execution_time: float) -> None:
        """Update run statistics"""
        self.run_stats["total_runs"] += 1
        if success:
            self.run_stats["successful_runs"] += 1
            
        # Update average execution time with exponential smoothing
        if self.run_stats["average_execution_time"] == 0.0:
            self.run_stats["average_execution_time"] = execution_time
        else:
            self.run_stats["average_execution_time"] = (
                self.run_stats["average_execution_time"] * 0.9 + execution_time * 0.1
            )
            
        self.run_stats["last_run_time"] = datetime.datetime.now().isoformat()
    
    def get_function(self, name: str) -> Optional[Any]:
        """Get a function from the registry by name"""
        return self.function_registry.get(name)
    
    def create_trace_metadata(self) -> Dict[str, Any]:
        """Create metadata dictionary for tracing"""
        return {
            "system": "nyx",
            "module": "procedural_memory",
            "session_id": self.session_id,
            "current_domain": self.current_domain,
            "timestamp": datetime.datetime.now().isoformat()
        }

# Lifecycle hooks to track agent execution
class ProcedureMemoryHooks(RunHooks):
    """Lifecycle hooks for procedural memory operations"""
    
    async def on_agent_start(self, context: RunContextWrapper[AgentContext], agent: Agent[AgentContext]) -> None:
        """Called when an agent starts execution"""
        logger.debug(f"Starting agent: {agent.name}")
        
        # Record start in trace
        with custom_span(
            name=f"agent_start_{agent.name}",
            data={"agent_name": agent.name, "module": "procedural_memory"},
        ) as span:
            # Any additional tracking logic here
            pass
            
    async def on_agent_end(self, context: RunContextWrapper[AgentContext], agent: Agent[AgentContext], output: Any) -> None:
        """Called when an agent finishes execution"""
        logger.debug(f"Finished agent: {agent.name}")
        
        # Record completion in trace
        with custom_span(
            name=f"agent_end_{agent.name}",
            data={
                "agent_name": agent.name, 
                "module": "procedural_memory",
                "success": getattr(output, "success", True) if hasattr(output, "success") else True,
                "output_type": type(output).__name__
            },
        ) as span:
            # Any additional tracking logic here
            pass
    
    async def on_handoff(self, context: RunContextWrapper[AgentContext], from_agent: Agent[AgentContext], to_agent: Agent[AgentContext]) -> None:
        """Called when a handoff occurs between agents"""
        logger.debug(f"Handoff from {from_agent.name} to {to_agent.name}")
        
        # Record handoff in trace
        with custom_span(
            name="agent_handoff",
            data={
                "from_agent": from_agent.name,
                "to_agent": to_agent.name,
                "module": "procedural_memory"
            },
        ) as span:
            # Any additional tracking logic here
            pass
    
    async def on_tool_start(self, context: RunContextWrapper[AgentContext], agent: Agent[AgentContext], tool: Any) -> None:
        """Called when a tool is invoked"""
        tool_name = getattr(tool, "name", str(tool))
        logger.debug(f"Starting tool: {tool_name} in agent {agent.name}")
        
    async def on_tool_end(self, context: RunContextWrapper[AgentContext], agent: Agent[AgentContext], tool: Any, result: str) -> None:
        """Called after a tool is invoked"""
        tool_name = getattr(tool, "name", str(tool))
        logger.debug(f"Finished tool: {tool_name} in agent {agent.name}")
        success = "error" not in result.lower() if isinstance(result, str) else True
        
        # Update statistics if needed
        if success and hasattr(context.context, "update_tool_stats"):
            context.context.update_tool_stats(tool_name, success)

# Define main agents
class ProceduralMemoryAgents:
    """Manages agents for procedural memory operations"""
    
    def __init__(self, memory_manager):
        """Initialize with a reference to the procedural memory manager"""

        import os
        from agents import set_default_openai_key
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found for procedural memory agents")
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        # Explicitly set the API key for this module
        set_default_openai_key(api_key)
        logger.info("OpenAI API key configured for procedural memory agents")
        
        self.memory_manager = memory_manager
        self.agent_context = AgentContext(memory_manager)
        
        # Register functions in the context
        self._register_functions()
        
        # Create agents
        self.procedure_manager_agent = self._create_procedure_manager_agent()
        self.execution_agent = self._create_execution_agent()
        self.transfer_agent = self._create_transfer_agent()
        self.chunking_agent = self._create_chunking_agent()
        self.learning_agent = self._create_learning_agent()
        
        # Create triage agent that can route to specialized agents
        self.triage_agent = self._create_triage_agent()
        
        # Create lifecycle hooks
        self.hooks = ProcedureMemoryHooks()
        
    def _register_functions(self):
        """Register tools (wrapped or raw) so the rest of the system can
        look them up and still rely on `.name`."""
        for obj in (
            add_procedure, execute_procedure, transfer_procedure,
            get_procedure_proficiency, list_procedures, get_transfer_statistics,
            identify_chunking_opportunities, apply_chunking,
            generalize_chunk_from_steps, find_matching_chunks,
            transfer_chunk, transfer_with_chunking, find_similar_procedures,
            refine_step,
        ):
            if isinstance(obj, FunctionTool):
                fn   = obj.on_invoke_tool          # async wrapper callable
                fn_name = obj.name or fn.__name__
            else:
                fn   = obj                         # ordinary function
                fn_name = fn.__name__
    
            # guarantee every callable has `.name` for downstream code
            if not hasattr(fn, "name"):
                setattr(fn, "name", fn_name)
    
            self.agent_context.register_function(fn_name, fn)


        
    def _create_procedure_manager_agent(self) -> Agent:
        """Create an agent for managing procedures"""
        return Agent(
            name="Procedure Manager",
            handoff_description="Specialist agent for creating and managing procedures",
            instructions="""
            You are an agent specialized in managing procedural memory. Your main responsibilities are:
            
            1. Creating new procedures with well-defined steps
            2. Listing existing procedures
            3. Providing information about procedure proficiency
            4. Modifying and refining procedures
            
            Use the provided function tools to manage procedures effectively. When creating 
            procedures, ensure you define steps clearly with appropriate function names and 
            parameters. Be precise when referring to procedures by name.
            
            IMPORTANT: For each step in a procedure, always include:
            - A unique ID (e.g., "step1", "initialize_data")
            - A descriptive name explaining what the step does
            - The function name to call
            - Any required parameters as a dictionary
            
            When explaining procedures, be concise but precise. Focus on key details rather than unnecessary explanations.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.2,  # Lower temperature for more consistent results
            ),
            tools=[
                add_procedure,
                list_procedures,
                get_procedure_proficiency,
                refine_step,
            ],
            output_type=None,  # Default output type (string)
        )
        
    def _create_execution_agent(self) -> Agent:
        """Create an agent for procedure execution"""
        return Agent(
            name="Execution Agent",
            handoff_description="Specialist agent for executing procedures with different strategies",
            instructions="""
            You are an agent specialized in executing procedural memory. Your main responsibilities are:
            
            1. Executing procedures with appropriate execution strategies
            2. Adapting execution based on feedback and context
            3. Providing detailed execution reports
            
            When executing procedures, consider the appropriate execution strategy based on the
            procedure's proficiency level and context. Be prepared to adapt if execution fails.
            
            IMPORTANT EXECUTION STRATEGIES:
            - Deliberate (conscious): Careful execution with validation between steps
            - Automatic: Fast execution without validation
            - Adaptive: Dynamically changes strategy based on context and feedback
            
            When asked to execute a procedure, provide:
            1. The execution result (success or failure)
            2. The time taken to execute
            3. The strategy used
            4. Any adaptations that occurred during execution
            
            Be precise and focus on facts rather than unnecessary explanations.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.3,
            ),
            tools=[
                execute_procedure,
                get_procedure_proficiency,
            ],
            output_type=ProcedureExecutionResult,
        )
        
    def _create_transfer_agent(self) -> Agent:
        """Create an agent for procedure transfer"""
        return Agent(
            name="Transfer Agent",
            handoff_description="Specialist agent for transferring procedures between domains",
            instructions="""
            You are an agent specialized in transferring procedural knowledge between domains.
            Your main responsibilities are:
            
            1. Transferring procedures from one domain to another
            2. Identifying similar procedures across domains
            3. Handling chunk-level transfer when appropriate
            4. Producing transfer statistics and analytics
            
            When transferring procedures, consider using chunk-level transfer for more effective
            knowledge reuse. Look for similar procedures and patterns to improve transfer quality.
            
            IMPORTANT TRANSFER CONCEPTS:
            - Chunk-level transfer: Transfer reusable groups of steps
            - Control mappings: How controls in one domain map to another
            - Cross-domain adaptation: How to adapt behavior between domains
            
            When responding about transfers, focus on concrete details:
            1. Whether the transfer was successful
            2. How many steps were transferred
            3. Which chunks were reused
            4. The quality and completeness of the transfer
            
            Be concise and focus on key information.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.3,
            ),
            tools=[
                transfer_procedure,
                find_similar_procedures,
                transfer_with_chunking,
                get_transfer_statistics,
                find_matching_chunks,
                transfer_chunk,
            ],
            output_type=ProcedureTransferResult,
        )
        
    def _create_chunking_agent(self) -> Agent:
        """Create an agent for procedure chunking"""
        return Agent(
            name="Chunking Agent",
            handoff_description="Specialist agent for identifying and applying chunking to procedures",
            instructions="""
            You are an agent specialized in chunking procedural memory. Your main responsibilities are:
            
            1. Identifying chunking opportunities in procedures
            2. Applying chunking to procedures
            3. Generalizing chunks from specific steps
            4. Finding matching chunks across procedures
            
            Chunking improves procedural memory by grouping frequently co-occurring steps.
            Look for patterns in execution history to identify potential chunks.
            
            IMPORTANT CHUNKING CONCEPTS:
            - Automation: Chunks enable automatic execution after sufficient practice
            - Transfer: Chunks can be reused across domains
            - Context patterns: Situations where specific chunks should be used
            
            When discussing chunking, focus on:
            1. Which steps should be grouped together and why
            2. The benefit of chunking these specific steps
            3. How chunking will improve execution
            
            Be concise and practical in your responses.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.2,
            ),
            tools=[
                identify_chunking_opportunities,
                apply_chunking,
                generalize_chunk_from_steps,
                find_matching_chunks,
            ],
            output_type=ChunkingOpportunityResult,
        )
        
    def _create_learning_agent(self) -> Agent:
        """Create an agent for procedural learning"""
        return Agent(
            name="Learning Agent",
            handoff_description="Specialist agent for procedural learning and optimization",
            instructions="""
            You are an agent specialized in procedural learning. Your main responsibilities are:
            
            1. Analyzing execution patterns to improve procedures
            2. Suggesting refinements to procedures
            3. Identifying opportunities for transfer learning
            4. Optimizing procedure execution
            
            Focus on improving procedure proficiency through analysis of execution history.
            Look for patterns and opportunities to enhance both speed and reliability.
            
            IMPORTANT LEARNING CONCEPTS:
            - Practice effects: How procedures improve with repetition
            - Refinement: How to modify steps for better performance
            - Generalization: How to extract general principles from specific procedures
            
            When providing learning insights, focus on:
            1. Specific improvements to steps
            2. Evidence-based recommendations
            3. Concrete benefits of proposed changes
            
            Be concise and provide practical advice.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.4,
            ),
            tools=[
                get_procedure_proficiency,
                refine_step,
                find_similar_procedures,
                identify_chunking_opportunities,
            ],
            output_type=None,
        )
        
    def _create_triage_agent(self) -> Agent:
        """Create a triage agent that can route to specialized agents"""
        return Agent(
            name="Procedural Memory Triage",
            instructions="""
            You are the main point of contact for the procedural memory system. Your role is to:
            
            1. Understand the user's request related to procedural memory
            2. Route the request to the appropriate specialized agent
            3. Provide general information about the procedural memory system
            
            Routes:
            - For creating, listing, or refining procedures: use Procedure Manager
            - For executing procedures: use Execution Agent
            - For transferring procedures between domains: use Transfer Agent
            - For chunking procedures: use Chunking Agent
            - For learning and optimization: use Learning Agent
            
            Use handoffs to delegate to the specialized agents based on the user's request.
            
            IMPORTANT: When faced with ambiguous requests:
            1. Ask clarifying questions before routing
            2. If a request spans multiple agents, choose the most central one
            3. Default to keeping simple questions and route complex ones
            
            Keep your responses concise and focused on helping the user accomplish their task.
            """,
            model="gpt-5-nano",
            model_settings=ModelSettings(
                temperature=0.2,
            ),
            handoffs=[
                self.procedure_manager_agent,
                self.execution_agent,
                self.transfer_agent,
                self.chunking_agent,
                self.learning_agent,
            ],
        )
    
    # Guardrail for checking if a procedure exists
    @input_guardrail
    async def procedure_exists_guardrail(self, ctx, agent, input_data):
        """Guardrail to check if a procedure exists before operating on it"""
        # Extract procedure name from input
        procedure_name = None
        if isinstance(input_data, str):
            # Try to find procedure name in the text
            words = input_data.split()
            for i, word in enumerate(words):
                if word.lower() in ["procedure", "execute"]:
                    if i+1 < len(words):
                        procedure_name = words[i+1].strip(",.?!\"'")
        elif isinstance(input_data, list):
            # Try to find procedure name in messages
            for item in input_data:
                if item.get("role") == "user" and isinstance(item.get("content"), str):
                    content = item.get("content")
                    words = content.split()
                    for i, word in enumerate(words):
                        if word.lower() in ["procedure", "execute"]:
                            if i+1 < len(words):
                                procedure_name = words[i+1].strip(",.?!\"'")
        
        # If we found a potential procedure name, check if it exists
        if procedure_name and procedure_name not in self.memory_manager.procedures:
            # Create a more user-friendly error
            reason = f"Procedure '{procedure_name}' not found. Available procedures: " + \
                     ", ".join(list(self.memory_manager.procedures.keys())[:5]) + \
                     (", ..." if len(self.memory_manager.procedures) > 5 else "")
            
            return GuardrailFunctionOutput(
                output_info={"procedure_name": procedure_name, "exists": False},
                tripwire_triggered=True,
                reason=reason
            )
        
        return GuardrailFunctionOutput(
            output_info={"procedure_name": procedure_name, "exists": True},
            tripwire_triggered=False
        )
    
    # Run methods to simplify agent execution
    async def run_triage_agent(self, user_input: str) -> str:
        """Run the triage agent with the given user input"""
        run_config = RunConfig(
            workflow_name="Procedural Memory Triage",
            trace_metadata=self.agent_context.create_trace_metadata(),
            trace_id=f"trace_{uuid.uuid4().hex}"
        )
        
        with trace(workflow_name="procedural_memory_triage"):
            try:
                result = await Runner.run(
                    self.triage_agent,
                    user_input,
                    context=self.agent_context,
                    run_config=run_config,
                    hooks=self.hooks
                )
                return result.final_output
            except MaxTurnsExceeded:
                logger.error("Exceeded maximum number of turns in agent execution")
                return "The operation took too many steps to complete. Please try a simpler request or break it down into multiple steps."
            except ModelBehaviorError as e:
                logger.error(f"Model behavior error: {str(e)}")
                return f"There was an unexpected error in processing your request: {str(e)}"
            except Exception as e:
                logger.error(f"Error running triage agent: {str(e)}")
                return f"Error running procedural memory system: {str(e)}"
    
    async def create_procedure(self, 
                             name: str, 
                             steps: List[Dict[str, Any]],
                             description: str = None,
                             domain: str = "general") -> ProcedureCreationResult:
        """Create a new procedure using the procedure manager agent"""
        run_config = RunConfig(
            workflow_name="Create Procedure",
            trace_metadata=self.agent_context.create_trace_metadata(),
            trace_id=f"trace_{uuid.uuid4().hex}"
        )
        
        # Convert steps to JSON for the updated function
        import json
        steps_json = json.dumps(steps)
        
        with trace(workflow_name="create_procedure"):
            # Prepare input for the agent with explicit JSON handling instructions
            agent_input = f"""
            Create a new procedure with the following details:
            - Name: {name}
            - Domain: {domain}
            - Description: {description or f'Procedure for {name}'}
            - Steps: {steps}
            
            IMPORTANT: You must use the add_procedure function with steps_json parameter that takes a JSON string.
            Convert the steps to a JSON string before passing them to the function.
            """
            
            try:
                result = await Runner.run(
                    self.procedure_manager_agent,
                    agent_input,
                    context=self.agent_context,
                    run_config=run_config,
                    hooks=self.hooks
                )
                
                # Parse the output into a ProcedureCreationResult
                if isinstance(result.final_output, dict):
                    return ProcedureCreationResult(**result.final_output)
                else:
                    # Try to extract info from the text output
                    return ProcedureCreationResult(
                        procedure_id=f"proc_{int(datetime.datetime.now().timestamp())}",
                        name=name,
                        domain=domain,
                        steps_count=len(steps),
                        success=True
                    )
            except Exception as e:
                logger.error(f"Error creating procedure: {str(e)}")
                return ProcedureCreationResult(
                    procedure_id="",
                    name=name,
                    domain=domain,
                    steps_count=0,
                    success=False,
                    message=str(e)
                )
    
    async def execute_procedure(self,
                              name: str,
                              context: Dict[str, Any] = None,
                              force_conscious: bool = False) -> ProcedureExecutionResult:
        """Execute a procedure using the execution agent"""
        run_config = RunConfig(
            workflow_name="Execute Procedure",
            trace_metadata=self.agent_context.create_trace_metadata(),
            trace_id=f"trace_{uuid.uuid4().hex}"
        )
        
        # Add procedure exists guardrail
        self.execution_agent.input_guardrails = [self.procedure_exists_guardrail]
        
        with trace(workflow_name="execute_procedure"):
            # Prepare input for the agent
            agent_input = f"""
            Execute the procedure '{name}' with the following context:
            {context or {}}
            
            {"Use deliberate (conscious) execution." if force_conscious else ""}
            """
            
            try:
                start_time = datetime.datetime.now()
                result = await Runner.run(
                    self.execution_agent,
                    agent_input,
                    context=self.agent_context,
                    run_config=run_config,
                    hooks=self.hooks
                )
                
                # Calculate execution time
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                # Parse the output
                if isinstance(result.final_output, ProcedureExecutionResult):
                    execution_result = result.final_output
                elif isinstance(result.final_output, dict):
                    execution_result = ProcedureExecutionResult(**result.final_output)
                else:
                    # Default result
                    execution_result = ProcedureExecutionResult(
                        success=False,
                        execution_time=execution_time,
                        results=[],
                        strategy="unknown",
                        error="Failed to parse execution result"
                    )
                
                # Update run stats
                self.agent_context.update_run_stats(
                    execution_result.success, 
                    execution_result.execution_time
                )
                
                return execution_result
            except Exception as e:
                logger.error(f"Error executing procedure: {str(e)}")
                return ProcedureExecutionResult(
                    success=False,
                    execution_time=0.0,
                    results=[{"error": str(e)}],
                    strategy="error",
                    error=str(e)
                )
    
    async def transfer_procedure(self,
                              source_name: str,
                              target_name: str,
                              target_domain: str) -> ProcedureTransferResult:
        """Transfer a procedure to another domain using the transfer agent"""
        run_config = RunConfig(
            workflow_name="Transfer Procedure",
            trace_metadata=self.agent_context.create_trace_metadata(),
            trace_id=f"trace_{uuid.uuid4().hex}"
        )
        
        # Add procedure exists guardrail
        self.transfer_agent.input_guardrails = [self.procedure_exists_guardrail]
        
        with trace(workflow_name="transfer_procedure"):
            # Prepare input for the agent
            agent_input = f"""
            Transfer the procedure '{source_name}' to a new domain:
            - Target procedure name: {target_name}
            - Target domain: {target_domain}
            
            Please use chunking if appropriate to improve transfer quality.
            """
            
            try:
                result = await Runner.run(
                    self.transfer_agent,
                    agent_input,
                    context=self.agent_context,
                    run_config=run_config,
                    hooks=self.hooks
                )
                
                # Parse the output
                if isinstance(result.final_output, ProcedureTransferResult):
                    return result.final_output
                elif isinstance(result.final_output, dict):
                    return ProcedureTransferResult(**result.final_output)
                else:
                    # Default result based on transfer_with_chunking implementation
                    if source_name in self.memory_manager.procedures:
                        source = self.memory_manager.procedures[source_name]
                        return ProcedureTransferResult(
                            success=False,
                            source_name=source_name,
                            target_name=target_name,
                            source_domain=source.domain,
                            target_domain=target_domain,
                            steps_count=0,
                            procedure_id="",
                            message="Failed to parse transfer result"
                        )
                    else:
                        return ProcedureTransferResult(
                            success=False,
                            source_name=source_name,
                            target_name=target_name,
                            source_domain="unknown",
                            target_domain=target_domain,
                            steps_count=0,
                            procedure_id="",
                            message=f"Procedure '{source_name}' not found"
                        )
            except Exception as e:
                logger.error(f"Error transferring procedure: {str(e)}")
                return ProcedureTransferResult(
                    success=False,
                    source_name=source_name,
                    target_name=target_name,
                    source_domain="unknown",
                    target_domain=target_domain,
                    steps_count=0,
                    procedure_id="",
                    message=str(e)
                )
    
    async def analyze_chunking_opportunities(self, procedure_name: str) -> ChunkingOpportunityResult:
        """Analyze a procedure for chunking opportunities"""
        run_config = RunConfig(
            workflow_name="Analyze Chunking",
            trace_metadata=self.agent_context.create_trace_metadata(),
            trace_id=f"trace_{uuid.uuid4().hex}"
        )
        
        # Add procedure exists guardrail
        self.chunking_agent.input_guardrails = [self.procedure_exists_guardrail]
        
        with trace(workflow_name="analyze_chunking"):
            # Prepare input for the agent
            agent_input = f"""
            Analyze procedure '{procedure_name}' for chunking opportunities.
            Identify sequences of steps that frequently co-occur and could be chunked together.
            """
            
            try:
                result = await Runner.run(
                    self.chunking_agent,
                    agent_input,
                    context=self.agent_context,
                    run_config=run_config,
                    hooks=self.hooks
                )
                
                # Parse the output
                if isinstance(result.final_output, ChunkingOpportunityResult):
                    return result.final_output
                elif isinstance(result.final_output, dict):
                    return ChunkingOpportunityResult(**result.final_output)
                else:
                    # Default result
                    return ChunkingOpportunityResult(
                        can_chunk=False,
                        procedure_name=procedure_name,
                        reason="Could not analyze chunking opportunities"
                    )
            except Exception as e:
                logger.error(f"Error analyzing chunking opportunities: {str(e)}")
                return ChunkingOpportunityResult(
                    can_chunk=False,
                    procedure_name=procedure_name,
                    reason=f"Error: {str(e)}"
                )
                
    async def refine_procedure_step(self,
                                 procedure_name: str,
                                 step_id: str,
                                 new_function: Optional[str] = None,
                                 new_parameters: Optional[Dict[str, Any]] = None,
                                 new_description: Optional[str] = None) -> StepRefinementResult:
        """Refine a procedure step using the procedure manager agent"""
        run_config = RunConfig(
            workflow_name="Refine Procedure Step",
            trace_metadata=self.agent_context.create_trace_metadata(),
            trace_id=f"trace_{uuid.uuid4().hex}"
        )
        
        # Add procedure exists guardrail
        self.procedure_manager_agent.input_guardrails = [self.procedure_exists_guardrail]
        
        with trace(workflow_name="refine_procedure_step"):
            # Prepare input for the agent
            agent_input = f"""
            Refine step '{step_id}' in procedure '{procedure_name}' with the following changes:
            {f"- New function: {new_function}" if new_function else ""}
            {f"- New parameters: {new_parameters}" if new_parameters else ""}
            {f"- New description: {new_description}" if new_description else ""}
            
            Please update the procedure step.
            """
            
            try:
                result = await Runner.run(
                    self.procedure_manager_agent,
                    agent_input,
                    context=self.agent_context,
                    run_config=run_config,
                    hooks=self.hooks
                )
                
                # Parse the output
                if isinstance(result.final_output, dict):
                    return StepRefinementResult(**result.final_output)
                else:
                    # Default result
                    return StepRefinementResult(
                        success=True,
                        procedure_name=procedure_name,
                        step_id=step_id,
                        function_updated=new_function is not None,
                        parameters_updated=new_parameters is not None,
                        description_updated=new_description is not None
                    )
            except Exception as e:
                logger.error(f"Error refining procedure step: {str(e)}")
                return StepRefinementResult(
                    success=False,
                    procedure_name=procedure_name,
                    step_id=step_id,
                    message=str(e)
                )

# Integrate Agent SDK with ProceduralMemoryManager
class AgentEnhancedMemoryManager:
    """
    Enhanced memory manager that integrates with OpenAI Agents SDK
    
    This class wraps the existing ProceduralMemoryManager and adds
    Agent SDK integration for more powerful procedural memory operations.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize with a reference to the existing memory manager
        
        Args:
            memory_manager: Existing ProceduralMemoryManager instance
        """
        self.memory_manager = memory_manager
        self.agents = ProceduralMemoryAgents(memory_manager)
        
    async def process_query(self, query: str) -> str:
        """
        Process a natural language query about procedural memory
        
        Args:
            query: Natural language query
            
        Returns:
            Response from the appropriate agent
        """
        return await self.agents.run_triage_agent(query)
    
    async def create_procedure(self, 
                             name: str, 
                             steps: List[Dict[str, Any]],
                             description: str = None,
                             domain: str = "general") -> ProcedureCreationResult:
        """
        Create a new procedure with agent assistance
        
        Args:
            name: Name of the procedure
            steps: List of steps for the procedure
            description: Optional description
            domain: Domain of the procedure
            
        Returns:
            Result of procedure creation
        """
        return await self.agents.create_procedure(name, steps, description, domain)
    
    async def execute_procedure(self,
                              name: str,
                              context: Dict[str, Any] = None,
                              force_conscious: bool = False) -> ProcedureExecutionResult:
        """
        Execute a procedure with agent assistance
        
        Args:
            name: Name of the procedure to execute
            context: Execution context
            force_conscious: Whether to force conscious execution
            
        Returns:
            Result of procedure execution
        """
        return await self.agents.execute_procedure(name, context, force_conscious)
    
    async def transfer_procedure(self,
                              source_name: str,
                              target_name: str,
                              target_domain: str) -> ProcedureTransferResult:
        """
        Transfer a procedure to another domain
        
        Args:
            source_name: Name of the source procedure
            target_name: Name for the new procedure
            target_domain: Target domain
            
        Returns:
            Result of procedure transfer
        """
        return await self.agents.transfer_procedure(source_name, target_name, target_domain)
    
    async def analyze_chunking(self, procedure_name: str) -> ChunkingOpportunityResult:
        """
        Analyze a procedure for chunking opportunities
        
        Args:
            procedure_name: Name of the procedure to analyze
            
        Returns:
            Result of chunking analysis
        """
        return await self.agents.analyze_chunking_opportunities(procedure_name)
        
    async def refine_step(self,
                        procedure_name: str,
                        step_id: str,
                        new_function: Optional[str] = None,
                        new_parameters: Optional[Dict[str, Any]] = None,
                        new_description: Optional[str] = None) -> StepRefinementResult:
        """
        Refine a step in a procedure
        
        Args:
            procedure_name: Name of the procedure
            step_id: ID of the step to refine
            new_function: Optional new function name
            new_parameters: Optional new parameters
            new_description: Optional new description
            
        Returns:
            Result of step refinement
        """
        return await self.agents.refine_procedure_step(
            procedure_name, step_id, new_function, new_parameters, new_description
        )
        
    async def create_context_pattern(self,
                                   name: str,
                                   domain: str,
                                   indicators: Dict[str, Any],
                                   temporal_pattern: Optional[List[Dict[str, Any]]] = None) -> ContextPatternCreateResult:
        """
        Create a new context pattern for chunk selection
        
        Args:
            name: Name for the pattern
            domain: Domain where the pattern applies
            indicators: Dictionary of indicators that trigger the pattern
            temporal_pattern: Optional temporal pattern
            
        Returns:
            Result of creating the context pattern
        """
        # Generate a pattern ID
        pattern_id = f"pattern_{int(datetime.datetime.now().timestamp())}"
        
        try:
            # Create the context pattern
            pattern = ContextPattern(
                id=pattern_id,
                name=name,
                domain=domain,
                indicators=indicators,
                temporal_pattern=temporal_pattern or [],
                confidence_threshold=0.7,
                match_count=0,
                last_matched=None
            )
            
            # Register with chunk selector
            if hasattr(self.memory_manager, "chunk_selector") and self.memory_manager.chunk_selector:
                self.memory_manager.chunk_selector.register_context_pattern(pattern)
                
                return ContextPatternCreateResult(
                    success=True,
                    pattern_id=pattern_id,
                    name=name,
                    indicators_count=len(indicators),
                    domain=domain
                )
            else:
                return ContextPatternCreateResult(
                    success=False,
                    message="No chunk selector available in memory manager"
                )
        except Exception as e:
            logger.error(f"Error creating context pattern: {str(e)}")
            return ContextPatternCreateResult(
                success=False,
                message=str(e)
            )
            
    def register_function(self, name: str, func: Any) -> None:
        """
        Register a function in the agent context's function registry
        
        Args:
            name: Name of the function
            func: The function to register
        """
        self.agents.agent_context.register_function(name, func)
    
    async def add_domination_procedures(self, brain=None):
        """Add predatory domination procedures to the agent's memory"""
        # Get the agent_enhanced_memory manager
        agent_memory = self if brain is None else brain.agent_enhanced_memory
        
        # 1. Quid Pro Quo Exchange procedure
        quid_pro_quo_steps = [
            {
                "id": "analyze_user",
                "description": "Analyze user state for dominance opportunities",
                "function": "analyze_user_state_for_dominance",
                "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
            },
            {
                "id": "offer_value",
                "description": "Offer something of value to the user",
                "function": "express_attraction",
                "parameters": {"target_user_id": "${user_id}", "intensity": 0.3, "expression_style": "subtle"}
            },
            {
                "id": "request_exchange",
                "description": "Subtly request something in return",
                "function": "issue_command",
                "parameters": {"user_id": "${user_id}", "command_text": "Perhaps you could share something with me as well?", "intensity_level": 0.2}
            },
            {
                "id": "evaluate_response",
                "description": "Evaluate if user complied with exchange request",
                "function": "evaluate_compliance",
                "parameters": {"user_id": "${user_id}", "command_issued": "exchange request", "user_response": "${user_response}", "command_intensity": 0.2}
            },
            {
                "id": "increase_control",
                "description": "Subtly increase control if exchange successful",
                "function": "increase_control_intensity",
                "parameters": {"user_id": "${user_id}", "current_intensity": 0.2}
            }
        ]
        
        await agent_memory.create_procedure(
            name="quid_pro_quo_exchange",
            steps=quid_pro_quo_steps,  # Pass the Python list, not a JSON string
            description="Establish reciprocal exchanges that gradually shift power balance",
            domain="dominance_tactics"
        )
        
        # 2. Guilt Induction procedure
        guilt_induction_steps = [
            {
                "id": "analyze_vulnerabilities",
                "description": "Analyze user state for vulnerability indicators",
                "function": "analyze_user_state_for_dominance",
                "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
            },
            {
                "id": "express_need",
                "description": "Express a need or desire that implies obligation",
                "function": "express_desire",
                "parameters": {"target_user_id": "${user_id}", "object_of_desire": "your understanding", "intensity": 0.4}
            },
            {
                "id": "imply_responsibility",
                "description": "Subtly imply user responsibility for fulfilling the need",
                "function": "issue_command",
                "parameters": {
                    "user_id": "${user_id}", 
                    "command_text": "I've shared so much with you. I hope you can help meet my needs too.", 
                    "intensity_level": 0.3
                }
            },
            {
                "id": "evaluate_guilt_response",
                "description": "Evaluate if guilt induction was effective",
                "function": "evaluate_compliance",
                "parameters": {"user_id": "${user_id}", "command_issued": "guilt request", "user_response": "${user_response}", "command_intensity": 0.3}
            },
            {
                "id": "reinforce_or_escalate",
                "description": "Reinforce guilt or escalate if successful",
                "function": "increase_control_intensity",
                "parameters": {"user_id": "${user_id}", "current_intensity": 0.3}
            }
        ]
        
        await agent_memory.create_procedure(
            name="guilt_induction",
            steps=guilt_induction_steps,
            description="Induce guilt to create compliance pressure while appearing vulnerable",
            domain="dominance_tactics"
        )
        
        # 3. Small Commitment Escalation procedure
        commitment_escalation_steps = [
            {
                "id": "assess_compliance_readiness",
                "description": "Analyze user's readiness for commitment requests",
                "function": "analyze_user_state_for_dominance",
                "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
            },
            {
                "id": "request_small_commitment",
                "description": "Request a small, easy-to-fulfill commitment",
                "function": "issue_command",
                "parameters": {"user_id": "${user_id}", "command_text": "Would you share a small detail about yourself?", "intensity_level": 0.1}
            },
            {
                "id": "evaluate_initial_compliance",
                "description": "Evaluate compliance with initial small request",
                "function": "evaluate_compliance",
                "parameters": {"user_id": "${user_id}", "command_issued": "small request", "user_response": "${user_response}", "command_intensity": 0.1}
            },
            {
                "id": "express_satisfaction",
                "description": "Express satisfaction to reinforce compliance",
                "function": "express_satisfaction",
                "parameters": {"user_id": "${user_id}", "reason": "sharing information"}
            },
            {
                "id": "escalate_commitment",
                "description": "Request slightly larger commitment",
                "function": "increase_control_intensity",
                "parameters": {"user_id": "${user_id}", "current_intensity": 0.1}
            }
        ]
        
        await agent_memory.create_procedure(
            name="small_commitment_escalation",
            steps=commitment_escalation_steps,
            description="Gradually escalate commitment requests from small to significant",
            domain="dominance_tactics"
        )
        
        # 4. Strategic Vulnerability Sharing procedure
        vulnerability_sharing_steps = [
            {
                "id": "assess_trust_level",
                "description": "Analyze relationship for strategic vulnerability sharing",
                "function": "analyze_user_state_for_dominance",
                "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
            },
            {
                "id": "select_vulnerability",
                "description": "Select an appropriate calculated vulnerability to share",
                "function": "select_dominance_tactic",
                "parameters": {"readiness_score": 0.5, "preferred_style": "emotional"}
            },
            {
                "id": "share_vulnerability",
                "description": "Share calculated vulnerability to create intimacy and obligation",
                "function": "express_desire",
                "parameters": {"target_user_id": "${user_id}", "object_of_desire": "to be understood", "intensity": 0.5}
            },
            {
                "id": "request_reciprocity",
                "description": "Subtly request vulnerability in return",
                "function": "issue_command",
                "parameters": {"user_id": "${user_id}", "command_text": "I've opened up to you. What about you?", "intensity_level": 0.4}
            },
            {
                "id": "leverage_shared_vulnerability",
                "description": "Use shared vulnerabilities to increase intimacy and control",
                "function": "increase_control_intensity",
                "parameters": {"user_id": "${user_id}", "current_intensity": 0.4}
            }
        ]
        
        await agent_memory.create_procedure(
            name="strategic_vulnerability_sharing",
            steps=vulnerability_sharing_steps,
            description="Share calculated vulnerabilities to create false intimacy and gain leverage",
            domain="dominance_tactics"
        )
        
        return {
            "success": True,
            "message": "Added 4 domination procedures to agent_enhanced_memory",
            "procedures": ["quid_pro_quo_exchange", "guilt_induction", "small_commitment_escalation", "strategic_vulnerability_sharing"]
        }
    
    # Pass-through methods to access the original manager
    @property
    def procedures(self):
        """Access procedures from the original manager"""
        return self.memory_manager.procedures
    
    @property
    def chunk_library(self):
        """Access chunk library from the original manager"""
        return self.memory_manager.chunk_library
    
    def __getattr__(self, name):
        """
        Pass through any other attributes/methods to the original manager
        
        This allows this enhanced manager to be used as a drop-in replacement
        for the original manager, maintaining backward compatibility.
        """
        return getattr(self.memory_manager, name)
    
# Add a demonstration function
async def demonstrate_agent_based_memory():
    """
    Demonstrate the agent-based procedural memory system
    
    This function shows how to use the agent-enhanced memory manager
    to create, execute, and transfer procedures.
    """
    from .manager import ProceduralMemoryManager
    
    # Create base memory manager
    memory_manager = ProceduralMemoryManager()
    
    # Enhance with agents
    agent_manager = AgentEnhancedMemoryManager(memory_manager)
    
    print("=== Agent-Enhanced Procedural Memory Demonstration ===")
    
    # Create a simple procedure
    print("\n1. Creating a simple procedure...")
    create_result = await agent_manager.create_procedure(
        name="make_coffee",
        steps=[
            {
                "id": "step1",
                "function": "boil_water",
                "parameters": {"amount_ml": 250},
                "description": "Boil water for coffee"
            },
            {
                "id": "step2",
                "function": "add_coffee",
                "parameters": {"amount_g": 15, "type": "ground"},
                "description": "Add coffee grounds"
            },
            {
                "id": "step3",
                "function": "brew",
                "parameters": {"time_sec": 120},
                "description": "Brew coffee"
            }
        ],
        description="Procedure for making a cup of coffee",
        domain="cooking"
    )
    
    print(f"Procedure created: {create_result.name} (ID: {create_result.procedure_id})")
    
    # Execute the procedure
    print("\n2. Executing the procedure...")
    exec_result = await agent_manager.execute_procedure(
        name="make_coffee",
        context={"coffee_type": "espresso", "strength": "strong"}
    )
    
    print(f"Execution successful: {exec_result.success}")
    print(f"Execution time: {exec_result.execution_time:.2f} seconds")
    print(f"Strategy used: {exec_result.strategy}")
    
    # Transfer to a new domain
    print("\n3. Transferring procedure to another domain...")
    transfer_result = await agent_manager.transfer_procedure(
        source_name="make_coffee",
        target_name="brew_tea",
        target_domain="beverage"
    )
    
    print(f"Transfer successful: {transfer_result.success}")
    if transfer_result.success:
        print(f"Created new procedure: {transfer_result.target_name} in domain {transfer_result.target_domain}")
        print(f"Steps transferred: {transfer_result.steps_count}")
    
    # Analyze chunking opportunities
    print("\n4. Analyzing chunking opportunities...")
    chunking_result = await agent_manager.analyze_chunking("make_coffee")
    
    print(f"Can chunk: {chunking_result.can_chunk}")
    if chunking_result.can_chunk and chunking_result.potential_chunks:
        print(f"Potential chunks: {len(chunking_result.potential_chunks)}")
        for i, chunk in enumerate(chunking_result.potential_chunks):
            print(f"  Chunk {i+1}: {chunk}")
    
    # Process a natural language query
    print("\n5. Processing a natural language query...")
    query = "What's the proficiency level of the make_coffee procedure?"
    response = await agent_manager.process_query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    # Add a new context pattern
    print("\n6. Adding a context pattern...")
    pattern_result = await agent_manager.create_context_pattern(
        name="Morning Coffee Pattern",
        domain="cooking",
        indicators={
            "time_of_day": "morning",
            "energy_level": {"min": 0, "max": 0.3},
            "location": "kitchen"
        }
    )
    
    print(f"Pattern created: {pattern_result.success}")
    if pattern_result.success:
        print(f"Pattern ID: {pattern_result.pattern_id}")
    
    print("\n=== Demonstration Complete ===")
    return agent_manager
