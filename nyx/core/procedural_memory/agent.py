# nyx/core/procedural_memory/agent.py

import asyncio
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import traceback

# OpenAI Agents SDK imports
from agents import (
    Agent, Runner, ModelSettings, RunConfig,
    handoff, function_tool, custom_span, trace,
    input_guardrail, output_guardrail, GuardrailFunctionOutput
)
from agents.exceptions import UserError, MaxTurnsExceeded
from pydantic import BaseModel, Field

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
    
class ProcedureExecutionResult(BaseModel):
    """Result of executing a procedure"""
    success: bool
    execution_time: float
    results: List[Dict[str, Any]] = []
    strategy: str = "default"
    adaptations: List[Dict[str, Any]] = []
    
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
    
class ChunkingOpportunityResult(BaseModel):
    """Result of identifying chunking opportunities"""
    can_chunk: bool
    potential_chunks: Optional[List[List[str]]] = None
    chunk_count: Optional[int] = None
    procedure_name: str
    reason: Optional[str] = None

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

# Define main agents
class ProceduralMemoryAgents:
    """Manages agents for procedural memory operations"""
    
    def __init__(self, memory_manager):
        """Initialize with a reference to the procedural memory manager"""
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
        
    def _register_functions(self):
        """Register function tools in the agent context"""
        # Register all the function tools
        for func in [
            add_procedure, execute_procedure, transfer_procedure,
            get_procedure_proficiency, list_procedures, get_transfer_statistics,
            identify_chunking_opportunities, apply_chunking,
            generalize_chunk_from_steps, find_matching_chunks,
            transfer_chunk, transfer_with_chunking, find_similar_procedures,
            refine_step
        ]:
            name = func.__name__
            self.agent_context.register_function(name, func)
            
        # Register any additional utility functions if needed
        # self.agent_context.register_function("utility_func", utility_func)
        
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
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.2,  # Lower temperature for more consistent results
            ),
            tools=[
                function_tool(add_procedure),
                function_tool(list_procedures),
                function_tool(get_procedure_proficiency),
                function_tool(refine_step),
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
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.3,
            ),
            tools=[
                function_tool(execute_procedure),
                function_tool(get_procedure_proficiency),
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
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.3,
            ),
            tools=[
                function_tool(transfer_procedure),
                function_tool(find_similar_procedures),
                function_tool(transfer_with_chunking),
                function_tool(get_transfer_statistics),
                function_tool(find_matching_chunks),
                function_tool(transfer_chunk),
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
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.2,
            ),
            tools=[
                function_tool(identify_chunking_opportunities),
                function_tool(apply_chunking),
                function_tool(generalize_chunk_from_steps),
                function_tool(find_matching_chunks),
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
            """,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.4,
            ),
            tools=[
                function_tool(get_procedure_proficiency),
                function_tool(refine_step),
                function_tool(find_similar_procedures),
                function_tool(identify_chunking_opportunities),
            ],
            output_type=None,  # Default output type (string)
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
            """,
            model="gpt-4o",
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
            trace_metadata={"system": "nyx", "module": "procedural_memory"}
        )
        
        with trace(workflow_name="procedural_memory_triage"):
            try:
                result = await Runner.run(
                    self.triage_agent,
                    user_input,
                    context=self.agent_context,
                    run_config=run_config
                )
                return result.final_output
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
            trace_metadata={"system": "nyx", "module": "procedural_memory"}
        )
        
        with trace(workflow_name="create_procedure"):
            # Prepare input for the agent
            agent_input = f"""
            Create a new procedure with the following details:
            - Name: {name}
            - Domain: {domain}
            - Description: {description or f'Procedure for {name}'}
            - Steps: {steps}
            
            Please add this procedure to the system.
            """
            
            try:
                result = await Runner.run(
                    self.procedure_manager_agent,
                    agent_input,
                    context=self.agent_context,
                    run_config=run_config
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
                    success=False
                )
    
    async def execute_procedure(self,
                              name: str,
                              context: Dict[str, Any] = None,
                              force_conscious: bool = False) -> ProcedureExecutionResult:
        """Execute a procedure using the execution agent"""
        run_config = RunConfig(
            workflow_name="Execute Procedure",
            trace_metadata={"system": "nyx", "module": "procedural_memory"}
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
                    run_config=run_config
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
                        strategy="unknown"
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
                    strategy="error"
                )
    
    async def transfer_procedure(self,
                              source_name: str,
                              target_name: str,
                              target_domain: str) -> ProcedureTransferResult:
        """Transfer a procedure to another domain using the transfer agent"""
        run_config = RunConfig(
            workflow_name="Transfer Procedure",
            trace_metadata={"system": "nyx", "module": "procedural_memory"}
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
                    run_config=run_config
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
                            procedure_id=""
                        )
                    else:
                        return ProcedureTransferResult(
                            success=False,
                            source_name=source_name,
                            target_name=target_name,
                            source_domain="unknown",
                            target_domain=target_domain,
                            steps_count=0,
                            procedure_id=""
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
                )
    
    async def analyze_chunking_opportunities(self, procedure_name: str) -> ChunkingOpportunityResult:
        """Analyze a procedure for chunking opportunities"""
        run_config = RunConfig(
            workflow_name="Analyze Chunking",
            trace_metadata={"system": "nyx", "module": "procedural_memory"}
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
                    run_config=run_config
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
    
    print("\n=== Demonstration Complete ===")
    return agent_manager
