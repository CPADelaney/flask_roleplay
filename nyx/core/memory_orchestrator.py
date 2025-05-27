# nyx/core/memory_orchestrator.py

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union

from agents import Agent, Runner, trace, function_tool, handoff, FunctionTool, InputGuardrail, GuardrailFunctionOutput, RunConfig, RunContextWrapper
from pydantic import BaseModel, Field

from nyx.core.memory_core import MemoryCoreAgents, MemoryCoreContext, add_memory, get_memory, retrieve_memories_with_formatting, retrieve_relevant_experiences, update_memory, delete_memory, apply_memory_decay, retrieve_memories, create_reflection_from_memories, create_abstraction_from_memories, retrieve_relevant_experiences, generate_conversational_recall,construct_narrative_from_memories, consolidate_memory_clusters

logger = logging.getLogger(__name__)

# Pydantic models for input/output validation
class MemoryQuery(BaseModel):
    query: str
    memory_types: List[str] = Field(default_factory=lambda: ["observation", "reflection", "abstraction", "experience"])
    limit: int = 5
    min_significance: int = 3
    include_archived: bool = False
    entities: Optional[List[str]] = None

class ReflectionRequest(BaseModel):
    topic: Optional[str] = None

class NarrativeRequest(BaseModel):
    topic: str
    chronological: bool = True
    limit: int = 5

class ExperienceRequest(BaseModel):
    query: str
    scenario_type: str = ""
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    entities: List[str] = Field(default_factory=list)
    limit: int = 3

class MemoryOrchestrator:
    """
    Memory orchestration system that coordinates all memory-related operations
    using specialized agents built with the OpenAI Agents SDK.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Create memory core
        self.memory_core = MemoryCoreAgents(user_id, conversation_id)
        
        # Trace group ID for connecting traces
        self.trace_group_id = f"nyx-memory-{user_id}-{conversation_id}"
        
        # Initialize initialized flag
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory orchestration system"""
        if self.initialized:
            return
            
        logger.info(f"Initializing memory orchestration system for user {self.user_id}")
        with trace(workflow_name="Memory System Initialization", group_id=self.trace_group_id):
            await self.memory_core.initialize()
            self._init_agents()
            self.initialized = True
            logger.info("Memory orchestration system initialized")
    
    def _init_agents(self):
        """Initialize all the specialized agents"""
        # Initialize retrieval agent
        self.retrieval_agent = Agent(
            name="Memory Retrieval Agent",
            handoff_description="Handles searching and retrieving memories based on queries",
            instructions="""You are specialized in retrieving memories from Nyx's memory system.
            Use the provided tools to effectively search for and retrieve memories based on the
            user's query parameters. Return the most relevant memories with appropriate formatting.""",
            tools=[
                # Don't double-wrap functions that are already decorated with @function_tool
                retrieve_memories,  # Already has @function_tool
                get_memory,  # Already has @function_tool
                retrieve_memories_with_formatting,  # Already has @function_tool
                retrieve_relevant_experiences  # Already has @function_tool
            ],
            output_type=Dict[str, Any]
        )
        
        # Initialize creation agent
        self.creation_agent = Agent(
            name="Memory Creation Agent",
            handoff_description="Handles creating and storing new memories",
            instructions="""You are specialized in creating and storing new memories for Nyx.
            When a user wants to create a new memory, help them structure it properly with
            appropriate metadata, tags, and significance levels. Ensure the memory is properly
            categorized and indexed.""",
            tools=[
                add_memory,  # Already has @function_tool
                update_memory,  # Already has @function_tool
                delete_memory  # Already has @function_tool
            ],
            output_type=Dict[str, Any]
        )
        
        # Initialize reflection agent
        self.reflection_agent = Agent(
            name="Reflection Agent",
            handoff_description="Creates reflections and abstractions from memories",
            instructions="""You are specialized in generating reflections and abstractions
            from Nyx's memories. Your role is to analyze memories, find patterns, and create
            higher-level insights that connect individual experiences.""",
            tools=[
                create_reflection_from_memories,  # Already has @function_tool
                create_abstraction_from_memories,  # Already has @function_tool
                retrieve_memories  # Already has @function_tool
            ],
            output_type=Dict[str, Any]
        )
        
        # Initialize experience agent
        self.experience_agent = Agent(
            name="Experience Agent",
            handoff_description="Manages experience recall and narrative generation",
            instructions="""You are specialized in managing Nyx's experiences and narratives.
            Your role is to retrieve relevant experiences, generate natural conversational
            recalls, and construct coherent narratives from memories.""",
            tools=[
                retrieve_relevant_experiences,  # Already has @function_tool
                generate_conversational_recall,  # Already has @function_tool
                construct_narrative_from_memories  # Already has @function_tool
            ],
            output_type=Dict[str, Any]
        )
        
        # Initialize maintenance agent
        self.maintenance_agent = Agent(
            name="Memory Maintenance Agent",
            handoff_description="Handles memory system maintenance, decay, and consolidation",
            instructions="""You are specialized in maintaining the health and efficiency of
            Nyx's memory system. Your role is to apply memory decay, consolidate similar
            memories, archive old memories, and ensure the overall system stays optimized.""",
            tools=[
                # For methods on self.memory_core, we need to wrap them
                function_tool(self.memory_core.run_maintenance),
                apply_memory_decay,  # Already has @function_tool
                consolidate_memory_clusters,  # Already has @function_tool
                function_tool(self.memory_core.get_memory_stats)
            ],
            output_type=Dict[str, Any]
        )
        
        # Initialize the main memory orchestrator agent
        self.main_agent = Agent(
            name="Memory Orchestrator",
            instructions="""You are the central memory orchestration system for Nyx AI.
            Your role is to coordinate memory operations, deciding which specialized
            agent to use for each memory task. Analyze the user's request and direct
            it to the appropriate specialized agent.""",
            handoffs=[
                handoff(self.retrieval_agent,
                       tool_name_override="retrieve_memories",
                       tool_description_override="Search and retrieve memories based on queries"),
                handoff(self.creation_agent,
                       tool_name_override="create_memory",
                       tool_description_override="Create and store new memories"),
                handoff(self.reflection_agent,
                       tool_name_override="create_reflection",
                       tool_description_override="Generate reflections and abstractions"),
                handoff(self.experience_agent,
                       tool_name_override="manage_experiences",
                       tool_description_override="Retrieve experiences and generate narratives"),
                handoff(self.maintenance_agent,
                       tool_name_override="maintenance_tasks",
                       tool_description_override="Perform memory system maintenance")
            ],
            output_type=Dict[str, Any]
        )
    
    # Input guardrail for request validation
    async def validate_memory_request(self, 
                                    ctx: RunContextWrapper[Any],
                                    agent: Agent,
                                    input_data: Union[str, List[Dict[str, Any]]]) -> GuardrailFunctionOutput:
        """Validate memory request input"""
        try:
            if isinstance(input_data, str):
                # Simple validation for string inputs
                if len(input_data.strip()) == 0:
                    return GuardrailFunctionOutput(
                        output_info={"error": "Empty request"},
                        tripwire_triggered=True
                    )
                return GuardrailFunctionOutput(
                    output_info={"request": input_data},
                    tripwire_triggered=False
                )
            else:
                # More complex validation for structured inputs
                return GuardrailFunctionOutput(
                    output_info={"request": input_data},
                    tripwire_triggered=False
                )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"error": str(e)},
                tripwire_triggered=True
            )
    
    async def process_memory_request(self, request_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a memory-related request using the appropriate agent
        
        Args:
            request_type: Type of memory request (retrieve, create, reflect, etc.)
            params: Parameters for the request
            
        Returns:
            Response from the appropriate memory agent
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="Memory Request", group_id=self.trace_group_id):
            # Create the appropriate request format based on request type
            if request_type == "retrieve":
                request = f"Retrieve memories matching this query: {params.get('query', '')}"
                if params.get('memory_types'):
                    request += f"\nMemory types: {', '.join(params['memory_types'])}"
                if params.get('limit'):
                    request += f"\nLimit: {params['limit']}"
            
            elif request_type == "create":
                memory_text = params.get('memory_text', '')
                memory_type = params.get('memory_type', 'observation')
                request = f"Create a new {memory_type} memory with this content: {memory_text}"
            
            elif request_type == "reflect":
                topic = params.get('topic', '')
                request = f"Create a reflection about this topic: {topic}" if topic else "Create a general reflection based on recent memories"
            
            elif request_type == "experience":
                query = params.get('query', '')
                scenario = params.get('scenario_type', '')
                request = f"Retrieve experiences related to: {query}"
                if scenario:
                    request += f"\nScenario type: {scenario}"
            
            elif request_type == "maintenance":
                operation = params.get('operation', 'run_maintenance')
                request = f"Perform memory maintenance operation: {operation}"
            
            else:
                request = f"Process this memory-related request of type {request_type} with these parameters: {params}"
            
            # Configure run with tracing
            run_config = RunConfig(
                trace_id=f"memory-request-{request_type}-{self.user_id}",
                workflow_name=f"Memory {request_type.capitalize()} Request",
                group_id=self.trace_group_id,
                input_guardrails=[InputGuardrail(guardrail_function=self.validate_memory_request)],
                trace_metadata={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                    "request_type": request_type
                }
            )
            
            # Run the request through the main orchestrator agent
            result = await Runner.run(
                self.main_agent, 
                request,
                context={"user_id": self.user_id, "conversation_id": self.conversation_id},
                run_config=run_config
            )
            
            # If the result is a string, wrap it in a dictionary
            if isinstance(result.final_output, str):
                return {"result": result.final_output}
            
            return result.final_output
    
    # Convenience methods for common operations

    async def retrieve_memories_parallel(self, 
                                      query: str, 
                                      memory_types: List[str] = None,
                                      limit_per_type: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve memories of different types in parallel
        
        Args:
            query: Search query
            memory_types: List of memory types to retrieve
            limit_per_type: Maximum number of memories per type
            
        Returns:
            Dictionary of memories by type
        """
        if not self.initialized:
            await self.initialize()
        
        # Default memory types if not specified
        if memory_types is None:
            memory_types = ["observation", "reflection", "abstraction", "experience"]
        
        # Create tasks for each memory type
        tasks = {}
        for memory_type in memory_types:
            tasks[memory_type] = asyncio.create_task(
                self.memory_core.retrieve_memories(
                    query=query,
                    memory_types=[memory_type],
                    limit=limit_per_type
                )
            )
        
        # Wait for all tasks to complete
        results = {}
        for memory_type, task in tasks.items():
            try:
                memories = await task
                results[memory_type] = memories
            except Exception as e:
                logger.error(f"Error retrieving {memory_type} memories: {str(e)}")
                results[memory_type] = []
        
        return results
    
    async def retrieve_memories_with_prioritization(self,
                                                 query: str,
                                                 memory_types: List[str] = None,
                                                 prioritization: Dict[str, float] = None,
                                                 limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories with type prioritization
        
        Args:
            query: Search query
            memory_types: List of memory types to retrieve
            prioritization: Priority weights for each memory type
            limit: Total number of memories to return
            
        Returns:
            List of memories prioritized by type
        """
        # Retrieve memories in parallel
        memories_by_type = await self.retrieve_memories_parallel(
            query=query,
            memory_types=memory_types,
            limit_per_type=max(1, limit // (len(memory_types) if memory_types else 1))
        )
        
        # Default prioritization if not provided
        if prioritization is None:
            prioritization = {
                "experience": 0.4,
                "reflection": 0.3,
                "abstraction": 0.2,
                "observation": 0.1
            }
        
        # Calculate allocated slots based on prioritization
        total_priority = sum(prioritization.values())
        allocated_slots = {}
        
        for memory_type, priority in prioritization.items():
            # Skip if memory type wasn't retrieved
            if memory_type not in memories_by_type:
                continue
                
            # Calculate allocated slots proportionally
            allocated_slots[memory_type] = max(1, int(round((priority / total_priority) * limit)))
        
        # Ensure we don't exceed total limit
        while sum(allocated_slots.values()) > limit:
            # Find type with most slots and reduce by 1
            max_type = max(allocated_slots.items(), key=lambda x: x[1])[0]
            allocated_slots[max_type] -= 1
        
        # Combine memories according to allocation
        combined_memories = []
        
        for memory_type, slot_count in allocated_slots.items():
            # Get available memories for this type
            available_memories = memories_by_type.get(memory_type, [])
            
            # Take up to allocated slot count
            combined_memories.extend(available_memories[:slot_count])
        
        # Sort by relevance
        combined_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Limit to requested number
        return combined_memories[:limit]
    
    async def retrieve_memories(self, query: str, memory_types: List[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories matching a query"""
        params = {
            "query": query,
            "memory_types": memory_types,
            "limit": limit
        }
        with trace(workflow_name="Retrieve Memories", group_id=self.trace_group_id):
            result = await self.process_memory_request("retrieve", params)
            return result.get("memories", [])
    
    async def create_memory(self, memory_text: str, memory_type: str = "observation", 
                          tags: List[str] = None, significance: int = 5) -> str:
        """Create a new memory"""
        params = {
            "memory_text": memory_text,
            "memory_type": memory_type,
            "tags": tags or [],
            "significance": significance
        }
        with trace(workflow_name="Create Memory", group_id=self.trace_group_id):
            result = await self.process_memory_request("create", params)
            return result.get("memory_id", "")
    
    async def create_reflection(self, topic: str = None) -> Dict[str, Any]:
        """Create a reflection on a topic"""
        params = {"topic": topic}
        with trace(workflow_name="Create Reflection", group_id=self.trace_group_id):
            return await self.process_memory_request("reflect", params)
    
    async def retrieve_experiences(self, query: str, scenario_type: str = "", 
                                 entities: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a query"""
        params = {
            "query": query,
            "scenario_type": scenario_type,
            "entities": entities or []
        }
        with trace(workflow_name="Retrieve Experiences", group_id=self.trace_group_id):
            result = await self.process_memory_request("experience", params)
            return result.get("experiences", [])
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run memory system maintenance"""
        params = {"operation": "run_maintenance"}
        with trace(workflow_name="Memory Maintenance", group_id=self.trace_group_id):
            return await self.process_memory_request("maintenance", params)
