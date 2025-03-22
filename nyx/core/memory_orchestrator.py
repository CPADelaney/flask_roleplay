# nyx/core/memory_orchestrator.py

import asyncio
import logging
from typing import Dict, List, Any, Optional

from agents import Agent, Runner, trace, function_tool, handoff
from pydantic import BaseModel, Field

from nyx.core.memory_system import MemoryCore

logger = logging.getLogger(__name__)

class MemoryQuery(BaseModel):
    query: str
    memory_types: List[str] = Field(default_factory=lambda: ["observation", "reflection", "abstraction", "experience"])
    limit: int = 5
    min_significance: int = 3

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
        self.memory_core = MemoryCore(user_id, conversation_id)
        
        # Initialize agent dictionary
        self._agents = {}
        
        # Trace group ID for connecting traces
        self.trace_group_id = f"nyx-memory-{user_id}-{conversation_id}"
    
    async def initialize(self):
        """Initialize the memory orchestration system"""
        logger.info(f"Initializing memory orchestration system for user {self.user_id}")
        await self.memory_core.initialize()
        self._init_agents()
        logger.info("Memory orchestration system initialized")
    
    def _init_agents(self):
        """Initialize all the specialized agents"""
        # Initialize the main memory orchestrator agent
        self._agents["main"] = Agent(
            name="Memory Orchestrator",
            instructions="""You are the central memory orchestration system for Nyx AI.
            Your role is to coordinate memory operations, deciding which specialized
            agent to use for each memory task. Analyze the user's request and direct
            it to the appropriate specialized agent.""",
            handoffs=[
                self._get_retrieval_agent(),
                self._get_creation_agent(),
                self._get_reflection_agent(),
                self._get_experience_agent(),
                self._get_maintenance_agent()
            ]
        )
    
    def _get_retrieval_agent(self) -> Agent:
        """Get or create the memory retrieval agent"""
        if "retrieval" not in self._agents:
            self._agents["retrieval"] = Agent(
                name="Memory Retrieval Agent",
                handoff_description="Handles searching and retrieving memories based on queries",
                instructions="""You are specialized in retrieving memories from Nyx's memory system.
                Use the provided tools to effectively search for and retrieve memories based on the
                user's query parameters. Return the most relevant memories with appropriate formatting.""",
                tools=[
                    function_tool(self.memory_core.retrieve_memories),
                    function_tool(self.memory_core.get_memory),
                    function_tool(self.memory_core.retrieve_memories_with_formatting)
                ]
            )
        return self._agents["retrieval"]
    
    def _get_creation_agent(self) -> Agent:
        """Get or create the memory creation agent"""
        if "creation" not in self._agents:
            self._agents["creation"] = Agent(
                name="Memory Creation Agent",
                handoff_description="Handles creating and storing new memories",
                instructions="""You are specialized in creating and storing new memories for Nyx.
                When a user wants to create a new memory, help them structure it properly with
                appropriate metadata, tags, and significance levels. Ensure the memory is properly
                categorized and indexed.""",
                tools=[
                    function_tool(self.memory_core.add_memory),
                    function_tool(self.memory_core.update_memory),
                    function_tool(self.memory_core.delete_memory)
                ]
            )
        return self._agents["creation"]
    
    def _get_reflection_agent(self) -> Agent:
        """Get or create the reflection agent"""
        if "reflection" not in self._agents:
            self._agents["reflection"] = Agent(
                name="Reflection Agent",
                handoff_description="Creates reflections and abstractions from memories",
                instructions="""You are specialized in generating reflections and abstractions
                from Nyx's memories. Your role is to analyze memories, find patterns, and create
                higher-level insights that connect individual experiences.""",
                tools=[
                    function_tool(self.memory_core.create_reflection_from_memories),
                    function_tool(self.memory_core.create_abstraction_from_memories),
                    function_tool(self.memory_core.retrieve_memories)
                ]
            )
        return self._agents["reflection"]
    
    def _get_experience_agent(self) -> Agent:
        """Get or create the experience agent"""
        if "experience" not in self._agents:
            self._agents["experience"] = Agent(
                name="Experience Agent",
                handoff_description="Manages experience recall and narrative generation",
                instructions="""You are specialized in managing Nyx's experiences and narratives.
                Your role is to retrieve relevant experiences, generate natural conversational
                recalls, and construct coherent narratives from memories.""",
                tools=[
                    function_tool(self.memory_core.retrieve_relevant_experiences),
                    function_tool(self.memory_core.generate_conversational_recall),
                    function_tool(self.memory_core.construct_narrative_from_memories)
                ]
            )
        return self._agents["experience"]
    
    def _get_maintenance_agent(self) -> Agent:
        """Get or create the maintenance agent"""
        if "maintenance" not in self._agents:
            self._agents["maintenance"] = Agent(
                name="Memory Maintenance Agent",
                handoff_description="Handles memory system maintenance, decay, and consolidation",
                instructions="""You are specialized in maintaining the health and efficiency of
                Nyx's memory system. Your role is to apply memory decay, consolidate similar
                memories, archive old memories, and ensure the overall system stays optimized.""",
                tools=[
                    function_tool(self.memory_core.run_maintenance),
                    function_tool(self.memory_core.apply_memory_decay),
                    function_tool(self.memory_core.consolidate_memory_clusters),
                    function_tool(self.memory_core.get_memory_stats)
                ]
            )
        return self._agents["maintenance"]
        
    async def process_memory_request(self, request_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a memory-related request using the appropriate agent
        
        Args:
            request_type: Type of memory request (retrieve, create, reflect, etc.)
            params: Parameters for the request
            
        Returns:
            Response from the appropriate memory agent
        """
        with trace(workflow_name="Memory Request", group_id=self.trace_group_id):
            if not self.memory_core.initialized:
                await self.memory_core.initialize()
            
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
            
            # Run the request through the main orchestrator agent
            result = await Runner.run(
                self._agents["main"], 
                request,
                run_config={
                    "trace_id": f"memory-request-{request_type}-{self.user_id}",
                    "workflow_name": f"Memory {request_type.capitalize()} Request",
                    "group_id": self.trace_group_id
                }
            )
            
            # If the result is a string, wrap it in a dictionary
            if isinstance(result.final_output, str):
                return {"result": result.final_output}
            
            return result.final_output
    
    # Convenience methods for common operations
    
    async def retrieve_memories(self, query: str, memory_types: List[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories matching a query"""
        params = {
            "query": query,
            "memory_types": memory_types,
            "limit": limit
        }
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
        result = await self.process_memory_request("create", params)
        return result.get("memory_id", "")
    
    async def create_reflection(self, topic: str = None) -> Dict[str, Any]:
        """Create a reflection on a topic"""
        params = {"topic": topic}
        return await self.process_memory_request("reflect", params)
    
    async def retrieve_experiences(self, query: str, scenario_type: str = "", 
                                 entities: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a query"""
        params = {
            "query": query,
            "scenario_type": scenario_type,
            "entities": entities or []
        }
        result = await self.process_memory_request("experience", params)
        return result.get("experiences", [])
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run memory system maintenance"""
        params = {"operation": "run_maintenance"}
        return await self.process_memory_request("maintenance", params)
