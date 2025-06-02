# nyx/core/memory_orchestrator.py

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union

from agents import Agent, Runner, handoff, RunConfig
from pydantic import BaseModel, Field

from nyx.core.memory_core import MemoryCoreAgents, MemoryContext

logger = logging.getLogger(__name__)

# ==================== Request Models ====================

class MemoryRequest(BaseModel):
    """Base memory request"""
    request_type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class RetrieveRequest(BaseModel):
    """Memory retrieval request"""
    query: str
    memory_types: List[str] = Field(default_factory=lambda: ["observation", "reflection", "experience"])
    limit: int = 10
    min_significance: int = 3

class CreateRequest(BaseModel):
    """Memory creation request"""
    memory_text: str
    memory_type: str = "observation"
    memory_scope: str = "game"
    significance: int = 5
    tags: List[str] = Field(default_factory=list)

class ReflectionRequest(BaseModel):
    """Reflection creation request"""
    topic: Optional[str] = None
    memory_ids: Optional[List[str]] = None

class MaintenanceRequest(BaseModel):
    """Maintenance request"""
    operation: str = "full"  # full, decay, stats

# ==================== Specialized Agents ====================

def create_orchestrator_agent(sub_agents: List[Agent]) -> Agent:
    """Create the main orchestrator agent"""
    handoffs_list = []
    
    for agent in sub_agents:
        # Create handoff for each sub-agent
        handoffs_list.append(
            handoff(
                agent,
                tool_description_override=f"Delegate to {agent.name}"
            )
        )
    
    return Agent(
        name="Memory Orchestrator",
        instructions="""You coordinate memory operations for Nyx. 
        Analyze requests and delegate to the appropriate specialist:
        - Memory Manager: General operations, creation, updates
        - Retrieval Specialist: Searching and retrieving memories  
        - Reflection Specialist: Creating reflections and abstractions
        - Maintenance Specialist: System maintenance and statistics
        
        Parse the request and delegate appropriately.""",
        handoffs=handoffs_list
    )

def create_maintenance_agent(memory_core: MemoryCoreAgents) -> Agent:
    """Create maintenance specialist agent"""
    from agents import function_tool
    
    @function_tool
    async def get_stats(ctx) -> Dict[str, Any]:
        """Get memory system statistics"""
        return await memory_core.get_memory_stats()
    
    @function_tool
    async def run_full_maintenance(ctx) -> Dict[str, Any]:
        """Run full maintenance cycle"""
        return await memory_core.run_maintenance()
    
    @function_tool
    async def run_memory_decay(ctx) -> Dict[str, Any]:
        """Run memory decay"""
        from nyx.core.memory_core import apply_decay
        return await apply_decay(ctx)
    
    @function_tool
    async def run_consolidation(ctx) -> Dict[str, Any]:
        """Run memory consolidation"""
        from nyx.core.memory_core import consolidate_memories
        return await consolidate_memories(ctx)
    
    return Agent(
        name="Maintenance Specialist",
        instructions="""You handle memory system maintenance.
        You can run maintenance cycles, apply decay, consolidate memories,
        and provide system statistics.""",
        tools=[get_stats, run_full_maintenance, run_memory_decay, run_consolidation]
    )

# ==================== Memory Orchestrator ====================

class MemoryOrchestrator:
    """Simplified memory orchestration using agents"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Create memory core
        self.memory_core = MemoryCoreAgents(user_id, conversation_id)
        
        # Will be initialized on first use
        self.initialized = False
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize the orchestration system"""
        if self.initialized:
            return
            
        logger.info(f"Initializing memory orchestrator for user {self.user_id}")
        
        # Initialize memory core
        await self.memory_core.initialize()
        
        # Get agents from memory core
        memory_agent = self.memory_core.memory_agent
        retrieval_agent = self.memory_core.retrieval_agent
        reflection_agent = self.memory_core.reflection_agent
        
        # Create maintenance agent
        maintenance_agent = create_maintenance_agent(self.memory_core)
        
        # Create orchestrator with all sub-agents
        self.orchestrator = create_orchestrator_agent([
            memory_agent,
            retrieval_agent,
            reflection_agent,
            maintenance_agent
        ])
        
        self.initialized = True
        logger.info("Memory orchestrator initialized")
    
    # ==================== Public API - Maintains Compatibility ====================
    
    async def retrieve_memories(self, query: str, memory_types: List[str] = None, 
                              limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories matching a query"""
        await self.initialize()
        
        # Direct call to memory core for simplicity
        return await self.memory_core.retrieve_memories(
            query=query,
            memory_types=memory_types,
            limit=limit
        )
    
    async def create_memory(self, memory_text: str, memory_type: str = "observation",
                          tags: List[str] = None, significance: int = 5) -> str:
        """Create a new memory"""
        await self.initialize()
        
        # Direct call to memory core
        return await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            tags=tags or [],
            significance=significance
        )
    
    async def create_reflection(self, topic: str = None) -> Dict[str, Any]:
        """Create a reflection on a topic"""
        await self.initialize()
        
        # Direct call to memory core
        return await self.memory_core.create_reflection(topic=topic)
    
    async def retrieve_experiences(self, query: str, scenario_type: str = "",
                                 entities: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant experiences"""
        await self.initialize()
        
        # Direct call to memory core
        return await self.memory_core.retrieve_experiences(
            query=query,
            scenario_type=scenario_type,
            limit=3
        )
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run memory system maintenance"""
        await self.initialize()
        
        # Direct call to memory core
        return await self.memory_core.run_maintenance()
    
    async def process_memory_request(self, request_type: str, 
                                   params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a memory request using the orchestrator"""
        await self.initialize()
        
        # Format request for orchestrator
        if request_type == "retrieve":
            request = f"Retrieve memories: {params}"
        elif request_type == "create":
            request = f"Create memory: {params}"
        elif request_type == "reflect":
            request = f"Create reflection: {params}"
        elif request_type == "maintenance":
            request = f"Run maintenance: {params}"
        else:
            request = f"Process {request_type} with params: {params}"
        
        # Run through orchestrator
        result = await Runner.run(
            self.orchestrator,
            request,
            context=self.memory_core.context
        )
        
        # Return result
        if isinstance(result.final_output, dict):
            return result.final_output
        return {"result": str(result.final_output)}
    
    # ==================== Additional Convenience Methods ====================
    
    async def retrieve_memories_parallel(self, query: str, 
                                       memory_types: List[str] = None,
                                       limit_per_type: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve memories of different types in parallel"""
        await self.initialize()
        
        if memory_types is None:
            memory_types = ["observation", "reflection", "experience"]
        
        # Create parallel tasks
        tasks = {}
        for memory_type in memory_types:
            tasks[memory_type] = asyncio.create_task(
                self.memory_core.retrieve_memories(
                    query=query,
                    memory_types=[memory_type],
                    limit=limit_per_type
                )
            )
        
        # Gather results
        results = {}
        for memory_type, task in tasks.items():
            try:
                results[memory_type] = await task
            except Exception as e:
                logger.error(f"Error retrieving {memory_type}: {e}")
                results[memory_type] = []
        
        return results
    
    async def retrieve_memories_with_prioritization(self, query: str,
                                                  memory_types: List[str] = None,
                                                  prioritization: Dict[str, float] = None,
                                                  limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories with type prioritization"""
        # Get memories by type
        memories_by_type = await self.retrieve_memories_parallel(
            query=query,
            memory_types=memory_types,
            limit_per_type=max(3, limit // len(memory_types or ["observation"]))
        )
        
        # Default prioritization
        if prioritization is None:
            prioritization = {
                "experience": 0.4,
                "reflection": 0.3,
                "observation": 0.2,
                "abstraction": 0.1
            }
        
        # Combine with prioritization
        all_memories = []
        for memory_type, memories in memories_by_type.items():
            weight = prioritization.get(memory_type, 0.1)
            for memory in memories:
                memory["_weight"] = weight
                all_memories.append(memory)
        
        # Sort by weighted relevance
        all_memories.sort(
            key=lambda m: m.get("relevance", 0.5) * m.get("_weight", 1.0),
            reverse=True
        )
        
        # Clean up and return
        for memory in all_memories:
            memory.pop("_weight", None)
        
        return all_memories[:limit]
