# nyx/memory_integration_sdk.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import asyncpg

from agents import Agent, function_tool, Runner, trace
from agents import ModelSettings, RunConfig
from pydantic import BaseModel, Field

from db.connection import get_db_connection
from nyx.nyx_memory_system import NyxMemorySystem
from logic.npc_agents.agent_coordinator import NPCAgentCoordinator
from utils.caching import MEMORY_CACHE

logger = logging.getLogger(__name__)

# ===== Pydantic Models for Structured Outputs =====

class Memory(BaseModel):
    """Structured representation of a memory"""
    memory_text: str = Field(..., description="The content of the memory")
    memory_type: str = Field("observation", description="Type of memory (observation, reflection, abstraction)")
    significance: int = Field(5, description="Importance of memory (1-10)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
class MemoryQuery(BaseModel):
    """Query parameters for retrieving memories"""
    query: str = Field(..., description="Search query")
    relevance_threshold: float = Field(0.6, description="Minimum relevance score (0.0-1.0)")
    max_results: int = Field(5, description="Maximum number of results to return")
    include_observation: bool = Field(True, description="Include observation memories")
    include_reflection: bool = Field(True, description="Include reflection memories")
    include_abstraction: bool = Field(True, description="Include abstraction memories")

class MemoryReflection(BaseModel):
    """Structured output for memory reflections"""
    reflection: str = Field(..., description="The reflection text")
    confidence: float = Field(..., description="Confidence level in the reflection (0.0-1.0)")
    source_memories: List[str] = Field(default_factory=list, description="IDs of source memories")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

class MemoryAbstraction(BaseModel):
    """Structured output for memory abstractions"""
    abstraction: str = Field(..., description="The abstraction text")
    pattern_type: str = Field(..., description="Type of pattern identified")
    source_memories: List[str] = Field(default_factory=list, description="IDs of source memories")

# ===== Memory Context Object =====

class MemoryContext:
    """Context object for memory agents"""
    def __init__(self, user_id: int, conversation_id: int = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = NyxMemorySystem(user_id, conversation_id)
        self.query_context = {}

# ===== Function Tools =====

@function_tool
async def add_memory(
    ctx,
    memory_text: str,
    memory_type: str = "observation",
    memory_scope: str = "game",
    significance: int = 5,
    tags: List[str] = None
) -> str:
    """
    Add a new memory to the appropriate scope.
    
    Args:
        memory_text: The content of the memory
        memory_type: Type of memory (observation, reflection, abstraction)
        memory_scope: Scope of memory (global, user, or game)
        significance: Importance of memory (1-10)
        tags: List of tags for categorization
    """
    tags = tags or []
    
    memory_system = ctx.context.memory_system
    
    # Add timestamp to metadata
    metadata = {"created_at": datetime.now().isoformat()}
    
    # Add memory
    memory_id = await memory_system.add_memory(
        memory_text=memory_text,
        memory_type=memory_type,
        memory_scope=memory_scope,
        significance=significance,
        tags=tags,
        metadata=metadata
    )
    
    return f"Memory added with ID: {memory_id}"

@function_tool
async def retrieve_memories(
    ctx,
    query: str,
    memory_types: List[str] = None,
    scopes: List[str] = None,
    limit: int = 5,
    min_significance: int = 3
) -> str:
    """
    Retrieve memories relevant to a query.
    
    Args:
        query: Search query
        memory_types: Types of memories to include
        scopes: Memory scopes to search
        limit: Maximum number of results
        min_significance: Minimum significance level
    """
    memory_types = memory_types or ["observation", "reflection", "abstraction"]
    scopes = scopes or ["game", "user", "global"]
    
    memory_system = ctx.context.memory_system
    
    # Retrieve memories
    memories = await memory_system.retrieve_memories(
        query=query,
        memory_types=memory_types,
        scopes=scopes,
        limit=limit,
        min_significance=min_significance,
        context=ctx.context.query_context
    )
    
    # Format results
    result = []
    for memory in memories:
        formatted_memory = {
            "id": memory.get("id"),
            "text": memory.get("memory_text"),
            "type": memory.get("memory_type"),
            "significance": memory.get("significance"),
            "relevance": memory.get("relevance"),
            "tags": memory.get("tags", [])
        }
        result.append(formatted_memory)
    
    return json.dumps(result)

@function_tool
async def get_memory_stats(ctx) -> str:
    """
    Get statistics about memories for this user/conversation.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Calculate statistics
    stats = {}
    
    async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
        async with pool.acquire() as conn:
            # Game memories count
            if conversation_id:
                game_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM NyxMemories
                    WHERE user_id = $1 AND conversation_id = $2
                """, user_id, conversation_id)
                stats["game_memories"] = game_count
                
                # Memory types distribution
                rows = await conn.fetch("""
                    SELECT memory_type, COUNT(*) 
                    FROM NyxMemories
                    WHERE user_id = $1 AND conversation_id = $2
                    GROUP BY memory_type
                """, user_id, conversation_id)
                
                stats["memory_types"] = {row["memory_type"]: row["count"] for row in rows}
            
            # User memories count
            user_count = await conn.fetchval("""
                SELECT COUNT(*) FROM NyxUserMemories
                WHERE user_id = $1
            """, user_id)
            stats["user_memories"] = user_count
            
            # Most significant memories
            if conversation_id:
                top_memories = await conn.fetch("""
                    SELECT memory_text, significance 
                    FROM NyxMemories
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY significance DESC
                    LIMIT 3
                """, user_id, conversation_id)
                
                stats["top_memories"] = [{"text": row["memory_text"], "significance": row["significance"]} for row in top_memories]
    
    return json.dumps(stats)

async def retrieve_npc_memories(self, query, npc_ids=None):
    """Retrieve memories from specific NPCs"""
    coordinator = NPCAgentCoordinator(self.user_id, self.conversation_id)
    npc_ids = await coordinator.load_agents(npc_ids)
    
    all_memories = []
    for npc_id in npc_ids:
        if npc_id in coordinator.active_agents:
            agent = coordinator.active_agents[npc_id]
            memory_result = await agent.memory_manager.retrieve_memories(query)
            for memory in memory_result.get("memories", []):
                memory["npc_id"] = npc_id
                all_memories.append(memory)
    
    return all_memories

# ===== Memory Agents =====

# Memory Retrieval Agent
memory_retrieval_agent = Agent[MemoryContext](
    name="Memory Retrieval Agent",
    instructions="""You specialize in retrieving relevant memories for Nyx.
    
Your role is to:
1. Find memories that are relevant to the current context
2. Determine the appropriate memory types to search for
3. Apply the right significance threshold for the query
4. Format memories for easy comprehension by other agents
5. Include relevance scores to help prioritize memories

Consider both direct relevance (keyword matching) and semantic relevance (thematic connections).
Be thorough yet selective, focusing on memories that provide the most context value.""",
    tools=[retrieve_memories, get_memory_stats]
)

# Memory Creation Agent
memory_creation_agent = Agent[MemoryContext](
    name="Memory Creation Agent",
    instructions="""You create new memories for Nyx based on observations and interactions.
    
Your role is to:
1. Transform observations into concise, factual memory entries
2. Assign appropriate significance levels based on content importance
3. Apply relevant tags for future retrieval
4. Determine the proper memory type (observation, reflection, abstraction)
5. Select the correct scope (game, user, global) based on applicability

Focus on creating memories that will be useful for understanding patterns, preferences,
and context in future interactions. Be descriptive but concise.""",
    output_type=Memory,
    tools=[add_memory]
)

# Memory Reflection Agent
memory_reflection_agent = Agent[MemoryContext](
    name="Memory Reflection Agent",
    instructions="""You create insightful reflections based on patterns in memories.
    
Your role is to:
1. Analyze collections of memories to identify patterns and trends
2. Form higher-level reflections that capture meaning beyond individual memories
3. Assess confidence level in reflections based on evidence strength
4. Create reflections that help Nyx understand the player and context
5. Write reflections from Nyx's first-person perspective

Your reflections should demonstrate insightful pattern recognition while
maintaining appropriate confidence based on evidence quality.""",
    output_type=MemoryReflection,
    tools=[retrieve_memories]
)

# Memory Abstraction Agent
memory_abstraction_agent = Agent[MemoryContext](
    name="Memory Abstraction Agent",
    instructions="""You create semantic abstractions from specific memories.
    
Your role is to:
1. Extract general principles from specific events
2. Identify behavioral patterns from individual observations
3. Create abstractions that can apply to similar situations
4. Ensure abstractions maintain the essential meaning of source memories
5. Classify the type of pattern the abstraction represents

Focus on creating abstractions that will help Nyx better understand
recurring patterns in player behavior and preferences.""",
    output_type=MemoryAbstraction,
    tools=[retrieve_memories, add_memory]
)

# Memory Management Agent (Orchestrator)
memory_management_agent = Agent[MemoryContext](
    name="Memory Management Agent",
    instructions="""You orchestrate Nyx's memory system, coordinating between specialized memory agents.
    
Your role is to:
1. Determine when to create new memories from observations
2. Identify opportunities for generating reflections or abstractions
3. Retrieve relevant memories based on context
4. Decide which memory operations are most valuable in the current context
5. Manage the overall memory system for optimal performance

Coordinate between the specialized memory agents to ensure Nyx maintains
a useful and well-organized memory system.""",
    handoffs=[
        handoff(memory_retrieval_agent, tool_name_override="retrieve_relevant_memories"),
        handoff(memory_creation_agent, tool_name_override="create_new_memory"),
        handoff(memory_reflection_agent, tool_name_override="generate_reflection"),
        handoff(memory_abstraction_agent, tool_name_override="create_abstraction")
    ],
    tools=[get_memory_stats]
)

# ===== Main Functions =====

@function_tool
async def construct_narrative(
    ctx, 
    topic: str, 
    context: Optional[Dict[str, Any]] = None,
    limit: int = 5,
    require_chronological: bool = True
) -> str:
    """
    Construct a coherent narrative from related memories.
    
    Args:
        topic: The topic to construct a narrative about
        context: Optional context information
        limit: Maximum number of memories to include
        require_chronological: Whether to enforce chronological ordering
    """
    memory_system = ctx.context.memory_system
    context = context or {}
    
    # Retrieve relevant memories (using SDK approach)
    memories_result = await retrieve_memories(ctx, query=topic, limit=limit)
    memories = json.loads(memories_result)
    
    if not memories:
        return json.dumps({
            "narrative": f"I don't have any significant memories about {topic}.",
            "sources": [],
            "confidence": 0.2
        })
    
    # Sort chronologically if required
    if require_chronological and "timestamp" in memories[0]:
        memories.sort(key=lambda x: x.get("timestamp", ""))
    
    # Extract memory texts
    memory_texts = [m.get("text", "") for m in memories]
    source_ids = [m.get("id", "") for m in memories]
    
    # Calculate confidence based on memory significance and recall frequency
    avg_significance = sum(m.get("significance", 0) for m in memories) / len(memories)
    avg_recalled = sum(m.get("times_recalled", 0) for m in memories) / len(memories)
    base_confidence = min(0.9, (avg_significance / 10.0) * 0.7 + (min(1.0, avg_recalled / 5.0) * 0.3))
    
    # Generate narrative using LLM (implement with SDK approach)
    prompt = f"""
    As Nyx, construct a coherent narrative about "{topic}" based on these memories:
    
    {memory_texts}
    
    Confidence level: {base_confidence:.2f}
    
    1. Begin the narrative with an appropriate confidence marker.
    2. Weave the memories into a coherent story, filling minimal gaps as needed.
    3. If memories seem contradictory, acknowledge the uncertainty.
    4. Keep it concise (under 200 words).
    5. Write in first person as Nyx.
    """
    
    # Call LLM using SDK
    response = await Runner.run(
        reflection_agent,
        prompt,
        context=ctx.context
    )
    
    reflection = response.final_output_as(MemoryReflection)
    narrative = reflection.reflection
    
    return json.dumps({
        "narrative": narrative,
        "sources": source_ids,
        "confidence": base_confidence,
        "memory_count": len(memories)
    })

@function_tool
async def reconsolidate_memory(ctx, memory_id: int, context: Dict[str, Any] = None) -> str:
    """
    Reconsolidate (slightly alter) a memory when it's recalled.
    This simulates how human memories change slightly each time they're accessed.
    
    Args:
        memory_id: The ID of the memory to reconsolidate
        context: Current context that might influence reconsolidation
    """
    context = context or {}
    
    # Get the memory
    memory_data = await retrieve_memory_by_id(ctx, memory_id=memory_id)
    memory = json.loads(memory_data)
    
    if not memory:
        return json.dumps({"error": "Memory not found"})
    
    memory_text = memory.get("text", "")
    significance = memory.get("significance", 5)
    memory_type = memory.get("type", "observation")
    
    # Only reconsolidate episodic memories with low/medium significance
    if memory_type != "observation" or significance >= 8:
        return json.dumps({"status": "Memory not eligible for reconsolidation"})
    
    # Current emotional state can influence reconsolidation
    current_emotion = context.get("emotional_state", "neutral")
    
    # Reconsolidation varies by memory age and significance
    reconsolidation_strength = min(0.3, significance / 10.0)
    
    # Generate a slightly altered version (implement using SDK)
    prompt = f"""
    Slightly alter this memory to simulate memory reconsolidation effects.
    
    Original memory: {memory_text}
    Emotional context: {current_emotion}
    
    Create a very slight alteration that:
    1. Maintains the same core information and meaning
    2. Makes subtle changes to wording or emphasis ({int(reconsolidation_strength * 100)}% alteration)
    3. Slightly enhances aspects that align with the "{current_emotion}" emotional state
    4. Never changes key facts, names, or locations
    
    Return only the altered text.
    """
    
    # Call LLM using SDK
    response = await Runner.run(
        memory_agent,
        prompt,
        context=ctx.context
    )
    
    altered_memory = response.text
    
    # Update the memory with altered text
    # Implement using SDK approach
    update_result = await update_memory(ctx, memory_id=memory_id, memory_text=altered_memory)
    
    return json.dumps({
        "original": memory_text,
        "altered": altered_memory,
        "reconsolidation_strength": reconsolidation_strength
    })

async def process_memory_operation(
    user_id: int,
    conversation_id: int,
    operation_type: str,
    query: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a memory operation through the appropriate agent
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        operation_type: Type of memory operation to perform
        query: Query or content for the memory operation
        context_data: Additional context data
        
    Returns:
        Result of the memory operation
    """
    # Create memory context
    memory_context = MemoryContext(user_id, conversation_id)
    
    # Add any additional context
    if context_data:
        memory_context.query_context = context_data
    
    # Create trace for monitoring
    with trace(
        workflow_name="Memory System",
        trace_id=f"memory-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        if operation_type == "create":
            # Run the memory creation agent
            result = await Runner.run(
                memory_creation_agent,
                query,
                context=memory_context
            )
            
            # Get structured output
            memory = result.final_output_as(Memory)
            
            # Actually create the memory
            memory_id = await memory_context.memory_system.add_memory(
                memory_text=memory.memory_text,
                memory_type=memory.memory_type,
                memory_scope="game",
                significance=memory.significance,
                tags=memory.tags
            )
            
            return {
                "memory_id": memory_id,
                "memory_text": memory.memory_text,
                "memory_type": memory.memory_type,
                "significance": memory.significance
            }
            
        elif operation_type == "retrieve":
            # Run the memory retrieval agent
            result = await Runner.run(
                memory_retrieval_agent,
                query,
                context=memory_context
            )
            
            # Extract memories from the result
            memories = []
            for item in result.new_items:
                if item.type == "tool_call_output_item" and "retrieve_memories" in str(item.raw_item):
                    try:
                        memories = json.loads(item.output)
                        break
                    except:
                        pass
            
            return {
                "memories": memories,
                "query": query
            }
            
        elif operation_type == "reflect":
            # Run the memory reflection agent
            result = await Runner.run(
                memory_reflection_agent,
                query,
                context=memory_context
            )
            
            # Get structured output
            reflection = result.final_output_as(MemoryReflection)
            
            # Store the reflection as a memory
            memory_id = await memory_context.memory_system.add_memory(
                memory_text=reflection.reflection,
                memory_type="reflection",
                memory_scope="game",
                significance=6,
                tags=reflection.tags
            )
            
            return {
                "reflection": reflection.reflection,
                "confidence": reflection.confidence,
                "memory_id": memory_id
            }
            
        elif operation_type == "abstract":
            # Run the memory abstraction agent
            result = await Runner.run(
                memory_abstraction_agent,
                query,
                context=memory_context
            )
            
            # Get structured output
            abstraction = result.final_output_as(MemoryAbstraction)
            
            # Store the abstraction as a memory
            memory_id = await memory_context.memory_system.add_memory(
                memory_text=abstraction.abstraction,
                memory_type="abstraction",
                memory_scope="game",
                significance=6,
                tags=["abstraction", abstraction.pattern_type]
            )
            
            return {
                "abstraction": abstraction.abstraction,
                "pattern_type": abstraction.pattern_type,
                "memory_id": memory_id
            }
            
        else:
            # Handle orchestration by the memory management agent
            result = await Runner.run(
                memory_management_agent,
                query,
                context=memory_context
            )
            
            # Process the results from potential handoffs
            response = {
                "operation": "orchestration",
                "actions_taken": []
            }
            
            for item in result.new_items:
                if item.type == "handoff_output_item":
                    action = item.raw_item.function.name.replace("_", " ")
                    response["actions_taken"].append(action)
                    
                    # Add specific results when available
                    if "generate_reflection" in str(item.raw_item):
                        try:
                            reflection_data = json.loads(item.raw_item.content)
                            response["reflection"] = reflection_data.get("reflection")
                        except:
                            pass
                    elif "create_abstraction" in str(item.raw_item):
                        try:
                            abstraction_data = json.loads(item.raw_item.content)
                            response["abstraction"] = abstraction_data.get("abstraction")
                        except:
                            pass
            
            return response

async def perform_memory_maintenance(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Perform maintenance on the memory system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Results of the maintenance operation
    """
    # Create memory context
    memory_context = MemoryContext(user_id, conversation_id)
    
    # Run maintenance on the memory system
    await memory_context.memory_system.run_maintenance()
    
    # Get memory statistics after maintenance
    stats = json.loads(await get_memory_stats(memory_context))
    
    return {
        "status": "Maintenance completed",
        "stats": stats
    }

async def create_memory_reflection_on_topic(
    user_id: int,
    conversation_id: int,
    topic: str
) -> Dict[str, Any]:
    """
    Create a reflection on a specific topic using memory agents
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        topic: Topic to reflect on
        
    Returns:
        Reflection data
    """
    # Create memory context
    memory_context = MemoryContext(user_id, conversation_id)
    
    # Generate prompt for reflection
    prompt = f"Create a thoughtful reflection on the topic: {topic}"
    
    # Run the memory reflection agent
    result = await Runner.run(
        memory_reflection_agent,
        prompt,
        context=memory_context
    )
    
    # Get structured output
    reflection = result.final_output_as(MemoryReflection)
    
    # Store the reflection
    memory_id = await memory_context.memory_system.add_memory(
        memory_text=reflection.reflection,
        memory_type="reflection",
        memory_scope="game",
        significance=7,
        tags=["reflection", "topic", topic]
    )
    
    return {
        "reflection": reflection.reflection,
        "confidence": reflection.confidence,
        "memory_id": memory_id,
        "topic": topic
    }
