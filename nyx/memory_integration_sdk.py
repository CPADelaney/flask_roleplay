# nyx/memory_integration_sdk.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncpg

from agents import Agent, function_tool, Runner, trace
from agents import ModelSettings, RunConfig
from pydantic import BaseModel, Field

from db.connection import get_db_connection
from memory.memory_nyx_integration import MemoryNyxBridge
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
    def __init__(self, user_id: int, conversation_id: int = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        # Don't initialize memory_system here—just set to None
        self.memory_system = None
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
    Add a new memory to the appropriate scope (NEW version).
    This version routes everything through Nyx governance (remember_through_nyx).
    """
    from memory.memory_nyx_integration import remember_through_nyx

    tags = tags or []
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id

    # Decide how to interpret memory_scope => entity_type/entity_id
    if memory_scope == "game":
        entity_type = "nyx"   # or "global"
        entity_id = 0
    else:
        entity_type = "player"
        entity_id = user_id
    
    # Map significance -> importance string for bridging calls
    importance_map = {
        range(1, 3):  "low",
        range(3, 6):  "medium",
        range(6, 9):  "high",
        range(9, 11): "critical"
    }
    importance = "medium"
    for rng, label in importance_map.items():
        if significance in rng:
            importance = label
            break

    # Now call the bridging function
    result = await remember_through_nyx(
        user_id=user_id,
        conversation_id=conversation_id,
        entity_type=entity_type,
        entity_id=entity_id,
        memory_text=memory_text,
        importance=importance,
        emotional=True,  # if you want emotional analysis
        tags=tags
    )

    # If governance denied the op, we get an error. Otherwise, success.
    if "error" in result:
        return f"Memory creation not approved: {result['error']}"
    return f"Memory added with ID: {result.get('memory_id')}"

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

# Add to memory_integration_sdk.py

@function_tool
async def consolidate_memories(ctx) -> str:
    """
    Consolidate related memories into higher-level semantic memories.
    This simulates how episodic memories transform into semantic knowledge over time.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Find clusters of related memories to consolidate
            memory_rows = await conn.fetch("""
                SELECT id, memory_text, tags, embedding
                FROM NyxMemories
                WHERE user_id = $1 
                AND conversation_id = $2
                AND memory_type = 'observation'
                AND is_archived = FALSE
                AND timestamp < NOW() - INTERVAL '3 days'
                AND times_recalled >= 2
            """, user_id, conversation_id)
            
            if not memory_rows:
                return "No memories found suitable for consolidation"
                
            # Convert to easily workable format
            memories = []
            for row in memory_rows:
                memories.append({
                    "id": row["id"],
                    "text": row["memory_text"],
                    "tags": row["tags"],
                    "embedding": row["embedding"],
                })
            
            # Find clusters using embedding similarity
            clusters = cluster_memories_by_similarity(memories)
            
            # For each significant cluster, create a consolidated memory
            consolidated_count = 0
            for cluster in clusters:
                if len(cluster) >= 3:  # Only consolidate clusters with several memories
                    cluster_ids = [m["id"] for m in cluster]
                    cluster_texts = [m["text"] for m in cluster]
                    
                    # Extract all tags from the cluster
                    all_tags = []
                    for m in cluster:
                        all_tags.extend(m.get("tags", []))
                    unique_tags = list(set(all_tags))
                    
                    # Generate a consolidated summary
                    summary = await generate_consolidated_summary(cluster_texts)
                    
                    # Store the consolidated memory
                    consolidated_id = await conn.fetchval("""
                        INSERT INTO NyxMemories (
                            user_id, conversation_id, memory_text, memory_type,
                            significance, embedding, timestamp,
                            tags, times_recalled, is_archived,
                            metadata
                        )
                        VALUES ($1, $2, $3, 'consolidated', 6, $4, CURRENT_TIMESTAMP, 
                                $5, 0, FALSE, $6)
                        RETURNING id
                    """,
                        user_id,
                        conversation_id,
                        summary,
                        await generate_embedding(summary),
                        unique_tags + ["consolidated"],
                        json.dumps({"source_memory_ids": cluster_ids})
                    )
                    
                    # Update the original memories to mark them as consolidated
                    # But don't archive them yet - they're still valuable
                    await conn.execute("""
                        UPDATE NyxMemories
                        SET is_consolidated = TRUE
                        WHERE id = ANY($1)
                    """, cluster_ids)
                    
                    consolidated_count += 1
    
    return f"Created {consolidated_count} consolidated memories from {len(clusters)} memory clusters"

@function_tool
async def apply_memory_decay(ctx) -> str:
    """
    Apply decay to memories based on age, significance, and recall frequency.
    This simulates how human memories fade over time, especially less important ones.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Get memories that haven't been recalled recently
            memory_rows = await conn.fetch("""
                SELECT id, significance, times_recalled, 
                       EXTRACT(EPOCH FROM (NOW() - timestamp)) / 86400 AS days_old,
                       EXTRACT(EPOCH FROM (NOW() - COALESCE(last_recalled, timestamp))) / 86400 AS days_since_recall
                FROM NyxMemories
                WHERE user_id = $1 
                AND conversation_id = $2
                AND is_archived = FALSE
                AND memory_type = 'observation'  -- Only decay episodic memories
            """, user_id, conversation_id)
            
            decayed_count = 0
            archived_count = 0
            
            for row in memory_rows:
                memory_id = row["id"]
                significance = row["significance"]
                times_recalled = row["times_recalled"]
                days_old = row["days_old"]
                days_since_recall = row["days_since_recall"]
                
                # Calculate decay factors
                age_factor = min(1.0, days_old / 30.0)  # Older memories decay more
                recall_factor = max(0.0, 1.0 - (times_recalled / 10.0))  # Frequently recalled memories decay less
                
                # How much to reduce significance
                # Memories decay faster if they're old AND haven't been recalled recently
                decay_rate = 0.1 * age_factor * recall_factor
                if days_since_recall > 7:
                    decay_rate *= 1.5  # Extra decay for memories not recalled in a week
                
                # Apply decay with a floor of 1 for significance
                new_significance = max(1.0, significance - decay_rate)
                
                # If significance drops below threshold, archive the memory
                if new_significance < 2.0 and days_old > 14:
                    await conn.execute("""
                        UPDATE NyxMemories
                        SET is_archived = TRUE
                        WHERE id = $1
                    """, memory_id)
                    archived_count += 1
                else:
                    # Otherwise just update the significance
                    await conn.execute("""
                        UPDATE NyxMemories
                        SET significance = $1
                        WHERE id = $2
                    """, new_significance, memory_id)
                    decayed_count += 1
    
    return f"Applied memory decay to {decayed_count} memories and archived {archived_count} memories"

@function_tool
async def reconsolidate_memory(ctx, memory_id: int) -> str:
    """
    Reconsolidate (slightly alter) a memory when it's recalled.
    This simulates how human memories change slightly each time they're accessed.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    context = ctx.context.query_context or {}
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Fetch the memory
            row = await conn.fetchrow("""
                SELECT memory_text, metadata, significance, memory_type, embedding
                FROM NyxMemories
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """, memory_id, user_id, conversation_id)
            
            if not row:
                return f"Memory with ID {memory_id} not found"
                
            memory_text = row["memory_text"]
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            significance = row["significance"]
            memory_type = row["memory_type"]
            
            # Only reconsolidate episodic memories with low/medium significance
            # High significance memories are more stable
            if memory_type != "observation" or significance >= 8:
                return f"Memory with ID {memory_id} is not eligible for reconsolidation"
                
            # Get original form if available
            original_form = metadata.get("original_form", memory_text)
            
            # Current emotional state can influence reconsolidation
            current_emotion = context.get("emotional_state", "neutral")
            
            # Reconsolidation varies by memory age
            reconsolidation_strength = min(0.3, significance / 10.0)  # Cap at 0.3
            
            # Generate a slightly altered version
            altered_memory = await alter_memory_text(
                memory_text, 
                original_form,
                reconsolidation_strength,
                current_emotion
            )
            
            # Update metadata to track changes
            if "reconsolidation_history" not in metadata:
                metadata["reconsolidation_history"] = []
                
            metadata["reconsolidation_history"].append({
                "previous_text": memory_text,
                "timestamp": datetime.now().isoformat(),
                "emotional_context": current_emotion
            })
            
            # Only store last 3 versions to avoid metadata bloat
            if len(metadata["reconsolidation_history"]) > 3:
                metadata["reconsolidation_history"] = metadata["reconsolidation_history"][-3:]
            
            # Update the memory
            await conn.execute("""
                UPDATE NyxMemories
                SET memory_text = $1, metadata = $2, embedding = $3
                WHERE id = $4
            """, 
                altered_memory, 
                json.dumps(metadata),
                await generate_embedding(altered_memory),
                memory_id
            )
    
    return f"Memory with ID {memory_id} has been reconsolidated"

async def cluster_memories_by_similarity(memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group memories into clusters based on embedding similarity.
    This is a simplified version of clustering.
    """
    # We'll use a greedy approach for simplicity
    clusters = []
    unclustered = memories.copy()
    
    while unclustered:
        # Take the first memory as a seed
        seed = unclustered.pop(0)
        current_cluster = [seed]
        
        # Find all similar memories
        i = 0
        while i < len(unclustered):
            memory = unclustered[i]
            
            # Calculate cosine similarity
            similarity = np.dot(seed["embedding"], memory["embedding"])
            if similarity > 0.85:  # Threshold for similarity
                current_cluster.append(memory)
                unclustered.pop(i)
            else:
                i += 1
        
        # If we found a significant cluster, add it
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
    
    return clusters

async def generate_consolidated_summary(memory_texts: List[str]) -> str:
    """
    Generate a consolidated summary of related memories.
    """
    joined_texts = "\n".join(memory_texts)
    
    prompt = f"""
    Consolidate these related memory fragments into a single coherent memory:
    
    {joined_texts}
    
    Create a single paragraph that:
    1. Captures the essential pattern or theme across these memories
    2. Generalizes the specific details into broader understanding
    3. Retains the most significant elements
    4. Begins with "I've observed that..." or similar phrase
    """
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You consolidate memory fragments into coherent patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback
        return f"I've observed a pattern across several memories: {memory_texts[0]}..."

async def alter_memory_text(
    memory_text: str, 
    original_form: str,
    alteration_strength: float,
    emotional_context: str
) -> str:
    """
    Slightly alter a memory text based on emotional context.
    The closer to the original form, the less alteration.
    """
    # For minimal changes, we'll use GPT with a specific prompt
    prompt = f"""
    Slightly alter this memory to simulate memory reconsolidation effects.
    
    Original memory: {original_form}
    Current memory: {memory_text}
    Emotional context: {emotional_context}
    
    Create a very slight alteration that:
    1. Maintains the same core information and meaning
    2. Makes subtle changes to wording or emphasis ({int(alteration_strength * 100)}% alteration)
    3. Slightly enhances aspects that align with the "{emotional_context}" emotional state
    4. Never changes key facts, names, or locations
    
    Return only the altered text with no explanation.
    """
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You subtly alter memories to simulate reconsolidation effects."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # If GPT fails, make minimal random changes
        words = memory_text.split()
        for i in range(len(words)):
            if random.random() < alteration_strength * 0.2:
                # Minimal changes like adding "very" or changing emphasis words
                if words[i] in ["a", "the", "was", "is"]:
                    continue  # Skip essential words
                if "good" in words[i]:
                    words[i] = "very " + words[i]
                elif "bad" in words[i]:
                    words[i] = "quite " + words[i]
        
        return " ".join(words)

@function_tool
async def create_semantic_abstraction(ctx, memory_text: str, source_id: int) -> str:
    """
    Create a semantic memory (higher-level abstraction) from an episodic memory.
    This converts concrete experiences into generalized knowledge.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Create a prompt for generating a semantic abstraction
    prompt = f"""
    Convert this specific observation into a general insight or pattern:
    
    Observation: {memory_text}
    
    Create a concise semantic memory that:
    1. Extracts the general principle or pattern from this specific event
    2. Forms a higher-level abstraction that could apply to similar situations
    3. Phrases it as a generalized insight rather than a specific event
    4. Keeps it under 50 words
    
    Example transformation:
    Observation: "Chase hesitated when Monica asked him about his past, changing the subject quickly."
    Semantic abstraction: "Chase appears uncomfortable discussing his past and employs deflection when questioned about it."
    """
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that extracts semantic meaning from specific observations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=100
        )
        
        abstraction = response.choices[0].message.content.strip()
        
        # Store the semantic memory with a reference to its source
        async with asyncpg.create_pool(dsn=DB_DSN) as pool:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO NyxMemories (
                        user_id, conversation_id, memory_text, memory_type,
                        significance, embedding, timestamp,
                        tags, times_recalled, is_archived,
                        metadata
                    )
                    VALUES ($1, $2, $3, 'semantic', $4, $5, CURRENT_TIMESTAMP, $6, 0, FALSE, $7)
                """,
                    user_id,
                    conversation_id,
                    abstraction,
                    5,  # Moderate significance for semantic memories
                    await generate_embedding(abstraction),
                    ["semantic", "abstraction"],
                    json.dumps({"source_memory_id": source_id})
                )
        
        return f"Created semantic abstraction: '{abstraction}'"
        
    except Exception as e:
        return f"Error creating semantic abstraction: {str(e)}"

@function_tool
async def enhance_context_with_memories(ctx, base_context: Dict[str, Any], query: str) -> str:
    """
    Enhance the provided context with relevant memories.
    This makes responses more consistent and personalized.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    memory_system = ctx.context.memory_system
    
    # Retrieve relevant memories based on the query
    memories = await memory_system.retrieve_memories(
        query=query,
        scopes=["game", "user"],
        memory_types=["observation", "reflection", "abstraction"],
        limit=5,
        min_significance=3,
        context=base_context
    )
    
    # Extract memory texts and IDs
    memory_texts = [m["memory_text"] for m in memories]
    memory_ids = [m["id"] for m in memories]
    
    # Generate a narrative about the topic if we have memories
    narrative = None
    if memories:
        narrative_result = await construct_narrative(
            ctx, 
            topic=query, 
            context=base_context,
            limit=5
        )
        narrative_data = json.loads(narrative_result)
        narrative = narrative_data.get("narrative")
    
    # Generate introspection about Nyx's understanding
    introspection = await generate_introspection(ctx)
    introspection_data = json.loads(introspection)
    
    # Format the enhancement
    enhancement = {
        "memory_context": "\n\n### Nyx's Relevant Memories ###\n" + 
                         "\n".join([f"- {text}" for text in memory_texts]) if memory_texts else "",
        "narrative_context": f"\n\n### Nyx's Narrative Understanding ###\n{narrative}" if narrative else "",
        "introspection_context": f"\n\n### Nyx's Self-Reflection ###\n{introspection_data.get('introspection', '')}" 
                               if introspection_data and "introspection" in introspection_data else "",
        "referenced_memory_ids": memory_ids
    }
    
    # Combine all enhancements
    combined_text = ""
    if enhancement["memory_context"]:
        combined_text += enhancement["memory_context"]
    if enhancement["narrative_context"]:
        combined_text += enhancement["narrative_context"]
    if enhancement["introspection_context"]:
        combined_text += enhancement["introspection_context"]
    
    # Also include the enhancement object for additional processing
    enhancement["text"] = combined_text
    
    # Create enhanced context
    enhanced_context = base_context.copy()
    enhanced_context["memory_context"] = combined_text
    enhanced_context["memory_ids"] = memory_ids
    
    return json.dumps(enhanced_context)

@function_tool
async def generate_introspection(ctx) -> str:
    """
    Generate Nyx's introspection about her own memory and knowledge.
    This adds metacognitive awareness to the DM character.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Count memories by type and significance
            stats = await conn.fetch("""
                SELECT 
                    memory_type, 
                    COUNT(*) as count,
                    AVG(significance) as avg_significance,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM NyxMemories
                WHERE user_id = $1 AND conversation_id = $2 AND is_archived = FALSE
                GROUP BY memory_type
            """, user_id, conversation_id)
            
            # Get most frequently recalled memories
            top_memories = await conn.fetch("""
                SELECT memory_text, times_recalled
                FROM NyxMemories
                WHERE user_id = $1 AND conversation_id = $2 AND is_archived = FALSE
                ORDER BY times_recalled DESC
                LIMIT 3
            """, user_id, conversation_id)
            
            # Get player model
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxPlayerModel'
            """, user_id, conversation_id)
            
            player_model = json.loads(row["value"]) if row and row["value"] else {}
            
            # Calculate memory health metrics
            memory_health = {
                "total_memories": sum(r["count"] for r in stats),
                "episodic_ratio": next((r["count"] for r in stats if r["memory_type"] == "observation"), 0) / 
                                sum(r["count"] for r in stats) if sum(r["count"] for r in stats) > 0 else 0,
                "average_significance": sum(r["count"] * r["avg_significance"] for r in stats) / 
                                      sum(r["count"] for r in stats) if sum(r["count"] for r in stats) > 0 else 0,
                "memory_span_days": (max(r["newest"] for r in stats) - min(r["oldest"] for r in stats)).days 
                                   if stats else 0,
                "top_recalled_memories": [{"text": m["memory_text"], "recalled": m["times_recalled"]} for m in top_memories]
            }
            
            # Generate introspection text
            introspection = await generate_introspection_text(memory_health, player_model)
            
            # Return combination of metrics and generated text
            return json.dumps({
                "memory_stats": {r["memory_type"]: {"count": r["count"], "avg_significance": r["avg_significance"]} for r in stats},
                "memory_health": memory_health,
                "player_understanding": player_model.get("play_style", {}),
                "introspection": introspection,
                "confidence": min(1.0, memory_health["total_memories"] / 100)  # More memories = more confidence
            })

async def generate_introspection_text(memory_health: Dict[str, Any], player_model: Dict[str, Any]) -> str:
    """
    Generate natural language introspection about Nyx's own memory state.
    """
    # Format key metrics for the prompt
    memory_count = memory_health["total_memories"]
    span_days = memory_health["memory_span_days"]
    avg_sig = memory_health["average_significance"]
    
    # Format player understanding
    play_style = player_model.get("play_style", {})
    play_style_str = ", ".join([f"{style}: {count}" for style, count in play_style.items() if count > 0])
    
    # Top recalled memories
    top_memories = memory_health.get("top_recalled_memories", [])
    top_memories_str = "\n".join([f"- {m['text']} (recalled {m['recalled']} times)" for m in top_memories])
    
    prompt = f"""
    As Nyx, generate an introspective reflection on your memory state using these metrics:
    
    - Total memories: {memory_count}
    - Memory span: {span_days} days
    - Average significance: {avg_sig:.1f}/10
    - Player tendencies: {play_style_str}
    
    Most frequently recalled memories:
    {top_memories_str}
    
    Create a first-person introspection that:
    1. Reflects on your understanding of Chase (the player)
    2. Notes any gaps or uncertainties in your knowledge
    3. Acknowledges how your perspective might be biased or incomplete
    4. Expresses metacognitive awareness of your role as the narrative guide
    
    Keep it natural and conversational, as if you're thinking to yourself.
    """
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Nyx, the dungeon master, reflecting on your memories and understanding."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback template if GPT fails
        return f"""
        I've accumulated {memory_count} memories about Chase and our interactions over {span_days} days.
        I notice that he tends to be {max(play_style.items(), key=lambda x: x[1])[0] if play_style else 'unpredictable'} in his approach.
        My understanding feels {'strong' if avg_sig > 7 else 'moderate' if avg_sig > 4 else 'limited'}, 
        though I wonder what I might be missing or misinterpreting.
        """

@function_tool
async def _format_memories_for_context(ctx, memories: List[Dict[str, Any]]) -> str:
    """Format memories for inclusion in context."""
    memory_texts = []
    for memory in memories:
        relevance = memory.get("relevance", 0.5)
        confidence_marker = "vividly recall" if relevance > 0.8 else \
                          "remember" if relevance > 0.6 else \
                          "think I recall" if relevance > 0.4 else \
                          "vaguely remember"
        
        memory_texts.append(f"I {confidence_marker}: {memory['memory_text']}")
    
    return "\n".join(memory_texts)

    
# Add to appropriate SDK module

@function_tool
async def check_maintenance_needs(ctx) -> str:
    """
    Check if memory maintenance is needed based on game state.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Check last maintenance time
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'LastMaintenanceTime'
            """, user_id, conversation_id)
            
            last_maintenance_time = None
            if row and row["value"]:
                try:
                    last_maintenance_time = datetime.fromisoformat(row["value"])
                except:
                    pass
            
            # Check memory count
            memory_count = await conn.fetchval("""
                SELECT COUNT(*) FROM NyxMemories
                WHERE user_id = $1 AND conversation_id = $2 AND is_archived = FALSE
            """, user_id, conversation_id)
            
            # Determine if maintenance is needed
            needs_maintenance = False
            reason = ""
            
            # Time-based: Run maintenance every 6 hours or more
            if not last_maintenance_time or (datetime.now() - last_maintenance_time).total_seconds() > 21600:  # 6 hours
                needs_maintenance = True
                reason = "Time-based maintenance"
            
            # Volume-based: Run maintenance if we have a lot of memories
            elif memory_count > 100:
                needs_maintenance = True
                reason = "Volume-based maintenance"
            
            # Run maintenance if needed
            if needs_maintenance:
                # Perform maintenance
                await perform_memory_maintenance(ctx)
                
                # Update last maintenance time
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LastMaintenanceTime', $3)
                    ON CONFLICT (user_id, conversation_id, key) 
                    DO UPDATE SET value = $3
                """, user_id, conversation_id, datetime.now().isoformat())
                
                return json.dumps({
                    "maintenance_performed": True,
                    "reason": reason,
                    "memory_count": memory_count
                })
            else:
                return json.dumps({
                    "maintenance_performed": False,
                    "memory_count": memory_count
                })

@function_tool
async def perform_memory_maintenance(ctx) -> str:
    """
    Perform comprehensive memory maintenance.
    This includes consolidation, decay, and archiving.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # First, apply memory decay
    decay_result = await apply_memory_decay(ctx)
    
    # Next, consolidate memories
    consolidation_result = await consolidate_memories(ctx)
    
    # Finally, archive very old memories
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Archive memories that are very old and insignificant
            archive_result = await conn.execute("""
                UPDATE NyxMemories
                SET is_archived = TRUE
                WHERE user_id = $1 
                AND conversation_id = $2
                AND significance < 3
                AND timestamp < NOW() - INTERVAL '30 days'
                AND times_recalled < 3
            """, user_id, conversation_id)
            
            # Get maintenance statistics
            stats = await conn.fetch("""
                SELECT 
                    memory_type, 
                    COUNT(*) as count,
                    AVG(significance) as avg_significance
                FROM NyxMemories
                WHERE user_id = $1 AND conversation_id = $2 AND is_archived = FALSE
                GROUP BY memory_type
            """, user_id, conversation_id)
            
            # Generate a maintenance reflection
            maintenance_reflection = f"""
            I've organized my memories, strengthening important ones and letting less significant ones fade.
            This helps me maintain a clearer picture of our interactions and narrative development.
            """
            
            # Store maintenance reflection
            await conn.execute("""
                INSERT INTO NyxMemories (
                    user_id, conversation_id, memory_text, memory_type,
                    significance, embedding, timestamp,
                    tags, times_recalled, is_archived,
                    metadata
                )
                VALUES ($1, $2, $3, 'reflection', 4, $4, CURRENT_TIMESTAMP, 
                        $5, 0, FALSE, $6)
            """,
                user_id,
                conversation_id,
                maintenance_reflection,
                await generate_embedding(maintenance_reflection),
                ["maintenance", "reflection", "system"],
                json.dumps({"maintenance": True, "timestamp": datetime.now().isoformat()})
            )
    
    return json.dumps({
        "decay_applied": True,
        "consolidation_performed": True,
        "memory_stats": [dict(row) for row in stats]
    })

def _process_memory_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Process a memory chunk for integration"""
    processed = {
        "content": chunk.get("content", ""),
        "type": chunk.get("type", "general"),
        "timestamp": chunk.get("timestamp", datetime.now()),
        "metadata": {},
        "connections": [],
        "relevance": 0.0
    }
    
    # Extract metadata
    processed["metadata"] = self._extract_metadata(chunk)
    
    # Find connections
    processed["connections"] = self._find_connections(chunk)
    
    # Calculate relevance
    processed["relevance"] = self._calculate_relevance(chunk)
    
    return processed

def _integrate_memory_chunk(self, chunk: Dict[str, Any]):
    """Integrate a processed memory chunk into the system"""
    try:
        # Store the memory
        memory_id = self._store_memory(chunk)
        
        # Update connections
        self._update_connections(memory_id, chunk["connections"])
        
        # Update indices
        self._update_indices(memory_id, chunk)
        
        # Trigger consolidation if needed
        if self._should_consolidate():
            self._consolidate_memories()
            
    except Exception as e:
        logger.error(f"Failed to integrate memory chunk: {e}")
        raise

def _consolidate_memories(self):
    """Consolidate and optimize memory storage"""
    try:
        # Get memories for consolidation
        memories = self._get_consolidation_candidates()
        
        # Group related memories
        groups = self._group_related_memories(memories)
        
        # Merge and optimize groups
        for group in groups:
            self._merge_memory_group(group)
            
        # Update indices
        self._update_consolidated_indices()
        
    except Exception as e:
        logger.error(f"Memory consolidation failed: {e}")
        raise

def _process_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a memory query"""
    try:
        # Parse query parameters
        params = self._parse_query_params(query)
        
        # Find matching memories
        matches = self._find_matching_memories(params)
        
        # Rank results
        ranked_results = self._rank_results(matches, params)
        
        # Format response
        return self._format_query_response(ranked_results)
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise

class MemoryIntegrationSDK:
    def__init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_bridge = MemoryNyxBridge(user_id, conversation_id)
        self.initialized = False

    async def initialize(self):
        """Initialize the memory integration SDK"""
        if self.initialized:
            return
    
        await self.memory_bridge.initialize()
        self.initialized = True
        logger.info(f"Memory integratioNSDK initialized for user {self.user_id}")

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        return await self.memory_bridge.get_memory(memory_id)

    async def query_memories(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query memories based on criteria"""
        return await self.memory_bridge.query_memories(query)
