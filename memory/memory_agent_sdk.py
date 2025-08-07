# memory/memory_agent_sdk.py

import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool,
    InputGuardrail,
    GuardrailFunctionOutput,
    RunContextWrapper
)

from memory.memory_agent_wrapper import MemoryAgentWrapper
# Import the memory system components
from memory.wrapper import MemorySystem
from memory.core import MemorySignificance

# Models for input/output validation
class MemoryInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    memory_text: str = Field(description="The memory text to record")
    importance: str = Field(default="medium", description="Importance level: 'trivial', 'low', 'medium', 'high', 'critical'")
    emotional: bool = Field(default=True, description="Whether to analyze emotional content")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for the memory")

class MemoryQueryInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    query: Optional[str] = Field(default=None, description="Optional search query")
    context: Optional[str] = Field(default=None, description="Current context that might influence recall")
    limit: int = Field(default=5, description="Maximum number of memories to return")

class BeliefInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    belief_text: str = Field(description="The belief statement")
    confidence: float = Field(default=0.7, description="Confidence in this belief (0.0-1.0)")

class BeliefsQueryInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    topic: Optional[str] = Field(default=None, description="Optional topic filter")

class MaintenanceInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")

class AnalysisInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")

class MemorySystemContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None

# Function tools for memory operations - REMOVED DEFAULT VALUES
@function_tool
async def remember(ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int, memory_text: str, 
                  importance: str, emotional: bool, 
                  tags: Optional[List[str]]) -> Dict[str, Any]:
    """
    Record a new memory for an entity.
    
    Args:
        ctx: Run context wrapper containing memory system context
        entity_type: Type of entity ("player", "nyx", etc.)
        entity_id: ID of the entity
        memory_text: The memory text to record
        importance: Importance level ("trivial", "low", "medium", "high", "critical")
        emotional: Whether to analyze emotional content
        tags: Optional tags for the memory
        
    Returns:
        Information about the created memory
    """
    # Handle defaults inside the function
    importance = importance or "medium"
    emotional = True if emotional is None else emotional
    
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    result = await ctx.context.memory_system.remember(
        entity_type=entity_type,
        entity_id=entity_id,
        memory_text=memory_text,
        importance=importance,
        emotional=emotional,
        tags=tags
    )
    
    # Format the result for better readability
    formatted_result = {
        "memory_id": result.get("memory_id"),
        "memory_text": memory_text,
        "importance": importance
    }
    
    # Add emotional analysis if available
    if "emotion_analysis" in result:
        formatted_result["emotional_analysis"] = {
            "primary_emotion": result["emotion_analysis"].get("primary_emotion"),
            "intensity": result["emotion_analysis"].get("intensity"),
            "valence": result["emotion_analysis"].get("valence")
        }
    
    return formatted_result

@function_tool
async def recall(
    ctx: RunContextWrapper[MemorySystemContext], 
    entity_type: str, 
    entity_id: int, 
    query: Optional[str] = None, 
    context: Optional[str] = None, 
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Recall memories for an entity, optionally filtered by a query.
    
    Args:
        ctx: Run context wrapper containing memory system context
        entity_type: Type of entity ("player", "nyx", etc.)
        entity_id: ID of the entity
        query: Optional search query
        context: Current context that might influence recall
        limit: Maximum number of memories to return
        
    Returns:
        Retrieved memories and related information
    """
    # Handle defaults inside the function
    limit = limit or 5
    
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    result = await ctx.context.memory_system.recall(
        entity_type=entity_type,
        entity_id=entity_id,
        query=query,
        context=context,
        limit=limit
    )
    
    # Extract memories into a more readable format
    memories = []
    for memory in result.get("memories", []):
        memories.append({
            "id": memory.get("id"),
            "text": memory.get("text"),
            "type": memory.get("type"),
            "significance": memory.get("significance"),
            "emotional_intensity": memory.get("emotional_intensity"),
            "timestamp": memory.get("timestamp")
        })
    
    formatted_result = {
        "memories": memories,
        "count": len(memories)
    }
    
    # Add special cases like flashbacks or mood-congruent recall if present
    if "flashback" in result:
        formatted_result["flashback"] = {
            "text": result["flashback"].get("text"),
            "source_memory_id": result["flashback"].get("source_memory_id")
        }
    
    if "mood_congruent_recall" in result and result["mood_congruent_recall"]:
        formatted_result["mood_influenced"] = True
        formatted_result["current_emotion"] = result.get("current_emotion", {}).get("primary", "neutral")
    
    return formatted_result

@function_tool
async def create_belief(ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int, 
                       belief_text: str, confidence: float) -> Dict[str, Any]:
    """
    Create a belief for an entity based on their experiences.
    
    Args:
        ctx: Run context wrapper containing memory system context
        entity_type: Type of entity ("player", "nyx", etc.)
        entity_id: ID of the entity
        belief_text: The belief statement
        confidence: Confidence in this belief (0.0-1.0)
        
    Returns:
        Created belief information
    """
    # Handle defaults inside the function
    confidence = confidence or 0.7
    
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    result = await ctx.context.memory_system.create_belief(
        entity_type=entity_type,
        entity_id=entity_id,
        belief_text=belief_text,
        confidence=confidence
    )
    
    return {
        "belief_id": result.get("belief_id"),
        "belief_text": result.get("belief_text"),
        "confidence": result.get("confidence")
    }

@function_tool
async def get_beliefs(ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int, 
                     topic: Optional[str]) -> List[Dict[str, Any]]:
    """
    Get beliefs held by an entity.
    
    Args:
        ctx: Run context wrapper containing memory system context
        entity_type: Type of entity ("player", "nyx", etc.)
        entity_id: ID of the entity
        topic: Optional topic filter
        
    Returns:
        List of beliefs
    """
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    beliefs = await ctx.context.memory_system.get_beliefs(
        entity_type=entity_type,
        entity_id=entity_id,
        topic=topic
    )
    
    return beliefs

@function_tool
async def run_maintenance(ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int) -> Dict[str, Any]:
    """
    Run maintenance tasks on an entity's memories (consolidation, decay, etc.).
    
    Args:
        ctx: Run context wrapper containing memory system context
        entity_type: Type of entity ("player", "nyx", etc.)
        entity_id: ID of the entity
        
    Returns:
        Results of maintenance operations
    """
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    result = await ctx.context.memory_system.maintain(
        entity_type=entity_type,
        entity_id=entity_id
    )
    
    return result

@function_tool
async def analyze_memories(ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int) -> Dict[str, Any]:
    """
    Perform a comprehensive analysis of an entity's memories.
    
    Args:
        ctx: Run context wrapper containing memory system context
        entity_type: Type of entity ("player", "nyx", etc.)
        entity_id: ID of the entity
        
    Returns:
        Memory analysis results
    """
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    result = await ctx.context.memory_system.analyze_entity_memory(
        entity_type=entity_type,
        entity_id=entity_id
    )
    
    return result

@function_tool
async def generate_schemas(ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int) -> Dict[str, Any]:
    """
    Generate schemas by analyzing memory patterns.
    
    Args:
        ctx: Run context wrapper containing memory system context
        entity_type: Type of entity ("player", "nyx", etc.)
        entity_id: ID of the entity
        
    Returns:
        Generated schemas information
    """
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    result = await ctx.context.memory_system.generate_schemas(
        entity_type=entity_type,
        entity_id=entity_id
    )
    
    return result

@function_tool
async def add_journal_entry(ctx: RunContextWrapper[MemorySystemContext], player_name: str, entry_text: str,
                           entry_type: str, fantasy_flag: bool,
                           intensity_level: int) -> Dict[str, Any]:
    """
    Add a journal entry to a player's memory.
    
    Args:
        ctx: Run context wrapper containing memory system context
        player_name: Name of the player
        entry_text: The journal entry text
        entry_type: Type of entry
        fantasy_flag: Whether this is a fantasy/dream
        intensity_level: Emotional intensity (0-5)
        
    Returns:
        Information about the created journal entry
    """
    # Handle defaults inside the function
    entry_type = entry_type or "observation"
    fantasy_flag = fantasy_flag if fantasy_flag is not None else False
    intensity_level = intensity_level or 0
    
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    journal_id = await ctx.context.memory_system.add_journal_entry(
        player_name=player_name,
        entry_text=entry_text,
        entry_type=entry_type,
        fantasy_flag=fantasy_flag,
        intensity_level=intensity_level
    )
    
    return {
        "journal_entry_id": journal_id,
        "player_name": player_name,
        "entry_text": entry_text,
        "entry_type": entry_type
    }

@function_tool
async def get_journal_history(ctx: RunContextWrapper[MemorySystemContext], player_name: str, entry_type: Optional[str],
                             limit: int) -> List[Dict[str, Any]]:
    """
    Get a player's journal entries.
    
    Args:
        ctx: Run context wrapper containing memory system context
        player_name: Name of the player
        entry_type: Optional filter by entry type
        limit: Maximum number of entries to return
        
    Returns:
        List of journal entries
    """
    # Handle defaults inside the function
    limit = limit or 10
    
    if ctx.context.memory_system is None:
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    
    journal_entries = await ctx.context.memory_system.get_journal_history(
        player_name=player_name,
        entry_type=entry_type,
        limit=limit
    )
    
    return journal_entries

# Input validation guardrail
async def validate_entity_input(ctx: RunContextWrapper[MemorySystemContext], agent: Agent[MemorySystemContext], input_data):
    """
    Validate entity information in input.
    
    Args:
        ctx: Run context wrapper containing memory system context
        agent: The memory manager agent
        input_data: User's input message (can be string or list)
    
    Returns:
        Guardrail validation result
    """
    # Handle input_data that might be a list or a string
    if isinstance(input_data, list):
        # Extract text content if possible from a list of messages
        input_text = ""
        for item in input_data:
            if isinstance(item, dict) and "content" in item:
                input_text += item["content"] + " "
            elif isinstance(item, str):
                input_text += item + " "
        input_text = input_text.strip()
    else:
        input_text = str(input_data)
    
    # Check if there's mention of entity but without clear entity_id
    if ("entity" in input_text.lower() or 
        "memory" in input_text.lower()) and not ("entity_id" in input_text):
        return GuardrailFunctionOutput(
            output_info={
                "valid": False,
                "reason": "Missing entity identification"
            },
            tripwire_triggered=True
        )
    
    return GuardrailFunctionOutput(
        output_info={
            "valid": True
        },
        tripwire_triggered=False
    )

# Create the Memory Agent
def create_memory_agent(user_id: int, conversation_id: int):
    """Create the memory agent with all tools and guardrails."""
    memory_context = MemorySystemContext(user_id, conversation_id)
    
    base_agent = Agent[MemorySystemContext](
        name="Memory Manager",
        instructions="""
        You are a memory management assistant that helps manage, retrieve, and analyze memories.
        You have access to a sophisticated memory system that stores and organizes memories for
        different entities. Each entity has a type (such as "player" or "nyx") and an entity_id.
        
        You can:
        1. Record new memories with varying levels of importance
        2. Retrieve memories based on queries or context
        3. Create and manage beliefs derived from memories
        4. Run maintenance on memories (consolidation, decay, etc.)
        5. Analyze memory patterns and generate schemas
        6. Manage journal entries for players
        
        Always ask for the entity_type and entity_id when performing memory operations.
        When describing memories, focus on their content, emotional aspects, and significance.
        
        For memory importance levels, use:
        - "trivial": Everyday minor details
        - "low": Minor but somewhat memorable events
        - "medium": Standard memories of moderate importance
        - "high": Important memories that stand out
        - "critical": Extremely important, life-changing memories
        """,
        tools=[
            remember,
            recall,
            create_belief,
            get_beliefs,
            run_maintenance,
            analyze_memories,
            generate_schemas,
            add_journal_entry,
            get_journal_history
        ],
        input_guardrails=[
            InputGuardrail(guardrail_function=validate_entity_input)
        ],
        model_settings=ModelSettings(temperature=0.3),
        model="gpt-5-nano"
    )
    
    memory_agent = MemoryAgentWrapper(base_agent, memory_context)
    return memory_agent

# Main function to run the memory agent
async def main():
    # Initialize the memory system context
    user_id = 1  # Example user_id
    conversation_id = 1  # Example conversation_id
    
    memory_context = MemorySystemContext(user_id, conversation_id)
    
    # Create the agent
    memory_agent = create_memory_agent(user_id, conversation_id)
    
    # Run the agent with a sample query
    result = await Runner.run(
        memory_agent, 
        "I want to record a new memory for the player with entity_id 101. The memory is 'The player discovered a hidden passage behind the bookshelf that led to an ancient library.'",
        context=memory_context
    )
    
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
