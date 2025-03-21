# nyx/api/function_tools.py

import logging
import json
from typing import Dict, List, Any, Optional
from agents import function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

# --- Memory Functions ---

@function_tool
async def add_memory(ctx: RunContextWrapper[Any],
                    memory_text: str,
                    memory_type: str = "observation",
                    significance: int = 5,
                    tags: Optional[List[str]] = None) -> str:
    """
    Add a new memory.
    
    Args:
        memory_text: Text content of the memory
        memory_type: Type of memory (observation, reflection, abstraction, experience)
        significance: Importance level (1-10)
        tags: Optional list of tags for categorization
    """
    brain = ctx.context
    
    memory_id = await brain.memory_core.add_memory(
        memory_text=memory_text,
        memory_type=memory_type,
        memory_scope="game",
        significance=significance,
        tags=tags or [],
        metadata={
            "emotional_context": brain.emotional_core.get_formatted_emotional_state()
        }
    )
    
    return json.dumps({
        "success": True,
        "memory_id": memory_id,
        "memory_type": memory_type
    })

@function_tool
async def retrieve_memories(ctx: RunContextWrapper[Any],
                          query: str,
                          memory_types: Optional[List[str]] = None,
                          limit: int = 5) -> str:
    """
    Retrieve memories based on query.
    
    Args:
        query: Search query
        memory_types: Types of memories to include
        limit: Maximum number of results
    """
    brain = ctx.context
    
    memories = await brain.memory_core.retrieve_memories(
        query=query,
        memory_types=memory_types or ["observation", "reflection", "abstraction", "experience"],
        limit=limit,
        context={
            "emotional_state": brain.emotional_core.get_formatted_emotional_state()
        }
    )
    
    # Format memories for output
    formatted_memories = []
    for memory in memories:
        confidence = memory.get("confidence_marker", "remember")
        formatted = {
            "id": memory["id"],
            "text": memory["memory_text"],
            "type": memory["memory_type"],
            "significance": memory["significance"],
            "confidence": confidence,
            "relevance": memory.get("relevance", 0.5),
            "tags": memory.get("tags", [])
        }
        formatted_memories.append(formatted)
    
    return json.dumps(formatted_memories)

@function_tool
async def create_reflection(ctx: RunContextWrapper[Any],
                          topic: Optional[str] = None) -> str:
    """
    Create a reflection on a specific topic.
    
    Args:
        topic: Optional topic to reflect on
    """
    brain = ctx.context
    
    reflection_result = await brain.create_reflection(topic=topic)
    
    return json.dumps({
        "reflection": reflection_result["reflection"],
        "confidence": reflection_result["confidence"],
        "topic": reflection_result["topic"],
        "reflection_id": reflection_result["reflection_id"]
    })

@function_tool
async def create_abstraction(ctx: RunContextWrapper[Any],
                           memory_ids: List[str],
                           pattern_type: str = "behavior") -> str:
    """
    Create an abstraction from specific memories.
    
    Args:
        memory_ids: List of memory IDs to abstract from
        pattern_type: Type of pattern to identify (behavior, preference, emotional, relationship)
    """
    brain = ctx.context
    
    abstraction_result = await brain.create_abstraction(
        memory_ids=memory_ids,
        pattern_type=pattern_type
    )
    
    return json.dumps({
        "abstraction": abstraction_result["abstraction"],
        "pattern_type": abstraction_result["pattern_type"],
        "confidence": abstraction_result["confidence"],
        "abstraction_id": abstraction_result["abstraction_id"]
    })

@function_tool
async def construct_narrative(ctx: RunContextWrapper[Any],
                            topic: str,
                            chronological: bool = True,
                            limit: int = 5) -> str:
    """
    Construct a narrative from memories about a topic.
    
    Args:
        topic: Topic for narrative construction
        chronological: Whether to maintain chronological order
        limit: Maximum number of memories to include
    """
    brain = ctx.context
    
    narrative_result = await brain.construct_narrative(
        topic=topic,
        chronological=chronological,
        limit=limit
    )
    
    return json.dumps({
        "narrative": narrative_result["narrative"],
        "confidence": narrative_result["confidence"],
        "experience_count": narrative_result["experience_count"]
    })

# --- Experience Functions ---

@function_tool
async def retrieve_experiences(ctx: RunContextWrapper[Any],
                             query: str,
                             scenario_type: Optional[str] = None,
                             limit: int = 3) -> str:
    """
    Retrieve experiences relevant to current context.
    
    Args:
        query: Search query
        scenario_type: Optional scenario type
        limit: Maximum number of results
    """
    brain = ctx.context
    
    experiences_result = await brain.retrieve_experiences(
        query=query,
        scenario_type=scenario_type,
        limit=limit
    )
    
    # Format experiences for output
    formatted_experiences = []
    for exp in experiences_result["experiences"]:
        formatted = {
            "content": exp.get("content", ""),
            "scenario_type": exp.get("scenario_type", ""),
            "confidence_marker": exp.get("confidence_marker", ""),
            "relevance_score": exp.get("relevance_score", 0.5),
            "experiential_richness": exp.get("experiential_richness", 0.5)
        }
        formatted_experiences.append(formatted)
    
    return json.dumps(formatted_experiences)

@function_tool
async def share_experience(ctx: RunContextWrapper[Any],
                         query: str,
                         context_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Process a request to share experiences.
    
    Args:
        query: User's query text
        context_data: Additional context data
    """
    brain = ctx.context
    
    result = await brain.experience_engine.handle_experience_sharing_request(
        user_query=query,
        context_data=context_data or {
            "emotional_state": brain.emotional_core.get_formatted_emotional_state()
        }
    )
    
    return json.dumps({
        "success": result["success"],
        "has_experience": result.get("has_experience", False),
        "response_text": result.get("response_text", ""),
        "experience_count": result.get("experience_count", 0)
    })

# --- Emotional Functions ---

@function_tool
async def get_emotional_state(ctx: RunContextWrapper[Any]) -> str:
    """
    Get the current emotional state.
    """
    brain = ctx.context
    
    emotional_state = brain.emotional_core.get_emotional_state()
    dominant_emotion, dominant_value = brain.emotional_core.get_dominant_emotion()
    
    return json.dumps({
        "emotional_state": emotional_state,
        "dominant_emotion": {
            "emotion": dominant_emotion,
            "intensity": dominant_value
        },
        "valence": brain.emotional_core.get_emotional_valence(),
        "arousal": brain.emotional_core.get_emotional_arousal()
    })

@function_tool
async def update_emotion(ctx: RunContextWrapper[Any],
                       emotion: str,
                       value: float) -> str:
    """
    Update a specific emotion value.
    
    Args:
        emotion: The emotion to update
        value: The delta change in emotion value (-1.0 to 1.0)
    """
    brain = ctx.context
    
    # Validate input
    if not -1.0 <= value <= 1.0:
        return json.dumps({
            "error": "Value must be between -1.0 and 1.0"
        })
    
    if emotion not in brain.emotional_core.emotions:
        return json.dumps({
            "error": f"Unknown emotion: {emotion}",
            "available_emotions": list(brain.emotional_core.emotions.keys())
        })
    
    # Update emotion
    brain.emotional_core.update_emotion(emotion, value)
    
    # Get updated state
    updated_state = brain.emotional_core.get_emotional_state()
    
    return json.dumps({
        "success": True,
        "updated_emotion": emotion,
        "change": value,
        "new_value": updated_state[emotion],
        "emotional_state": updated_state
    })

@function_tool
async def set_emotion(ctx: RunContextWrapper[Any],
                    emotion: str,
                    value: float) -> str:
    """
    Set a specific emotion to an absolute value.
    
    Args:
        emotion: The emotion to set
        value: The absolute value (0.0 to 1.0)
    """
    brain = ctx.context
    
    # Validate input
    if not 0.0 <= value <= 1.0:
        return json.dumps({
            "error": "Value must be between 0.0 and 1.0"
        })
    
    if emotion not in brain.emotional_core.emotions:
        return json.dumps({
            "error": f"Unknown emotion: {emotion}",
            "available_emotions": list(brain.emotional_core.emotions.keys())
        })
    
    # Set emotion to absolute value
    brain.emotional_core.set_emotion(emotion, value)
    
    # Get updated state
    updated_state = brain.emotional_core.get_emotional_state()
    
    return json.dumps({
        "success": True,
        "set_emotion": emotion,
        "value": value,
        "emotional_state": updated_state
    })

# --- System Functions ---

@function_tool
async def process_input(ctx: RunContextWrapper[Any],
                      user_input: str,
                      context_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Process user input through all systems.
    
    Args:
        user_input: User's input text
        context_data: Additional context information
    """
    brain = ctx.context
    
    result = await brain.process_input(
        user_input=user_input,
        context=context_data
    )
    
    # Format memories for output
    formatted_memories = []
    for memory in result["memories"]:
        formatted_memories.append({
            "id": memory["id"],
            "text": memory["memory_text"],
            "relevance": memory.get("relevance", 0.5)
        })
    
    return json.dumps({
        "emotional_state": result["emotional_state"],
        "memories": formatted_memories,
        "memory_count": result["memory_count"],
        "has_experience": result["has_experience"],
        "experience_response": result["experience_response"],
        "memory_id": result["memory_id"],
        "response_time": result["response_time"]
    })

@function_tool
async def generate_response(ctx: RunContextWrapper[Any],
                          user_input: str,
                          context_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a complete response to user input.
    
    Args:
        user_input: User's input text
        context_data: Additional context information
    """
    brain = ctx.context
    
    response_data = await brain.generate_response(
        user_input=user_input,
        context=context_data
    )
    
    return json.dumps({
        "message": response_data["message"],
        "response_type": response_data["response_type"],
        "emotional_expression": response_data["emotional_expression"],
        "memories_used": response_data["memories_used"],
        "memory_count": response_data["memory_count"]
    })

@function_tool
async def run_maintenance(ctx: RunContextWrapper[Any]) -> str:
    """
    Run maintenance on all systems.
    """
    brain = ctx.context
    
    maintenance_result = await brain.run_maintenance()
    
    return json.dumps({
        "memory_maintenance": {
            "memories_decayed": maintenance_result["memory_maintenance"]["memories_decayed"],
            "clusters_consolidated": maintenance_result["memory_maintenance"]["clusters_consolidated"],
            "memories_archived": maintenance_result["memory_maintenance"]["memories_archived"]
        },
        "maintenance_time": maintenance_result["maintenance_time"]
    })

@function_tool
async def get_system_stats(ctx: RunContextWrapper[Any]) -> str:
    """
    Get statistics about all systems.
    """
    brain = ctx.context
    
    stats = await brain.get_system_stats()
    
    return json.dumps({
        "memory_stats": {
            "total_memories": stats["memory_stats"]["total_memories"],
            "type_counts": stats["memory_stats"]["type_counts"],
            "archived_count": stats["memory_stats"]["archived_count"]
        },
        "emotional_state": {
            "dominant_emotion": stats["emotional_state"]["dominant_emotion"],
            "dominant_value": stats["emotional_state"]["dominant_value"],
            "valence": stats["emotional_state"]["valence"],
            "arousal": stats["emotional_state"]["arousal"]
        },
        "interaction_stats": stats["interaction_stats"],
        "performance_metrics": stats["performance_metrics"],
        "introspection": stats["introspection"]["introspection"]
    })
