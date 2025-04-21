# nyx/api/unified_function_tools.py

import logging
import json
from typing import Dict, List, Any, Optional
from agents import function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

@function_tool
async def configure_thinking(ctx: RunContextWrapper[Any],
                          enabled: bool = True) -> Dict[str, Any]:
    """
    Configure the thinking capability settings
    
    Args:
        enabled: Whether thinking is enabled at all
    
    Returns:
        Updated configuration
    """
    brain = ctx.context
    
    # Update thinking configuration
    brain.thinking_config["thinking_enabled"] = enabled
    
    return {
        "thinking_enabled": brain.thinking_config["thinking_enabled"],
        "updated_at": datetime.datetime.now().isoformat()
    }

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
    
    # Directly use the memory_core's add_memory method
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
    
    # Use the enhanced memory retrieval method
    formatted_memories = await brain.memory_core.retrieve_memories_with_formatting(
        query=query,
        memory_types=memory_types or ["observation", "reflection", "abstraction", "experience"],
        limit=limit
    )
    
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
    
    # Use the memory_core's create_reflection method
    reflection_result = await brain.memory_core.create_reflection_from_memories(topic=topic)
    
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
    
    # Use the memory_core's create_abstraction method
    abstraction_result = await brain.memory_core.create_abstraction_from_memories(
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
    
    # Use the memory_core's construct_narrative method
    narrative_result = await brain.memory_core.construct_narrative_from_memories(
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
    
    # Use the experience_interface's retrieval method
    experiences = await brain.experience_interface.retrieve_experiences_enhanced(
        query=query,
        scenario_type=scenario_type,
        limit=limit
    )
    
    return json.dumps(experiences)

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
    
    # Add emotional state to context if not provided
    if context_data is None:
        context_data = {}
    if "emotional_state" not in context_data:
        context_data["emotional_state"] = brain.emotional_core.get_formatted_emotional_state()
    
    # Use the experience_interface's sharing method
    result = await brain.experience_interface.share_experience_enhanced(
        query=query,
        context_data=context_data
    )
    
    return json.dumps(result)

# --- Emotional Functions ---

@function_tool
async def get_emotional_state(ctx: RunContextWrapper[Any]) -> str:
    """
    Get the current emotional state.
    """
    brain = ctx.context
    
    # Get emotional state directly from emotional_core
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
    
    # Use the emotional_core's async update method
    result = await brain.emotional_core.update_emotion_async(emotion, value)
    
    return json.dumps(result)

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
    
    # Use the emotional_core's async set method
    result = await brain.emotional_core.set_emotion_async(emotion, value)
    
    return json.dumps(result)

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
    
    # Use the brain's enhanced processing method
    result = await brain.process_user_input_enhanced(
        user_input=user_input,
        context=context_data
    )
    
    return json.dumps(result)

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
    
    # Use the brain's enhanced response generation method
    response_data = await brain.generate_enhanced_response(
        user_input=user_input,
        context=context_data
    )
    
    return json.dumps(response_data)

@function_tool
async def run_maintenance(ctx: RunContextWrapper[Any]) -> str:
    """
    Run maintenance on all systems.
    """
    brain = ctx.context
    
    # Run maintenance directly using brain method
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
    
    # Get system stats directly using brain method
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

# --- Enhanced System Functions ---

@function_tool
async def adapt_to_context(ctx: RunContextWrapper[Any],
                         user_input: str,
                         context_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Adapt system behavior based on context changes.
    
    Args:
        user_input: User's input text
        context_data: Additional context information
    """
    brain = ctx.context
    
    # Create adaptable context
    adaptable_context = {
        "user_input": user_input,
        **({} if context_data is None else context_data)
    }
    
    # Add emotional state if available
    if hasattr(brain, 'emotional_core'):
        adaptable_context["emotional_state"] = brain.emotional_core.get_formatted_emotional_state()
    
    # Detect context change using brain's dynamic adaptation system
    if hasattr(brain, 'dynamic_adaptation'):
        # Detect context change
        change_result = await brain.dynamic_adaptation.detect_context_change(adaptable_context)
        
        # Monitor performance
        performance = await brain.dynamic_adaptation.monitor_performance({
            "success_rate": context_data.get("success_rate", 0.5) if context_data else 0.5,
            "user_satisfaction": context_data.get("user_satisfaction", 0.5) if context_data else 0.5,
            "efficiency": context_data.get("efficiency", 0.5) if context_data else 0.5,
            "response_quality": context_data.get("response_quality", 0.5) if context_data else 0.5
        })
        
        # Select strategy if significant change
        strategy = None
        if change_result[0]:  # significant change
            strategy = await brain.dynamic_adaptation.select_strategy(adaptable_context, performance)
        
        return json.dumps({
            "context_change": change_result,
            "performance": performance,
            "strategy": strategy
        })
    
    return json.dumps({
        "error": "Dynamic adaptation system not available"
    })

@function_tool
async def evaluate_response(ctx: RunContextWrapper[Any],
                         response: str,
                         context_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Evaluate a response using internal feedback system.
    
    Args:
        response: The response to evaluate
        context_data: Additional context information
    """
    brain = ctx.context
    
    # Use internal feedback system if available
    if hasattr(brain, 'internal_feedback'):
        # Track performance metrics
        metrics = {
            "response_quality": context_data.get("response_quality", 0.5) if context_data else 0.5,
            "user_satisfaction": context_data.get("user_satisfaction", 0.5) if context_data else 0.5
        }
        
        quality_stats = {}
        for metric, value in metrics.items():
            quality_stats[metric] = await brain.internal_feedback.track_performance(metric, value)
        
        # Evaluate confidence
        confidence_eval = await brain.internal_feedback.evaluate_confidence(
            context_data.get("confidence", 0.7) if context_data else 0.7,
            context_data.get("success", True) if context_data else True
        )
        
        # Create evaluable content
        evaluable_content = {
            "text": response,
            "type": context_data.get("response_type", "general") if context_data else "general",
            "metrics": metrics
        }
        
        # Critic evaluation
        critic_evals = {}
        for aspect in ["consistency", "effectiveness", "efficiency"]:
            critic_evals[aspect] = await brain.internal_feedback.critic_evaluate(
                aspect, evaluable_content, context_data or {}
            )
        
        return json.dumps({
            "quality_stats": quality_stats,
            "confidence_eval": confidence_eval,
            "critic_evals": critic_evals
        })
    
    return json.dumps({
        "error": "Internal feedback system not available"
    })
@function_tool
async def get_hormone_levels(ctx) -> Dict[str, float]:
    """
    Get current digital hormone levels
    
    Returns:
        Dictionary of hormone levels
    """
    brain = ctx.context
    
    if not brain.hormone_system:
        return {"error": "Hormone system not initialized"}
    
    # Get current hormone levels
    hormone_levels = {name: data["value"] for name, data in brain.hormone_system.hormones.items()}
    
    return hormone_levels

@function_tool
async def get_hormone_state(ctx) -> Dict[str, Any]:
    """
    Get detailed state of the hormone system
    
    Returns:
        Full hormone system state with cycles and environmental factors
    """
    brain = ctx.context
    
    if not brain.hormone_system:
        return {"error": "Hormone system not initialized"}
    
    # Get hormone levels
    hormone_levels = {name: data["value"] for name, data in brain.hormone_system.hormones.items()}
    
    # Get cycle phases
    cycle_phases = {name: data["cycle_phase"] for name, data in brain.hormone_system.hormones.items()}
    
    # Get environmental factors
    environmental_factors = brain.hormone_system.environmental_factors.copy()
    
    return {
        "hormone_levels": hormone_levels,
        "cycle_phases": cycle_phases,
        "environmental_factors": environmental_factors
    }

@function_tool
async def update_hormone_cycles(ctx) -> Dict[str, Any]:
    """
    Manually trigger an update of hormone cycles
    
    Returns:
        Updated hormone values
    """
    brain = ctx.context
    
    if not brain.hormone_system:
        return {"error": "Hormone system not initialized"}
    
    # Update hormone cycles
    result = await brain.hormone_system.update_hormone_cycles(ctx)
    
    return result

@function_tool
async def set_environmental_factor(ctx, factor_name: str, value: float) -> Dict[str, Any]:
    """
    Set an environmental factor for the hormone system
    
    Args:
        factor_name: Name of the factor to update
        value: New value (0.0-1.0)
        
    Returns:
        Updated environmental factors
    """
    brain = ctx.context
    
    if not brain.hormone_system:
        return {"error": "Hormone system not initialized"}
    
    # Validate factor name
    if factor_name not in brain.hormone_system.environmental_factors:
        return {
            "error": f"Unknown environmental factor: {factor_name}",
            "available_factors": list(brain.hormone_system.environmental_factors.keys())
        }
    
    # Validate value
    if not 0.0 <= value <= 1.0:
        return {"error": "Value must be between 0.0 and 1.0"}
    
    # Set the factor
    brain.hormone_system.environmental_factors[factor_name] = value
    
    return {
        "updated": True,
        "factor": factor_name,
        "value": value,
        "environmental_factors": brain.hormone_system.environmental_factors
    }

@function_tool
async def get_hormone_impact_on_identity(ctx) -> Dict[str, Any]:
    """
    Get the impact of hormones on identity
    
    Returns:
        Recent hormone-driven identity changes
    """
    brain = ctx.context
    
    if not brain.identity_evolution:
        return {"error": "Identity evolution system not initialized"}
    
    # Get evolution history
    evolution_history = brain.identity_evolution.identity_profile.get("evolution_history", [])
    
    # Filter for hormone-driven changes
    hormone_driven_changes = []
    
    for entry in evolution_history:
        if entry.get("type") == "hormone_influence":
            hormone_driven_changes.append(entry)
    
    # If no hormone-driven changes, return empty result
    if not hormone_driven_changes:
        return {
            "message": "No hormone-driven identity changes found",
            "changes": []
        }
    
    # Get the most recent changes
    recent_changes = hormone_driven_changes[-5:]
    
    # Format the changes
    formatted_changes = []
    
    for change in recent_changes:
        updates = change.get("updates", {})
        hormone_levels = change.get("hormone_levels", {})
        
        # Find dominant hormone
        dominant_hormone = None
        max_level = 0
        for hormone, level in hormone_levels.items():
            if level > max_level:
                max_level = level
                dominant_hormone = hormone
        
        formatted_changes.append({
            "timestamp": change.get("timestamp"),
            "dominant_hormone": dominant_hormone,
            "hormone_levels": hormone_levels,
            "trait_updates": updates.get("traits", {}),
            "preference_updates": updates.get("preferences", {})
        })
    
    return {
        "hormone_driven_changes": formatted_changes,
        "total_changes": len(hormone_driven_changes),
        "recent_changes": len(recent_changes)
    }

@function_tool
async def generate_hormone_reflection(ctx) -> str:
    """
    Generate a reflection on how hormones are affecting identity
    
    Returns:
        Hormone reflection text
    """
    brain = ctx.context
    
    if not brain.identity_evolution:
        return "Identity evolution system not initialized."
    
    try:
        reflection = await brain.identity_evolution.generate_identity_reflection()
        return reflection
    except Exception as e:
        return f"Unable to generate hormone reflection: {str(e)}"
