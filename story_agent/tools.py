# story_agent/tools.py

"""
Organized tools for the Story Director agent.
"""

import logging
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple # Ensure Tuple is imported if used elsewhere
from pydantic import BaseModel, Field

# ****** ADD THIS IMPORT ******
from agents import function_tool, RunContextWrapper
# ****** OPTIONAL: Import specific context if no circular dependency ******
# from story_agent.story_director_agent import StoryDirectorContext # Use this if possible
from logic.conflict_system.conflict_tools import _internal_add_conflict_to_narrative_logic

from context.context_config import get_config

from db.connection import get_db_connection_context
from logic.narrative_progression import (
    get_current_narrative_stage,
    check_for_personal_revelations,
    check_for_narrative_moments,
    check_for_npc_revelations,
    add_dream_sequence,
    add_moment_of_clarity,
    NARRATIVE_STAGES
)
from logic.social_links import (
    get_social_link_tool,
    get_relationship_summary_tool,
    check_for_crossroads_tool,
    check_for_ritual_tool,
    apply_crossroads_choice_tool
)

# Context system imports
from context.context_service import get_context_service, get_comprehensive_context
from context.memory_manager import get_memory_manager, search_memories_tool, MemorySearchRequest
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

logger = logging.getLogger(__name__)

# Define the context type alias for easier use
# Use 'Any' if StoryDirectorContext causes circular imports
ContextType = Any # Or StoryDirectorContext

# ----- NEW: Context Tools -----

@function_tool
# @track_performance("get_optimized_context") # Uncomment if track_performance is defined/imported
async def get_optimized_context(
    ctx: RunContextWrapper[ContextType],
    # 1. Change parameters with defaults to Optional[Type] = None
    query_text: Optional[str] = None,
    use_vector: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Get optimized context using the comprehensive context system.

    Args:
        query_text: Optional query text for relevance scoring. Defaults to "" if not provided. # 2. Update docstrings
        use_vector: Whether to use vector search for relevance. Defaults to True if not provided.

    Returns:
        Dictionary with comprehensive context information.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 3. Handle default values inside the function
    actual_query_text = query_text if query_text is not None else ""
    actual_use_vector = use_vector if use_vector is not None else True

    try:
        context_service = await get_context_service(user_id, conversation_id)
        # from context.context_config import get_config # Already imported above
        config = get_config()
        token_budget = config.get_token_budget("default")

        # 4. Use the 'actual_' variables in the function call
        context_data = await context_service.get_context(
            input_text=actual_query_text,
            context_budget=token_budget,
            use_vector_search=actual_use_vector
        )

        # Safely access performance monitor
        perf_monitor = None
        if hasattr(context, 'performance_monitor'):
            perf_monitor = context.performance_monitor
        else:
            try:
                # Attempt to get instance if not present on context
                perf_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
            except Exception as pm_err:
                 logger.warning(f"Could not get performance monitor instance: {pm_err}")

        if perf_monitor and "token_usage" in context_data:
            try:
                # Ensure token_usage is a dict before summing
                usage = context_data["token_usage"]
                if isinstance(usage, dict):
                    total_tokens = sum(usage.values())
                    perf_monitor.record_token_usage(total_tokens)
                else:
                    logger.warning(f"Unexpected format for token_usage: {type(usage)}")
            except Exception as token_err:
                 logger.warning(f"Error recording token usage: {token_err}")


        return context_data
    except Exception as e:
        logger.error(f"Error getting optimized context: {str(e)}", exc_info=True)
        # Return consistent error structure
        return {
            "error": str(e),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "context_data": None # Indicate data retrieval failed
        }

@function_tool
# @track_performance("") # Uncomment if track_performance is defined/imported
async def retrieve_relevant_memories(
    ctx: RunContextWrapper[ContextType],
    query_text: str,
    memory_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memories using vector search.
    
    Args:
        query_text: Query text for relevance matching.
        memory_type: Optional type filter (observation, event, etc.).
        limit: Maximum number of memories to return. Defaults to 5 if not provided.
        
    Returns:
        List of relevant memories.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Handle the default value inside the function
    actual_limit = limit if limit is not None else 5
    
    try:
        # Create a MemorySearchRequest object
        request = MemorySearchRequest(
            query_text=query_text,
            memory_types=[memory_type] if memory_type else None,
            limit=actual_limit,
            use_vector=True
        )
        
        # Call the standalone function with the right parameters
        memory_result = await search_memories_tool(ctx, user_id, conversation_id, request)
        
        # Process the result
        memory_dicts = []
        if memory_result and hasattr(memory_result, 'memories'):
            for memory in memory_result.memories:
                if hasattr(memory, 'to_dict'):
                    memory_dicts.append(memory.to_dict())
                elif isinstance(memory, dict):
                    memory_dicts.append(memory)
                    
        return memory_dicts
    except Exception as e:
        logger.error(f"Error retrieving relevant memories: {str(e)}", exc_info=True)
        return []
        
@function_tool
# @track_performance("store_narrative_memory") # Uncomment if track_performance is defined/imported
async def store_narrative_memory(
    ctx: RunContextWrapper[ContextType],
    content: str,
    # 1. Change parameters with defaults to Optional[Type] = None
    memory_type: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[List[str]] = None # This one was already correct
) -> Dict[str, Any]:
    """
    Store a narrative memory in the memory system.

    Args:
        content: Content of the memory.
        memory_type: Type of memory. Defaults to "observation" if not provided. # 2. Update docstrings
        importance: Importance score (0.0-1.0). Defaults to 0.6 if not provided.
        tags: Optional tags for categorization. Defaults to [memory_type, "story_director"].

    Returns:
        Stored memory information or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 3. Handle default values inside the function
    actual_memory_type = memory_type if memory_type is not None else "observation"
    actual_importance = importance if importance is not None else 0.6
    # Handle default for tags based on actual_memory_type
    actual_tags = tags if tags is not None else [actual_memory_type, "story_director"]

    try:
        memory_manager = await get_memory_manager(user_id, conversation_id)

        # 4. Use the 'actual_' variables in the function call
        memory_id = await memory_manager.add_memory(
            content=content,
            memory_type=actual_memory_type,
            importance=actual_importance,
            tags=actual_tags, # Use handled tags default
            metadata={"source": "story_director_tool", "timestamp": datetime.now().isoformat()}
        )

        # Safely check for narrative_manager
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            try:
                await context.narrative_manager.add_interaction(
                    content=content,
                    importance=actual_importance, # Use actual_
                    tags=actual_tags # Use actual_
                )
            except Exception as nm_err:
                 logger.warning(f"Error calling narrative_manager.add_interaction: {nm_err}")


        return {
            "memory_id": memory_id,
            "content": content,
            "memory_type": actual_memory_type, # Return actual used value
            "importance": actual_importance, # Return actual used value
            "success": True
        }
    except Exception as e:
        logger.error(f"Error storing narrative memory: {str(e)}", exc_info=True)
        return {"error": str(e), "success": False}
        
@function_tool
# @track_performance("search_by_vector") # Uncomment if track_performance is defined/imported
async def search_by_vector(
    ctx: RunContextWrapper[ContextType],
    query_text: str,
    entity_types: Optional[List[str]] = None, # This one was already correct
    # 1. Change signature: top_k: Optional[int] = None
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Search for entities by semantic similarity using vector search.

    Args:
        query_text: Query text for semantic search.
        entity_types: Types of entities to search for. Defaults to ["npc", "location", "memory", "narrative"].
        top_k: Maximum number of results to return. Defaults to 5 if not provided. # 2. Update docstring

    Returns:
        List of semantically similar entities.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 3. Handle the default value inside the function
    actual_top_k = top_k if top_k is not None else 5
    # Handle default for entity_types as well (already done in original code, but good to be explicit)
    actual_entity_types = entity_types if entity_types is not None else ["npc", "location", "memory", "narrative"]


    try:
        vector_service = await get_vector_service(user_id, conversation_id)
        # Check if vector service is enabled AFTER getting it
        if not vector_service or not vector_service.enabled:
            logger.info("Vector service is not enabled or available. Skipping vector search.")
            return []

        # 4. Use the 'actual_' variables in the function call
        results = await vector_service.search_entities(
            query_text=query_text,
            entity_types=actual_entity_types, # Use handled default
            top_k=actual_top_k, # Use actual_top_k here
            hybrid_ranking=True # Assuming this is always true
        )
        return results
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}", exc_info=True)
        return [] # Return empty list on error as per original logic

@function_tool
# @track_performance("get_summarized_narrative_context") # Uncomment if track_performance is defined/imported
async def get_summarized_narrative_context(
    ctx: RunContextWrapper[ContextType],
    query: str,
    # 1. Change signature: max_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get automatically summarized narrative context using progressive summarization.

    Args:
        query: Query for relevance matching.
        max_tokens: Maximum tokens for context. Defaults to 1000 if not provided. # 2. Update docstring

    Returns:
        Summarized narrative context or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 3. Handle the default value inside the function
    actual_max_tokens = max_tokens if max_tokens is not None else 1000

    try:
        narrative_manager = None
        # Safely check for and access narrative_manager
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            narrative_manager = context.narrative_manager
        else:
            try:
                # Ensure progressive_summarization is importable
                from story_agent.progressive_summarization import RPGNarrativeManager
                dsn = 'DATABASE_URL_NOT_FOUND' # Default DSN
                try:
                    # Attempt to get DSN safely
                    async with get_db_connection_context() as conn:
                         # Check if conn and _pool exist and have the attribute
                         pool = getattr(conn, '_pool', None)
                         connect_kwargs = getattr(pool, '_connect_kwargs', {}) if pool else {}
                         dsn = connect_kwargs.get('dsn', dsn)
                except Exception as db_conn_err:
                     logger.warning(f"Could not get DSN from DB connection: {db_conn_err}")


                narrative_manager = RPGNarrativeManager(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    db_connection_string=dsn
                )
                await narrative_manager.initialize()

                # Try storing on context if it's mutable and designed for it
                try:
                    # Check if context is suitable for attribute assignment
                    if hasattr(context, '__dict__') or isinstance(context, object): # Basic check
                         context.narrative_manager = narrative_manager
                    else:
                         logger.warning("Context object does not support attribute assignment for narrative_manager.")
                except Exception as assign_err: # Catch potential errors during assignment
                     logger.warning(f"Could not store narrative_manager on context: {assign_err}")

            except ImportError:
                 logger.error("Module 'story_agent.progressive_summarization' not found.")
                 return {"error": "Narrative manager component not available.", "memories": [], "arcs": []}
            except Exception as init_error:
                logger.error(f"Error initializing narrative manager: {init_error}", exc_info=True)
                return {"error": "Narrative manager initialization failed.", "memories": [], "arcs": []}

        # Ensure narrative_manager was successfully initialized before using
        if not narrative_manager:
             return {"error": "Narrative manager could not be initialized.", "memories": [], "arcs": []}


        # 4. Use the 'actual_' variable in the function call
        context_data = await narrative_manager.get_current_narrative_context(
            query,  # or input_text parameter as expected by the method
            actual_max_tokens
        )
        return context_data
    except Exception as e:
        logger.error(f"Error getting summarized narrative context: {str(e)}", exc_info=True)
        # Ensure consistent error structure
        return {"error": str(e), "memories": [], "arcs": []}
        
@function_tool
@track_performance("analyze_activity")
async def analyze_activity(
    ctx: RunContextWrapper[ContextType],
    activity_text: str,
    setting_context: Optional[str] = None,
    apply_effects: bool = False
) -> Dict[str, Any]:
    """
    Analyze an activity to determine its resource effects.

    Args:
        activity_text: Description of the activity
        setting_context: Optional context about the current setting
        apply_effects: Whether to immediately apply the determined effects

    Returns:
        Dict with activity analysis and effects
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.activity_analyzer import ActivityAnalyzer
        analyzer = ActivityAnalyzer(user_id, conversation_id)
        result = await analyzer.analyze_activity(activity_text, setting_context, apply_effects)

        if hasattr(context, 'add_narrative_memory'):
            effects_description = []
            for resource_type, value in result.get("effects", {}).items():
                if value:
                    direction = "increased" if value > 0 else "decreased"
                    effects_description.append(f"{resource_type} {direction} by {abs(value)}")
            effects_text = ", ".join(effects_description) if effects_description else "no significant effects"
            memory_content = f"Analyzed activity: {activity_text[:100]}... with effects: {effects_text}"
            await context.add_narrative_memory(memory_content, "activity_analysis", 0.5)

        return result
    except Exception as e:
        logger.error(f"Error analyzing activity: {str(e)}", exc_info=True)
        return {"activity_type": "unknown", "activity_details": "", "effects": {}, "description": f"Error analyzing activity: {str(e)}", "error": str(e)}

@function_tool
@track_performance("get_filtered_activities")
async def get_filtered_activities(
    ctx: RunContextWrapper[ContextType],
    npc_archetypes: List[str] = [],
    meltdown_level: int = 0,
    setting: str = ""
) -> List[Dict[str, Any]]:
    """
    Get a list of activities filtered by NPC archetypes, meltdown level, and setting.

    Args:
        npc_archetypes: List of NPC archetypes (e.g., "Giantess", "Mommy Domme")
        meltdown_level: Meltdown level (0-5)
        setting: Current setting/location

    Returns:
        List of filtered activities
    """
    context = ctx.context
    try:
        from logic.activities_logic import filter_activities_for_npc, build_short_summary

        user_stats = None
        if hasattr(context, 'resource_manager'):
            try:
                resources = await context.resource_manager.get_resources()
                vitals = await context.resource_manager.get_vitals()
                user_stats = {**resources, **vitals}
            except Exception as stats_error:
                logger.warning(f"Could not get user stats: {stats_error}")

        activities = await filter_activities_for_npc(
            npc_archetypes=npc_archetypes, meltdown_level=meltdown_level, user_stats=user_stats, setting=setting
        )
        for activity in activities:
            activity["short_summary"] = build_short_summary(activity)

        return activities
    except Exception as e:
        logger.error(f"Error getting filtered activities: {str(e)}", exc_info=True)
        return []


@function_tool
@track_performance("analyze_activity_impact")
async def analyze_activity_impact(
    ctx: RunContextWrapper[ContextType],
    activity_text: str,
    check_conflicts: bool = True,
    check_relationships: bool = True
) -> Dict[str, Any]:
    """
    Analyze the comprehensive impact of an activity across multiple systems.

    Args:
        activity_text: Description of the activity
        check_conflicts: Whether to check impact on active conflicts
        check_relationships: Whether to check impact on relationships

    Returns:
        Dict with comprehensive impact analysis
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.activity_analyzer import ActivityAnalyzer
        analyzer = ActivityAnalyzer(user_id, conversation_id)
        base_analysis = await analyzer.analyze_activity(activity_text, apply_effects=False)

        results = {"resource_effects": base_analysis.get("effects", {}), "description": base_analysis.get("description", ""), "conflict_impacts": [], "relationship_impacts": []}

        if check_conflicts and hasattr(context, 'conflict_manager'):
            try:
                conflict_manager = context.conflict_manager
                active_conflicts = await conflict_manager.get_active_conflicts()
                for conflict in active_conflicts:
                    conflict_keywords = [k for k in [conflict.get('conflict_name', ''), conflict.get('faction_a_name', ''), conflict.get('faction_b_name', '')] if k]
                    matching_keywords = [k for k in conflict_keywords if k.lower() in activity_text.lower()]
                    if matching_keywords:
                        resource_sum = sum(abs(effect) for effect in base_analysis.get("effects", {}).values())
                        impact_level = 3 if resource_sum > 20 else (2 if resource_sum > 10 else 1)
                        results["conflict_impacts"].append({"conflict_id": conflict.get('conflict_id'), "conflict_name": conflict.get('conflict_name', ''), "is_relevant": True, "matching_keywords": matching_keywords, "estimated_impact_level": impact_level, "suggested_progress_change": impact_level * 2})
            except Exception as conflict_error:
                logger.warning(f"Error checking conflict impacts: {conflict_error}")

        if check_relationships:
            try:
                async with get_db_connection_context() as conn:
                    npcs = await conn.fetch("SELECT npc_id, npc_name FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE", user_id, conversation_id)
                    mentioned_npcs = [{"npc_id": npc['npc_id'], "npc_name": npc['npc_name']} for npc in npcs if npc['npc_name'] and npc['npc_name'].lower() in activity_text.lower()]
                    for npc in mentioned_npcs:
                        impact_level = 2 if any(word in activity_text.lower() for word in ["help", "assist", "support", "gift"]) else (-2 if any(word in activity_text.lower() for word in ["refuse", "reject", "ignore", "insult"]) else 1)
                        results["relationship_impacts"].append({"npc_id": npc["npc_id"], "npc_name": npc["npc_name"], "estimated_impact": impact_level, "suggested_relationship_change": impact_level * 3})
            except Exception as relationship_error:
                logger.warning(f"Error checking relationship impacts: {relationship_error}")

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Analyzed comprehensive impact of activity: {activity_text[:100]}..."
            await context.add_narrative_memory(memory_content, "activity_impact_analysis", 0.6)

        return results
    except Exception as e:
        logger.error(f"Error analyzing activity impact: {str(e)}", exc_info=True)
        return {"resource_effects": {}, "description": f"Error analyzing activity impact: {str(e)}", "conflict_impacts": [], "relationship_impacts": [], "error": str(e)}

@function_tool
@track_performance("get_all_activities")
async def get_all_activities(ctx: RunContextWrapper[ContextType]) -> List[Dict[str, Any]]:
    """
    Get a list of all available activities from the database.

    Returns:
        List of all activities
    """
    try:
        from logic.activities_logic import get_all_activities as get_activities, build_short_summary
        activities = await get_activities()
        for activity in activities:
            activity["short_summary"] = build_short_summary(activity)
        return activities
    except Exception as e:
        logger.error(f"Error getting all activities: {str(e)}", exc_info=True)
        return []

@function_tool
@track_performance("generate_activity_suggestion")
async def generate_activity_suggestion(
    ctx: RunContextWrapper[ContextType],
    npc_name: str,
    intensity_level: int = 2,
    archetypes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a suggested activity for an NPC interaction based on archetypes and intensity.

    Args:
        npc_name: Name of the NPC
        intensity_level: Desired intensity level (1-5)
        archetypes: Optional list of NPC archetypes

    Returns:
        Dict with suggested activity details
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        if not archetypes:
            try:
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow("SELECT archetype FROM NPCStats WHERE npc_name=$1 AND user_id=$2 AND conversation_id=$3", npc_name, user_id, conversation_id)
                    if row and row['archetype']:
                        if isinstance(row['archetype'], str):
                            try: archetype_data = json.loads(row['archetype']); archetypes = archetype_data if isinstance(archetype_data, list) else (archetype_data.get("types") if isinstance(archetype_data, dict) else [row['archetype']])
                            except: archetypes = [row['archetype']]
                        elif isinstance(row['archetype'], list): archetypes = row['archetype']
                        elif isinstance(row['archetype'], dict) and "types" in row['archetype']: archetypes = row['archetype']["types"]
            except Exception as archetype_error: logger.warning(f"Error getting NPC archetypes: {archetype_error}")
        if not archetypes: archetypes = ["Dominance", "Femdom"]

        setting = "Default"
        if hasattr(context, 'get_comprehensive_context'):
            try: comprehensive_context = await context.get_comprehensive_context(); current_location = comprehensive_context.get("current_location"); setting = current_location or setting
            except Exception as context_error: logger.warning(f"Error getting location from context: {context_error}")

        from logic.activities_logic import filter_activities_for_npc, build_short_summary, get_all_activities as get_activities
        activities = await filter_activities_for_npc(npc_archetypes=archetypes, meltdown_level=max(0, intensity_level-1), setting=setting)
        if not activities: activities = await get_activities(); activities = random.sample(activities, min(3, len(activities)))

        selected_activity = random.choice(activities) if activities else None
        if not selected_activity: return {"npc_name": npc_name, "success": False, "error": "No suitable activities found"}

        intensity_tiers = selected_activity.get("intensity_tiers", []); tier_text = ""
        if intensity_tiers: idx = min(intensity_level - 1, len(intensity_tiers) - 1); idx = max(0, idx); tier_text = intensity_tiers[idx]

        suggestion = {"npc_name": npc_name, "activity_name": selected_activity.get("name", ""), "purpose": selected_activity.get("purpose", [])[0] if selected_activity.get("purpose") else "", "intensity_tier": tier_text, "intensity_level": intensity_level, "short_summary": build_short_summary(selected_activity), "archetypes_used": archetypes, "setting": setting, "success": True}

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Generated activity suggestion for {npc_name}: {suggestion['activity_name']} (Intensity: {intensity_level})"
            await context.add_narrative_memory(memory_content, "activity_suggestion", 0.5)

        return suggestion
    except Exception as e:
        logger.error(f"Error generating activity suggestion: {str(e)}", exc_info=True)
        return {"npc_name": npc_name, "success": False, "error": str(e)}

# ----- Story State Tools -----

@function_tool
@track_performance("get_key_npcs")
# 1. Change signature: limit: Optional[int] = None
async def get_key_npcs(ctx: RunContextWrapper[ContextType], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get the key NPCs in the current game state, ordered by importance.

    Args:
        limit: Maximum number of NPCs to return. Defaults to 5 if not provided. # 2. Update docstring

    Returns:
        List of NPC information dictionaries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 3. Handle the default value inside the function
    actual_limit = limit if limit is not None else 5

    async with get_db_connection_context() as conn:
        try:
            # 4. Use the actual_limit in the query
            rows = await conn.fetch(
                """
                SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                ORDER BY dominance DESC
                LIMIT $3
                """,
                user_id, conversation_id, actual_limit # Use actual_limit here
            )

            npcs = []
            for row in rows:
                dynamics = {}
                # Check if the relationship tool is available before calling
                if HAS_REL_TOOL:
                    try:
                        relationship = await get_relationship_summary_tool(
                            user_id, conversation_id, "player", user_id, "npc", row['npc_id']
                        )
                        dynamics = relationship.get('dynamics', {}) if relationship else {}
                    except Exception as rel_err:
                         logger.warning(f"Error getting relationship summary for NPC {row['npc_id']} in get_key_npcs: {rel_err}")
                else:
                    logger.debug("Skipping relationship summary in get_key_npcs as tool is not available.")


                npcs.append({
                    "npc_id": row['npc_id'],
                    "npc_name": row['npc_name'],
                    "dominance": row['dominance'],
                    "cruelty": row['cruelty'],
                    "closeness": row['closeness'],
                    "trust": row['trust'],
                    "respect": row['respect'],
                    "relationship_dynamics": dynamics
                })
            return npcs
        except Exception as e:
            logger.error(f"Error fetching key NPCs: {str(e)}", exc_info=True)
            return []

@function_tool
@track_performance("get_narrative_stages")
async def get_narrative_stages(ctx: RunContextWrapper[ContextType]) -> List[Dict[str, str]]:
    """
    Get information about all narrative stages in the game.

    Returns:
        List of narrative stages with their descriptions
    """
    # context = ctx.context # Context not needed for this function
    return [{"name": stage.name, "description": stage.description} for stage in NARRATIVE_STAGES]

@function_tool
@track_performance("analyze_narrative_and_activity")
async def analyze_narrative_and_activity(
    ctx: RunContextWrapper[ContextType],
    narrative_text: str,
    player_activity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive analysis of narrative text and player activity to determine
    impacts on conflicts, resources, and story progression.

    Args:
        narrative_text: The narrative description
        player_activity: Optional specific player activity description

    Returns:
        Comprehensive analysis results
    """
    context = ctx.context
    conflict_manager = context.conflict_manager

    try:
        # Directly call the internal logic function you want to execute
        conflict_analysis = await _internal_add_conflict_to_narrative_logic(ctx, narrative_text)
        results = {"conflict_analysis": conflict_analysis, "activity_effects": None, "relationship_impacts": [], "resource_changes": {}, "conflict_progression": []}

        if player_activity:
            activity_analyzer = context.activity_analyzer
            activity_effects = await activity_analyzer.analyze_activity(player_activity, apply_effects=False)
            results["activity_effects"] = activity_effects
            active_conflicts = await conflict_manager.get_active_conflicts()
            for conflict in active_conflicts:
                conflict_keywords = [k for k in [conflict['conflict_name'], conflict['faction_a_name'], conflict['faction_b_name']] if k]
                if any(keyword.lower() in player_activity.lower() for keyword in conflict_keywords):
                    progress_increment = 10 if "actively" in player_activity.lower() or "directly" in player_activity.lower() else 5
                    if conflict['conflict_type'] == "major": progress_increment *= 0.5
                    elif conflict['conflict_type'] == "minor": progress_increment *= 1.5
                    results["conflict_progression"].append({"conflict_id": conflict['conflict_id'], "conflict_name": conflict['conflict_name'], "is_relevant": True, "suggested_progress_increment": progress_increment})

        return results
    except Exception as e:
        logger.error(f"Error analyzing narrative and activity: {str(e)}", exc_info=True)
        return {"error": str(e), "conflict_analysis": {"conflict_generated": False}, "activity_effects": None, "relationship_impacts": [], "resource_changes": {}, "conflict_progression": []}

@function_tool
@track_performance("get_relationship_summary_wrapper")
async def get_relationship_summary_wrapper(
    ctx: RunContextWrapper[ContextType],
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Dict[str, Any]:
    """
    Get a summary of a relationship between two entities.

    Args:
        entity1_type: Type of the first entity (player, npc)
        entity1_id: ID of the first entity
        entity2_type: Type of the second entity (player, npc)
        entity2_id: ID of the second entity

    Returns:
        Relationship summary
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.social_links import get_relationship_summary_tool # Already imported, but good practice
        result = await get_relationship_summary_tool(user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)

        if hasattr(context, 'add_narrative_memory'):
            entity1_name, entity2_name = entity1_type, entity2_type # Defaults
            try: # Try to get names
                 from logic.relationship_integration import RelationshipIntegration
                 integration = RelationshipIntegration(user_id, conversation_id)
                 if entity1_type == "player": entity1_name = "player"
                 elif entity1_type == "npc": entity1_name = await integration.get_entity_name(entity1_type, entity1_id) or entity1_name
                 if entity2_type == "player": entity2_name = "player"
                 elif entity2_type == "npc": entity2_name = await integration.get_entity_name(entity2_type, entity2_id) or entity2_name
            except Exception as name_err: logger.warning(f"Could not get entity names for memory: {name_err}")

            memory_content = f"Retrieved relationship summary between {entity1_name} and {entity2_name}"
            await context.add_narrative_memory(memory_content, "relationship_analysis", 0.4)

        return result or {"error": "No relationship found", "entity1_type": entity1_type, "entity1_id": entity1_id, "entity2_type": entity2_type, "entity2_id": entity2_id}
    except Exception as e:
        logger.error(f"Error getting relationship summary: {str(e)}", exc_info=True)
        return {"error": str(e), "entity1_type": entity1_type, "entity1_id": entity1_id, "entity2_type": entity2_type, "entity2_id": entity2_id}

@function_tool
@track_performance("update_relationship_dimensions")
async def update_relationship_dimensions(
    ctx: RunContextWrapper[ContextType],
    link_id: int,
    dimension_changes: Dict[str, int],
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update specific dimensions of a relationship.

    Args:
        link_id: ID of the relationship link
        dimension_changes: Dictionary of dimension changes
        reason: Reason for the changes

    Returns:
        Result of the update
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.relationship_integration import RelationshipIntegration
        integration = RelationshipIntegration(user_id, conversation_id)
        result = await integration.update_dimensions(link_id, dimension_changes, reason)

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Updated relationship dimensions for link {link_id}: {dimension_changes}"
            if reason: memory_content += f" Reason: {reason}"
            await context.add_narrative_memory(memory_content, "relationship_update", 0.5)

        return result
    except Exception as e:
        logger.error(f"Error updating relationship dimensions: {str(e)}", exc_info=True)
        return {"error": str(e), "link_id": link_id}

@function_tool
@track_performance("get_player_relationships")
async def get_player_relationships(ctx: RunContextWrapper[ContextType]) -> Dict[str, Any]:
    """
    Get all relationships for the player.

    Returns:
        List of player relationships
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.relationship_integration import RelationshipIntegration
        integration = RelationshipIntegration(user_id, conversation_id)
        relationships = await integration.get_player_relationships()
        return {"relationships": relationships, "count": len(relationships)}
    except Exception as e:
        logger.error(f"Error getting player relationships: {str(e)}", exc_info=True)
        return {"error": str(e), "relationships": [], "count": 0}

@function_tool
@track_performance("generate_relationship_evolution")
async def generate_relationship_evolution(ctx: RunContextWrapper[ContextType], link_id: int) -> Dict[str, Any]:
    """
    Generate relationship evolution information.

    Args:
        link_id: ID of the relationship link

    Returns:
        Relationship evolution information
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.relationship_integration import RelationshipIntegration
        integration = RelationshipIntegration(user_id, conversation_id)
        result = await integration.generate_relationship_evolution(link_id)

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Generated relationship evolution for link {link_id}"
            await context.add_narrative_memory(memory_content, "relationship_evolution", 0.6)

        return result
    except Exception as e:
        logger.error(f"Error generating relationship evolution: {str(e)}", exc_info=True)
        return {"error": str(e), "link_id": link_id}

# ----- Conflict Tools -----

@function_tool
@track_performance("generate_conflict")
async def generate_conflict(ctx: RunContextWrapper[ContextType], conflict_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a new conflict of the specified type, or determine the appropriate type
    based on current game state if none specified.

    Args:
        conflict_type: Optional type of conflict to generate (major, minor, standard, catastrophic)

    Returns:
        Information about the generated conflict
    """
    context = ctx.context
    conflict_manager = context.conflict_manager

    try:
        conflict = await conflict_manager.generate_conflict(conflict_type)
        if hasattr(context, 'add_narrative_memory'):
            await context.add_narrative_memory(f"Generated new {conflict['conflict_type']} conflict: {conflict['conflict_name']}", "conflict_generation", 0.7)
        return {"conflict_id": conflict['conflict_id'], "conflict_name": conflict['conflict_name'], "conflict_type": conflict['conflict_type'], "description": conflict['description'], "success": True, "message": "Conflict generated successfully"}
    except Exception as e:
        logger.error(f"Error generating conflict: {str(e)}", exc_info=True)
        return {"conflict_id": 0, "conflict_name": "", "conflict_type": conflict_type or "unknown", "description": "", "success": False, "message": f"Failed to generate conflict: {str(e)}"}

@function_tool
@track_performance("update_conflict_progress")
async def update_conflict_progress(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    progress_increment: float
) -> Dict[str, Any]:
    """
    Update the progress of a conflict.

    Args:
        conflict_id: ID of the conflict to update
        progress_increment: Amount to increment the progress (0-100)

    Returns:
        Updated conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager

    try:
        old_conflict = await conflict_manager.get_conflict(conflict_id)
        old_phase = old_conflict['phase'] if old_conflict else 'unknown'
        updated_conflict = await conflict_manager.update_conflict_progress(conflict_id, progress_increment)

        if hasattr(context, 'add_narrative_memory'):
            memory_importance = 0.7 if updated_conflict['phase'] != old_phase else 0.5
            memory_content = f"Updated conflict {updated_conflict['conflict_name']} progress by {progress_increment} points to {updated_conflict['progress']}%. "
            if updated_conflict['phase'] != old_phase: memory_content += f"Phase advanced from {old_phase} to {updated_conflict['phase']}."
            await context.add_narrative_memory(memory_content, "conflict_progression", memory_importance)

        return {"conflict_id": conflict_id, "new_progress": updated_conflict['progress'], "new_phase": updated_conflict['phase'], "phase_changed": updated_conflict['phase'] != old_phase, "success": True}
    except Exception as e:
        logger.error(f"Error updating conflict progress: {str(e)}", exc_info=True)
        return {"conflict_id": conflict_id, "new_progress": 0, "new_phase": "unknown", "phase_changed": False, "success": False, "error": str(e)}

@function_tool
@track_performance("resolve_conflict")
async def resolve_conflict(ctx: RunContextWrapper[ContextType], conflict_id: int) -> Dict[str, Any]:
    """
    Resolve a conflict and apply consequences.

    Args:
        conflict_id: ID of the conflict to resolve

    Returns:
        Information about the conflict resolution
    """
    context = ctx.context
    conflict_manager = context.conflict_manager

    try:
        result = await conflict_manager.resolve_conflict(conflict_id)
        consequences = [con.get('description', '') for con in result.get('consequences', [])]

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Resolved conflict {result.get('conflict_name', f'ID: {conflict_id}')} with outcome: {result.get('outcome', 'unknown')}. Consequences: {'; '.join(consequences)}"
            await context.add_narrative_memory(memory_content, "conflict_resolution", 0.8)

        return {"conflict_id": conflict_id, "outcome": result.get('outcome', 'unknown'), "consequences": consequences, "success": True}
    except Exception as e:
        logger.error(f"Error resolving conflict: {str(e)}", exc_info=True)
        return {"conflict_id": conflict_id, "outcome": "error", "consequences": [f"Error: {str(e)}"], "success": False}

@function_tool
@track_performance("analyze_narrative_for_conflict")
async def analyze_narrative_for_conflict(ctx: RunContextWrapper[ContextType], narrative_text: str) -> Dict[str, Any]:
    """
    Analyze a narrative text to see if it should trigger a conflict.

    Args:
        narrative_text: The narrative text to analyze

    Returns:
        Analysis results and possibly a new conflict
    """
    context = ctx.context
    # conflict_manager = context.conflict_manager # Not directly used in this specific path anymore

    try:
        # ****** MODIFIED CALL ******
        # result = await add_conflict_to_narrative(ctx, narrative_text) # OLD - This was calling the FunctionTool object
        result = await _internal_add_conflict_to_narrative_logic(ctx, narrative_text) # NEW - This calls the callable async function
        
        if hasattr(context, 'add_narrative_memory') and result.get("trigger_conflict", False): # Adjusted key based on refactored logic
            conflict_info = { # Reconstruct a simple conflict_info if needed for memory
                "conflict_type": result.get("conflict_type", "unknown"),
                "conflict_name": result.get("conflict_name", "Unnamed conflict")
            }
            memory_content = (
                f"Analysis detected conflict in narrative and generated new "
                f"{conflict_info.get('conflict_type', 'unknown')} conflict: "
                f"{conflict_info.get('conflict_name', 'Unnamed conflict')}"
            )
            await context.add_narrative_memory(memory_content, "conflict_analysis", 0.6)
        return result
    except Exception as e:
        logger.error(f"Error analyzing narrative for conflict: {str(e)}", exc_info=True)
        # Ensure the error return structure is consistent if needed
        return {"analysis": {"conflict_intensity": 0, "matched_keywords": []}, "trigger_conflict": False, "error": str(e)}


@function_tool
@track_performance("set_player_involvement")
async def set_player_involvement(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    involvement_level: str,
    # 1. Change parameters with defaults to Optional[Type] = None
    faction: Optional[str] = None,
    money_committed: Optional[int] = None,
    supplies_committed: Optional[int] = None,
    influence_committed: Optional[int] = None,
    action: Optional[str] = None # This one was already correct
) -> Dict[str, Any]:
    """
    Set the player's involvement in a conflict.

    Args:
        conflict_id: ID of the conflict.
        involvement_level: Level of involvement (none, observing, participating, leading).
        faction: Which faction to support (a, b, neutral). Defaults to 'neutral'. # 2. Update docstrings
        money_committed: Money committed to the conflict. Defaults to 0.
        supplies_committed: Supplies committed to the conflict. Defaults to 0.
        influence_committed: Influence committed to the conflict. Defaults to 0.
        action: Optional specific action taken. Defaults to None.

    Returns:
        Updated conflict information or error dictionary.
    """
    context = ctx.context
    # Ensure managers are available on the context
    if not hasattr(context, 'conflict_manager') or not hasattr(context, 'resource_manager'):
         logger.error("Context missing conflict_manager or resource_manager in set_player_involvement")
         return {"conflict_id": conflict_id, "error": "Internal context setup error", "success": False}

    conflict_manager = context.conflict_manager
    resource_manager = context.resource_manager

    # 3. Handle default values inside the function
    actual_faction = faction if faction is not None else "neutral"
    actual_money = money_committed if money_committed is not None else 0
    actual_supplies = supplies_committed if supplies_committed is not None else 0
    actual_influence = influence_committed if influence_committed is not None else 0
    # 'action' is already Optional, no need for 'actual_action' unless you want to default it differently

    try:
        # 4. Use the 'actual_' variables in function logic
        resource_check = await resource_manager.check_resources(
            actual_money, actual_supplies, actual_influence
        )
        if not resource_check.get('has_resources', False):
            # Ensure 'success' field exists for consistency
            resource_check['success'] = False
            resource_check['error'] = "Insufficient resources to commit"
            return resource_check

        conflict_info = await conflict_manager.get_conflict(conflict_id)
        result = await conflict_manager.set_player_involvement(
            conflict_id,
            involvement_level,
            actual_faction, # Use actual_
            actual_money, # Use actual_
            actual_supplies, # Use actual_
            actual_influence, # Use actual_
            action # Use original optional action
        )

        # Add memory log using 'actual_' values
        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if actual_money > 0: resources_text.append(f"{actual_money} money")
            if actual_supplies > 0: resources_text.append(f"{actual_supplies} supplies")
            if actual_influence > 0: resources_text.append(f"{actual_influence} influence")
            resources_committed = ", ".join(resources_text) if resources_text else "no resources"

            conflict_name = conflict_info.get('conflict_name', f'ID: {conflict_id}') if conflict_info else f'ID: {conflict_id}'
            memory_content = (
                f"Player set involvement in conflict {conflict_name} "
                f"to {involvement_level}, supporting {actual_faction} faction " # Use actual_
                f"with {resources_committed}."
            )
            if action: memory_content += f" Action taken: {action}"
            await context.add_narrative_memory(memory_content, "conflict_involvement", 0.7)

        # Ensure result format consistency
        if isinstance(result, dict):
            result["success"] = True # Ensure success flag is present
        else:
            # If the underlying call didn't return a dict, construct one
            result = {
                "conflict_id": conflict_id,
                "involvement_level": involvement_level,
                "faction": actual_faction,
                "resources_committed": {
                    "money": actual_money,
                    "supplies": actual_supplies,
                    "influence": actual_influence
                },
                "action": action,
                "success": True,
                # Add any other relevant info if the underlying call returned something else
                "raw_result": result
            }

        return result
    except Exception as e:
        logger.error(f"Error setting involvement for conflict {conflict_id}: {str(e)}", exc_info=True)
        return {"conflict_id": conflict_id, "error": str(e), "success": False}

@function_tool
@track_performance("get_conflict_details")
async def get_conflict_details(ctx: RunContextWrapper[ContextType], conflict_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific conflict.

    Args:
        conflict_id: ID of the conflict

    Returns:
        Detailed conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager

    try:
        conflict = await conflict_manager.get_conflict(conflict_id)
        involved_npcs = await conflict_manager.get_conflict_npcs(conflict_id)
        player_involvement = await conflict_manager.get_player_involvement(conflict_id)

        result = {
            "conflict_id": conflict_id,
            "conflict_name": conflict.get('conflict_name', ''),
            "conflict_type": conflict.get('conflict_type', ''),
            "description": conflict.get('description', ''),
            "phase": conflict.get('phase', ''),
            "progress": conflict.get('progress', 0),
            "faction_a_name": conflict.get('faction_a_name', ''),
            "faction_b_name": conflict.get('faction_b_name', ''),
            "involved_npcs": involved_npcs,
            "player_involvement": player_involvement,
            "start_day": conflict.get('start_day', 0),
            "estimated_duration": conflict.get('estimated_duration', 0),
            "resources_required": conflict.get('resources_required', {}),
            "success_rate": conflict.get('success_rate', 0)
        }
        return result
    except Exception as e:
        logger.error(f"Error getting conflict details: {str(e)}", exc_info=True)
        return {"conflict_id": conflict_id, "error": f"Failed to get conflict details: {str(e)}", "success": False}

# ----- Resource Tools -----

@function_tool
@track_performance("check_resources")
async def check_resources(
    ctx: RunContextWrapper[ContextType],
    # 1. Change parameters with defaults to Optional[Type] = None
    money: Optional[int] = None,
    supplies: Optional[int] = None,
    influence: Optional[int] = None
) -> Dict[str, Any]:
    """
    Check if player has sufficient resources.

    Args:
        money: Required amount of money. Defaults to 0 if not provided. # 2. Update docstrings
        supplies: Required amount of supplies. Defaults to 0 if not provided.
        influence: Required amount of influence. Defaults to 0 if not provided.

    Returns:
        Dictionary with resource check results.
    """
    context = ctx.context
    # Ensure resource_manager is available on the context
    if not hasattr(context, 'resource_manager'):
         logger.error("Context missing resource_manager in check_resources")
         return {"has_resources": False, "error": "Internal context setup error", "current": {}}
    resource_manager = context.resource_manager

    # 3. Handle default values inside the function
    actual_money = money if money is not None else 0
    actual_supplies = supplies if supplies is not None else 0
    actual_influence = influence if influence is not None else 0

    try:
        # 4. Use the 'actual_' variables in the function call
        result = await resource_manager.check_resources(
            actual_money, actual_supplies, actual_influence
        )

        # Format money if present in the result
        current_res = result.get('current', {})
        if current_res.get('money') is not None:
            try:
                # Ensure get_formatted_money exists and handles potential errors
                formatted_money = await resource_manager.get_formatted_money(current_res['money'])
                current_res['formatted_money'] = formatted_money
                result['current'] = current_res # Update the dict in result
            except Exception as format_err:
                logger.warning(f"Could not format money in check_resources: {format_err}")
                # Optionally add formatted_money: None or keep it absent

        # Ensure consistent return structure
        if 'has_resources' not in result:
            result['has_resources'] = False # Assume false if not explicitly set
        if 'current' not in result:
            result['current'] = {}

        return result
    except Exception as e:
        logger.error(f"Error checking resources: {str(e)}", exc_info=True)
        return {"has_resources": False, "error": str(e), "current": {}}
        
@function_tool
@track_performance("commit_resources_to_conflict")
async def commit_resources_to_conflict(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    # 1. Change parameters with defaults to Optional[Type] = None
    money: Optional[int] = None,
    supplies: Optional[int] = None,
    influence: Optional[int] = None
) -> Dict[str, Any]:
    """
    Commit player resources to a conflict.

    Args:
        conflict_id: ID of the conflict.
        money: Amount of money to commit. Defaults to 0 if not provided. # 2. Update docstrings
        supplies: Amount of supplies to commit. Defaults to 0 if not provided.
        influence: Amount of influence to commit. Defaults to 0 if not provided.

    Returns:
        Result of committing resources or error dictionary.
    """
    context = ctx.context
    # Ensure managers are available on the context
    if not hasattr(context, 'resource_manager'):
         logger.error("Context missing resource_manager in commit_resources_to_conflict")
         return {"success": False, "error": "Internal context setup error"}
    resource_manager = context.resource_manager

    # 3. Handle default values inside the function
    actual_money = money if money is not None else 0
    actual_supplies = supplies if supplies is not None else 0
    actual_influence = influence if influence is not None else 0

    try:
        conflict_info = None
        # Check for conflict_manager safely
        if hasattr(context, 'conflict_manager') and context.conflict_manager:
            try:
                conflict_info = await context.conflict_manager.get_conflict(conflict_id)
            except Exception as conflict_error:
                logger.warning(f"Could not get conflict info for conflict {conflict_id}: {conflict_error}")
        else:
             logger.warning("Context missing conflict_manager in commit_resources_to_conflict")


        # 4. Use the 'actual_' variables in the function call
        result = await resource_manager.commit_resources_to_conflict(
            conflict_id, actual_money, actual_supplies, actual_influence
        )

        # Format money if money was committed and result is successful
        if actual_money > 0 and result.get('success', False) and result.get('money_result'):
            money_result = result['money_result']
            # Check if old/new values exist before formatting
            if 'old_value' in money_result and 'new_value' in money_result:
                try:
                    old_formatted = await resource_manager.get_formatted_money(money_result['old_value'])
                    new_formatted = await resource_manager.get_formatted_money(money_result['new_value'])
                    # Ensure change exists before formatting
                    change_val = money_result.get('change')
                    formatted_change = await resource_manager.get_formatted_money(change_val) if change_val is not None else None

                    money_result['formatted_old_value'] = old_formatted
                    money_result['formatted_new_value'] = new_formatted
                    if formatted_change is not None:
                        money_result['formatted_change'] = formatted_change
                    result['money_result'] = money_result
                except Exception as format_err:
                    logger.warning(f"Could not format money in commit_resources_to_conflict: {format_err}")

        # Add memory log using 'actual_' values
        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if actual_money > 0: resources_text.append(f"{actual_money} money")
            if actual_supplies > 0: resources_text.append(f"{actual_supplies} supplies")
            if actual_influence > 0: resources_text.append(f"{actual_influence} influence")
            resources_committed = ", ".join(resources_text) if resources_text else "No resources"

            conflict_name = conflict_info.get('conflict_name', f"ID: {conflict_id}") if conflict_info else f"ID: {conflict_id}"
            memory_content = f"Committed {resources_committed} to conflict {conflict_name}"
            await context.add_narrative_memory(memory_content, "resource_commitment", 0.6)

        # Ensure success flag is present in the final result
        if 'success' not in result:
             result['success'] = True # Assume success if no error occurred

        return result
    except Exception as e:
        logger.error(f"Error committing resources for conflict {conflict_id}: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@function_tool
@track_performance("get_player_resources")
async def get_player_resources(ctx: RunContextWrapper[ContextType]) -> Dict[str, Any]:
    """
    Get the current player resources and vitals.

    Returns:
        Current resource status
    """
    context = ctx.context
    resource_manager = context.resource_manager

    try:
        resources = await resource_manager.get_resources()
        vitals = await resource_manager.get_vitals()
        formatted_money = await resource_manager.get_formatted_money()
        updated_at = resources.get('updated_at', datetime.now())
        updated_at_iso = updated_at.isoformat() if isinstance(updated_at, datetime) else str(updated_at)

        return {"money": resources.get('money', 0), "supplies": resources.get('supplies', 0), "influence": resources.get('influence', 0), "energy": vitals.get('energy', 0), "hunger": vitals.get('hunger', 0), "formatted_money": formatted_money, "updated_at": updated_at_iso}
    except Exception as e:
        logger.error(f"Error getting player resources: {str(e)}", exc_info=True)
        return {"error": str(e), "money": 0, "supplies": 0, "influence": 0, "energy": 0, "hunger": 0, "formatted_money": "0"}

@function_tool
@track_performance("analyze_activity_effects")
async def analyze_activity_effects(ctx: RunContextWrapper[ContextType], activity_text: str) -> Dict[str, Any]:
    """
    Analyze an activity to determine its effects on player resources.

    Args:
        activity_text: Description of the activity

    Returns:
        Activity effects
    """
    context = ctx.context
    activity_analyzer = context.activity_analyzer

    try:
        result = await activity_analyzer.analyze_activity(activity_text, apply_effects=False) # Analyze only
        effects = result.get('effects', {})

        if hasattr(context, 'add_narrative_memory'):
            effects_description = [f"{res} {('increased' if val > 0 else 'decreased')} by {abs(val)}" for res, val in effects.items() if val]
            effects_text = ", ".join(effects_description) if effects_description else "no significant effects"
            memory_content = f"Analyzed activity: {activity_text[:100]}... with {effects_text}"
            await context.add_narrative_memory(memory_content, "activity_analysis", 0.4)

        return {"activity_type": result.get('activity_type', 'unknown'), "activity_details": result.get('activity_details', ''), "hunger_effect": effects.get('hunger'), "energy_effect": effects.get('energy'), "money_effect": effects.get('money'), "supplies_effect": effects.get('supplies'), "influence_effect": effects.get('influence'), "description": result.get('description', f"Effects of {activity_text}")}
    except Exception as e:
        logger.error(f"Error analyzing activity effects: {str(e)}", exc_info=True)
        return {"activity_type": "unknown", "activity_details": "", "description": f"Failed to analyze: {str(e)}", "error": str(e)}

# This seems like the intended apply function.
@function_tool
@track_performance("apply_activity_effects")
async def apply_activity_effects(ctx: RunContextWrapper[ContextType], activity_text: str) -> Dict[str, Any]:
    """
    Analyze and apply the effects of an activity to player resources.

    Args:
        activity_text: Description of the activity

    Returns:
        Results of applying activity effects
    """
    context = ctx.context
    activity_analyzer = context.activity_analyzer

    try:
        result = await activity_analyzer.analyze_activity(activity_text, apply_effects=True) # Analyze AND apply

        if 'effects' in result and 'money' in result['effects']:
            resource_manager = context.resource_manager
            resources = await resource_manager.get_resources()
            result['formatted_money'] = await resource_manager.get_formatted_money(resources.get('money', 0))

        if hasattr(context, 'add_narrative_memory'):
            effects = result.get('effects', {})
            effects_description = [f"{res} {('increased' if val > 0 else 'decreased')} by {abs(val)}" for res, val in effects.items() if val]
            effects_text = ", ".join(effects_description) if effects_description else "no significant effects"
            memory_content = f"Applied activity effects for: {activity_text[:100]}... with {effects_text}"
            await context.add_narrative_memory(memory_content, "activity_application", 0.5)

        return result
    except Exception as e:
        logger.error(f"Error applying activity effects: {str(e)}", exc_info=True)
        return {"error": str(e), "activity_type": "unknown", "activity_details": "", "effects": {}}

@function_tool
@track_performance("get_resource_history")
async def get_resource_history(
    ctx: RunContextWrapper[ContextType],
    resource_type: Optional[str] = None, # This one was already correct
    # 1. Change signature: limit: Optional[int] = None
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get the history of resource changes.

    Args:
        resource_type: Optional filter for specific resource type (money, supplies, influence, energy, hunger).
        limit: Maximum number of history entries to return. Defaults to 10 if not provided. # 2. Update docstring

    Returns:
        List of resource change history entries.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Ensure resource_manager is available for formatting money
    if not hasattr(context, 'resource_manager'):
         logger.error("Context missing resource_manager in get_resource_history")
         # Decide if you can proceed without it or return error
         # For now, let's proceed but log a warning later if needed for formatting
         resource_manager = None # Set to None to check later
    else:
        resource_manager = context.resource_manager


    # 3. Handle the default value inside the function
    actual_limit = limit if limit is not None else 10

    async with get_db_connection_context() as conn:
        try:
            base_query = "SELECT resource_type, old_value, new_value, amount_changed, source, description, timestamp FROM ResourceHistoryLog WHERE user_id=$1 AND conversation_id=$2"
            params = [user_id, conversation_id]

            # 4. Use the actual_limit in query construction
            if resource_type:
                base_query += " AND resource_type=$3 ORDER BY timestamp DESC LIMIT $4"
                params.extend([resource_type, actual_limit]) # Use actual_limit
            else:
                base_query += " ORDER BY timestamp DESC LIMIT $3"
                params.append(actual_limit) # Use actual_limit

            rows = await conn.fetch(base_query, *params)
            history = []

            for row in rows:
                formatted_old, formatted_new, formatted_change = None, None, None
                # Safely check resource_manager before using
                if row['resource_type'] == "money" and resource_manager:
                    try:
                        formatted_old = await resource_manager.get_formatted_money(row['old_value'])
                        formatted_new = await resource_manager.get_formatted_money(row['new_value'])
                        formatted_change = await resource_manager.get_formatted_money(row['amount_changed'])
                    except Exception as format_err:
                         logger.warning(f"Could not format money in get_resource_history: {format_err}")
                elif row['resource_type'] == "money" and not resource_manager:
                     logger.warning("Cannot format money in get_resource_history: resource_manager missing from context.")


                timestamp = row['timestamp']
                timestamp_iso = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

                history.append({
                    "resource_type": row['resource_type'],
                    "old_value": row['old_value'],
                    "new_value": row['new_value'],
                    "amount_changed": row['amount_changed'],
                    "formatted_old_value": formatted_old,
                    "formatted_new_value": formatted_new,
                    "formatted_change": formatted_change,
                    "source": row['source'],
                    "description": row['description'],
                    "timestamp": timestamp_iso
                })
            return history
        except Exception as e:
            logger.error(f"Error getting resource history: {str(e)}", exc_info=True)
            return []

# ----- Narrative Tools -----

@function_tool
@track_performance("generate_personal_revelation")
async def generate_personal_revelation(ctx: RunContextWrapper[ContextType], npc_name: str, revelation_type: str) -> Dict[str, Any]:
    """
    Generate a personal revelation for the player about their relationship with an NPC.

    Args:
        npc_name: Name of the NPC involved in the revelation
        revelation_type: Type of revelation (dependency, obedience, corruption, willpower, confidence)

    Returns:
        A personal revelation
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Define revelation templates based on type
    templates = {
        "dependency": [
            f"I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
            f"I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
            f"The thought of spending a day without talking to {npc_name} makes me anxious. I should be concerned about that, shouldn't I?"
        ],
        "obedience": [
            f"I caught myself automatically rearranging my schedule when {npc_name} hinted she wanted to see me. I didn't even think twice about it.",
            f"Today I changed my opinion the moment I realized it differed from {npc_name}'s. That's... not like me. Or is it becoming like me?",
            f"{npc_name} gave me that look, and I immediately stopped what I was saying. When did her disapproval start carrying so much weight?"
        ],
        "corruption": [
            f"I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
            f"Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
            f"Sometimes I catch glimpses of my old self, like a stranger I used to know. When did I change so fundamentally?"
        ],
        "willpower": [
            f"I had every intention of saying no to {npc_name} today. The 'yes' came out before I even realized I was speaking.",
            f"I've been trying to remember what it felt like to disagree with {npc_name}. The memory feels distant, like it belongs to someone else.",
            f"I made a list of boundaries I wouldn't cross. Looking at it now, I've broken every single one at {npc_name}'s suggestion."
        ],
        "confidence": [
            f"I opened my mouth to speak in the meeting, then saw {npc_name} watching me. I suddenly couldn't remember what I was going to say.",
            f"I used to trust my judgment. Now I find myself second-guessing every thought that {npc_name} hasn't explicitly approved.",
            f"When did I start feeling this small? This uncertain? I can barely remember how it felt to be sure of myself."
        ]
    }


    try:
        revelation_templates = templates.get(revelation_type.lower(), templates["dependency"])
        inner_monologue = random.choice(revelation_templates) # No need to format again if f-string used above

        async with get_db_connection_context() as conn:
            try:
                journal_id = await conn.fetchval("INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, revelation_types, timestamp) VALUES ($1, $2, 'personal_revelation', $3, $4, CURRENT_TIMESTAMP) RETURNING id", user_id, conversation_id, inner_monologue, revelation_type)

                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(f"Personal revelation about {npc_name}: {inner_monologue}", "personal_revelation", 0.8, tags=[revelation_type, "revelation", npc_name.lower().replace(" ", "_")])
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
                    await context.narrative_manager.add_revelation(content=inner_monologue, revelation_type=revelation_type, importance=0.8, tags=[revelation_type, "revelation"])

                return {"type": "personal_revelation", "name": f"{revelation_type.capitalize()} Awareness", "inner_monologue": inner_monologue, "journal_id": journal_id, "success": True}
            except Exception as db_error: logger.error(f"Database error recording personal revelation: {db_error}"); raise
    except Exception as e:
        logger.error(f"Error generating personal revelation: {str(e)}", exc_info=True)
        return {"type": "personal_revelation", "name": f"{revelation_type.capitalize()} Awareness", "inner_monologue": f"Error generating revelation: {str(e)}", "success": False}


@function_tool
@track_performance("generate_dream_sequence")
async def generate_dream_sequence(ctx: RunContextWrapper[ContextType], npc_names: List[str]) -> Dict[str, Any]:
    """
    Generate a symbolic dream sequence based on player's current state.

    Args:
        npc_names: List of NPC names to include in the dream

    Returns:
        A dream sequence
    """
    while len(npc_names) < 3: npc_names.append(f"Unknown Figure {len(npc_names) + 1}")
    npc1, npc2, npc3 = npc_names[:3]
    
    # Dream templates
    dream_templates = [
        f"You're sitting in a chair as {npc1} circles you slowly. \"Show me your hands,\" she says. You extend them, surprised to find intricate strings wrapped around each finger, extending upward. \"Do you see who's holding them?\" she asks. You look up, but the ceiling is mirrored, showing only your own face looking back down at you, smiling with an expression that isn't yours.",
        f"You're searching your home frantically, calling {npc1}'s name. The rooms shift and expand, doorways leading to impossible spaces. Your phone rings. It's {npc1}. \"Where are you?\" you ask desperately. \"I'm right here,\" she says, her voice coming both from the phone and from behind you. \"I've always been right here. You're the one who's lost.\"",
        f"You're trying to walk away from {npc1}, but your feet sink deeper into the floor with each step. \"I don't understand why you're struggling,\" she says, not moving yet somehow keeping pace beside you. \"You stopped walking on your own long ago.\" You look down to find your legs have merged with the floor entirely, indistinguishable from the material beneath.",
        f"You're giving a presentation to a room full of people, but every time you speak, your voice comes out as {npc1}'s voice, saying words you didn't intend. The audience nods approvingly. \"Much better,\" whispers {npc2} from beside you. \"Your ideas were never as good as hers anyway.\"",
        f"You're walking through an unfamiliar house, opening doors that should lead outside but only reveal more rooms. In each room, {npc1} is engaged in a different activity, wearing a different expression. In the final room, all versions of her turn to look at you simultaneously. \"Which one is real?\" they ask in unison. \"The one you needed, or the one who needed you?\"",
        f"You're swimming in deep water. Below you, {npc1} and {npc2} walk along the bottom, looking up at you and conversing, their voices perfectly clear despite the water. \"They still think they're above it all,\" says {npc1}, and they both laugh. You realize you can't remember how to reach the surface."
    ]


    try:
        dream_text = random.choice(dream_templates) # Already formatted
        context = ctx.context
        user_id = context.user_id
        conversation_id = context.conversation_id

        async with get_db_connection_context() as conn:
            try:
                journal_id = await conn.fetchval("INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp) VALUES ($1, $2, 'dream_sequence', $3, CURRENT_TIMESTAMP) RETURNING id", user_id, conversation_id, dream_text)

                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(f"Dream sequence: {dream_text}", "dream_sequence", 0.7, tags=["dream", "symbolic"] + [npc.lower().replace(" ", "_") for npc in npc_names[:3]])
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
                    await context.narrative_manager.add_dream_sequence(content=dream_text, symbols=[npc1, npc2, npc3, "control", "manipulation"], importance=0.7, tags=["dream", "symbolic"])

                return {"type": "dream_sequence", "text": dream_text, "journal_id": journal_id, "success": True}
            except Exception as db_error: logger.error(f"Database error recording dream sequence: {db_error}"); raise
    except Exception as e:
        logger.error(f"Error generating dream sequence: {str(e)}", exc_info=True)
        return {"type": "dream_sequence", "text": f"Error generating dream: {str(e)}", "success": False}

@function_tool
@track_performance("check_relationship_events")
async def check_relationship_events(ctx: RunContextWrapper[ContextType]) -> Dict[str, Any]:
    """
    Check for relationship events like crossroads or rituals.

    Returns:
        Dictionary with any triggered relationship events
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        crossroads = await check_for_crossroads_tool(user_id, conversation_id)
        ritual = await check_for_ritual_tool(user_id, conversation_id)

        if (crossroads or ritual) and hasattr(context, 'add_narrative_memory'):
            event_type = "crossroads" if crossroads else "ritual"; npc_name = "Unknown"
            if crossroads: npc_name = crossroads.get("npc_name", "Unknown")
            elif ritual: npc_name = ritual.get("npc_name", "Unknown")
            memory_content = f"Relationship {event_type} detected with {npc_name}"
            await context.add_narrative_memory(memory_content, f"relationship_{event_type}", 0.8, tags=[event_type, "relationship", npc_name.lower().replace(" ", "_")])

        return {"crossroads": crossroads, "ritual": ritual, "has_events": crossroads is not None or ritual is not None}
    except Exception as e:
        logger.error(f"Error checking relationship events: {str(e)}", exc_info=True)
        return {"error": str(e), "crossroads": None, "ritual": None, "has_events": False}

@function_tool
@track_performance("apply_crossroads_choice_tool")
async def apply_crossroads_choice_tool(
    ctx: RunContextWrapper[ContextType],
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> Dict[str, Any]:
    """
    Apply a chosen effect from a triggered relationship crossroads.

    Args:
        link_id: ID of the social link
        crossroads_name: Name of the crossroads event
        choice_index: Index of the chosen option

    Returns:
        Result of applying the choice
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        result = await apply_crossroads_choice_tool(user_id, conversation_id, link_id, crossroads_name, choice_index)

        if hasattr(context, 'add_narrative_memory'):
            npc_name = "Unknown"
            try:
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow("SELECT entity2_id FROM SocialLinks WHERE link_id = $1 AND entity2_type = 'npc'", link_id)
                    if row: npc_id = row['entity2_id']; npc_row = await conn.fetchrow("SELECT npc_name FROM NPCStats WHERE npc_id = $1", npc_id); npc_name = npc_row['npc_name'] if npc_row else npc_name
            except Exception as db_error: logger.warning(f"Could not get NPC name for memory: {db_error}")

            memory_content = f"Applied crossroads choice {choice_index} for '{crossroads_name}' with {npc_name}"
            await context.add_narrative_memory(memory_content, "crossroads_choice", 0.8, tags=["crossroads", "relationship", npc_name.lower().replace(" ", "_")])
            if hasattr(context, 'narrative_manager') and context.narrative_manager:
                await context.narrative_manager.add_interaction(content=memory_content, npc_name=npc_name, importance=0.8, tags=["crossroads", "relationship_choice"])

        return result
    except Exception as e:
        logger.error(f"Error applying crossroads choice: {str(e)}", exc_info=True)
        return {"link_id": link_id, "crossroads_name": crossroads_name, "choice_index": choice_index, "success": False, "error": str(e)}

@function_tool
@track_performance("check_npc_relationship")
async def check_npc_relationship(ctx: RunContextWrapper[ContextType], npc_id: int) -> Dict[str, Any]:
    """
    Get the relationship between the player and an NPC.

    Args:
        npc_id: ID of the NPC

    Returns:
        Relationship summary
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        relationship = await get_relationship_summary_tool(user_id, conversation_id, "player", user_id, "npc", npc_id)
        if not relationship:
            try:
                from logic.social_links_agentic import create_social_link
                link_id = await create_social_link(user_id, conversation_id, "player", user_id, "npc", npc_id)
                relationship = await get_relationship_summary_tool(user_id, conversation_id, "player", user_id, "npc", npc_id) # Fetch again
            except Exception as link_error:
                logger.error(f"Error creating social link: {link_error}")
                return {"error": f"Failed to create relationship: {str(link_error)}", "npc_id": npc_id}

        return relationship or {"error": "Could not get or create relationship", "npc_id": npc_id}
    except Exception as e:
        logger.error(f"Error checking NPC relationship: {str(e)}", exc_info=True)
        return {"error": str(e), "npc_id": npc_id}

@function_tool
@track_performance("add_moment_of_clarity")
async def add_moment_of_clarity(ctx: RunContextWrapper[ContextType], realization_text: str) -> Dict[str, Any]:
    """
    Add a moment of clarity where the player briefly becomes aware of their situation.

    Args:
        realization_text: The specific realization the player has

    Returns:
        The created moment of clarity
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.narrative_progression import add_moment_of_clarity as add_clarity # Already imported
        result = await add_clarity(user_id, conversation_id, realization_text)

        if hasattr(context, 'add_narrative_memory'):
            await context.add_narrative_memory(f"Moment of clarity: {realization_text}", "moment_of_clarity", 0.9, tags=["clarity", "realization", "awareness"])
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            await context.narrative_manager.add_revelation(content=realization_text, revelation_type="clarity", importance=0.9, tags=["clarity", "realization"])

        return {"type": "moment_of_clarity", "content": result, "success": True}
    except Exception as e:
        logger.error(f"Error adding moment of clarity: {str(e)}", exc_info=True)
        return {"type": "moment_of_clarity", "content": None, "success": False, "error": str(e)}

@function_tool
@track_performance("get_player_journal_entries")
async def get_player_journal_entries(
    ctx: RunContextWrapper[ContextType],
    entry_type: Optional[str] = None, # This one was already correct
    # 1. Change signature: limit: Optional[int] = None
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get entries from the player's journal.

    Args:
        entry_type: Optional filter for entry type (personal_revelation, dream_sequence, moment_of_clarity, etc.).
        limit: Maximum number of entries to return. Defaults to 10 if not provided. # 2. Update docstring

    Returns:
        List of journal entries.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 3. Handle the default value inside the function
    actual_limit = limit if limit is not None else 10

    async with get_db_connection_context() as conn:
        try:
            base_query = "SELECT id, entry_type, entry_text, revelation_types, narrative_moment, fantasy_flag, intensity_level, timestamp FROM PlayerJournal WHERE user_id=$1 AND conversation_id=$2"
            params = [user_id, conversation_id]

            # 4. Use the actual_limit in query construction
            if entry_type:
                base_query += " AND entry_type=$3 ORDER BY timestamp DESC LIMIT $4"
                params.extend([entry_type, actual_limit]) # Use actual_limit
            else:
                base_query += " ORDER BY timestamp DESC LIMIT $3"
                params.append(actual_limit) # Use actual_limit

            rows = await conn.fetch(base_query, *params)
            entries = []
            for row in rows:
                 timestamp = row['timestamp']
                 timestamp_iso = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
                 # Ensure all keys exist or use .get() for safety
                 entries.append({
                     "id": row.get('id'),
                     "entry_type": row.get('entry_type'),
                     "entry_text": row.get('entry_text'),
                     "revelation_types": row.get('revelation_types'),
                     "narrative_moment": row.get('narrative_moment'),
                     "fantasy_flag": row.get('fantasy_flag'),
                     "intensity_level": row.get('intensity_level'),
                     "timestamp": timestamp_iso
                 })
            return entries
        except Exception as e:
            logger.error(f"Error getting player journal entries: {str(e)}", exc_info=True)
            return []
            
@function_tool
async def analyze_conflict_potential(ctx: RunContextWrapper[ContextType], narrative_text: str) -> Dict[str, Any]:
    """
    Analyze narrative text for conflict potential.

    Args:
        narrative_text: The narrative text to analyze

    Returns:
        Conflict potential analysis
    """
    context = ctx.context # Use attribute access
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        conflict_keywords = [
            "argument", "disagreement", "tension", "rivalry", "competition",
            "dispute", "feud", "clash", "confrontation", "battle", "fight",
            "war", "conflict", "power struggle", "contest", "strife"
        ]
        
        matched_keywords = []
        for keyword in conflict_keywords:
            if keyword in narrative_text.lower():
                matched_keywords.append(keyword)
        
        # Calculate conflict intensity based on keyword matches
        conflict_intensity = min(10, len(matched_keywords) * 2)
        
        # Check for NPC mentions
        from logic.conflict_system.conflict_manager import ConflictManager
        conflict_manager = ConflictManager(user_id, conversation_id)
        npcs = await conflict_manager._get_available_npcs()
        
        mentioned_npcs = []
        for npc in npcs:
            if npc["npc_name"] in narrative_text:
                mentioned_npcs.append({
                    "npc_id": npc["npc_id"],
                    "npc_name": npc["npc_name"],
                    "dominance": npc["dominance"],
                    "faction_affiliations": npc.get("faction_affiliations", [])
                })
        
        # Look for faction mentions
        mentioned_factions = []
        for npc in mentioned_npcs:
            for affiliation in npc.get("faction_affiliations", []):
                faction_name = affiliation.get("faction_name")
                if faction_name and faction_name in narrative_text:
                    mentioned_factions.append({
                        "faction_id": affiliation.get("faction_id"),
                        "faction_name": faction_name
                    })
        
        # Check for relationship indicators between NPCs
        npc_relationships = []
        for i, npc1 in enumerate(mentioned_npcs):
            for npc2 in mentioned_npcs[i+1:]:
                # Look for both NPCs in the same sentence
                sentences = narrative_text.split('.')
                for sentence in sentences:
                    if npc1["npc_name"] in sentence and npc2["npc_name"] in sentence:
                        # Check for relationship indicators
                        relationship_type = "unknown"
                        for word in ["allies", "friends", "partners", "together"]:
                            if word in sentence.lower():
                                relationship_type = "alliance"
                                break
                        for word in ["enemies", "rivals", "hate", "against"]:
                            if word in sentence.lower():
                                relationship_type = "rivalry"
                                break
                        
                        npc_relationships.append({
                            "npc1_id": npc1["npc_id"],
                            "npc1_name": npc1["npc_name"],
                            "npc2_id": npc2["npc_id"],
                            "npc2_name": npc2["npc_name"],
                            "relationship_type": relationship_type,
                            "sentence": sentence.strip()
                        })
        
        # Determine appropriate conflict type based on analysis
        conflict_type = "major" if conflict_intensity >= 8 else ("standard" if conflict_intensity >= 5 else "minor")
        internal_faction_conflict = None # Logic seems complex, assume correct for now

        return {"conflict_intensity": conflict_intensity, "matched_keywords": matched_keywords, "mentioned_npcs": mentioned_npcs, "mentioned_factions": mentioned_factions, "npc_relationships": npc_relationships, "recommended_conflict_type": conflict_type, "potential_internal_faction_conflict": internal_faction_conflict, "has_conflict_potential": conflict_intensity >= 4}
    except Exception as e:
        logging.error(f"Error analyzing conflict potential: {e}")
        return {"conflict_intensity": 0, "matched_keywords": [], "mentioned_npcs": [], "mentioned_factions": [], "npc_relationships": [], "recommended_conflict_type": "minor", "potential_internal_faction_conflict": None, "has_conflict_potential": False, "error": str(e)}

@function_tool
# 1. Change signature: analysis: Optional[Dict[str, Any]] = None
async def generate_conflict_from_analysis(
    ctx: RunContextWrapper[ContextType],
    analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a conflict based on analysis provided by analyze_conflict_potential.

    Args:
        analysis: Conflict potential analysis dictionary (required). # 3. Update docstring

    Returns:
        Generated conflict details or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 2. Add internal check for the required parameter
    if analysis is None:
        logger.error("generate_conflict_from_analysis called without 'analysis' parameter.")
        return {
            "generated": False,
            "reason": "Missing required 'analysis' parameter.",
            "analysis": None
        }

    try:
        # Now proceed with the original logic, using the validated 'analysis' dict
        if not analysis.get("has_conflict_potential", False):
            return {"generated": False, "reason": "Insufficient conflict potential", "analysis": analysis}

        # Assuming ConflictSystemIntegration is imported correctly
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        conflict_type = analysis.get("recommended_conflict_type", "standard")
        # Assuming generate_new_conflict exists and works
        conflict = await conflict_integration.generate_new_conflict(conflict_type)

        internal_faction_conflict = None
        potential_internal = analysis.get("potential_internal_faction_conflict")
        # Check if conflict generation was successful before proceeding
        if potential_internal and conflict and conflict.get("conflict_id"):
            internal_data = potential_internal
            try:
                # Assuming initiate_faction_power_struggle exists and works
                internal_faction_conflict = await conflict_integration.initiate_faction_power_struggle(
                    conflict["conflict_id"],
                    internal_data["faction_id"],
                    internal_data["challenger_npc_id"],
                    internal_data["target_npc_id"],
                    internal_data["prize"],
                    internal_data["approach"],
                    False # Not public by default
                )
            except Exception as e:
                # Log the specific error for the internal conflict generation
                logger.error(f"Error generating internal faction conflict within generate_conflict_from_analysis: {e}")
                # Decide if this should prevent returning the main conflict or just be logged
                # For now, just log and continue

        # Ensure conflict is included even if internal conflict fails
        return {
            "generated": True,
            "conflict": conflict, # Return the main conflict info
            "internal_faction_conflict": internal_faction_conflict # May be None if generation failed
        }
    except Exception as e:
        logger.error(f"Error generating conflict from analysis: {e}", exc_info=True) # Add exc_info
        return {"generated": False, "reason": f"Error: {str(e)}", "analysis": analysis}

@function_tool
async def analyze_npc_manipulation_potential(ctx: RunContextWrapper[ContextType], conflict_id: int, npc_id: int) -> Dict[str, Any]:
    """
    Analyze an NPC's potential to manipulate the player within a conflict.

    Args:
        conflict_id: ID of the conflict
        npc_id: ID of the NPC

    Returns:
        Manipulation potential analysis
    """
    context = ctx.context # Use attribute access
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        potential = await conflict_integration.analyze_manipulation_potential(npc_id) # Assuming this method exists
        conflict = await conflict_integration.get_conflict_details(conflict_id) # Assuming this method exists
        involvement = conflict.get("player_involvement") if conflict else None

        makes_sense = True; reason = "NPC could manipulate player"
        if involvement and involvement.get("involvement_level") != "none" and involvement.get("is_manipulated"):
            manipulator_id = involvement.get("manipulated_by", {}).get("npc_id")
            if manipulator_id == npc_id: makes_sense = False; reason = "NPC is already manipulating player"

        goal = {"faction": "neutral", "involvement_level": "observing"}
        if potential.get("femdom_compatible"): goal["involvement_level"] = "participating"
            # Faction logic seems complex, assume correct for now

        return {"npc_id": npc_id, "conflict_id": conflict_id, "manipulation_potential": potential, "makes_sense": makes_sense, "reason": reason, "recommended_goal": goal, "current_involvement": involvement}
    except Exception as e:
        logging.error(f"Error analyzing manipulation potential: {e}")
        return {"npc_id": npc_id, "conflict_id": conflict_id, "manipulation_potential": {}, "makes_sense": False, "reason": f"Error: {str(e)}", "recommended_goal": {}, "current_involvement": None}

@function_tool
# @track_performance("generate_manipulation_attempt") # Uncomment if track_performance is defined/imported
async def generate_manipulation_attempt(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    npc_id: int,
    manipulation_type: str,
    # 1. Change signature: goal: Optional[Dict[str, Any]] = None
    goal: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a manipulation attempt by an NPC in a conflict.

    Args:
        conflict_id: ID of the conflict.
        npc_id: ID of the NPC.
        manipulation_type: Type of manipulation (domination, blackmail, seduction, etc.).
        goal: What the NPC wants the player to do (required dictionary). # 3. Update docstring

    Returns:
        Generated manipulation attempt details or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 2. Add internal check for the required parameter
    if goal is None:
        logger.error("generate_manipulation_attempt called without 'goal' parameter.")
        return {
            "generated": False,
            "reason": "Missing required 'goal' parameter.",
            "npc_id": npc_id,
            "manipulation_type": manipulation_type
        }

    try:
        # Now proceed with the original logic, using the validated 'goal' dict
        # Assuming ConflictSystemIntegration is imported correctly
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        # Assuming these methods exist and work
        suggestion = await conflict_integration.suggest_manipulation_content(
            npc_id, conflict_id, manipulation_type, goal
        )
        attempt = await conflict_integration.create_manipulation_attempt(
            conflict_id,
            npc_id,
            manipulation_type,
            suggestion["content"],
            goal, # Use the validated goal
            suggestion["leverage_used"],
            suggestion["intimacy_level"]
        )

        # Ensure suggestion has expected keys before accessing
        npc_name = suggestion.get("npc_name", "Unknown NPC")
        content = suggestion.get("content", "No content generated.")

        return {
            "generated": True,
            "attempt": attempt,
            "npc_id": npc_id,
            "npc_name": npc_name,
            "manipulation_type": manipulation_type,
            "content": content
        }
    except Exception as e:
        # Log the specific error with context
        logger.error(f"Error generating manipulation attempt for NPC {npc_id} in conflict {conflict_id}: {e}", exc_info=True)
        return {
            "generated": False,
            "reason": f"Error: {str(e)}",
            "npc_id": npc_id,
            "manipulation_type": manipulation_type
        }
        
@function_tool
# @track_performance("track_conflict_story_beat") # Uncomment if track_performance is defined/imported
async def track_conflict_story_beat(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    path_id: str,
    beat_description: str,
    involved_npcs: List[int],
    # 1. Change signature: progress_value: Optional[float] = None
    progress_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Track a story beat for a resolution path, advancing progress.

    Args:
        conflict_id: ID of the conflict.
        path_id: ID of the resolution path.
        beat_description: Description of what happened.
        involved_npcs: List of NPC IDs involved.
        progress_value: Progress value (0-100). Defaults to 5.0 if not provided. # 2. Update docstring

    Returns:
        Updated path information or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # 3. Handle the default value inside the function
    actual_progress_value = progress_value if progress_value is not None else 5.0

    try:
        # Assuming ConflictSystemIntegration is imported correctly
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        # 4. Use the actual_progress_value in the call
        result = await conflict_integration.track_story_beat(
            conflict_id,
            path_id,
            beat_description,
            involved_npcs,
            actual_progress_value # Use the handled value
        ) # Assuming this method exists

        # Ensure consistent return structure
        if isinstance(result, dict):
            # If the underlying method returns a dict, assume it's okay
            return {"tracked": True, "result": result}
        else:
            # If not a dict, wrap it
             return {"tracked": True, "result": {"raw_output": result}} # Or adjust as needed

    except Exception as e:
        logger.error(f"Error tracking story beat for conflict {conflict_id}, path {path_id}: {e}", exc_info=True) # Add exc_info
        return {"tracked": False, "reason": f"Error: {str(e)}"}

@function_tool
async def suggest_potential_manipulation(ctx: RunContextWrapper[ContextType], narrative_text: str) -> Dict[str, Any]:
    """
    Analyze narrative text and suggest potential NPC manipulation opportunities.

    Args:
        narrative_text: The narrative text to analyze

    Returns:
        Suggested manipulation opportunities
    """
    context = ctx.context # Use attribute access
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        active_conflicts = await conflict_integration.get_active_conflicts()
        if not active_conflicts: return {"opportunities": [], "reason": "No active conflicts"}

        from logic.conflict_system.conflict_manager import ConflictManager
        conflict_manager = ConflictManager(user_id, conversation_id)
        npcs = await conflict_manager._get_available_npcs() # Assuming this method exists
        mentioned_npcs = [npc for npc in npcs if npc["npc_name"] in narrative_text]
        if not mentioned_npcs: return {"opportunities": [], "reason": "No NPCs mentioned in narrative"}

        opportunities = []
        for conflict in active_conflicts:
            conflict_id = conflict["conflict_id"]
            for npc in mentioned_npcs:
                if npc.get("sex", "female") == "female" and npc.get("dominance", 0) > 60:
                    is_stakeholder = any(s["npc_id"] == npc["npc_id"] for s in conflict.get("stakeholders", []))
                    if is_stakeholder:
                        potential = await conflict_integration.analyze_manipulation_potential(npc["npc_id"]) # Assuming this method exists
                        if potential.get("overall_potential", 0) > 60:
                            opportunities.append({"conflict_id": conflict_id, "conflict_name": conflict["conflict_name"], "npc_id": npc["npc_id"], "npc_name": npc["npc_name"], "dominance": npc["dominance"], "manipulation_type": potential.get("most_effective_type"), "potential": potential.get("overall_potential")})

        return {"opportunities": opportunities, "total_opportunities": len(opportunities)}
    except Exception as e:
        logging.error(f"Error suggesting potential manipulation: {e}")
        return {"opportunities": [], "reason": f"Error: {str(e)}"}

# REMEMBER: get_story_state should NOT be in story_tools if defined elsewhere

# Story state and metadata tools
story_tools = [
    get_key_npcs,
    get_narrative_stages,
    analyze_narrative_and_activity
]

# Conflict management tools
conflict_tools = [
    generate_conflict,
    update_conflict_progress,
    resolve_conflict,
    analyze_narrative_for_conflict,
    set_player_involvement,
    get_conflict_details,
    analyze_conflict_potential,         # Added
    generate_conflict_from_analysis,    # Added
    analyze_npc_manipulation_potential, # Added
    generate_manipulation_attempt,      # Added
    track_conflict_story_beat,          # Added
    suggest_potential_manipulation      # Added
]

# Resource management tools
resource_tools = [
    check_resources,
    commit_resources_to_conflict,
    get_player_resources,
    # analyze_activity_effects, # Keep only one version
    apply_activity_effects,   # Keep only one version (Assuming this is the one you want)
    get_resource_history
]

# Narrative element tools
narrative_tools = [
    generate_personal_revelation,
    generate_dream_sequence,
    check_relationship_events,
    # apply_crossroads_choice_tool, # Moved to relationship_tools
    # check_npc_relationship,       # Moved to relationship_tools
    add_moment_of_clarity,
    get_player_journal_entries
]

# Context management tools
context_tools = [
    get_optimized_context,
    retrieve_relevant_memories,
    store_narrative_memory,
    search_by_vector,
    get_summarized_narrative_context
]

# Relationship tools (Consolidated relationship-specific tools)
relationship_tools = [
    check_relationship_events,      # From narrative_tools
    apply_crossroads_choice_tool,   # From narrative_tools
    check_npc_relationship,         # From narrative_tools
    get_relationship_summary_wrapper,
    update_relationship_dimensions,
    get_player_relationships,
    generate_relationship_evolution
]

# Activity tools
activity_tools = [
    analyze_activity,
    get_filtered_activities,
    # apply_activity_effects, # Already in resource_tools, avoid duplication
    analyze_activity_impact,
    get_all_activities,
    generate_activity_suggestion
]
