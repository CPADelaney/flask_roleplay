# story_agent/tools.py

"""
Organized tools for the Story Director agent.

This module organizes tools into logical categories:
- story_tools: Tools for general story state and progression
- conflict_tools: Tools for managing conflicts
- resource_tools: Tools for resource management
- narrative_tools: Tools for narrative elements
- context_tools: NEW: Tools for context management
"""

import logging
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field

from agents import function_tool, RunContextWrapper

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
from logic.social_links_agentic import (
    get_social_link,
    get_relationship_summary,
    check_for_relationship_crossroads,
    check_for_relationship_ritual,
    apply_crossroads_choice
)

# NEW: Context system imports
from context.context_service import get_context_service, get_comprehensive_context
from context.memory_manager import get_memory_manager
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

logger = logging.getLogger(__name__)

# ----- NEW: Context Tools -----

@function_tool
async def get_optimized_context(ctx, query_text: str = "", use_vector: bool = True) -> Dict[str, Any]:
    """
    Get optimized context using the comprehensive context system.
    
    Args:
        query_text: Optional query text for relevance scoring
        use_vector: Whether to use vector search for relevance
        
    Returns:
        Dictionary with comprehensive context information
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    try:
        # Get context service
        context_service = await get_context_service(user_id, conversation_id)
        
        # Determine token budget and settings
        from context.context_config import get_config
        config = get_config()
        token_budget = config.get_token_budget("default")
        
        # Get context
        context = await context_service.get_context(
            input_text=query_text,
            context_budget=token_budget,
            use_vector_search=use_vector
        )
        
        # Track performance
        if hasattr(ctx.context, 'performance_monitor'):
            perf_monitor = ctx.context.performance_monitor
        else:
            perf_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
            
        # Record token usage
        if "token_usage" in context:
            total_tokens = sum(context["token_usage"].values())
            perf_monitor.record_token_usage(total_tokens)
        
        return context
    except Exception as e:
        logger.error(f"Error getting optimized context: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "user_id": user_id,
            "conversation_id": conversation_id
        }

@function_tool
async def retrieve_relevant_memories(
    ctx, 
    query_text: str, 
    memory_type: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memories using vector search.
    
    Args:
        query_text: Query text for relevance matching
        memory_type: Optional type filter (observation, event, etc.)
        limit: Maximum number of memories to return
        
    Returns:
        List of relevant memories
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    try:
        # Get memory manager
        memory_manager = await get_memory_manager(user_id, conversation_id)
        
        # Search for memories
        memory_types = [memory_type] if memory_type else None
        memories = await memory_manager.search_memories(
            query_text=query_text,
            memory_types=memory_types,
            limit=limit,
            use_vector=True
        )
        
        # Convert to dictionaries
        memory_dicts = []
        for memory in memories:
            if hasattr(memory, 'to_dict'):
                memory_dicts.append(memory.to_dict())
            else:
                memory_dicts.append(memory)
        
        return memory_dicts
    except Exception as e:
        logger.error(f"Error retrieving relevant memories: {str(e)}", exc_info=True)
        return []

@function_tool
async def store_narrative_memory(
    ctx,
    content: str,
    memory_type: str = "observation",
    importance: float = 0.6,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Store a narrative memory in the memory system.
    
    Args:
        content: Content of the memory
        memory_type: Type of memory
        importance: Importance score (0.0-1.0)
        tags: Optional tags for categorization
        
    Returns:
        Stored memory information
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    try:
        # Get memory manager
        memory_manager = await get_memory_manager(user_id, conversation_id)
        
        # Add memory
        memory_id = await memory_manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [memory_type, "story_director"],
            metadata={
                "source": "story_director_tool",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Check if there's a narrative manager for progressive summarization
        if hasattr(ctx.context, 'narrative_manager') and ctx.context.narrative_manager:
            # Add to narrative manager
            await ctx.context.narrative_manager.add_interaction(
                content=content,
                importance=importance,
                tags=tags or [memory_type, "story_director"]
            )
        
        return {
            "memory_id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error storing narrative memory: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }

@function_tool
async def search_by_vector(
    ctx,
    query_text: str,
    entity_types: Optional[List[str]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for entities by semantic similarity using vector search.
    
    Args:
        query_text: Query text for semantic search
        entity_types: Types of entities to search for
        top_k: Maximum number of results to return
        
    Returns:
        List of semantically similar entities
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    try:
        # Get vector service
        vector_service = await get_vector_service(user_id, conversation_id)
        
        # If vector service is not enabled, return empty list
        if not vector_service.enabled:
            return []
        
        # Search for entities
        entity_types = entity_types or ["npc", "location", "memory", "narrative"]
        
        results = await vector_service.search_entities(
            query_text=query_text,
            entity_types=entity_types,
            top_k=top_k,
            hybrid_ranking=True
        )
        
        return results
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}", exc_info=True)
        return []

@function_tool
async def get_summarized_narrative_context(
    ctx,
    query: str,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Get automatically summarized narrative context using progressive summarization.
    
    Args:
        query: Query for relevance matching
        max_tokens: Maximum tokens for context
        
    Returns:
        Summarized narrative context
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    try:
        # Check if narrative manager is available
        if hasattr(ctx.context, 'narrative_manager') and ctx.context.narrative_manager:
            # Use narrative manager's optimized context
            narrative_manager = ctx.context.narrative_manager
        else:
            # Try to import narrative manager
            try:
                from story_agent.progressive_summarization import RPGNarrativeManager
                
                # UPDATED: Get DB connection string
                async with get_db_connection_context() as conn:
                    dsn = conn.dsn
                
                narrative_manager = RPGNarrativeManager(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    db_connection_string=dsn
                )
                await narrative_manager.initialize()
                
                # Store for future use
                ctx.context.narrative_manager = narrative_manager
            except Exception as import_error:
                logger.error(f"Error initializing narrative manager: {import_error}")
                return {
                    "error": "Narrative manager not available",
                    "memories": [],
                    "arcs": []
                }
        
        # Get optimal context
        context = await narrative_manager.get_optimal_narrative_context(
            query=query,
            max_tokens=max_tokens
        )
        
        return context
    except Exception as e:
        logger.error(f"Error getting summarized narrative context: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "memories": [],
            "arcs": []
        }

# ----- Story State Tools -----

@function_tool
@track_performance("get_story_state")  # NEW: Performance tracking
async def get_story_state(ctx) -> Dict[str, Any]:
    """
    Get the current state of the story, including active conflicts, narrative stage, 
    resources, and any pending narrative events.
    
    Returns:
        A dictionary containing the current story state
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    conflict_manager = context.conflict_manager
    resource_manager = context.resource_manager
    
    try:
        # NEW: Get comprehensive context first
        comprehensive_context = None
        
        # Try to use context service if available
        if hasattr(context, 'get_comprehensive_context'):
            try:
                comprehensive_context = await context.get_comprehensive_context()
            except Exception as context_error:
                logger.warning(f"Error getting comprehensive context: {context_error}")
        
        # If we couldn't get comprehensive context, fall back to individual components
        
        # Get current narrative stage
        narrative_stage = await get_current_narrative_stage(user_id, conversation_id)
        stage_info = None
        if narrative_stage:
            stage_info = {
                "name": narrative_stage.name,
                "description": narrative_stage.description
            }
        
        # Get active conflicts
        active_conflicts = await conflict_manager.get_active_conflicts()
        conflict_infos = []
        for conflict in active_conflicts:
            conflict_infos.append({
                "conflict_id": conflict['conflict_id'],
                "conflict_name": conflict['conflict_name'],
                "conflict_type": conflict['conflict_type'],
                "description": conflict['description'],
                "phase": conflict['phase'],
                "progress": conflict['progress'],
                "faction_a_name": conflict['faction_a_name'],
                "faction_b_name": conflict['faction_b_name']
            })
        
        # Get key NPCs
        key_npcs = await get_key_npcs(ctx, limit=5)
        
        # Get player resources and vitals
        resources = await resource_manager.get_resources()
        vitals = await resource_manager.get_vitals()
        
        # Format currency for display
        formatted_money = await resource_manager.get_formatted_money()
        
        resource_status = {
            "money": resources.get('money', 0),
            "supplies": resources.get('supplies', 0),
            "influence": resources.get('influence', 0),
            "energy": vitals.get('energy', 0),
            "hunger": vitals.get('hunger', 0),
            "formatted_money": formatted_money
        }
        
        # Check for narrative events
        narrative_events = []
        
        # Personal revelations
        personal_revelation = await check_for_personal_revelations(user_id, conversation_id)
        if personal_revelation:
            narrative_events.append({
                "event_type": "personal_revelation",
                "content": personal_revelation,
                "should_present": True,
                "priority": 8
            })
        
        # Narrative moments
        narrative_moment = await check_for_narrative_moments(user_id, conversation_id)
        if narrative_moment:
            narrative_events.append({
                "event_type": "narrative_moment",
                "content": narrative_moment,
                "should_present": True,
                "priority": 9
            })
        
        # NPC revelations
        npc_revelation = await check_for_npc_revelations(user_id, conversation_id)
        if npc_revelation:
            narrative_events.append({
                "event_type": "npc_revelation",
                "content": npc_revelation,
                "should_present": True,
                "priority": 7
            })
        
        # Check for relationship events
        crossroads = await check_for_relationship_crossroads(user_id, conversation_id)
        ritual = await check_for_relationship_ritual(user_id, conversation_id)
        
        # NEW: Get relevant memories
        relevant_memories = []
        if hasattr(context, 'get_relevant_memories'):
            try:
                relevant_memories = await context.get_relevant_memories(
                    "current story state overview recent events",
                    limit=3
                )
            except Exception as mem_error:
                logger.warning(f"Error getting relevant memories: {mem_error}")
        
        # Generate key observations based on current state
        key_observations = []
        
        # If at a higher corruption stage, add observation
        if narrative_stage and narrative_stage.name in ["Creeping Realization", "Veil Thinning", "Full Revelation"]:
            key_observations.append(f"Player has progressed to {narrative_stage.name} stage, indicating significant corruption")
        
        # If multiple active conflicts, note this
        if len(conflict_infos) > 2:
            key_observations.append(f"Player is juggling {len(conflict_infos)} active conflicts, which may be overwhelming")
        
        # If any major or catastrophic conflicts, highlight them
        major_conflicts = [c for c in conflict_infos if c["conflict_type"] in ["major", "catastrophic"]]
        if major_conflicts:
            conflict_names = ", ".join([c["conflict_name"] for c in major_conflicts])
            key_observations.append(f"Major conflicts in progress: {conflict_names}")
        
        # If resources are low, note this
        if resource_status["money"] < 30:
            key_observations.append("Player is low on money, which may limit conflict involvement options")
        
        if resource_status["energy"] < 30:
            key_observations.append("Player energy is low, which may affect capability in conflicts")
        
        if resource_status["hunger"] < 30:
            key_observations.append("Player is hungry, which may distract from conflict progress")
        
        # Determine overall story direction
        story_direction = ""
        if narrative_stage:
            if narrative_stage.name == "Innocent Beginning":
                story_direction = "Introduce subtle hints of control dynamics while maintaining a veneer of normalcy"
            elif narrative_stage.name == "First Doubts":
                story_direction = "Create situations that highlight inconsistencies in NPC behavior, raising questions"
            elif narrative_stage.name == "Creeping Realization":
                story_direction = "NPCs should be more open about their manipulative behavior, testing boundaries"
            elif narrative_stage.name == "Veil Thinning":
                story_direction = "Dominant characters should drop pretense more frequently, openly directing the player"
            elif narrative_stage.name == "Full Revelation":
                story_direction = "The true nature of relationships should be explicit, with NPCs acknowledging their control"
        
        # NEW: Create a memory about this state retrieval
        if hasattr(context, 'add_narrative_memory'):
            await context.add_narrative_memory(
                f"Retrieved story state with {len(conflict_infos)} conflicts and narrative stage: {stage_info['name'] if stage_info else 'Unknown'}",
                "story_state_retrieval",
                0.4
            )
        
        # NEW: Track performance if context has performance monitor
        if hasattr(context, 'performance_monitor'):
            context.performance_monitor.record_memory_usage()
        
        return {
            "narrative_stage": stage_info,
            "active_conflicts": conflict_infos,
            "narrative_events": narrative_events,
            "key_npcs": key_npcs,
            "resources": resource_status,
            "key_observations": key_observations,
            "relationship_crossroads": crossroads,
            "relationship_ritual": ritual,
            "story_direction": story_direction,
            "memories": relevant_memories,  # NEW: Include relevant memories
            "last_updated": datetime.now().isoformat(),
            "context_source": "integrated" if comprehensive_context else "direct"
        }
    except Exception as e:
        logger.error(f"Error getting story state: {str(e)}", exc_info=True)
        return {
            "error": f"Failed to get story state: {str(e)}",
            "narrative_stage": None,
            "active_conflicts": [],
            "narrative_events": [],
            "key_npcs": [],
            "resources": {},
            "last_updated": datetime.now().isoformat()
        }

@function_tool
@track_performance("get_key_npcs")
async def get_key_npcs(ctx, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get the key NPCs in the current game state, ordered by importance.
    
    Args:
        limit: Maximum number of NPCs to return
        
    Returns:
        List of NPC information dictionaries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # UPDATED: Use async connection
    async with get_db_connection_context() as conn:
        try:
            # Get NPCs ordered by dominance (a proxy for importance)
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                ORDER BY dominance DESC
                LIMIT $3
            """, user_id, conversation_id, limit)
            
            npcs = []
            for row in rows:
                npc_id, npc_name = row['npc_id'], row['npc_name']
                dominance, cruelty = row['dominance'], row['cruelty']
                closeness, trust, respect = row['closeness'], row['trust'], row['respect']
                
                # Get relationship with player
                relationship = await get_relationship_summary(
                    user_id, conversation_id, 
                    "player", user_id, "npc", npc_id
                )
                
                dynamics = {}
                if relationship and 'dynamics' in relationship:
                    dynamics = relationship['dynamics']
                
                npcs.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust,
                    "respect": respect,
                    "relationship_dynamics": dynamics
                })
            
            return npcs
        except Exception as e:
            logger.error(f"Error fetching key NPCs: {str(e)}", exc_info=True)
            return []

@function_tool
@track_performance("get_narrative_stages")
async def get_narrative_stages(ctx) -> List[Dict[str, str]]:
    """
    Get information about all narrative stages in the game.
    
    Returns:
        List of narrative stages with their descriptions
    """
    stages = []
    for stage in NARRATIVE_STAGES:
        stages.append({
            "name": stage.name,
            "description": stage.description
        })
    return stages

@function_tool
@track_performance("analyze_narrative_and_activity")
async def analyze_narrative_and_activity(
    ctx,
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
        # Start with conflict analysis
        conflict_analysis = await conflict_manager.add_conflict_to_narrative(narrative_text)
        
        results = {
            "conflict_analysis": conflict_analysis,
            "activity_effects": None,
            "relationship_impacts": [],
            "resource_changes": {},
            "conflict_progression": []
        }
        
        # If player activity is provided, analyze it
        if player_activity:
            activity_analyzer = context.activity_analyzer
            activity_effects = await activity_analyzer.analyze_activity(
                player_activity, apply_effects=False
            )
            results["activity_effects"] = activity_effects
            
            # Check if this activity might progress any conflicts
            active_conflicts = await conflict_manager.get_active_conflicts()
            
            for conflict in active_conflicts:
                # Simple relevance check - see if keywords from conflict appear in activity
                conflict_keywords = [
                    conflict['conflict_name'],
                    conflict['faction_a_name'],
                    conflict['faction_b_name']
                ]
                
                relevant = any(keyword.lower() in player_activity.lower() for keyword in conflict_keywords if keyword)
                
                if relevant:
                    # Determine an appropriate progress increment
                    progress_increment = 5  # Default increment
                    
                    if "actively" in player_activity.lower() or "directly" in player_activity.lower():
                        progress_increment = 10
                    
                    if conflict['conflict_type'] == "major":
                        progress_increment = progress_increment * 0.5  # Major conflicts progress slower
                    elif conflict['conflict_type'] == "minor":
                        progress_increment = progress_increment * 1.5  # Minor conflicts progress faster
                    
                    # Add to results
                    results["conflict_progression"].append({
                        "conflict_id": conflict['conflict_id'],
                        "conflict_name": conflict['conflict_name'],
                        "is_relevant": True,
                        "suggested_progress_increment": progress_increment
                    })
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing narrative and activity: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "conflict_analysis": {"conflict_generated": False},
            "activity_effects": None,
            "relationship_impacts": [],
            "resource_changes": {},
            "conflict_progression": []
        }

# ----- Conflict Tools -----

@function_tool
@track_performance("generate_conflict")
async def generate_conflict(ctx, conflict_type: Optional[str] = None) -> Dict[str, Any]:
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
        
        # NEW: Store this as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            await context.add_narrative_memory(
                f"Generated new {conflict['conflict_type']} conflict: {conflict['conflict_name']}",
                "conflict_generation",
                0.7
            )
        
        return {
            "conflict_id": conflict['conflict_id'],
            "conflict_name": conflict['conflict_name'],
            "conflict_type": conflict['conflict_type'],
            "description": conflict['description'],
            "success": True,
            "message": "Conflict generated successfully"
        }
    except Exception as e:
        logger.error(f"Error generating conflict: {str(e)}", exc_info=True)
        return {
            "conflict_id": 0,
            "conflict_name": "",
            "conflict_type": conflict_type or "unknown",
            "description": "",
            "success": False,
            "message": f"Failed to generate conflict: {str(e)}"
        }

@function_tool
@track_performance("update_conflict_progress")
async def update_conflict_progress(
    ctx, 
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
        # Get current conflict info
        old_conflict = await conflict_manager.get_conflict(conflict_id)
        old_phase = old_conflict['phase']
        
        # Update progress
        updated_conflict = await conflict_manager.update_conflict_progress(conflict_id, progress_increment)
        
        # NEW: Store this update as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            memory_importance = 0.5  # Default importance
            
            # Increase importance if phase changed
            if updated_conflict['phase'] != old_phase:
                memory_importance = 0.7
                
            memory_content = (
                f"Updated conflict {updated_conflict['conflict_name']} progress by {progress_increment} points "
                f"to {updated_conflict['progress']}%. "
            )
            
            if updated_conflict['phase'] != old_phase:
                memory_content += f"Phase advanced from {old_phase} to {updated_conflict['phase']}."
                
            await context.add_narrative_memory(
                memory_content,
                "conflict_progression",
                memory_importance
            )
        
        return {
            "conflict_id": conflict_id,
            "new_progress": updated_conflict['progress'],
            "new_phase": updated_conflict['phase'],
            "phase_changed": updated_conflict['phase'] != old_phase,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error updating conflict progress: {str(e)}", exc_info=True)
        return {
            "conflict_id": conflict_id,
            "new_progress": 0,
            "new_phase": "unknown",
            "phase_changed": False,
            "success": False,
            "error": str(e)
        }

@function_tool
@track_performance("resolve_conflict")
async def resolve_conflict(ctx, conflict_id: int) -> Dict[str, Any]:
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
        
        consequences = []
        for consequence in result.get('consequences', []):
            consequences.append(consequence.get('description', ''))
        
        # NEW: Store this resolution as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            memory_content = (
                f"Resolved conflict {result.get('conflict_name', f'ID: {conflict_id}')} "
                f"with outcome: {result.get('outcome', 'unknown')}. "
                f"Consequences: {'; '.join(consequences)}"
            )
            
            await context.add_narrative_memory(
                memory_content,
                "conflict_resolution",
                0.8  # High importance for conflict resolutions
            )
        
        return {
            "conflict_id": conflict_id,
            "outcome": result.get('outcome', 'unknown'),
            "consequences": consequences,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error resolving conflict: {str(e)}", exc_info=True)
        return {
            "conflict_id": conflict_id,
            "outcome": "error",
            "consequences": [f"Error: {str(e)}"],
            "success": False
        }

@function_tool
@track_performance("analyze_narrative_for_conflict")
async def analyze_narrative_for_conflict(ctx, narrative_text: str) -> Dict[str, Any]:
    """
    Analyze a narrative text to see if it should trigger a conflict.
    
    Args:
        narrative_text: The narrative text to analyze
        
    Returns:
        Analysis results and possibly a new conflict
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        result = await conflict_manager.add_conflict_to_narrative(narrative_text)
        
        # NEW: Store this analysis as a memory if possible
        if hasattr(context, 'add_narrative_memory') and result.get("conflict_generated", False):
            conflict_info = result.get("conflict", {})
            
            memory_content = (
                f"Analysis detected conflict in narrative and generated new "
                f"{conflict_info.get('conflict_type', 'unknown')} conflict: "
                f"{conflict_info.get('conflict_name', 'Unnamed conflict')}"
            )
            
            await context.add_narrative_memory(
                memory_content,
                "conflict_analysis",
                0.6
            )
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing narrative for conflict: {str(e)}", exc_info=True)
        return {
            "analysis": {
                "conflict_intensity": 0,
                "matched_keywords": []
            },
            "conflict_generated": False,
            "error": str(e)
        }

@function_tool
@track_performance("set_player_involvement")
async def set_player_involvement(
    ctx, 
    conflict_id: int, 
    involvement_level: str,
    faction: str = "neutral",
    money_committed: int = 0,
    supplies_committed: int = 0,
    influence_committed: int = 0,
    action: Optional[str] = None
) -> Dict[str, Any]:
    """
    Set the player's involvement in a conflict.
    
    Args:
        conflict_id: ID of the conflict
        involvement_level: Level of involvement (none, observing, participating, leading)
        faction: Which faction to support (a, b, neutral)
        money_committed: Money committed to the conflict
        supplies_committed: Supplies committed to the conflict
        influence_committed: Influence committed to the conflict
        action: Optional specific action taken
        
    Returns:
        Updated conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        # First check if player has sufficient resources
        resource_manager = context.resource_manager
        resource_check = await resource_manager.check_resources(
            money_committed, supplies_committed, influence_committed
        )
        
        if not resource_check['has_resources']:
            return {
                "error": "Insufficient resources to commit",
                "missing": resource_check.get('missing', {}),
                "current": resource_check.get('current', {}),
                "success": False
            }
        
        # Get conflict info for memory
        conflict_info = await conflict_manager.get_conflict(conflict_id)
        
        result = await conflict_manager.set_player_involvement(
            conflict_id, involvement_level, faction,
            money_committed, supplies_committed, influence_committed, action
        )
        
        # NEW: Store this involvement as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if money_committed > 0:
                resources_text.append(f"{money_committed} money")
            if supplies_committed > 0:
                resources_text.append(f"{supplies_committed} supplies")
            if influence_committed > 0:
                resources_text.append(f"{influence_committed} influence")
                
            resources_committed = ", ".join(resources_text) if resources_text else "no resources"
            
            memory_content = (
                f"Player set involvement in conflict {conflict_info.get('conflict_name', f'ID: {conflict_id}')} "
                f"to {involvement_level}, supporting {faction} faction with {resources_committed}."
            )
            
            if action:
                memory_content += f" Action taken: {action}"
                
            await context.add_narrative_memory(
                memory_content,
                "conflict_involvement",
                0.7
            )
        
        # Add success flag
        if isinstance(result, dict):
            result["success"] = True
        else:
            # If the result is not a dictionary, create a new one
            result = {
                "conflict_id": conflict_id,
                "involvement_level": involvement_level,
                "faction": faction,
                "resources_committed": {
                    "money": money_committed,
                    "supplies": supplies_committed,
                    "influence": influence_committed
                },
                "action": action,
                "success": True
            }
        
        return result
    except Exception as e:
        logger.error(f"Error setting involvement: {str(e)}", exc_info=True)
        return {
            "conflict_id": conflict_id,
            "error": str(e),
            "success": False
        }

@function_tool
@track_performance("get_conflict_details")
async def get_conflict_details(ctx, conflict_id: int) -> Dict[str, Any]:
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
        
        # Get additional information about involved NPCs
        involved_npcs = await conflict_manager.get_conflict_npcs(conflict_id)
        
        # Get player involvement
        player_involvement = await conflict_manager.get_player_involvement(conflict_id)
        
        # Create a comprehensive response
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
        return {
            "conflict_id": conflict_id,
            "error": f"Failed to get conflict details: {str(e)}",
            "success": False
        }

# ----- Resource Tools -----

@function_tool
@track_performance("check_resources")
async def check_resources(ctx, money: int = 0, supplies: int = 0, influence: int = 0) -> Dict[str, Any]:
    """
    Check if player has sufficient resources.
    
    Args:
        money: Required amount of money
        supplies: Required amount of supplies
        influence: Required amount of influence
        
    Returns:
        Dictionary with resource check results
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    try:
        result = await resource_manager.check_resources(money, supplies, influence)
        # Add formatted money
        if result.get('current', {}).get('money') is not None:
            formatted_money = await resource_manager.get_formatted_money(result['current']['money'])
            result['current']['formatted_money'] = formatted_money
        
        return result
    except Exception as e:
        logger.error(f"Error checking resources: {str(e)}", exc_info=True)
        return {
            "has_resources": False,
            "error": str(e),
            "current": {}
        }

@function_tool
@track_performance("commit_resources_to_conflict")
async def commit_resources_to_conflict(
    ctx, 
    conflict_id: int, 
    money: int = 0,
    supplies: int = 0,
    influence: int = 0
) -> Dict[str, Any]:
    """
    Commit player resources to a conflict.
    
    Args:
        conflict_id: ID of the conflict
        money: Amount of money to commit
        supplies: Amount of supplies to commit
        influence: Amount of influence to commit
        
    Returns:
        Result of committing resources
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    try:
        # Get conflict info for memory
        conflict_info = None
        if hasattr(context, 'conflict_manager'):
            try:
                conflict_info = await context.conflict_manager.get_conflict(conflict_id)
            except Exception as conflict_error:
                logger.warning(f"Could not get conflict info: {conflict_error}")
        
        result = await resource_manager.commit_resources_to_conflict(
            conflict_id, money, supplies, influence
        )
        
        # Add formatted money if money was committed
        if money > 0 and result.get('money_result'):
            money_result = result['money_result']
            if 'old_value' in money_result and 'new_value' in money_result:
                old_formatted = await resource_manager.get_formatted_money(money_result['old_value'])
                new_formatted = await resource_manager.get_formatted_money(money_result['new_value'])
                money_result['formatted_old_value'] = old_formatted
                money_result['formatted_new_value'] = new_formatted
                money_result['formatted_change'] = await resource_manager.get_formatted_money(money_result['change'])
                result['money_result'] = money_result
        
        # NEW: Store this commitment as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if money > 0:
                resources_text.append(f"{money} money")
            if supplies > 0:
                resources_text.append(f"{supplies} supplies")
            if influence > 0:
                resources_text.append(f"{influence} influence")
                
            resources_committed = ", ".join(resources_text)
            
            conflict_name = conflict_info.get('conflict_name', f"ID: {conflict_id}") if conflict_info else f"ID: {conflict_id}"
            
            memory_content = (
                f"Committed {resources_committed} to conflict {conflict_name}"
            )
                
            await context.add_narrative_memory(
                memory_content,
                "resource_commitment",
                0.6
            )
        
        return result
    except Exception as e:
        logger.error(f"Error committing resources: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@function_tool
@track_performance("get_player_resources")
async def get_player_resources(ctx) -> Dict[str, Any]:
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
        
        # Get formatted money
        formatted_money = await resource_manager.get_formatted_money()
        
        return {
            "money": resources.get('money', 0),
            "supplies": resources.get('supplies', 0),
            "influence": resources.get('influence', 0),
            "energy": vitals.get('energy', 0),
            "hunger": vitals.get('hunger', 0),
            "formatted_money": formatted_money,
            "updated_at": resources.get('updated_at', datetime.now()).isoformat() if isinstance(resources.get('updated_at'), datetime) else str(resources.get('updated_at'))
        }
    except Exception as e:
        logger.error(f"Error getting player resources: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "money": 0,
            "supplies": 0,
            "influence": 0,
            "energy": 0,
            "hunger": 0,
            "formatted_money": "0"
        }

@function_tool
@track_performance("analyze_activity_effects")
async def analyze_activity_effects(ctx, activity_text: str) -> Dict[str, Any]:
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
        # Don't apply effects, just analyze them
        result = await activity_analyzer.analyze_activity(activity_text, apply_effects=False)
        
        effects = result.get('effects', {})
        
        # NEW: Store this analysis as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            effects_description = []
            
            for resource_type, value in effects.items():
                if value:
                    direction = "increased" if value > 0 else "decreased"
                    effects_description.append(f"{resource_type} {direction} by {abs(value)}")
            
            effects_text = ", ".join(effects_description) if effects_description else "no significant effects"
            
            memory_content = (
                f"Analyzed activity: {activity_text[:100]}... with {effects_text}"
            )
                
            await context.add_narrative_memory(
                memory_content,
                "activity_analysis",
                0.4  # Lower importance for analysis only
            )
        
        return {
            "activity_type": result.get('activity_type', 'unknown'),
            "activity_details": result.get('activity_details', ''),
            "hunger_effect": effects.get('hunger'),
            "energy_effect": effects.get('energy'),
            "money_effect": effects.get('money'),
            "supplies_effect": effects.get('supplies'),
            "influence_effect": effects.get('influence'),
            "description": result.get('description', f"Effects of {activity_text}")
        }
    except Exception as e:
        logger.error(f"Error analyzing activity effects: {str(e)}", exc_info=True)
        return {
            "activity_type": "unknown",
            "activity_details": "",
            "description": f"Failed to analyze: {str(e)}",
            "error": str(e)
        }

@function_tool
@track_performance("apply_activity_effects")
async def apply_activity_effects(ctx, activity_text: str) -> Dict[str, Any]:
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
        # Apply the effects
        result = await activity_analyzer.analyze_activity(activity_text, apply_effects=True)
        
        # Add formatted money if money was affected
        if 'effects' in result and 'money' in result['effects']:
            resource_manager = context.resource_manager
            resources = await resource_manager.get_resources()
            result['formatted_money'] = await resource_manager.get_formatted_money(resources.get('money', 0))
        
        # NEW: Store this application as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            effects = result.get('effects', {})
            effects_description = []
            
            for resource_type, value in effects.items():
                if value:
                    direction = "increased" if value > 0 else "decreased"
                    effects_description.append(f"{resource_type} {direction} by {abs(value)}")
            
            effects_text = ", ".join(effects_description) if effects_description else "no significant effects"
            
            memory_content = (
                f"Applied activity effects for: {activity_text[:100]}... with {effects_text}"
            )
                
            await context.add_narrative_memory(
                memory_content,
                "activity_application",
                0.5
            )
        
        return result
    except Exception as e:
        logger.error(f"Error applying activity effects: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "activity_type": "unknown",
            "activity_details": "",
            "effects": {}
        }

@function_tool
@track_performance("get_resource_history")
async def get_resource_history(ctx, resource_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get the history of resource changes.
    
    Args:
        resource_type: Optional filter for specific resource type
                      (money, supplies, influence, energy, hunger)
        limit: Maximum number of history entries to return
        
    Returns:
        List of resource change history entries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # UPDATED: Use async connection
    async with get_db_connection_context() as conn:
        try:
            if resource_type:
                rows = await conn.fetch("""
                    SELECT resource_type, old_value, new_value, amount_changed, 
                           source, description, timestamp
                    FROM ResourceHistoryLog
                    WHERE user_id=$1 AND conversation_id=$2 AND resource_type=$3
                    ORDER BY timestamp DESC
                    LIMIT $4
                """, user_id, conversation_id, resource_type, limit)
            else:
                rows = await conn.fetch("""
                    SELECT resource_type, old_value, new_value, amount_changed, 
                           source, description, timestamp
                    FROM ResourceHistoryLog
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """, user_id, conversation_id, limit)
            
            history = []
            for row in rows:
                resource_type = row['resource_type']
                old_value = row['old_value']
                new_value = row['new_value']
                amount_changed = row['amount_changed']
                source = row['source']
                description = row['description']
                timestamp = row['timestamp']
                
                # Format money values if resource_type is money
                formatted_old = None
                formatted_new = None
                formatted_change = None
                
                if resource_type == "money":
                    resource_manager = context.resource_manager
                    formatted_old = await resource_manager.get_formatted_money(old_value)
                    formatted_new = await resource_manager.get_formatted_money(new_value)
                    formatted_change = await resource_manager.get_formatted_money(amount_changed)
                
                history.append({
                    "resource_type": resource_type,
                    "old_value": old_value,
                    "new_value": new_value,
                    "amount_changed": amount_changed,
                    "formatted_old_value": formatted_old,
                    "formatted_new_value": formatted_new,
                    "formatted_change": formatted_change,
                    "source": source,
                    "description": description,
                    "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
                })
            
            return history
        except Exception as e:
            logger.error(f"Error getting resource history: {str(e)}", exc_info=True)
            return []

# ----- Narrative Tools -----

@function_tool
@track_performance("generate_personal_revelation")
async def generate_personal_revelation(ctx, npc_name: str, revelation_type: str) -> Dict[str, Any]:
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
            "I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
            "I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
            "The thought of spending a day without talking to {npc_name} makes me anxious. I should be concerned about that, shouldn't I?"
        ],
        "obedience": [
            "I caught myself automatically rearranging my schedule when {npc_name} hinted she wanted to see me. I didn't even think twice about it.",
            "Today I changed my opinion the moment I realized it differed from {npc_name}'s. That's... not like me. Or is it becoming like me?",
            "{npc_name} gave me that look, and I immediately stopped what I was saying. When did her disapproval start carrying so much weight?"
        ],
        "corruption": [
            "I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
            "Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
            "Sometimes I catch glimpses of my old self, like a stranger I used to know. When did I change so fundamentally?"
        ],
        "willpower": [
            "I had every intention of saying no to {npc_name} today. The 'yes' came out before I even realized I was speaking.",
            "I've been trying to remember what it felt like to disagree with {npc_name}. The memory feels distant, like it belongs to someone else.",
            "I made a list of boundaries I wouldn't cross. Looking at it now, I've broken every single one at {npc_name}'s suggestion."
        ],
        "confidence": [
            "I opened my mouth to speak in the meeting, then saw {npc_name} watching me. I suddenly couldn't remember what I was going to say.",
            "I used to trust my judgment. Now I find myself second-guessing every thought that {npc_name} hasn't explicitly approved.",
            "When did I start feeling this small? This uncertain? I can barely remember how it felt to be sure of myself."
        ]
    }
    
    try:
        # Default to dependency if type not found
        revelation_templates = templates.get(revelation_type.lower(), templates["dependency"])
        
        # Select a random template and format it
        inner_monologue = random.choice(revelation_templates).format(npc_name=npc_name)
        
        # Add to PlayerJournal
        # UPDATED: Use async connection
        async with get_db_connection_context() as conn:
            try:
                # Use asyncpg's execute() and fetchrow() methods
                journal_id = await conn.fetchval("""
                    INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, revelation_types, timestamp)
                    VALUES ($1, $2, 'personal_revelation', $3, $4, CURRENT_TIMESTAMP)
                    RETURNING id
                """, user_id, conversation_id, inner_monologue, revelation_type)
                
                # NEW: Store this as a memory if possible
                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(
                        f"Personal revelation about {npc_name}: {inner_monologue}",
                        "personal_revelation",
                        0.8,  # High importance for revelations
                        tags=[revelation_type, "revelation", npc_name.lower().replace(" ", "_")]
                    )
                
                # NEW: Check for narrative manager
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
                    # Add to narrative manager
                    await context.narrative_manager.add_revelation(
                        content=inner_monologue,
                        revelation_type=revelation_type,
                        importance=0.8,
                        tags=[revelation_type, "revelation"]
                    )
                
                return {
                    "type": "personal_revelation",
                    "name": f"{revelation_type.capitalize()} Awareness",
                    "inner_monologue": inner_monologue,
                    "journal_id": journal_id,
                    "success": True
                }
            except Exception as db_error:
                logger.error(f"Database error recording personal revelation: {db_error}")
                raise
    except Exception as e:
        logger.error(f"Error generating personal revelation: {str(e)}", exc_info=True)
        return {
            "type": "personal_revelation",
            "name": f"{revelation_type.capitalize()} Awareness",
            "inner_monologue": f"Error generating revelation: {str(e)}",
            "success": False
        }

@function_tool
@track_performance("generate_dream_sequence")
async def generate_dream_sequence(ctx, npc_names: List[str]) -> Dict[str, Any]:
    """
    Generate a symbolic dream sequence based on player's current state.
    
    Args:
        npc_names: List of NPC names to include in the dream
        
    Returns:
        A dream sequence
    """
    # Ensure we have at least 3 NPC names
    while len(npc_names) < 3:
        npc_names.append(f"Unknown Woman {len(npc_names) + 1}")
    
    npc1, npc2, npc3 = npc_names[:3]
    
    # Dream templates
    dream_templates = [
        "You're sitting in a chair as {npc1} circles you slowly. \"Show me your hands,\" she says. "
        "You extend them, surprised to find intricate strings wrapped around each finger, extending upward. "
        "\"Do you see who's holding them?\" she asks. You look up, but the ceiling is mirrored, "
        "showing only your own face looking back down at you, smiling with an expression that isn't yours.",
        
        "You're searching your home frantically, calling {npc1}'s name. The rooms shift and expand, "
        "doorways leading to impossible spaces. Your phone rings. It's {npc1}. \"Where are you?\" you ask desperately. "
        "\"I'm right here,\" she says, her voice coming both from the phone and from behind you. "
        "\"I've always been right here. You're the one who's lost.\"",
        
        "You're trying to walk away from {npc1}, but your feet sink deeper into the floor with each step. "
        "\"I don't understand why you're struggling,\" she says, not moving yet somehow keeping pace beside you. "
        "\"You stopped walking on your own long ago.\" You look down to find your legs have merged with the floor entirely, "
        "indistinguishable from the material beneath.",
        
        "You're giving a presentation to a room full of people, but every time you speak, your voice comes out as {npc1}'s voice, "
        "saying words you didn't intend. The audience nods approvingly. \"Much better,\" whispers {npc2} from beside you. "
        "\"Your ideas were never as good as hers anyway.\"",
        
        "You're walking through an unfamiliar house, opening doors that should lead outside but only reveal more rooms. "
        "In each room, {npc1} is engaged in a different activity, wearing a different expression. In the final room, "
        "all versions of her turn to look at you simultaneously. \"Which one is real?\" they ask in unison. \"The one you needed, or the one who needed you?\"",
        
        "You're swimming in deep water. Below you, {npc1} and {npc2} walk along the bottom, "
        "looking up at you and conversing, their voices perfectly clear despite the water. "
        "\"They still think they're above it all,\" says {npc1}, and they both laugh. You realize you can't remember how to reach the surface."
    ]
    
    try:
        # Select a random dream template
        dream_text = random.choice(dream_templates).format(npc1=npc1, npc2=npc2, npc3=npc3)
        
        # Add to PlayerJournal
        context = ctx.context
        user_id = context.user_id
        conversation_id = context.conversation_id
        
        # UPDATED: Use async connection
        async with get_db_connection_context() as conn:
            try:
                # Use asyncpg's fetchval method
                journal_id = await conn.fetchval("""
                    INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                    VALUES ($1, $2, 'dream_sequence', $3, CURRENT_TIMESTAMP)
                    RETURNING id
                """, user_id, conversation_id, dream_text)
                
                # NEW: Store this as a memory if possible
                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(
                        f"Dream sequence: {dream_text}",
                        "dream_sequence",
                        0.7,  # High importance for dreams
                        tags=["dream", "symbolic"] + [npc.lower().replace(" ", "_") for npc in npc_names[:3]]
                    )
                
                # NEW: Check for narrative manager
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
                    # Add to narrative manager
                    await context.narrative_manager.add_dream_sequence(
                        content=dream_text,
                        symbols=[npc1, npc2, npc3, "control", "manipulation"],
                        importance=0.7,
                        tags=["dream", "symbolic"]
                    )
                
                return {
                    "type": "dream_sequence",
                    "text": dream_text,
                    "journal_id": journal_id,
                    "success": True
                }
            except Exception as db_error:
                logger.error(f"Database error recording dream sequence: {db_error}")
                raise
    except Exception as e:
        logger.error(f"Error generating dream sequence: {str(e)}", exc_info=True)
        return {
            "type": "dream_sequence",
            "text": f"Error generating dream: {str(e)}",
            "success": False
        }

@function_tool
@track_performance("check_relationship_events")
async def check_relationship_events(ctx) -> Dict[str, Any]:
    """
    Check for relationship events like crossroads or rituals.
    
    Returns:
        Dictionary with any triggered relationship events
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Check for crossroads
        crossroads = await check_for_relationship_crossroads(user_id, conversation_id)
        
        # Check for rituals
        ritual = await check_for_relationship_ritual(user_id, conversation_id)
        
        # NEW: Store this as a memory if crossroads or ritual found
        if (crossroads or ritual) and hasattr(context, 'add_narrative_memory'):
            event_type = "crossroads" if crossroads else "ritual"
            npc_name = "Unknown"
            
            if crossroads:
                npc_name = crossroads.get("npc_name", "Unknown")
            elif ritual:
                npc_name = ritual.get("npc_name", "Unknown")
            
            memory_content = f"Relationship {event_type} detected with {npc_name}"
            
            await context.add_narrative_memory(
                memory_content,
                f"relationship_{event_type}",
                0.8,  # High importance for relationship events
                tags=[event_type, "relationship", npc_name.lower().replace(" ", "_")]
            )
        
        return {
            "crossroads": crossroads,
            "ritual": ritual,
            "has_events": crossroads is not None or ritual is not None
        }
    except Exception as e:
        logger.error(f"Error checking relationship events: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "crossroads": None,
            "ritual": None,
            "has_events": False
        }

@function_tool
@track_performance("apply_crossroads_choice")
async def apply_crossroads_choice(
    ctx,
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
        result = await apply_crossroads_choice(
            user_id, conversation_id, link_id, crossroads_name, choice_index
        )
        
        # NEW: Store this as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            # Get NPC from link ID if possible
            npc_name = "Unknown"
            try:
                # UPDATED: Use async connection
                async with get_db_connection_context() as conn:
                    # Use asyncpg's fetchrow method
                    row = await conn.fetchrow("""
                        SELECT entity2_id
                        FROM SocialLinks
                        WHERE link_id = $1 AND entity2_type = 'npc'
                    """, link_id)
                    
                    if row:
                        npc_id = row['entity2_id']
                        
                        # Fetch the NPC name
                        npc_row = await conn.fetchrow("""
                            SELECT npc_name
                            FROM NPCStats
                            WHERE npc_id = $1
                        """, npc_id)
                        
                        if npc_row:
                            npc_name = npc_row['npc_name']
            except Exception as db_error:
                logger.warning(f"Could not get NPC name for memory: {db_error}")
            
            memory_content = (
                f"Applied crossroads choice {choice_index} for '{crossroads_name}' "
                f"with {npc_name}"
            )
                
            await context.add_narrative_memory(
                memory_content,
                "crossroads_choice",
                0.8,  # High importance for relationship choices
                tags=["crossroads", "relationship", npc_name.lower().replace(" ", "_")]
            )
            
            # NEW: Check for narrative manager
            if hasattr(context, 'narrative_manager') and context.narrative_manager:
                # Add to narrative manager
                await context.narrative_manager.add_interaction(
                    content=memory_content,
                    npc_name=npc_name,
                    importance=0.8,
                    tags=["crossroads", "relationship_choice"]
                )
        
        return result
    except Exception as e:
        logger.error(f"Error applying crossroads choice: {str(e)}", exc_info=True)
        return {
            "link_id": link_id,
            "crossroads_name": crossroads_name,
            "choice_index": choice_index,
            "success": False,
            "error": str(e)
        }

@function_tool
@track_performance("check_npc_relationship")
async def check_npc_relationship(
    ctx, 
    npc_id: int
) -> Dict[str, Any]:
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
        relationship = await get_relationship_summary(
            user_id, conversation_id, 
            "player", user_id, "npc", npc_id
        )
        
        if not relationship:
            # If no relationship exists, create a basic one
            try:
                from logic.social_links_agentic import create_social_link
                link_id = await create_social_link(
                    user_id, conversation_id,
                    "player", user_id, "npc", npc_id
                )
                
                # Fetch again
                relationship = await get_relationship_summary(
                    user_id, conversation_id, 
                    "player", user_id, "npc", npc_id
                )
            except Exception as link_error:
                logger.error(f"Error creating social link: {link_error}")
                return {
                    "error": f"Failed to create relationship: {str(link_error)}",
                    "npc_id": npc_id
                }
        
        return relationship or {
            "error": "Could not get or create relationship",
            "npc_id": npc_id
        }
    except Exception as e:
        logger.error(f"Error checking NPC relationship: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "npc_id": npc_id
        }

@function_tool
@track_performance("add_moment_of_clarity")
async def add_moment_of_clarity(ctx, realization_text: str) -> Dict[str, Any]:
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
        from logic.narrative_progression import add_moment_of_clarity as add_clarity
        
        result = await add_clarity(user_id, conversation_id, realization_text)
        
        # NEW: Store this as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            await context.add_narrative_memory(
                f"Moment of clarity: {realization_text}",
                "moment_of_clarity",
                0.9,  # Very high importance for moments of clarity
                tags=["clarity", "realization", "awareness"]
            )
            
            # NEW: Check for narrative manager
            if hasattr(context, 'narrative_manager') and context.narrative_manager:
                # Add to narrative manager
                await context.narrative_manager.add_revelation(
                    content=realization_text,
                    revelation_type="clarity",
                    importance=0.9,
                    tags=["clarity", "realization"]
                )
        
        return {
            "type": "moment_of_clarity",
            "content": result,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error adding moment of clarity: {str(e)}", exc_info=True)
        return {
            "type": "moment_of_clarity",
            "content": None,
            "success": False,
            "error": str(e)
        }

@function_tool
@track_performance("get_player_journal_entries")
async def get_player_journal_entries(ctx, entry_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get entries from the player's journal.
    
    Args:
        entry_type: Optional filter for entry type 
                   (personal_revelation, dream_sequence, moment_of_clarity, etc.)
        limit: Maximum number of entries to return
        
    Returns:
        List of journal entries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # UPDATED: Use async connection
    async with get_db_connection_context() as conn:
        try:
            if entry_type:
                rows = await conn.fetch("""
                    SELECT id, entry_type, entry_text, revelation_types, 
                           narrative_moment, fantasy_flag, intensity_level, timestamp
                    FROM PlayerJournal
                    WHERE user_id=$1 AND conversation_id=$2 AND entry_type=$3
                    ORDER BY timestamp DESC
                    LIMIT $4
                """, user_id, conversation_id, entry_type, limit)
            else:
                rows = await conn.fetch("""
                    SELECT id, entry_type, entry_text, revelation_types, 
                           narrative_moment, fantasy_flag, intensity_level, timestamp
                    FROM PlayerJournal
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """, user_id, conversation_id, limit)
            
            entries = []
            for row in rows:
                id = row['id']
                entry_type = row['entry_type']
                entry_text = row['entry_text']
                revelation_types = row['revelation_types']
                narrative_moment = row['narrative_moment']
                fantasy_flag = row['fantasy_flag']
                intensity_level = row['intensity_level']
                timestamp = row['timestamp']
                
                entries.append({
                    "id": id,
                    "entry_type": entry_type,
                    "entry_text": entry_text,
                    "revelation_types": revelation_types,
                    "narrative_moment": narrative_moment,
                    "fantasy_flag": fantasy_flag,
                    "intensity_level": intensity_level,
                    "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
                })
            
            return entries
        except Exception as e:
            logger.error(f"Error getting player journal entries: {str(e)}", exc_info=True)
            return []

@function_tool
async def analyze_conflict_potential(ctx, narrative_text: str) -> Dict[str, Any]:
    """
    Analyze narrative text for conflict potential.
    
    Args:
        narrative_text: The narrative text to analyze
        
    Returns:
        Conflict potential analysis
    """
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
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
        conflict_type = "minor"
        if conflict_intensity >= 8:
            conflict_type = "major"
        elif conflict_intensity >= 5:
            conflict_type = "standard"
        
        # Check for potential internal faction conflict
        internal_faction_conflict = None
        if len(mentioned_npcs) >= 2 and len(mentioned_factions) > 0:
            for faction in mentioned_factions:
                faction_npcs = [npc for npc in mentioned_npcs 
                               if any(aff.get("faction_id") == faction["faction_id"] 
                                     for aff in npc.get("faction_affiliations", []))]
                
                if len(faction_npcs) >= 2:
                    # Check for potential challenger and target
                    faction_npcs.sort(key=lambda x: x["dominance"], reverse=True)
                    internal_faction_conflict = {
                        "faction_id": faction["faction_id"],
                        "faction_name": faction["faction_name"],
                        "challenger_npc_id": faction_npcs[1]["npc_id"],
                        "challenger_npc_name": faction_npcs[1]["npc_name"],
                        "target_npc_id": faction_npcs[0]["npc_id"],
                        "target_npc_name": faction_npcs[0]["npc_name"],
                        "prize": "leadership",
                        "approach": "subtle"
                    }
        
        return {
            "conflict_intensity": conflict_intensity,
            "matched_keywords": matched_keywords,
            "mentioned_npcs": mentioned_npcs,
            "mentioned_factions": mentioned_factions,
            "npc_relationships": npc_relationships,
            "recommended_conflict_type": conflict_type,
            "potential_internal_faction_conflict": internal_faction_conflict,
            "has_conflict_potential": conflict_intensity >= 4
        }
    except Exception as e:
        logging.error(f"Error analyzing conflict potential: {e}")
        return {
            "conflict_intensity": 0,
            "matched_keywords": [],
            "mentioned_npcs": [],
            "mentioned_factions": [],
            "npc_relationships": [],
            "recommended_conflict_type": "minor",
            "potential_internal_faction_conflict": None,
            "has_conflict_potential": False,
            "error": str(e)
        }

@function_tool
async def generate_conflict_from_analysis(ctx, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a conflict based on analysis.
    
    Args:
        analysis: Conflict potential analysis from analyze_conflict_potential
        
    Returns:
        Generated conflict details
    """
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
    try:
        if not analysis.get("has_conflict_potential", False):
            return {
                "generated": False,
                "reason": "Insufficient conflict potential",
                "analysis": analysis
            }
        
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        # Generate the conflict
        conflict_type = analysis.get("recommended_conflict_type", "standard")
        conflict = await conflict_integration.generate_new_conflict(conflict_type)
        
        # If there's potential for internal faction conflict, generate it
        internal_faction_conflict = None
        if analysis.get("potential_internal_faction_conflict") and conflict.get("conflict_id"):
            internal_data = analysis["potential_internal_faction_conflict"]
            try:
                internal_faction_conflict = await conflict_integration.initiate_faction_power_struggle(
                    conflict["conflict_id"],
                    internal_data["faction_id"],
                    internal_data["challenger_npc_id"],
                    internal_data["target_npc_id"],
                    internal_data["prize"],
                    internal_data["approach"],
                    False  # Not public by default
                )
            except Exception as e:
                logging.error(f"Error generating internal faction conflict: {e}")
        
        return {
            "generated": True,
            "conflict": conflict,
            "internal_faction_conflict": internal_faction_conflict
        }
    except Exception as e:
        logging.error(f"Error generating conflict from analysis: {e}")
        return {
            "generated": False,
            "reason": f"Error: {str(e)}",
            "analysis": analysis
        }

@function_tool
async def analyze_npc_manipulation_potential(ctx, conflict_id: int, npc_id: int) -> Dict[str, Any]:
    """
    Analyze an NPC's potential to manipulate the player within a conflict.
    
    Args:
        conflict_id: ID of the conflict
        npc_id: ID of the NPC
        
    Returns:
        Manipulation potential analysis
    """
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        potential = await conflict_integration.analyze_manipulation_potential(npc_id)
        
        # Get conflict details for context
        conflict = await conflict_integration.get_conflict_details(conflict_id)
        
        # Get player's involvement
        involvement = None
        if conflict and "player_involvement" in conflict:
            involvement = conflict["player_involvement"]
        
        # Determine if manipulation makes sense based on current involvement
        makes_sense = True
        reason = "NPC could manipulate player"
        
        if involvement and involvement["involvement_level"] != "none":
            if involvement["is_manipulated"]:
                manipulator_id = involvement["manipulated_by"].get("npc_id")
                if manipulator_id == npc_id:
                    makes_sense = False
                    reason = "NPC is already manipulating player"
        
        # Determine appropriate manipulation goal
        goal = {"faction": "neutral", "involvement_level": "observing"}
        if potential["femdom_compatible"]:
            # Dominant female NPC would want more involvement
            goal["involvement_level"] = "participating"
            
            # Determine faction based on NPC's faction
            for stakeholder in conflict.get("stakeholders", []):
                if stakeholder["npc_id"] == npc_id and stakeholder.get("faction_id"):
                    # Find which faction this is in the conflict
                    if conflict.get("faction_a_name") == stakeholder.get("faction_name"):
                        goal["faction"] = "a"
                    elif conflict.get("faction_b_name") == stakeholder.get("faction_name"):
                        goal["faction"] = "b"
                    break
        
        return {
            "npc_id": npc_id,
            "conflict_id": conflict_id,
            "manipulation_potential": potential,
            "makes_sense": makes_sense,
            "reason": reason,
            "recommended_goal": goal,
            "current_involvement": involvement
        }
    except Exception as e:
        logging.error(f"Error analyzing manipulation potential: {e}")
        return {
            "npc_id": npc_id,
            "conflict_id": conflict_id,
            "manipulation_potential": {},
            "makes_sense": False,
            "reason": f"Error: {str(e)}",
            "recommended_goal": {},
            "current_involvement": None
        }

@function_tool
async def generate_manipulation_attempt(
    ctx, 
    conflict_id: int, 
    npc_id: int, 
    manipulation_type: str,
    goal: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a manipulation attempt by an NPC in a conflict.
    
    Args:
        conflict_id: ID of the conflict
        npc_id: ID of the NPC
        manipulation_type: Type of manipulation (domination, blackmail, seduction, etc.)
        goal: What the NPC wants the player to do
        
    Returns:
        Generated manipulation attempt
    """
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        # Get suggested content
        suggestion = await conflict_integration.suggest_manipulation_content(
            npc_id, conflict_id, manipulation_type, goal
        )
        
        # Create the manipulation attempt
        attempt = await conflict_integration.create_manipulation_attempt(
            conflict_id,
            npc_id,
            manipulation_type,
            suggestion["content"],
            goal,
            suggestion["leverage_used"],
            suggestion["intimacy_level"]
        )
        
        return {
            "generated": True,
            "attempt": attempt,
            "npc_id": npc_id,
            "npc_name": suggestion["npc_name"],
            "manipulation_type": manipulation_type,
            "content": suggestion["content"]
        }
    except Exception as e:
        logging.error(f"Error generating manipulation attempt: {e}")
        return {
            "generated": False,
            "reason": f"Error: {str(e)}",
            "npc_id": npc_id,
            "manipulation_type": manipulation_type
        }

@function_tool
async def track_conflict_story_beat(
    ctx,
    conflict_id: int,
    path_id: str,
    beat_description: str,
    involved_npcs: List[int],
    progress_value: float = 5.0
) -> Dict[str, Any]:
    """
    Track a story beat for a resolution path, advancing progress.
    
    Args:
        conflict_id: ID of the conflict
        path_id: ID of the resolution path
        beat_description: Description of what happened
        involved_npcs: List of NPC IDs involved
        progress_value: Progress value (0-100)
        
    Returns:
        Updated path information
    """
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
    try:
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        result = await conflict_integration.track_story_beat(
            conflict_id, path_id, beat_description, involved_npcs, progress_value
        )
        
        return {
            "tracked": True,
            "result": result
        }
    except Exception as e:
        logging.error(f"Error tracking story beat: {e}")
        return {
            "tracked": False,
            "reason": f"Error: {str(e)}"
        }

@function_tool
async def suggest_potential_manipulation(ctx, narrative_text: str) -> Dict[str, Any]:
    """
    Analyze narrative text and suggest potential NPC manipulation opportunities.
    
    Args:
        narrative_text: The narrative text to analyze
        
    Returns:
        Suggested manipulation opportunities
    """
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
    try:
        # Get active conflicts
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        active_conflicts = await conflict_integration.get_active_conflicts()
        
        if not active_conflicts:
            return {
                "opportunities": [],
                "reason": "No active conflicts"
            }
        
        # Get NPCs mentioned in the narrative
        from logic.conflict_system.conflict_manager import ConflictManager
        conflict_manager = ConflictManager(user_id, conversation_id)
        npcs = await conflict_manager._get_available_npcs()
        
        mentioned_npcs = []
        for npc in npcs:
            if npc["npc_name"] in narrative_text:
                mentioned_npcs.append(npc)
        
        if not mentioned_npcs:
            return {
                "opportunities": [],
                "reason": "No NPCs mentioned in narrative"
            }
        
        # Find female NPCs with high dominance
        opportunities = []
        for conflict in active_conflicts:
            conflict_id = conflict["conflict_id"]
            
            for npc in mentioned_npcs:
                if npc.get("sex", "female") == "female" and npc.get("dominance", 0) > 60:
                    # Check if this NPC is a stakeholder in this conflict
                    is_stakeholder = False
                    for stakeholder in conflict.get("stakeholders", []):
                        if stakeholder["npc_id"] == npc["npc_id"]:
                            is_stakeholder = True
                            break
                    
                    if is_stakeholder:
                        # Analyze manipulation potential
                        potential = await conflict_integration.analyze_manipulation_potential(npc["npc_id"])
                        
                        if potential["overall_potential"] > 60:
                            opportunities.append({
                                "conflict_id": conflict_id,
                                "conflict_name": conflict["conflict_name"],
                                "npc_id": npc["npc_id"],
                                "npc_name": npc["npc_name"],
                                "dominance": npc["dominance"],
                                "manipulation_type": potential["most_effective_type"],
                                "potential": potential["overall_potential"]
                            })
        
        return {
            "opportunities": opportunities,
            "total_opportunities": len(opportunities)
        }
    except Exception as e:
        logging.error(f"Error suggesting potential manipulation: {e}")
        return {
            "opportunities": [],
            "reason": f"Error: {str(e)}"
        }

# ----- Exports -----

# Story state and metadata tools
story_tools = [
    get_story_state,
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
    get_conflict_details
]

# Resource management tools
resource_tools = [
    check_resources,
    commit_resources_to_conflict,
    get_player_resources,
    analyze_activity_effects,
    apply_activity_effects,
    get_resource_history
]

# Narrative element tools
narrative_tools = [
    generate_personal_revelation,
    generate_dream_sequence,
    check_relationship_events,
    apply_crossroads_choice,
    check_npc_relationship,
    add_moment_of_clarity,
    get_player_journal_entries
]

# NEW: Context management tools
context_tools = [
    get_optimized_context,
    retrieve_relevant_memories,
    store_narrative_memory,
    search_by_vector,
    get_summarized_narrative_context
]
