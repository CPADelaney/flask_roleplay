# story_agent/schemas.py
"""
Pydantic models for all story agent tool parameters.
Following strict schema validation with extra="forbid".
"""

from typing import Optional, Dict, List, Literal, Any
from pydantic import BaseModel, Field, ConfigDict

# ===== Relationship Models =====
class DimensionChanges(BaseModel):
    """Changes to relationship dimensions."""
    money: Optional[int] = Field(None, ge=-100, le=100)
    trust: Optional[int] = Field(None, ge=-100, le=100)
    respect: Optional[int] = Field(None, ge=-100, le=100)
    obedience: Optional[int] = Field(None, ge=-100, le=100)
    closeness: Optional[int] = Field(None, ge=-100, le=100)
    dominance: Optional[int] = Field(None, ge=-100, le=100)
    
    model_config = ConfigDict(extra="forbid")


# ===== Conflict Models =====

# Helper models for ConflictAnalysis
class FactionAffiliation(BaseModel):
    """NPC faction affiliation details."""
    faction_id: int
    faction_name: str
    
    model_config = ConfigDict(extra="forbid")


class MentionedNPCTyped(BaseModel):
    """Fully typed NPC mentioned in conflict analysis."""
    npc_id: int
    npc_name: str
    dominance: int
    faction_affiliations: List[FactionAffiliation] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")


class MentionedFaction(BaseModel):
    """Faction mentioned in conflict analysis."""
    faction_id: int
    faction_name: str
    
    model_config = ConfigDict(extra="forbid")


class NPCRelationship(BaseModel):
    """Relationship between NPCs in conflict analysis."""
    npc1_id: int
    npc1_name: str
    npc2_id: int
    npc2_name: str
    relationship_type: Literal["alliance", "rivalry", "unknown"]
    sentence: str
    
    model_config = ConfigDict(extra="forbid")


class InternalFactionConflict(BaseModel):
    """Internal faction conflict details."""
    faction_id: int
    challenger_npc_id: int
    target_npc_id: int
    prize: str
    approach: str
    
    model_config = ConfigDict(extra="forbid")


class ConflictAnalysis(BaseModel):
    """Analysis results from conflict potential detection."""
    conflict_intensity: int = Field(ge=0, le=10)
    matched_keywords: List[str]
    mentioned_npcs: List[MentionedNPCTyped]
    mentioned_factions: List[MentionedFaction]
    npc_relationships: List[NPCRelationship]
    recommended_conflict_type: Literal["major", "standard", "minor", "catastrophic"]
    potential_internal_faction_conflict: Optional[InternalFactionConflict] = None
    has_conflict_potential: bool
    
    model_config = ConfigDict(extra="forbid")


class ManipulationGoal(BaseModel):
    """Goal for NPC manipulation attempts."""
    faction: Literal["a", "b", "neutral"]
    involvement_level: Literal["none", "observing", "participating", "leading"]
    money_committed: int = Field(0, ge=0)
    supplies_committed: int = Field(0, ge=0)
    influence_committed: int = Field(0, ge=0)
    specific_action: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")


class ManipulationPotential(BaseModel):
    """Analysis of NPC manipulation potential."""
    overall_potential: int = Field(ge=0, le=100)
    most_effective_type: Literal["domination", "blackmail", "seduction", "gaslighting", "manipulation"]
    femdom_compatible: bool
    
    model_config = ConfigDict(extra="forbid")


class ManipulationOpportunity(BaseModel):
    """Suggested manipulation opportunity."""
    conflict_id: int
    conflict_name: str
    npc_id: int
    npc_name: str
    dominance: int
    manipulation_type: str
    potential: int
    
    model_config = ConfigDict(extra="forbid")


# ===== Context/Memory Models =====
class MemorySearchParams(BaseModel):
    """Parameters for memory search operations."""
    query_text: str
    memory_type: Optional[str] = None
    limit: int = Field(5, ge=1, le=100)
    use_vector: bool = True
    
    model_config = ConfigDict(extra="forbid")


class ContextQueryParams(BaseModel):
    """Parameters for context retrieval."""
    query_text: str = ""
    use_vector: bool = True
    max_tokens: Optional[int] = Field(None, ge=100, le=10000)
    
    model_config = ConfigDict(extra="forbid")


class StoreMemoryParams(BaseModel):
    """Parameters for storing narrative memories."""
    content: str
    memory_type: str = "observation"
    importance: float = Field(0.6, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=lambda: ["story_director"])
    
    model_config = ConfigDict(extra="forbid")


class VectorSearchParams(BaseModel):
    """Parameters for vector search operations."""
    query_text: str
    entity_types: List[str] = Field(
        default_factory=lambda: ["npc", "location", "memory", "narrative"]
    )
    top_k: int = Field(5, ge=1, le=50)
    
    model_config = ConfigDict(extra="forbid")


# ===== Resource Models =====
class ResourceCheck(BaseModel):
    """Resource requirements to check."""
    money: int = Field(0, ge=0)
    supplies: int = Field(0, ge=0)
    influence: int = Field(0, ge=0)
    
    model_config = ConfigDict(extra="forbid")


class ResourceCommitment(BaseModel):
    """Resources to commit to a conflict."""
    conflict_id: int
    money: int = Field(0, ge=0)
    supplies: int = Field(0, ge=0)
    influence: int = Field(0, ge=0)
    
    model_config = ConfigDict(extra="forbid")


# ===== Activity Models =====
class ActivityAnalysisParams(BaseModel):
    """Parameters for activity analysis."""
    activity_text: str
    setting_context: Optional[str] = None
    apply_effects: bool = False
    
    model_config = ConfigDict(extra="forbid")


class ActivityFilterParams(BaseModel):
    """Parameters for filtering activities."""
    npc_archetypes: List[str] = Field(default_factory=list)
    meltdown_level: int = Field(0, ge=0, le=5)
    setting: str = ""
    
    model_config = ConfigDict(extra="forbid")


class ActivitySuggestionParams(BaseModel):
    """Parameters for activity suggestions."""
    npc_name: str
    intensity_level: int = Field(2, ge=1, le=5)
    archetypes: Optional[List[str]] = None
    
    model_config = ConfigDict(extra="forbid")


# ===== Player Involvement Models =====
class PlayerInvolvementParams(BaseModel):
    """Parameters for setting player involvement in conflicts."""
    conflict_id: int
    involvement_level: Literal["none", "observing", "participating", "leading"]
    faction: Literal["a", "b", "neutral"] = "neutral"
    money_committed: int = Field(0, ge=0)
    supplies_committed: int = Field(0, ge=0)
    influence_committed: int = Field(0, ge=0)
    action: Optional[str] = None
    
    model_config = ConfigDict(extra="forbid")


class PlayerInvolvementData(BaseModel):
    """Current player involvement in a conflict."""
    involvement_level: Literal["none", "observing", "participating", "leading"]
    faction: Literal["a", "b", "neutral"]
    is_manipulated: bool = False
    manipulated_by: Optional[Dict[str, Any]] = None  # Could be further typed
    resources_committed: Optional[Dict[str, int]] = None
    
    model_config = ConfigDict(extra="forbid")


# ===== Story Beat Models =====
class StoryBeatParams(BaseModel):
    """Parameters for tracking conflict story beats."""
    conflict_id: int
    path_id: str
    beat_description: str
    involved_npcs: List[int]
    progress_value: float = Field(5.0, ge=0.0, le=100.0)
    
    model_config = ConfigDict(extra="forbid")


# story_agent/tools.py
"""
Refactored tools for the Story Director agent with strict schema validation.
"""

import logging
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field



# Original imports remain the same
from agents import function_tool, RunContextWrapper
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
ContextType = Any  # Or StoryDirectorContext

# TODO: Consider injecting random seed for deterministic testing
# TODO: Expose public wrapper for _get_available_npcs() or document private usage

# ===== REFACTORED CONTEXT TOOLS =====

@function_tool
async def get_optimized_context(
    ctx: RunContextWrapper[ContextType],
    params: Optional[ContextQueryParams] = None
) -> Dict[str, Any]:
    """
    Get optimized context using the comprehensive context system.

    Args:
        params: Query parameters for context retrieval

    Returns:
        Dictionary with comprehensive context information.
    """
    # Handle defaults
    if params is None:
        params = ContextQueryParams()
    
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        context_service = await get_context_service(user_id, conversation_id)
        config = get_config()
        token_budget = params.max_tokens or config.get_token_budget("default")

        context_data = await context_service.get_context(
            input_text=params.query_text,
            context_budget=token_budget,
            use_vector_search=params.use_vector
        )

        # Safely access performance monitor
        perf_monitor = None
        if hasattr(context, 'performance_monitor'):
            perf_monitor = context.performance_monitor
        else:
            try:
                perf_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
            except Exception as pm_err:
                logger.warning(f"Could not get performance monitor instance: {pm_err}")

        if perf_monitor and "token_usage" in context_data:
            try:
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
        return {
            "error": str(e),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "context_data": None
        }


@function_tool
async def retrieve_relevant_memories(
    ctx: RunContextWrapper[ContextType],
    params: MemorySearchParams
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memories using vector search.
    
    Args:
        params: Memory search parameters
        
    Returns:
        List of relevant memories.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Create a MemorySearchRequest object
        request = MemorySearchRequest(
            query_text=params.query_text,
            memory_types=[params.memory_type] if params.memory_type else None,
            limit=params.limit,
            use_vector=params.use_vector
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
async def store_narrative_memory(
    ctx: RunContextWrapper[ContextType],
    params: StoreMemoryParams
) -> Dict[str, Any]:
    """
    Store a narrative memory in the memory system.

    Args:
        params: Memory storage parameters

    Returns:
        Stored memory information or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Handle default tags
    tags = params.tags
    if not tags:
        tags = [params.memory_type, "story_director"]

    try:
        memory_manager = await get_memory_manager(user_id, conversation_id)

        memory_id = await memory_manager.add_memory(
            content=params.content,
            memory_type=params.memory_type,
            importance=params.importance,
            tags=tags,
            metadata={"source": "story_director_tool", "timestamp": datetime.now().isoformat()}
        )

        # Safely check for narrative_manager
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            try:
                await context.narrative_manager.add_interaction(
                    content=params.content,
                    importance=params.importance,
                    tags=tags
                )
            except Exception as nm_err:
                logger.warning(f"Error calling narrative_manager.add_interaction: {nm_err}")

        return {
            "memory_id": memory_id,
            "content": params.content,
            "memory_type": params.memory_type,
            "importance": params.importance,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error storing narrative memory: {str(e)}", exc_info=True)
        return {"error": str(e), "success": False}


@function_tool
async def search_by_vector(
    ctx: RunContextWrapper[ContextType],
    params: VectorSearchParams
) -> List[Dict[str, Any]]:
    """
    Search for entities by semantic similarity using vector search.

    Args:
        params: Vector search parameters

    Returns:
        List of semantically similar entities.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        vector_service = await get_vector_service(user_id, conversation_id)
        if not vector_service or not vector_service.enabled:
            logger.info("Vector service is not enabled or available. Skipping vector search.")
            return []

        results = await vector_service.search_entities(
            query_text=params.query_text,
            entity_types=params.entity_types,
            top_k=params.top_k,
            hybrid_ranking=True
        )
        return results
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}", exc_info=True)
        return []


@function_tool
async def get_summarized_narrative_context(
    ctx: RunContextWrapper[ContextType],
    query: str,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get automatically summarized narrative context using progressive summarization.

    Args:
        query: Query for relevance matching.
        max_tokens: Maximum tokens for context (default: 1000)

    Returns:
        Summarized narrative context or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Handle default
    actual_max_tokens = max_tokens if max_tokens is not None else 1000

    try:
        narrative_manager = None
        if hasattr(context, 'narrative_manager') and context.narrative_manager:
            narrative_manager = context.narrative_manager
        else:
            try:
                from story_agent.progressive_summarization import RPGNarrativeManager
                dsn = 'DATABASE_URL_NOT_FOUND'
                try:
                    async with get_db_connection_context() as conn:
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

                try:
                    if hasattr(context, '__dict__') or isinstance(context, object):
                        context.narrative_manager = narrative_manager
                    else:
                        logger.warning("Context object does not support attribute assignment for narrative_manager.")
                except Exception as assign_err:
                    logger.warning(f"Could not store narrative_manager on context: {assign_err}")

            except ImportError:
                logger.error("Module 'story_agent.progressive_summarization' not found.")
                return {"error": "Narrative manager component not available.", "memories": [], "arcs": []}
            except Exception as init_error:
                logger.error(f"Error initializing narrative manager: {init_error}", exc_info=True)
                return {"error": "Narrative manager initialization failed.", "memories": [], "arcs": []}

        if not narrative_manager:
            return {"error": "Narrative manager could not be initialized.", "memories": [], "arcs": []}

        context_data = await narrative_manager.get_current_narrative_context(
            query,
            actual_max_tokens
        )
        return context_data
    except Exception as e:
        logger.error(f"Error getting summarized narrative context: {str(e)}", exc_info=True)
        return {"error": str(e), "memories": [], "arcs": []}


# ===== REFACTORED ACTIVITY TOOLS =====

@function_tool
@track_performance("analyze_activity")
async def analyze_activity(
    ctx: RunContextWrapper[ContextType],
    params: ActivityAnalysisParams
) -> Dict[str, Any]:
    """
    Analyze an activity to determine its resource effects.

    Args:
        params: Activity analysis parameters

    Returns:
        Dict with activity analysis and effects
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        from logic.activity_analyzer import ActivityAnalyzer
        analyzer = ActivityAnalyzer(user_id, conversation_id)
        result = await analyzer.analyze_activity(
            params.activity_text, 
            params.setting_context, 
            params.apply_effects
        )

        if hasattr(context, 'add_narrative_memory'):
            effects_description = []
            for resource_type, value in result.get("effects", {}).items():
                if value:
                    direction = "increased" if value > 0 else "decreased"
                    effects_description.append(f"{resource_type} {direction} by {abs(value)}")
            effects_text = ", ".join(effects_description) if effects_description else "no significant effects"
            memory_content = f"Analyzed activity: {params.activity_text[:100]}... with effects: {effects_text}"
            await context.add_narrative_memory(memory_content, "activity_analysis", 0.5)

        return result
    except Exception as e:
        logger.error(f"Error analyzing activity: {str(e)}", exc_info=True)
        return {
            "activity_type": "unknown", 
            "activity_details": "", 
            "effects": {}, 
            "description": f"Error analyzing activity: {str(e)}", 
            "error": str(e)
        }


@function_tool
@track_performance("get_filtered_activities")
async def get_filtered_activities(
    ctx: RunContextWrapper[ContextType],
    params: Optional[ActivityFilterParams] = None
) -> List[Dict[str, Any]]:
    """
    Get a list of activities filtered by NPC archetypes, meltdown level, and setting.

    Args:
        params: Activity filter parameters

    Returns:
        List of filtered activities
    """
    if params is None:
        params = ActivityFilterParams()
    
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
            npc_archetypes=params.npc_archetypes, 
            meltdown_level=params.meltdown_level, 
            user_stats=user_stats, 
            setting=params.setting
        )
        for activity in activities:
            activity["short_summary"] = build_short_summary(activity)

        return activities
    except Exception as e:
        logger.error(f"Error getting filtered activities: {str(e)}", exc_info=True)
        return []


@function_tool
@track_performance("generate_activity_suggestion")
async def generate_activity_suggestion(
    ctx: RunContextWrapper[ContextType],
    params: ActivitySuggestionParams
) -> Dict[str, Any]:
    """
    Generate a suggested activity for an NPC interaction based on archetypes and intensity.

    Args:
        params: Activity suggestion parameters

    Returns:
        Dict with suggested activity details
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        archetypes = params.archetypes
        if not archetypes:
            try:
                async with get_db_connection_context() as conn:
                    row = await conn.fetchrow(
                        "SELECT archetype FROM NPCStats WHERE npc_name=$1 AND user_id=$2 AND conversation_id=$3", 
                        params.npc_name, user_id, conversation_id
                    )
                    if row and row['archetype']:
                        if isinstance(row['archetype'], str):
                            try: 
                                archetype_data = json.loads(row['archetype'])
                                archetypes = archetype_data if isinstance(archetype_data, list) else (
                                    archetype_data.get("types") if isinstance(archetype_data, dict) else [row['archetype']]
                                )
                            except: 
                                archetypes = [row['archetype']]
                        elif isinstance(row['archetype'], list): 
                            archetypes = row['archetype']
                        elif isinstance(row['archetype'], dict) and "types" in row['archetype']: 
                            archetypes = row['archetype']["types"]
            except Exception as archetype_error: 
                logger.warning(f"Error getting NPC archetypes: {archetype_error}")
        
        if not archetypes: 
            archetypes = ["Dominance", "Femdom"]

        setting = "Default"
        if hasattr(context, 'get_comprehensive_context'):
            try: 
                comprehensive_context = await context.get_comprehensive_context()
                current_location = comprehensive_context.get("current_location")
                setting = current_location or setting
            except Exception as context_error: 
                logger.warning(f"Error getting location from context: {context_error}")

        from logic.activities_logic import filter_activities_for_npc, build_short_summary, get_all_activities as get_activities
        activities = await filter_activities_for_npc(
            npc_archetypes=archetypes, 
            meltdown_level=max(0, params.intensity_level-1), 
            setting=setting
        )
        if not activities: 
            activities = await get_activities()
            activities = random.sample(activities, min(3, len(activities)))

        selected_activity = random.choice(activities) if activities else None
        if not selected_activity: 
            return {"npc_name": params.npc_name, "success": False, "error": "No suitable activities found"}

        intensity_tiers = selected_activity.get("intensity_tiers", [])
        tier_text = ""
        if intensity_tiers: 
            idx = min(params.intensity_level - 1, len(intensity_tiers) - 1)
            idx = max(0, idx)
            tier_text = intensity_tiers[idx]

        suggestion = {
            "npc_name": params.npc_name, 
            "activity_name": selected_activity.get("name", ""), 
            "purpose": selected_activity.get("purpose", [])[0] if selected_activity.get("purpose") else "", 
            "intensity_tier": tier_text, 
            "intensity_level": params.intensity_level, 
            "short_summary": build_short_summary(selected_activity), 
            "archetypes_used": archetypes, 
            "setting": setting, 
            "success": True
        }

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Generated activity suggestion for {params.npc_name}: {suggestion['activity_name']} (Intensity: {params.intensity_level})"
            await context.add_narrative_memory(memory_content, "activity_suggestion", 0.5)

        return suggestion
    except Exception as e:
        logger.error(f"Error generating activity suggestion: {str(e)}", exc_info=True)
        return {"npc_name": params.npc_name, "success": False, "error": str(e)}


# ===== REFACTORED RELATIONSHIP TOOLS =====

@function_tool
@track_performance("update_relationship_dimensions")
async def update_relationship_dimensions(
    ctx: RunContextWrapper[ContextType],
    link_id: int,
    dimension_changes: DimensionChanges,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update specific dimensions of a relationship.

    Args:
        link_id: ID of the relationship link
        dimension_changes: Dimension changes model
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
        
        # Convert model to dict, excluding None values
        changes_dict = dimension_changes.model_dump(exclude_none=True)
        
        result = await integration.update_dimensions(link_id, changes_dict, reason)

        if hasattr(context, 'add_narrative_memory'):
            memory_content = f"Updated relationship dimensions for link {link_id}: {changes_dict}"
            if reason: 
                memory_content += f" Reason: {reason}"
            await context.add_narrative_memory(memory_content, "relationship_update", 0.5)

        return result
    except Exception as e:
        logger.error(f"Error updating relationship dimensions: {str(e)}", exc_info=True)
        return {"error": str(e), "link_id": link_id}


# ===== REFACTORED CONFLICT TOOLS =====

@function_tool
async def generate_conflict_from_analysis(
    ctx: RunContextWrapper[ContextType],
    analysis: ConflictAnalysis
) -> Dict[str, Any]:
    """Generate a conflict based on analysis provided by analyze_conflict_potential."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        if not analysis.has_conflict_potential:
            return {
                "generated": False, 
                "reason": "Insufficient conflict potential", 
                "analysis": analysis.model_dump()
            }

        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        conflict = await conflict_integration.generate_new_conflict(
            analysis.recommended_conflict_type
        )

        internal_faction_conflict = None
        if analysis.potential_internal_faction_conflict and conflict and conflict.get("conflict_id"):
            internal_data = analysis.potential_internal_faction_conflict
            try:
                internal_faction_conflict = await conflict_integration.initiate_faction_power_struggle(
                    conflict["conflict_id"],
                    internal_data.faction_id,
                    internal_data.challenger_npc_id,
                    internal_data.target_npc_id,
                    internal_data.prize,
                    internal_data.approach,
                    False
                )
            except Exception as e:
                logger.error(f"Error generating internal faction conflict: {e}")

        return {
            "generated": True,
            "conflict": conflict,
            "internal_faction_conflict": internal_faction_conflict
        }
    except Exception as e:
        logger.error(f"Error generating conflict from analysis: {e}", exc_info=True)
        return {
            "generated": False, 
            "reason": f"Error: {str(e)}", 
            "analysis": analysis.model_dump()
        }


@function_tool
async def generate_manipulation_attempt(
    ctx: RunContextWrapper[ContextType],
    conflict_id: int,
    npc_id: int,
    manipulation_type: str,
    goal: ManipulationGoal
) -> Dict[str, Any]:
    """Generate a manipulation attempt by an NPC in a conflict."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        # Convert model to dict
        goal_dict = goal.model_dump()

        suggestion = await conflict_integration.suggest_manipulation_content(
            npc_id, conflict_id, manipulation_type, goal_dict
        )
        attempt = await conflict_integration.create_manipulation_attempt(
            conflict_id,
            npc_id,
            manipulation_type,
            suggestion["content"],
            goal_dict,
            suggestion["leverage_used"],
            suggestion["intimacy_level"]
        )

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
        logger.error(f"Error generating manipulation attempt: {e}", exc_info=True)
        return {
            "generated": False,
            "reason": f"Error: {str(e)}",
            "npc_id": npc_id,
            "manipulation_type": manipulation_type
        }


@function_tool
async def set_player_involvement(
    ctx: RunContextWrapper[ContextType],
    params: PlayerInvolvementParams
) -> Dict[str, Any]:
    """
    Set the player's involvement in a conflict.

    Args:
        params: Player involvement parameters

    Returns:
        Updated conflict information or error dictionary.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Use the conflict integration directly instead of expecting it on context
    from logic.conflict_system.conflict_integration import ConflictSystemIntegration
    from logic.conflict_system.conflict_tools import (
        get_conflict_details, update_player_involvement as update_involvement
    )
    
    try:
        # Check if we have a resource manager
        if not hasattr(context, 'resource_manager'):
            logger.error("Context missing resource_manager")
            return {"conflict_id": params.conflict_id, "error": "Internal context setup error", "success": False}
            
        resource_manager = context.resource_manager

        # Check resources
        resource_check = await resource_manager.check_resources(
            params.money_committed, params.supplies_committed, params.influence_committed
        )
        if not resource_check.get('has_resources', False):
            resource_check['success'] = False
            resource_check['error'] = "Insufficient resources to commit"
            return resource_check

        # Get conflict info using the tools
        conflict_info = await get_conflict_details(ctx, params.conflict_id)
        
        # Update player involvement using the tools
        involvement_data = {
            "involvement_level": params.involvement_level,
            "faction": params.faction,
            "resources_committed": {
                "money": params.money_committed,
                "supplies": params.supplies_committed,
                "influence": params.influence_committed
            },
            "actions_taken": [params.action] if params.action else []
        }
        
        result = await update_involvement(ctx, params.conflict_id, involvement_data)

        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if params.money_committed > 0: 
                resources_text.append(f"{params.money_committed} money")
            if params.supplies_committed > 0: 
                resources_text.append(f"{params.supplies_committed} supplies")
            if params.influence_committed > 0: 
                resources_text.append(f"{params.influence_committed} influence")
            resources_committed = ", ".join(resources_text) if resources_text else "no resources"

            conflict_name = conflict_info.get('conflict_name', f'ID: {params.conflict_id}') if conflict_info else f'ID: {params.conflict_id}'
            memory_content = (
                f"Player set involvement in conflict {conflict_name} "
                f"to {params.involvement_level}, supporting {params.faction} faction "
                f"with {resources_committed}."
            )
            if params.action: 
                memory_content += f" Action taken: {params.action}"
            await context.add_narrative_memory(memory_content, "conflict_involvement", 0.7)

        if isinstance(result, dict):
            result["success"] = True
        else:
            result = {
                "conflict_id": params.conflict_id,
                "involvement_level": params.involvement_level,
                "faction": params.faction,
                "resources_committed": {
                    "money": params.money_committed,
                    "supplies": params.supplies_committed,
                    "influence": params.influence_committed
                },
                "action": params.action,
                "success": True,
                "raw_result": result
            }

        return result
    except Exception as e:
        logger.error(f"Error setting involvement: {str(e)}", exc_info=True)
        return {"conflict_id": params.conflict_id, "error": str(e), "success": False}


@function_tool
async def track_conflict_story_beat(
    ctx: RunContextWrapper[ContextType],
    params: StoryBeatParams
) -> Dict[str, Any]:
    """Track a story beat for a resolution path, advancing progress."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)

        result = await conflict_integration.track_story_beat(
            params.conflict_id,
            params.path_id,
            params.beat_description,
            params.involved_npcs,
            params.progress_value
        )

        if isinstance(result, dict):
            return {"tracked": True, "result": result}
        else:
            return {"tracked": True, "result": {"raw_output": result}}

    except Exception as e:
        logger.error(f"Error tracking story beat: {e}", exc_info=True)
        return {"tracked": False, "reason": f"Error: {str(e)}"}


# ===== REFACTORED RESOURCE TOOLS =====

@function_tool
@track_performance("check_resources")
async def check_resources(
    ctx: RunContextWrapper[ContextType],
    params: Optional[ResourceCheck] = None
) -> Dict[str, Any]:
    """
    Check if player has sufficient resources.

    Args:
        params: Resource check parameters

    Returns:
        Dictionary with resource check results.
    """
    if params is None:
        params = ResourceCheck()
    
    context = ctx.context
    if not hasattr(context, 'resource_manager'):
        logger.error("Context missing resource_manager")
        return {"has_resources": False, "error": "Internal context setup error", "current": {}}
    
    resource_manager = context.resource_manager

    try:
        result = await resource_manager.check_resources(
            params.money, params.supplies, params.influence
        )

        current_res = result.get('current', {})
        if current_res.get('money') is not None:
            try:
                formatted_money = await resource_manager.get_formatted_money(current_res['money'])
                current_res['formatted_money'] = formatted_money
                result['current'] = current_res
            except Exception as format_err:
                logger.warning(f"Could not format money: {format_err}")

        if 'has_resources' not in result:
            result['has_resources'] = False
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
    params: ResourceCommitment
) -> Dict[str, Any]:
    """
    Commit player resources to a conflict.

    Args:
        params: Resource commitment parameters

    Returns:
        Result of committing resources or error dictionary.
    """
    context = ctx.context
    if not hasattr(context, 'resource_manager'):
        logger.error("Context missing resource_manager")
        return {"success": False, "error": "Internal context setup error"}
    
    resource_manager = context.resource_manager

    try:
        conflict_info = None
        if hasattr(context, 'conflict_manager') and context.conflict_manager:
            try:
                conflict_info = await context.conflict_manager.get_conflict(params.conflict_id)
            except Exception as conflict_error:
                logger.warning(f"Could not get conflict info: {conflict_error}")
        else:
            logger.warning("Context missing conflict_manager")

        result = await resource_manager.commit_resources_to_conflict(
            params.conflict_id, params.money, params.supplies, params.influence
        )

        if params.money > 0 and result.get('success', False) and result.get('money_result'):
            money_result = result['money_result']
            if 'old_value' in money_result and 'new_value' in money_result:
                try:
                    old_formatted = await resource_manager.get_formatted_money(money_result['old_value'])
                    new_formatted = await resource_manager.get_formatted_money(money_result['new_value'])
                    change_val = money_result.get('change')
                    formatted_change = await resource_manager.get_formatted_money(change_val) if change_val is not None else None

                    money_result['formatted_old_value'] = old_formatted
                    money_result['formatted_new_value'] = new_formatted
                    if formatted_change is not None:
                        money_result['formatted_change'] = formatted_change
                    result['money_result'] = money_result
                except Exception as format_err:
                    logger.warning(f"Could not format money: {format_err}")

        if hasattr(context, 'add_narrative_memory'):
            resources_text = []
            if params.money > 0: 
                resources_text.append(f"{params.money} money")
            if params.supplies > 0: 
                resources_text.append(f"{params.supplies} supplies")
            if params.influence > 0: 
                resources_text.append(f"{params.influence} influence")
            resources_committed = ", ".join(resources_text) if resources_text else "No resources"

            conflict_name = conflict_info.get('conflict_name', f"ID: {params.conflict_id}") if conflict_info else f"ID: {params.conflict_id}"
            memory_content = f"Committed {resources_committed} to conflict {conflict_name}"
            await context.add_narrative_memory(memory_content, "resource_commitment", 0.6)

        if 'success' not in result:
            result['success'] = True

        return result
    except Exception as e:
        logger.error(f"Error committing resources: {str(e)}", exc_info=True)
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
    resource_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get the history of resource changes.

    Args:
        resource_type: Optional filter for specific resource type (money, supplies, influence, energy, hunger).
        limit: Maximum number of history entries to return. Defaults to 10 if not provided.

    Returns:
        List of resource change history entries.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    resource_manager = getattr(context, 'resource_manager', None)

    # Handle the default value inside the function
    actual_limit = limit if limit is not None else 10

    async with get_db_connection_context() as conn:
        try:
            base_query = "SELECT resource_type, old_value, new_value, amount_changed, source, description, timestamp FROM ResourceHistoryLog WHERE user_id=$1 AND conversation_id=$2"
            params = [user_id, conversation_id]

            if resource_type:
                base_query += " AND resource_type=$3 ORDER BY timestamp DESC LIMIT $4"
                params.extend([resource_type, actual_limit])
            else:
                base_query += " ORDER BY timestamp DESC LIMIT $3"
                params.append(actual_limit)

            rows = await conn.fetch(base_query, *params)
            history = []

            for row in rows:
                formatted_old, formatted_new, formatted_change = None, None, None
                # Only format money if resource_manager is available
                if row['resource_type'] == "money" and resource_manager:
                    try:
                        formatted_old = await resource_manager.get_formatted_money(row['old_value'])
                        formatted_new = await resource_manager.get_formatted_money(row['new_value'])
                        formatted_change = await resource_manager.get_formatted_money(row['amount_changed'])
                    except Exception as format_err:
                         logger.warning(f"Could not format money in get_resource_history: {format_err}")

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
        inner_monologue = random.choice(revelation_templates)

        # Create a context object for canon
        class RevealContext:
            def __init__(self, user_id, conversation_id):
                self.user_id = user_id
                self.conversation_id = conversation_id
        
        canon_ctx = RevealContext(user_id, conversation_id)
        
        async with get_db_connection_context() as conn:
            try:
                # Use canon to create journal entry
                journal_id = await canon.create_journal_entry(
                    ctx=canon_ctx,
                    conn=conn,
                    entry_type='personal_revelation',
                    entry_text=inner_monologue,
                    revelation_types=revelation_type,
                    narrative_moment=None,
                    fantasy_flag=False,
                    intensity_level=0,
                    importance=0.8,
                    tags=[revelation_type, "revelation", npc_name.lower().replace(" ", "_")]
                )

                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(
                        f"Personal revelation about {npc_name}: {inner_monologue}", 
                        "personal_revelation", 
                        0.8, 
                        tags=[revelation_type, "revelation", npc_name.lower().replace(" ", "_")]
                    )
                    
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
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
async def generate_dream_sequence(ctx: RunContextWrapper[ContextType], npc_names: List[str]) -> Dict[str, Any]:
    """
    Generate a symbolic dream sequence based on player's current state.

    Args:
        npc_names: List of NPC names to include in the dream

    Returns:
        A dream sequence
    """
    while len(npc_names) < 3: 
        npc_names.append(f"Unknown Figure {len(npc_names) + 1}")
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
        dream_text = random.choice(dream_templates)
        context = ctx.context
        user_id = context.user_id
        conversation_id = context.conversation_id

        # Create a context object for canon
        class DreamContext:
            def __init__(self, user_id, conversation_id):
                self.user_id = user_id
                self.conversation_id = conversation_id
        
        canon_ctx = DreamContext(user_id, conversation_id)

        async with get_db_connection_context() as conn:
            try:
                # Use canon to create journal entry
                journal_id = await canon.create_journal_entry(
                    ctx=canon_ctx,
                    conn=conn,
                    entry_type='dream_sequence',
                    entry_text=dream_text,
                    revelation_types=None,
                    narrative_moment=True,
                    fantasy_flag=True,
                    intensity_level=0,
                    importance=0.7,
                    tags=["dream", "symbolic"] + [npc.lower().replace(" ", "_") for npc in npc_names[:3]]
                )

                if hasattr(context, 'add_narrative_memory'):
                    await context.add_narrative_memory(
                        f"Dream sequence: {dream_text}", 
                        "dream_sequence", 
                        0.7, 
                        tags=["dream", "symbolic"] + [npc.lower().replace(" ", "_") for npc in npc_names[:3]]
                    )
                    
                if hasattr(context, 'narrative_manager') and context.narrative_manager:
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
@track_performance("apply_crossroads_choice")
async def apply_crossroads_choice(
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
        from logic.narrative_progression import add_moment_of_clarity as add_clarity
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
    entry_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get entries from the player's journal.

    Args:
        entry_type: Optional filter for entry type (personal_revelation, dream_sequence, moment_of_clarity, etc.).
        limit: Maximum number of entries to return. Defaults to 10 if not provided.

    Returns:
        List of journal entries.
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    # Handle the default value inside the function
    actual_limit = limit if limit is not None else 10

    async with get_db_connection_context() as conn:
        try:
            base_query = "SELECT id, entry_type, entry_text, revelation_types, narrative_moment, fantasy_flag, intensity_level, timestamp FROM PlayerJournal WHERE user_id=$1 AND conversation_id=$2"
            params = [user_id, conversation_id]

            if entry_type:
                base_query += " AND entry_type=$3 ORDER BY timestamp DESC LIMIT $4"
                params.extend([entry_type, actual_limit])
            else:
                base_query += " ORDER BY timestamp DESC LIMIT $3"
                params.append(actual_limit)

            rows = await conn.fetch(base_query, *params)
            entries = []
            for row in rows:
                 timestamp = row['timestamp']
                 timestamp_iso = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
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

# Add this function near the other tool functions (around line 1250, before analyze_conflict_potential)

@function_tool
@track_performance("get_available_npcs")
async def get_available_npcs(
    ctx: RunContextWrapper[ContextType],
    include_unintroduced: bool = False,
    min_dominance: Optional[int] = None,
    gender_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get available NPCs that can be involved in conflicts or other interactions.
    
    Args:
        include_unintroduced: Whether to include NPCs that haven't been introduced yet
        min_dominance: Minimum dominance level filter
        gender_filter: Filter by gender ("male", "female", or None for all)
    
    Returns:
        List of available NPCs with their details
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Use the integrated NPC system
        from logic.fully_integrated_npc_system import IntegratedNPCSystem
        npc_system = IntegratedNPCSystem(user_id, conversation_id)
        await npc_system.initialize()
        
        # Get NPCs based on criteria
        async with get_db_connection_context() as conn:
            query = """
                SELECT 
                    n.npc_id, n.npc_name, n.dominance, n.cruelty, 
                    n.closeness, n.trust, n.respect, n.intensity, 
                    n.sex, n.faction_affiliations, n.archetype,
                    n.introduced
                FROM NPCStats n
                WHERE n.user_id = $1 AND n.conversation_id = $2
            """
            params = [user_id, conversation_id]
            
            # Add filters
            conditions = []
            if not include_unintroduced:
                conditions.append("n.introduced = TRUE")
            
            if min_dominance is not None:
                conditions.append(f"n.dominance >= ${len(params) + 1}")
                params.append(min_dominance)
            
            if gender_filter:
                conditions.append(f"n.sex = ${len(params) + 1}")
                params.append(gender_filter)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY n.dominance DESC, n.introduced DESC"
            
            rows = await conn.fetch(query, *params)
            
            npcs = []
            for row in rows:
                npc = dict(row)
                
                # Parse faction affiliations if stored as JSON string
                if isinstance(npc['faction_affiliations'], str):
                    try:
                        npc['faction_affiliations'] = json.loads(npc['faction_affiliations'])
                    except:
                        npc['faction_affiliations'] = []
                elif npc['faction_affiliations'] is None:
                    npc['faction_affiliations'] = []
                
                # Parse archetype if stored as JSON
                if isinstance(npc['archetype'], str):
                    try:
                        archetype_data = json.loads(npc['archetype'])
                        if isinstance(archetype_data, list):
                            npc['archetype_list'] = archetype_data
                        elif isinstance(archetype_data, dict) and 'types' in archetype_data:
                            npc['archetype_list'] = archetype_data['types']
                        else:
                            npc['archetype_list'] = [npc['archetype']]
                    except:
                        npc['archetype_list'] = [npc['archetype']]
                else:
                    npc['archetype_list'] = []
                
                # Get relationship with player
                try:
                    relationship = await npc_system.get_relationship_with_player(npc['npc_id'])
                    npc['relationship_with_player'] = relationship
                except:
                    npc['relationship_with_player'] = {
                        'closeness': npc.get('closeness', 0),
                        'trust': npc.get('trust', 0),
                        'respect': npc.get('respect', 0),
                        'intensity': npc.get('intensity', 0)
                    }
                
                npcs.append(npc)
            
            return npcs
            
    except Exception as e:
        logger.error(f"Error getting available NPCs: {str(e)}", exc_info=True)
        return []
            
@function_tool
async def analyze_conflict_potential(ctx: RunContextWrapper[ContextType], narrative_text: str) -> Dict[str, Any]:
    """Analyze narrative text for conflict potential."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Move import here
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
        npcs = await get_available_npcs(ctx)
        
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
        internal_faction_conflict = None

        return {"conflict_intensity": conflict_intensity, "matched_keywords": matched_keywords, "mentioned_npcs": mentioned_npcs, "mentioned_factions": mentioned_factions, "npc_relationships": npc_relationships, "recommended_conflict_type": conflict_type, "potential_internal_faction_conflict": internal_faction_conflict, "has_conflict_potential": conflict_intensity >= 4}
    except Exception as e:
        logger.error(f"Error analyzing conflict potential: {e}")
        return {"conflict_intensity": 0, "matched_keywords": [], "mentioned_npcs": [], "mentioned_factions": [], "npc_relationships": [], "recommended_conflict_type": "minor", "potential_internal_faction_conflict": None, "has_conflict_potential": False, "error": str(e)}

@function_tool
async def analyze_npc_manipulation_potential(ctx: RunContextWrapper[ContextType], conflict_id: int, npc_id: int) -> Dict[str, Any]:
    """Analyze an NPC's potential to manipulate the player within a conflict."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        potential = await conflict_integration.analyze_manipulation_potential(npc_id)
        conflict = await conflict_integration.get_conflict_details(conflict_id)
        involvement = conflict.get("player_involvement") if conflict else None

        makes_sense = True; reason = "NPC could manipulate player"
        if involvement and involvement.get("involvement_level") != "none" and involvement.get("is_manipulated"):
            manipulator_id = involvement.get("manipulated_by", {}).get("npc_id")
            if manipulator_id == npc_id: makes_sense = False; reason = "NPC is already manipulating player"

        goal = {"faction": "neutral", "involvement_level": "observing"}
        if potential.get("femdom_compatible"): goal["involvement_level"] = "participating"

        return {"npc_id": npc_id, "conflict_id": conflict_id, "manipulation_potential": potential, "makes_sense": makes_sense, "reason": reason, "recommended_goal": goal, "current_involvement": involvement}
    except Exception as e:
        logger.error(f"Error analyzing manipulation potential: {e}")
        return {"npc_id": npc_id, "conflict_id": conflict_id, "manipulation_potential": {}, "makes_sense": False, "reason": f"Error: {str(e)}", "recommended_goal": {}, "current_involvement": None}

@function_tool
async def get_npc_details(
    ctx: RunContextWrapper[ContextType],
    npc_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific NPC.
    
    Args:
        npc_id: ID of the NPC
        
    Returns:
        NPC details or None if not found
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        from logic.fully_integrated_npc_system import IntegratedNPCSystem
        npc_system = IntegratedNPCSystem(user_id, conversation_id)
        await npc_system.initialize()
        
        # Get NPC details
        npc_data = await npc_system.get_npc(npc_id)
        
        if npc_data:
            # Enhance with additional information
            npc_data['relationship_with_player'] = await npc_system.get_relationship_with_player(npc_id)
            
            # Get conflict involvement
            async with get_db_connection_context() as conn:
                conflicts = await conn.fetch("""
                    SELECT c.conflict_id, c.conflict_name, cs.faction_name, cs.involvement_level
                    FROM ConflictStakeholders cs
                    JOIN Conflicts c ON cs.conflict_id = c.conflict_id
                    WHERE cs.npc_id = $1 AND c.is_active = TRUE
                        AND c.user_id = $2 AND c.conversation_id = $3
                """, npc_id, user_id, conversation_id)
                
                npc_data['active_conflicts'] = [dict(c) for c in conflicts]
            
            return npc_data
            
    except Exception as e:
        logger.error(f"Error getting NPC details: {str(e)}", exc_info=True)
        return None

@function_tool
async def suggest_potential_manipulation(ctx: RunContextWrapper[ContextType], narrative_text: str) -> Dict[str, Any]:
    """Analyze narrative text and suggest potential NPC manipulation opportunities."""
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id

    try:
        # Move import here
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        active_conflicts = await conflict_integration.get_active_conflicts()
        if not active_conflicts: return {"opportunities": [], "reason": "No active conflicts"}

        # Only get introduced female NPCs with high dominance for manipulation
        npcs = await get_available_npcs(ctx, include_unintroduced=False, min_dominance=60, gender_filter="female")
        mentioned_npcs = [npc for npc in npcs if npc["npc_name"] in narrative_text]
        if not mentioned_npcs: return {"opportunities": [], "reason": "No NPCs mentioned in narrative"}

        opportunities = []
        for conflict in active_conflicts:
            conflict_id = conflict["conflict_id"]
            for npc in mentioned_npcs:
                if npc.get("sex", "female") == "female" and npc.get("dominance", 0) > 60:
                    is_stakeholder = any(s["npc_id"] == npc["npc_id"] for s in conflict.get("stakeholders", []))
                    if is_stakeholder:
                        potential = await conflict_integration.analyze_manipulation_potential(npc["npc_id"])
                        if potential.get("overall_potential", 0) > 60:
                            opportunities.append({"conflict_id": conflict_id, "conflict_name": conflict["conflict_name"], "npc_id": npc["npc_id"], "npc_name": npc["npc_name"], "dominance": npc["dominance"], "manipulation_type": potential.get("most_effective_type"), "potential": potential.get("overall_potential")})

        return {"opportunities": opportunities, "total_opportunities": len(opportunities)}
    except Exception as e:
        logger.error(f"Error suggesting potential manipulation: {e}")
        return {"opportunities": [], "reason": f"Error: {str(e)}"}

# Tool lists - cleaned up to only include defined functions

# Context management tools
context_tools = [
    get_optimized_context,
    retrieve_relevant_memories,
    store_narrative_memory,
    search_by_vector,
    get_summarized_narrative_context,
    get_available_npcs,  # Add this
    get_npc_details      # Add this
]

# Activity tools
activity_tools = [
    analyze_activity,
    get_filtered_activities,
    generate_activity_suggestion
]

# Relationship tools
relationship_tools = [
    check_relationship_events,
    apply_crossroads_choice,
    check_npc_relationship,
    update_relationship_dimensions,
]

# Conflict management tools
conflict_tools = [
    analyze_conflict_potential,
    generate_conflict_from_analysis,
    analyze_npc_manipulation_potential,
    generate_manipulation_attempt,
    set_player_involvement,
    track_conflict_story_beat,
    suggest_potential_manipulation
]

# Resource management tools
resource_tools = [
    check_resources,
    commit_resources_to_conflict,
    get_player_resources,
    apply_activity_effects,
    get_resource_history
]

# Narrative element tools
narrative_tools = [
    generate_personal_revelation,
    generate_dream_sequence,
    check_relationship_events,
    add_moment_of_clarity,
    get_player_journal_entries
]
