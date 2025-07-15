# story_agent/agent_interaction.py

"""
This module handles interactions between specialized agents in the Story Director system.
It provides orchestration for complex story development tasks that require input from
multiple specialized agents, with full integration of the comprehensive context system.
"""

import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from context.context_config import ContextConfig

from agents import Agent, Runner, trace, handoff
from agents.exceptions import AgentsException, ModelBehaviorError

from story_agent.specialized_agents import (
    initialize_specialized_agents,
    analyze_conflict,
    generate_narrative_element,
    ConflictAnalystContext,
    NarrativeCrafterContext
)

from story_agent.story_director_agent import (
    initialize_story_director,
    StoryDirectorContext
)

# Context system integration
from context.context_service import get_context_service, get_comprehensive_context
from context.memory_manager import get_memory_manager, Memory
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

# Progressive summarization integration
from story_agent.progressive_summarization import (
    RPGNarrativeManager,
    SummaryLevel
)

# Database connection
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ----- Orchestration Functions -----

@track_performance("orchestrate_conflict_analysis")
async def orchestrate_conflict_analysis_and_narrative(
    user_id: int, 
    conversation_id: int,
    conflict_id: int
) -> Dict[str, Any]:
    """
    Orchestrate a complex interaction that:
    1. Analyzes a conflict with the Conflict Analyst agent
    2. Generates narrative elements with the Narrative Crafter agent
    3. Integrates both into a comprehensive story update
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        conflict_id: ID of the conflict to analyze
        
    Returns:
        Dictionary with the orchestrated result
    """
    start_time = time.time()
    
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("conflict_analysis_orchestration")
    
    try:
        # Get context service for comprehensive context
        context_service = await get_context_service(user_id, conversation_id)
        
        # Initialize memory manager for sophisticated memory handling
        memory_manager = await get_memory_manager(user_id, conversation_id)
        
        # Initialize vector service for semantic search
        vector_service = await get_vector_service(user_id, conversation_id)
        
        # Initialize narrative manager for progressive summarization
        narrative_manager = None
        try:
            narrative_manager = RPGNarrativeManager(
                user_id=user_id,
                conversation_id=conversation_id,
                db_connection_string=None  # Will be initialized from config
            )
            await narrative_manager.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize narrative manager: {e}")
        
        # Create all necessary contexts with full integration
        conflict_context = ConflictAnalystContext(
            user_id=user_id, 
            conversation_id=conversation_id
        )
        
        narrative_context = NarrativeCrafterContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        director_context = StoryDirectorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Initialize context components with error handling
        await director_context.initialize_context_components()
        
        # Run tasks in parallel to improve performance
        # First, create a semantically relevant query for the conflict
        from logic.conflict_system.conflict_manager import ConflictManager
        conflict_manager = ConflictManager(user_id, conversation_id)
        conflict_details = await conflict_manager.get_conflict(conflict_id)
        
        # Form a query based on conflict details for vector search
        query = f"conflict analysis for {conflict_details.get('conflict_name', '')} involving {conflict_details.get('faction_a_name', '')} and {conflict_details.get('faction_b_name', '')}"
        
        # Launch parallel tasks
        tasks = [
            asyncio.create_task(analyze_conflict(conflict_id, conflict_context)),
            asyncio.create_task(context_service.get_context(
                input_text=query,
                use_vector_search=True,
                context_budget=context_service.config.get_token_budget("analysis")
            ))
        ]
        
        # Use vector search to find NPCs involved in this conflict
        involved_npcs_task = asyncio.create_task(
            vector_service.search_entities(
                query_text=f"conflict {conflict_details.get('conflict_name', '')}",
                entity_types=["npc"],
                top_k=5
            )
        )
        
        # Use vector search to find relevant memories
        relevant_memories_task = asyncio.create_task(
            memory_manager.search_memories(
                query_text=query,
                limit=5,
                use_vector=True
            )
        )
        
        # Get narrative stage in parallel
        from logic.narrative_progression import get_current_narrative_stage
        narrative_stage_task = asyncio.create_task(
            get_current_narrative_stage(user_id, conversation_id)
        )
        
        # Wait for all primary tasks to complete
        conflict_analysis, comprehensive_context = await asyncio.gather(*tasks)
        
        # Wait for secondary tasks
        involved_npcs_results = await involved_npcs_task
        relevant_memories = await relevant_memories_task
        narrative_stage = await narrative_stage_task
        
        # Process involved NPCs from vector search
        npc_names = []
        for result in involved_npcs_results:
            if "metadata" in result and "npc_name" in result["metadata"]:
                npc_names.append(result["metadata"]["npc_name"])
            elif "metadata" in result and "entity_id" in result["metadata"]:
                # Try to get NPC name from database
                try:
                    # Using the new async context manager for database connection
                    async with get_db_connection_context() as conn:
                        row = await conn.fetchrow(
                            "SELECT npc_name FROM NPCStats WHERE npc_id = $1",
                            int(result["metadata"]["entity_id"])
                        )
                        if row:
                            npc_names.append(row["npc_name"])
                except Exception as db_error:
                    logger.warning(f"Error retrieving NPC name: {db_error}")
        
        # If we still don't have enough NPCs, get from comprehensive context
        if len(npc_names) < 3:
            for npc in comprehensive_context.get("npcs", []):
                if "npc_name" in npc and npc["npc_name"] not in npc_names:
                    npc_names.append(npc["npc_name"])
                    if len(npc_names) >= 3:
                        break
        
        # Format memories for context
        recent_events = []
        for memory in relevant_memories:
            if hasattr(memory, 'content'):
                recent_events.append(memory.content)
            elif isinstance(memory, dict) and 'content' in memory:
                recent_events.append(memory['content'])
        
        # If we have a narrative manager, get optimal context
        if narrative_manager:
            narrative_context_result = await narrative_manager.get_optimal_narrative_context(
                query=query,
                max_tokens=1000
            )
            
            # Extract key events from optimal context
            for event in narrative_context_result.get("relevant_events", []):
                if "content" in event and event["content"] not in recent_events:
                    recent_events.append(event["content"])
        
        # Generate narrative based on conflict analysis with enriched context
        stage_name = narrative_stage.name if narrative_stage else "Unknown"
        
        # Pass context to narrative generation
        narrative_task = asyncio.create_task(
            generate_narrative_element(
                "conflict_narrative",
                {
                    "npc_names": npc_names[:3],  # Limit to top 3 most relevant NPCs
                    "narrative_stage": stage_name,
                    "recent_events": "\n".join(recent_events[:5]),  # Top 5 most relevant memories
                    "conflict_name": conflict_details.get("conflict_name", ""),
                    "faction_a": conflict_details.get("faction_a_name", ""),
                    "faction_b": conflict_details.get("faction_b_name", ""),
                    "conflict_analysis": conflict_analysis["analysis"],
                    "comprehensive_context": comprehensive_context  # Pass the complete context
                },
                narrative_context
            )
        )
        
        # Initialize the Story Director to integrate everything
        director_agent, _ = await initialize_story_director(user_id, conversation_id)
        
        # Wait for narrative generation to complete
        narrative_element = await narrative_task
        
        # Store in both memory systems for redundancy and richness
        
        # 1. Store in memory manager
        memory_id = await memory_manager.add_memory(
            content=narrative_element["content"],
            memory_type="narrative_element",
            importance=0.8,
            tags=["conflict", conflict_details.get("conflict_name", "").lower().replace(" ", "_"), "narrative"],
            metadata={
                "conflict_id": conflict_id,
                "conflict_name": conflict_details.get("conflict_name", ""),
                "source": "orchestration"
            }
        )
        
        # 2. Store in narrative manager for progressive summarization
        if narrative_manager:
            await narrative_manager.add_revelation(
                content=narrative_element["content"],
                revelation_type="conflict_narrative",
                importance=0.8,
                tags=["conflict", conflict_details.get("conflict_name", "").lower().replace(" ", "_"), "narrative"]
            )
        
        # Build an enhanced prompt with comprehensive context
        integration_prompt = f"""
        I need you to integrate conflict analysis and a narrative element into a comprehensive story update.
        
        CONFLICT ANALYSIS:
        {conflict_analysis["analysis"]}
        
        NARRATIVE ELEMENT:
        {narrative_element["content"]}
        
        CONFLICT DETAILS:
        - Name: {conflict_details.get('conflict_name', '')}
        - Type: {conflict_details.get('conflict_type', '')}
        - Phase: {conflict_details.get('phase', '')}
        - Progress: {conflict_details.get('progress', 0)}%
        - Faction A: {conflict_details.get('faction_a_name', '')}
        - Faction B: {conflict_details.get('faction_b_name', '')}
        
        KEY NPCs INVOLVED:
        {', '.join(npc_names[:5])}
        
        NARRATIVE STAGE:
        {stage_name}
        
        RELEVANT MEMORIES:
        {recent_events[0] if recent_events else "No relevant memories found."}
        
        Please create a comprehensive story update that:
        1. Summarizes the current state of the conflict
        2. Integrates the narrative element to advance the story
        3. Suggests clear next steps for the player
        4. Hints at possible consequences of different choices
        
        Format your response as a structured story update with clear sections.
        """
        
        # Run the Story Director to integrate everything with tracing
        with trace(workflow_name="StoryOrchestration", group_id=f"conflict_{conflict_id}"):
            integration_result = await Runner.run(
                director_agent,
                integration_prompt,
                context=director_context
            )
        
        # Track token usage for this operation
        if hasattr(integration_result, 'raw_responses') and integration_result.raw_responses:
            for response in integration_result.raw_responses:
                if hasattr(response, 'usage'):
                    performance_monitor.record_token_usage(response.usage.total_tokens)
        
        execution_time = time.time() - start_time
        
        # Store the final integrated result as a memory
        await memory_manager.add_memory(
            content=f"Integrated story update for conflict {conflict_details.get('conflict_name', '')}: {integration_result.final_output[:200]}...",
            memory_type="integrated_story_update",
            importance=0.8,
            tags=["integrated", "conflict", conflict_details.get("conflict_name", "").lower().replace(" ", "_")],
            metadata={
                "conflict_id": conflict_id,
                "execution_time": execution_time,
                "source": "orchestration"
            }
        )
        
        # Record performance metrics
        performance_monitor.stop_timer(timer_id)
        
        # Return the final orchestrated result with rich metadata
        return {
            "conflict_id": conflict_id,
            "conflict_name": conflict_details.get("conflict_name", ""),
            "conflict_analysis": conflict_analysis,
            "narrative_element": narrative_element,
            "integrated_update": integration_result.final_output,
            "execution_time": execution_time,
            "involved_npcs": npc_names[:5],
            "narrative_stage": stage_name,
            "metrics": {
                "conflict_analysis_metrics": conflict_context.get_metrics() if hasattr(conflict_context, "get_metrics") else {},
                "narrative_metrics": narrative_context.get_metrics() if hasattr(narrative_context, "get_metrics") else {},
                "director_metrics": director_context.metrics if hasattr(director_context, "metrics") else {},
                "performance": performance_monitor.get_metrics()
            }
        }
    except Exception as e:
        logger.error(f"Error in orchestrate_conflict_analysis_and_narrative: {e}", exc_info=True)
        
        # Attempt to record the error as a memory for future reference
        try:
            memory_manager = await get_memory_manager(user_id, conversation_id)
            await memory_manager.add_memory(
                content=f"Error during conflict analysis orchestration: {str(e)}",
                memory_type="error",
                importance=0.7,
                tags=["error", "orchestration", "conflict_analysis"],
                metadata={
                    "conflict_id": conflict_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as mem_error:
            logger.error(f"Failed to record error in memory system: {mem_error}")
            
        # Stop the performance timer if running
        if 'timer_id' in locals():
            performance_monitor.stop_timer(timer_id)
            
        return {
            "conflict_id": conflict_id,
            "error": str(e),
            "success": False,
            "execution_time": time.time() - start_time
        }

@track_performance("generate_story_beat")
async def generate_comprehensive_story_beat(
    user_id: int,
    conversation_id: int,
    story_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a comprehensive story beat involving multiple specialized agents.
    
    This function:
    1. Uses conflict analysis to understand the current conflict landscape
    2. Uses relationship analysis to understand character dynamics
    3. Uses narrative crafting to create emotionally resonant moments
    4. Integrates everything into a cohesive story beat
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        story_context: Dictionary with context about the current story state
        
    Returns:
        Dictionary with the comprehensive story beat
    """
    start_time = time.time()
    
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("story_beat_generation")
    
    # Store context manager for delta updates
    context_manager = get_context_manager()
    
    try:
        # Initialize context components with full integration
        context_service = await get_context_service(user_id, conversation_id)
        memory_manager = await get_memory_manager(user_id, conversation_id)
        vector_service = await get_vector_service(user_id, conversation_id)
        
        # Try to initialize narrative manager for progressive summarization
        narrative_manager = None
        try:
            narrative_manager = RPGNarrativeManager(
                user_id=user_id,
                conversation_id=conversation_id,
                db_connection_string=None  # Will be initialized from config
            )
            await narrative_manager.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize narrative manager: {e}")
        
        # Initialize specialized agents
        specialized_agents = initialize_specialized_agents()
        
        # Initialize contexts for each agent with context system integration
        director_context = StoryDirectorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        await director_context.initialize_context_components()
        
        # Build a semantically meaningful query based on story context
        query = story_context.get("narrative_focus", "current story state")
        
        # If there are key NPCs, add them to the query for better relevance
        if "key_npcs" in story_context:
            npcs = ", ".join(story_context["key_npcs"][:3])
            query = f"{query} involving {npcs}"
            
        # If there's a specific theme, add it for better relevance
        if "theme" in story_context:
            query = f"{query} with theme of {story_context['theme']}"
            
        # Get comprehensive context with vector search for relevance
        comprehensive_context = await context_service.get_context(
            input_text=query,
            use_vector_search=True,
            context_budget=context_service.config.get_token_budget("story_beat")
        )
        
        # Launch parallel tasks for efficiency
        narrative_stage_task = asyncio.create_task(
            get_narrative_stage_info(comprehensive_context)
        )
        
        relevant_memories_task = asyncio.create_task(
            memory_manager.search_memories(
                query_text=query,
                limit=7,  # Get more memories for richer context
                use_vector=True
            )
        )
        
        # Get active conflicts in parallel
        conflict_manager = director_context.conflict_manager
        active_conflicts_task = asyncio.create_task(
            conflict_manager.get_active_conflicts()
        )
        
        # Use vector search to find related entities
        vector_results_task = asyncio.create_task(
            vector_service.search_entities(
                query_text=query,
                entity_types=["npc", "location", "memory", "narrative"],
                top_k=10,
                hybrid_ranking=True  # Use hybrid ranking for better relevance
            )
        )
        
        # Wait for all parallel tasks
        narrative_stage_info = await narrative_stage_task
        relevant_memories = await relevant_memories_task
        active_conflicts = await active_conflicts_task
        vector_results = await vector_results_task
        
        # Extract and organize memory content
        memory_content = []
        for memory in relevant_memories:
            if hasattr(memory, 'content'):
                memory_content.append(memory.content)
            elif isinstance(memory, dict) and 'content' in memory:
                memory_content.append(memory['content'])
                
        # Get NPCs from comprehensive context and vector results
        npcs = comprehensive_context.get("npcs", [])
        npc_names = [npc.get("npc_name", "Unknown") for npc in npcs[:5]]
        
        # Add NPCs from vector results if not already included
        for result in vector_results:
            if "metadata" in result and "entity_type" in result["metadata"] and result["metadata"]["entity_type"] == "npc":
                npc_name = result["metadata"].get("npc_name", "")
                if npc_name and npc_name not in npc_names:
                    npc_names.append(npc_name)
                    if len(npc_names) >= 5:  # Limit to 5 NPCs
                        break
        
        # Use conflict analyst if active conflicts exist
        conflict_analysis = {"analysis": "No active conflicts found."}
        if active_conflicts:
            main_conflict = active_conflicts[0]  # Most important conflict
            conflict_context = ConflictAnalystContext(
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            conflict_analysis = await analyze_conflict(main_conflict["conflict_id"], conflict_context)
        
        # Determine appropriate narrative element type based on story context
        element_type = story_context.get("requested_element_type", "")
        if not element_type:
            # Choose appropriate element type based on narrative stage
            narrative_stage = narrative_stage_info.get("name", "Unknown")
            if narrative_stage in ["Innocent Beginning", "First Doubts"]:
                element_type = "subtle_manipulation"
            elif narrative_stage in ["Creeping Realization"]:
                element_type = "revelation"
            else:
                element_type = "explicit_control"
        
        # Get player stats for context
        player_stats = comprehensive_context.get("player_stats", {})
        
        # Initialize narrative context with full context integration
        narrative_context = NarrativeCrafterContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Get optimal narrative context if available
        narrative_context_data = {}
        if narrative_manager:
            try:
                narrative_context_data = await narrative_manager.get_optimal_narrative_context(
                    query=query,
                    max_tokens=1000
                )
            except Exception as narrative_error:
                logger.warning(f"Error getting narrative context: {narrative_error}")
        
        # Generate narrative element with rich context
        narrative_element = await generate_narrative_element(
            element_type,
            {
                "npc_names": npc_names,
                "narrative_stage": narrative_stage_info.get("name", "Unknown"),
                "player_stats": player_stats,
                "conflict_analysis": conflict_analysis["analysis"],
                "requested_focus": story_context.get("narrative_focus", ""),
                "recent_memories": "\n".join(memory_content[:3]),  # Top 3 memories
                "comprehensive_context": comprehensive_context,  # Pass complete context
                "narrative_context": narrative_context_data  # Pass optimal narrative context
            },
            narrative_context
        )
        
        # Store the narrative element in both memory systems
        memory_id = await memory_manager.add_memory(
            content=narrative_element["content"],
            memory_type="narrative_element",
            importance=0.8,
            tags=[element_type, "story_beat"],
            metadata={
                "element_type": element_type,
                "narrative_stage": narrative_stage_info.get("name", "Unknown"),
                "source": "story_beat_generation"
            }
        )
        
        # Add to narrative manager for progressive summarization
        if narrative_manager:
            try:
                await narrative_manager.add_revelation(
                    content=narrative_element["content"],
                    revelation_type=element_type,
                    importance=0.8,
                    tags=[element_type, "story_beat"]
                )
            except Exception as narrative_error:
                logger.warning(f"Error adding to narrative manager: {narrative_error}")
        
        # Create a context diff for the new narrative element
        context_manager.apply_targeted_change(
            path="/narrative_elements",
            value={
                "type": element_type,
                "content": narrative_element["content"],
                "timestamp": datetime.now().isoformat()
            },
            operation="add"
        )
        
        # Use the Story Director to integrate everything
        director_agent, _ = await initialize_story_director(user_id, conversation_id)
        
        # Build a rich, context-aware prompt
        integration_prompt = f"""
        Create a compelling story beat that integrates the following elements:
        
        CURRENT NARRATIVE STAGE: {narrative_stage_info.get("name", "Unknown")}
        {narrative_stage_info.get("description", "")}
        
        CONFLICT ANALYSIS: 
        {conflict_analysis["analysis"]}
        
        NARRATIVE ELEMENT:
        {narrative_element["content"]}
        
        PLAYER STATS:
        {json.dumps(player_stats, indent=2)}
        
        KEY NPCs:
        {", ".join(npc_names)}
        
        RECENT MEMORIES:
        {memory_content[0] if memory_content else "No recent memories"}
        
        USER REQUESTS:
        Focus: {story_context.get("narrative_focus", "No specific focus")}
        Tone: {story_context.get("tone", "Default tone")}
        
        Create a cohesive story beat that:
        1. Advances the narrative appropriately for the current stage
        2. Integrates conflict elements naturally
        3. Showcases characters authentically
        4. Presents meaningful choices to the player
        5. Maintains the overarching theme of subtle control and manipulation
        
        Format your response as a rich narrative scene followed by potential player choices and their implications.
        """
        
        with trace(workflow_name="StoryBeatGeneration", group_id=f"user_{user_id}"):
            story_beat = await Runner.run(
                director_agent,
                integration_prompt,
                context=director_context
            )
        
        # Track token usage and performance
        if hasattr(story_beat, 'raw_responses') and story_beat.raw_responses:
            for response in story_beat.raw_responses:
                if hasattr(response, 'usage'):
                    performance_monitor.record_token_usage(response.usage.total_tokens)
        
        # Store the story beat as a memory
        await memory_manager.add_memory(
            content=f"Generated story beat: {story_beat.final_output[:200]}...",
            memory_type="story_beat",
            importance=0.9,  # High importance for story beats
            tags=["story_beat", element_type, narrative_stage_info.get("name", "Unknown").lower().replace(" ", "_")],
            metadata={
                "element_type": element_type,
                "narrative_stage": narrative_stage_info.get("name", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        execution_time = time.time() - start_time
        
        # Stop performance tracking
        performance_monitor.stop_timer(timer_id)
        
        # Return the final comprehensive story beat with rich metadata
        return {
            "story_beat": story_beat.final_output,
            "narrative_stage": narrative_stage_info.get("name", "Unknown"),
            "element_type": element_type,
            "conflict_analysis": conflict_analysis,
            "narrative_element": narrative_element,
            "execution_time": execution_time,
            "success": True,
            "performance_metrics": performance_monitor.get_metrics(),
            "context_version": comprehensive_context.get("version", None)
        }
    except Exception as e:
        logger.error(f"Error generating comprehensive story beat: {e}", exc_info=True)
        
        # Stop the timer if still running
        if 'timer_id' in locals() and 'performance_monitor' in locals():
            performance_monitor.stop_timer(timer_id)
            
        # Try to record the error as a memory
        try:
            memory_manager = await get_memory_manager(user_id, conversation_id)
            await memory_manager.add_memory(
                content=f"Error generating story beat: {str(e)}",
                memory_type="error",
                importance=0.7,
                tags=["error", "story_beat"],
                metadata={
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as mem_error:
            logger.error(f"Failed to record error in memory system: {mem_error}")
            
        return {
            "error": str(e),
            "success": False,
            "execution_time": time.time() - start_time
        }

async def get_narrative_stage_info(comprehensive_context: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract narrative stage information from comprehensive context.
    
    Args:
        comprehensive_context: Comprehensive context from context service
        
    Returns:
        Dictionary with narrative stage name and description
    """
    narrative_stage = comprehensive_context.get("narrative_stage", {})
    if not narrative_stage or not isinstance(narrative_stage, dict):
        return {"name": "Unknown", "description": "No narrative stage information available"}
        
    return {
        "name": narrative_stage.get("name", "Unknown"),
        "description": narrative_stage.get("description", "No description available")
    }

# ----- Agent Communication Interface -----

@track_performance("agent_communicate")
async def agent_communicate(
    source_agent_type: str,
    target_agent_type: str,
    message: str,
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Facilitate communication between two specialized agents.
    
    Args:
        source_agent_type: Type of the source agent (e.g., "conflict_analyst")
        target_agent_type: Type of the target agent (e.g., "narrative_crafter")
        message: Message to send from source to target
        context_data: Dictionary with additional context (user_id, conversation_id, etc.)
        
    Returns:
        Response from the target agent
    """
    user_id = context_data.get("user_id")
    conversation_id = context_data.get("conversation_id")
    
    if not user_id or not conversation_id:
        raise ValueError("Missing user_id or conversation_id in context_data")
    
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("agent_communication")
    
    # Context manager for tracking changes
    context_manager = get_context_manager()
    
    try:
        # Initialize comprehensive context components
        context_service = await get_context_service(user_id, conversation_id)
        memory_manager = await get_memory_manager(user_id, conversation_id)
        vector_service = await get_vector_service(user_id, conversation_id)
        
        # Try to initialize narrative manager for progressive summarization
        narrative_manager = None
        try:
            narrative_manager = RPGNarrativeManager(
                user_id=user_id,
                conversation_id=conversation_id,
                db_connection_string=None  # Will be initialized from config
            )
            await narrative_manager.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize narrative manager: {e}")
        
        # Initialize specialized agents
        specialized_agents = initialize_specialized_agents()
        
        # Validate agent types
        if source_agent_type not in specialized_agents:
            raise ValueError(f"Invalid source agent type: {source_agent_type}")
        
        if target_agent_type not in specialized_agents:
            raise ValueError(f"Invalid target agent type: {target_agent_type}")
        
        source_agent = specialized_agents[source_agent_type]
        target_agent = specialized_agents[target_agent_type]
        
        # Create appropriate context for the target agent
        if target_agent_type == "conflict_analyst":
            target_context = ConflictAnalystContext(
                user_id=user_id,
                conversation_id=conversation_id
            )
        elif target_agent_type == "narrative_crafter":
            target_context = NarrativeCrafterContext(
                user_id=user_id,
                conversation_id=conversation_id
            )
        else:
            # Generic context for other agent types
            from story_agent.specialized_agents import AgentContext
            target_context = AgentContext(
                user_id=user_id,
                conversation_id=conversation_id
            )
        
        # Get comprehensive context with vector search for relevance
        comprehensive_context = await context_service.get_context(
            input_text=message,
            use_vector_search=True
        )
        
        # Get semantically relevant memories using vector search
        relevant_memories = await memory_manager.search_memories(
            query_text=message,
            limit=5,
            use_vector=True
        )
        
        # Format memories
        memory_text = ""
        for memory in relevant_memories:
            if hasattr(memory, 'content'):
                memory_text += f"- {memory.content[:200]}...\n"
            elif isinstance(memory, dict) and 'content' in memory:
                memory_text += f"- {memory['content'][:200]}...\n"
        
        # Get optimal narrative context if available
        narrative_context_text = ""
        if narrative_manager:
            try:
                narrative_context_data = await narrative_manager.get_optimal_narrative_context(
                    query=message,
                    max_tokens=800
                )
                
                for event in narrative_context_data.get("relevant_events", []):
                    if "content" in event:
                        narrative_context_text += f"- {event['content'][:150]}...\n"
            except Exception as narrative_error:
                logger.warning(f"Error getting narrative context: {narrative_error}")
        
        # Get key NPCs from context
        npcs = comprehensive_context.get("npcs", [])
        npc_text = ""
        if npcs:
            npc_text = "Key NPCs:\n"
            for npc in npcs[:3]:  # Limit to top 3
                npc_text += f"- {npc.get('npc_name', 'Unknown')}: {npc.get('description', '')[:100]}...\n"
        
        # Format the message with rich context
        formatted_message = f"""
        Message from {source_agent_type.replace('_', ' ').title()}:
        
        {message}
        
        Relevant context:
        - Current narrative stage: {comprehensive_context.get('narrative_stage', {}).get('name', 'Unknown')}
        - Current location: {comprehensive_context.get('current_location', 'Unknown')}
        
        {npc_text}
        
        Relevant memories:
        {memory_text}
        
        {narrative_context_text}
        
        Please respond with your thoughts and analysis based on your expertise as {target_agent_type.replace('_', ' ').title()}.
        """
        
        try:
            # Send the message to the target agent with tracing
            with trace(workflow_name="AgentCommunication", group_id=f"user_{user_id}"):
                response = await Runner.run(
                    target_agent,
                    formatted_message,
                    context=target_context
                )
            
            # Track token usage
            if hasattr(response, 'raw_responses') and response.raw_responses:
                for resp in response.raw_responses:
                    if hasattr(resp, 'usage'):
                        performance_monitor.record_token_usage(resp.usage.total_tokens)
            
            # Store this communication as a memory
            memory_id = await memory_manager.add_memory(
                content=f"Communication from {source_agent_type} to {target_agent_type}: {message[:100]}...\nResponse: {response.final_output[:100]}...",
                memory_type="agent_communication",
                importance=0.4,
                tags=[source_agent_type, target_agent_type, "communication"],
                metadata={
                    "source_agent": source_agent_type,
                    "target_agent": target_agent_type,
                    "full_message": message,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Add to narrative manager for progressive summarization
            if narrative_manager:
                try:
                    await narrative_manager.add_interaction(
                        content=f"Agent communication: {source_agent_type} -> {target_agent_type}\nMessage: {message[:100]}...\nResponse: {response.final_output[:100]}...",
                        importance=0.4,
                        tags=[source_agent_type, target_agent_type, "communication"]
                    )
                except Exception as narrative_error:
                    logger.warning(f"Error adding to narrative manager: {narrative_error}")
            
            # Create a context diff for the new communication
            await context_manager.apply_targeted_change(
                path="/agent_communications",
                value={
                    "source": source_agent_type,
                    "target": target_agent_type,
                    "message_fragment": message[:50],
                    "response_fragment": response.final_output[:50],
                    "timestamp": datetime.now().isoformat()
                },
                operation="add"
            )
            
            # Return the response with metadata
            return {
                "source_agent": source_agent_type,
                "target_agent": target_agent_type,
                "original_message": message,
                "response": response.final_output,
                "memory_id": memory_id,
                "success": True,
                "context_version": comprehensive_context.get("version", None)
            }
        except Exception as e:
            logger.error(f"Error in agent communication: {e}", exc_info=True)
            
            # Record the error as a memory
            await memory_manager.add_memory(
                content=f"Error in communication from {source_agent_type} to {target_agent_type}: {str(e)}",
                memory_type="error",
                importance=0.5,
                tags=[source_agent_type, target_agent_type, "error", "communication"],
                metadata={
                    "source_agent": source_agent_type,
                    "target_agent": target_agent_type,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return {
                "source_agent": source_agent_type,
                "target_agent": target_agent_type,
                "original_message": message,
                "error": str(e),
                "success": False
            }
    finally:
        # Stop the timer
        if 'timer_id' in locals() and 'performance_monitor' in locals():
            performance_monitor.stop_timer(timer_id)

# ----- API Interface Functions -----

@track_performance("get_agent_recommendations")
async def get_agent_recommendations(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Get recommendations from all specialized agents regarding the current story state.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary with recommendations from each agent
    """
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("get_agent_recommendations")
    
    try:
        # Initialize comprehensive context components
        context_service = await get_context_service(user_id, conversation_id)
        memory_manager = await get_memory_manager(user_id, conversation_id)
        vector_service = await get_vector_service(user_id, conversation_id)
        
        # Try to initialize narrative manager for progressive summarization
        narrative_manager = None
        try:
            narrative_manager = RPGNarrativeManager(
                user_id=user_id,
                conversation_id=conversation_id,
                db_connection_string=None  # Will be initialized from config
            )
            await narrative_manager.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize narrative manager: {e}")
        
        # Get comprehensive context with vector search
        comprehensive_context = await context_service.get_context(
            input_text="story state recommendations",
            use_vector_search=True,
            context_budget=context_service.config.get_token_budget("recommendations")
        )
        
        # Initialize specialized agents
        specialized_agents = initialize_specialized_agents()
        
        # Get current story state
        director_agent, director_context = await initialize_story_director(user_id, conversation_id)
        
        # Build an enhanced prompt with comprehensive context
        story_state_prompt = f"""
        Analyze the current state of the story and provide a comprehensive overview of:
        1. The current narrative stage ({comprehensive_context.get('narrative_stage', {}).get('name', 'Unknown')})
        2. Active conflicts and their status ({len(comprehensive_context.get('conflicts', []))})
        3. Key NPC relationships ({len(comprehensive_context.get('npcs', []))})
        4. Player stats and their implications
        5. Recent significant events
        
        Format this as a structured summary of the current story state.
        """
        
        with trace(workflow_name="StoryStateAnalysis", group_id=f"user_{user_id}"):
            story_state = await Runner.run(
                director_agent,
                story_state_prompt,
                context=director_context
            )
        
        # Track token usage
        if hasattr(story_state, 'raw_responses') and story_state.raw_responses:
            for resp in story_state.raw_responses:
                if hasattr(resp, 'usage'):
                    performance_monitor.record_token_usage(resp.usage.total_tokens)
        
        # Get optimal context in parallel for each agent
        if narrative_manager:
            optimal_contexts = {}
            for agent_type in specialized_agents.keys():
                try:
                    optimal_contexts[agent_type] = asyncio.create_task(
                        narrative_manager.get_optimal_narrative_context(
                            query=f"{agent_type} recommendation for current story state",
                            max_tokens=800
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error getting optimal context for {agent_type}: {e}")
        
        # Gather recommendations from each specialized agent
        recommendations = {}
        tasks = []
        
        for agent_type, agent in specialized_agents.items():
            # Create appropriate context for each agent
            if agent_type == "conflict_analyst":
                agent_context = ConflictAnalystContext(
                    user_id=user_id,
                    conversation_id=conversation_id
                )
            elif agent_type == "narrative_crafter":
                agent_context = NarrativeCrafterContext(
                    user_id=user_id,
                    conversation_id=conversation_id
                )
            else:
                # Generic context for other agent types
                from story_agent.specialized_agents import AgentContext
                agent_context = AgentContext(
                    user_id=user_id,
                    conversation_id=conversation_id
                )
            
            # Get semantically relevant memories for each agent
            relevant_memories_task = asyncio.create_task(
                memory_manager.search_memories(
                    query_text=f"{agent_type} recommendation",
                    limit=3,
                    use_vector=True
                )
            )
            
            # Wait for memories
            relevant_memories = await relevant_memories_task
            
            memory_text = ""
            for memory in relevant_memories:
                if hasattr(memory, 'content'):
                    memory_text += f"- {memory.content[:150]}...\n"
                elif isinstance(memory, dict) and 'content' in memory:
                    memory_text += f"- {memory['content'][:150]}...\n"
            
            # Get optimal context for this agent if available
            narrative_context_text = ""
            if narrative_manager and agent_type in optimal_contexts:
                try:
                    narrative_context_data = await optimal_contexts[agent_type]
                    
                    for event in narrative_context_data.get("relevant_events", []):
                        if "content" in event:
                            narrative_context_text += f"- {event['content'][:150]}...\n"
                except Exception as e:
                    logger.warning(f"Error retrieving optimal context for {agent_type}: {e}")
            
            # Create recommendation prompt for each agent with rich context
            agent_prompt = f"""
            Based on the current story state:
            
            {story_state.final_output}
            
            Relevant memories:
            {memory_text}
            
            {narrative_context_text}
            
            Current narrative stage: {comprehensive_context.get('narrative_stage', {}).get('name', 'Unknown')}
            
            As the {agent_type.replace('_', ' ').title()}, what do you recommend for advancing the story?
            
            Focus on your specific area of expertise and provide clear, actionable recommendations.
            """
            
            # Create task for this agent
            async def get_agent_recommendation(agent_name, agent_obj, context, prompt):
                try:
                    with trace(workflow_name=f"{agent_name.capitalize()}Recommendation", group_id=f"user_{user_id}"):
                        result = await Runner.run(
                            agent_obj,
                            prompt,
                            context=context
                        )
                    
                    # Track token usage
                    if hasattr(result, 'raw_responses') and result.raw_responses:
                        for resp in result.raw_responses:
                            if hasattr(resp, 'usage'):
                                performance_monitor.record_token_usage(resp.usage.total_tokens)
                    
                    return agent_name, result.final_output
                except Exception as e:
                    logger.error(f"Error getting recommendation from {agent_name}: {e}", exc_info=True)
                    return agent_name, f"Error: {str(e)}"
            
            tasks.append(
                asyncio.create_task(
                    get_agent_recommendation(agent_type, agent, agent_context, agent_prompt)
                )
            )
        
        # Wait for all recommendations
        results = await asyncio.gather(*tasks)
        
        # Compile results
        for agent_name, recommendation in results:
            recommendations[agent_name] = recommendation
        
        # Store this as a memory in both systems
        memory_id = await memory_manager.add_memory(
            content=f"Generated agent recommendations for story state",
            memory_type="recommendation_generation",
            importance=0.6,
            tags=["recommendations", "story_state"],
            metadata={
                "agents": list(recommendations.keys()),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Add to narrative manager for progressive summarization
        if narrative_manager:
            try:
                summary_content = f"Agent recommendations for story state:\n"
                for agent_type, rec in recommendations.items():
                    summary_content += f"- {agent_type}: {rec[:100]}...\n"
                
                await narrative_manager.add_interaction(
                    content=summary_content,
                    importance=0.6,
                    tags=["recommendations", "story_state"]
                )
            except Exception as narrative_error:
                logger.warning(f"Error adding to narrative manager: {narrative_error}")
        
        # Return compiled recommendations with rich metadata
        return {
            "story_state": story_state.final_output,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": performance_monitor.get_metrics(),
            "context_version": comprehensive_context.get("version", None),
            "memory_id": memory_id
        }
    except Exception as e:
        logger.error(f"Error getting agent recommendations: {e}", exc_info=True)
        
        # Try to record the error as a memory
        try:
            memory_manager = await get_memory_manager(user_id, conversation_id)
            await memory_manager.add_memory(
                content=f"Error getting agent recommendations: {str(e)}",
                memory_type="error",
                importance=0.5,
                tags=["error", "recommendations"],
                metadata={
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as mem_error:
            logger.error(f"Failed to record error in memory system: {mem_error}")
            
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    finally:
        # Stop the timer
        if 'timer_id' in locals() and 'performance_monitor' in locals():
            performance_monitor.stop_timer(timer_id)
