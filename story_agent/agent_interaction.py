# story_agent/agent_interaction.py

"""
This module handles interactions between specialized agents in the Story Director system.
It provides orchestration for complex story development tasks that require input from
multiple specialized agents.
"""

import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

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

# NEW: Context system integration
from context.context_service import get_context_service, get_comprehensive_context
from context.memory_manager import get_memory_manager, Memory
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager
from context.performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

# NEW: Progressive summarization integration
from story_agent.progressive_summarization import (
    RPGNarrativeManager,
    SummaryLevel
)

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
    
    # NEW: Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("conflict_analysis_orchestration")
    
    try:
        # NEW: Get context service
        context_service = await get_context_service(user_id, conversation_id)
        
        # First, create all necessary contexts
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
        
        # NEW: Initialize context components
        await director_context.initialize_context_components()
        
        # Run tasks in parallel to improve performance
        conflict_task = asyncio.create_task(
            analyze_conflict(conflict_id, conflict_context)
        )
        
        # NEW: Get comprehensive context while conflict analysis is running
        comprehensive_context_task = asyncio.create_task(
            context_service.get_context(
                input_text=f"conflict analysis for conflict {conflict_id}",
                use_vector_search=True
            )
        )
        
        # While conflict analysis is running, get NPCs and current stage for narrative
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get conflict details for narrative context
            cursor.execute("""
                SELECT conflict_name, faction_a_name, faction_b_name
                FROM Conflicts
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (conflict_id, user_id, conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Conflict {conflict_id} not found")
            
            conflict_name, faction_a, faction_b = row
            
            # Get involved NPCs
            cursor.execute("""
                SELECT n.npc_name
                FROM ConflictNPCs c
                JOIN NPCStats n ON c.npc_id = n.npc_id
                WHERE c.conflict_id = %s
                ORDER BY c.influence_level DESC
                LIMIT 3
            """, (conflict_id,))
            
            npc_names = [row[0] for row in cursor.fetchall()]
            
            from logic.narrative_progression import get_current_narrative_stage
            
            # Get current narrative stage
            narrative_stage = await get_current_narrative_stage(user_id, conversation_id)
            stage_name = narrative_stage.name if narrative_stage else "Innocent Beginning"
            
            # NEW: Get relevant memories using vector search instead of recent events
            memory_manager = await get_memory_manager(user_id, conversation_id)
            memories = await memory_manager.search_memories(
                query_text=f"conflict {conflict_name} {faction_a} {faction_b}",
                limit=5,
                use_vector=True
            )
            
            recent_events = []
            for memory in memories:
                if hasattr(memory, 'content'):
                    recent_events.append(memory.content)
                elif isinstance(memory, dict) and 'content' in memory:
                    recent_events.append(memory['content'])
            
        finally:
            cursor.close()
            conn.close()
        
        # Wait for conflict analysis to complete
        conflict_analysis = await conflict_task
        
        # Wait for comprehensive context
        comprehensive_context = await comprehensive_context_task
        
        # NEW: Check for narrative manager availability
        narrative_manager = None
        try:
            narrative_manager = RPGNarrativeManager(
                user_id=user_id,
                conversation_id=conversation_id,
                db_connection_string=get_db_connection()
            )
            await narrative_manager.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize narrative manager: {e}")
        
        # Generate narrative based on conflict analysis
        narrative_task = asyncio.create_task(
            generate_narrative_element(
                "conflict_narrative",
                {
                    "npc_names": npc_names,
                    "narrative_stage": stage_name,
                    "recent_events": "\n".join(recent_events),
                    "conflict_name": conflict_name,
                    "faction_a": faction_a,
                    "faction_b": faction_b,
                    "conflict_analysis": conflict_analysis["analysis"]
                },
                narrative_context
            )
        )
        
        # Initialize the Story Director to integrate everything
        director_agent, _ = await initialize_story_director(user_id, conversation_id)
        
        # Wait for narrative generation to complete
        narrative_element = await narrative_task
        
        # NEW: Store this as a memory
        if narrative_manager:
            await narrative_manager.add_revelation(
                content=narrative_element["content"],
                revelation_type="conflict_narrative",
                importance=0.8,
                tags=["conflict", conflict_name, "narrative"]
            )
        else:
            # Fallback to memory manager
            await memory_manager.add_memory(
                content=narrative_element["content"],
                memory_type="narrative_element",
                importance=0.8,
                tags=["conflict", conflict_name, "narrative"],
                metadata={
                    "conflict_id": conflict_id,
                    "conflict_name": conflict_name,
                    "source": "orchestration"
                }
            )
        
        # Integration prompt for the Story Director
        integration_prompt = f"""
        I need you to integrate conflict analysis and a narrative element into a comprehensive story update.
        
        CONFLICT ANALYSIS:
        {conflict_analysis["analysis"]}
        
        NARRATIVE ELEMENT:
        {narrative_element["content"]}
        
        Please create a comprehensive story update that:
        1. Summarizes the current state of the conflict
        2. Integrates the narrative element to advance the story
        3. Suggests clear next steps for the player
        4. Hints at possible consequences of different choices
        
        Format your response as a structured story update with clear sections.
        """
        
        # Run the Story Director to integrate everything
        with trace(workflow_name="StoryOrchestration", group_id=f"conflict_{conflict_id}"):
            integration_result = await Runner.run(
                director_agent,
                integration_prompt,
                context=director_context
            )
        
        execution_time = time.time() - start_time
        
        # NEW: Record performance metrics
        performance_monitor.stop_timer(timer_id)
        
        # Return the final orchestrated result
        return {
            "conflict_id": conflict_id,
            "conflict_analysis": conflict_analysis,
            "narrative_element": narrative_element,
            "integrated_update": integration_result.final_output,
            "execution_time": execution_time,
            "metrics": {
                "conflict_analysis_metrics": conflict_context.get_metrics() if hasattr(conflict_context, "get_metrics") else {},
                "narrative_metrics": narrative_context.get_metrics() if hasattr(narrative_context, "get_metrics") else {},
                "director_metrics": director_context.metrics if hasattr(director_context, "metrics") else {},
                "performance": performance_monitor.get_metrics()
            }
        }
    except Exception as e:
        logger.error(f"Error in orchestrate_conflict_analysis_and_narrative: {e}", exc_info=True)
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
    
    # NEW: Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("story_beat_generation")
    
    try:
        # NEW: Initialize context components
        context_service = await get_context_service(user_id, conversation_id)
        memory_manager = await get_memory_manager(user_id, conversation_id)
        vector_service = await get_vector_service(user_id, conversation_id)
        
        # Initialize specialized agents
        specialized_agents = initialize_specialized_agents()
        
        # Initialize contexts for each agent
        director_context = StoryDirectorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        await director_context.initialize_context_components()
        
        # NEW: Get comprehensive context with vector search
        query = story_context.get("narrative_focus", "current story state")
        comprehensive_context = await context_service.get_context(
            input_text=query,
            use_vector_search=True
        )
        
        # Extract key information from context
        narrative_stage_info = comprehensive_context.get("narrative_stage", {})
        narrative_stage = narrative_stage_info.get("name", "Unknown")
        npcs = comprehensive_context.get("npcs", [])
        npc_names = [npc.get("npc_name", "Unknown") for npc in npcs[:5]]
        
        # NEW: Get relevant memories using vector search
        relevant_memories = await memory_manager.search_memories(
            query_text=query,
            limit=5,
            use_vector=True
        )
        
        # Format memories for context
        memory_content = []
        for memory in relevant_memories:
            if hasattr(memory, 'content'):
                memory_content.append(memory.content)
            elif isinstance(memory, dict) and 'content' in memory:
                memory_content.append(memory['content'])
        
        # Get active conflicts
        active_conflicts = []
        if hasattr(director_context, 'conflict_manager'):
            active_conflicts = await director_context.conflict_manager.get_active_conflicts()
        
        # Use conflict analyst if active conflicts exist
        conflict_analysis = {"analysis": "No active conflicts found."}
        if active_conflicts:
            main_conflict = active_conflicts[0]  # Most important conflict
            conflict_context = ConflictAnalystContext(
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            conflict_analysis = await analyze_conflict(main_conflict["conflict_id"], conflict_context)
        
        # Generate appropriate narrative element
        element_type = story_context.get("requested_element_type", "")
        if not element_type:
            # Choose appropriate element type based on narrative stage
            if narrative_stage in ["Innocent Beginning", "First Doubts"]:
                element_type = "subtle_manipulation"
            elif narrative_stage in ["Creeping Realization"]:
                element_type = "revelation"
            else:
                element_type = "explicit_control"
        
        narrative_context = NarrativeCrafterContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # NEW: Get player stats from context
        player_stats = {}
        if "player_stats" in comprehensive_context:
            player_stats = comprehensive_context["player_stats"]
        
        narrative_element = await generate_narrative_element(
            element_type,
            {
                "npc_names": npc_names,
                "narrative_stage": narrative_stage,
                "player_stats": player_stats,
                "conflict_analysis": conflict_analysis["analysis"],
                "requested_focus": story_context.get("narrative_focus", ""),
                "recent_memories": "\n".join(memory_content[:3])  # Add memory context
            },
            narrative_context
        )
        
        # NEW: Store this narrative element as a memory
        await memory_manager.add_memory(
            content=narrative_element["content"],
            memory_type="narrative_element",
            importance=0.8,
            tags=[element_type, "story_beat"],
            metadata={
                "element_type": element_type,
                "narrative_stage": narrative_stage,
                "source": "story_beat_generation"
            }
        )
        
        # Try to use narrative manager if available
        try:
            narrative_manager = RPGNarrativeManager(
                user_id=user_id,
                conversation_id=conversation_id,
                db_connection_string=get_db_connection()
            )
            await narrative_manager.initialize()
            
            # Add to narrative manager
            await narrative_manager.add_revelation(
                content=narrative_element["content"],
                revelation_type=element_type,
                importance=0.8,
                tags=[element_type, "story_beat"]
            )
        except Exception as e:
            logger.warning(f"Could not use narrative manager: {e}")
        
        # Use the Story Director to integrate everything
        director_agent, _ = await initialize_story_director(user_id, conversation_id)
        
        # NEW: Use context-aware prompt
        integration_prompt = f"""
        Create a compelling story beat that integrates the following elements:
        
        CURRENT NARRATIVE STAGE: {narrative_stage}
        
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
        
        execution_time = time.time() - start_time
        
        # NEW: Track performance
        performance_monitor.stop_timer(timer_id)
        if hasattr(story_beat, 'raw_responses') and story_beat.raw_responses:
            for response in story_beat.raw_responses:
                if hasattr(response, 'usage'):
                    performance_monitor.record_token_usage(response.usage.total_tokens)
        
        # Return the final comprehensive story beat
        return {
            "story_beat": story_beat.final_output,
            "narrative_stage": narrative_stage,
            "element_type": element_type,
            "conflict_analysis": conflict_analysis,
            "narrative_element": narrative_element,
            "execution_time": execution_time,
            "success": True,
            "performance_metrics": performance_monitor.get_metrics()
        }
    except Exception as e:
        logger.error(f"Error generating comprehensive story beat: {e}", exc_info=True)
        # Stop the timer if still running
        if 'timer_id' in locals() and 'performance_monitor' in locals():
            performance_monitor.stop_timer(timer_id)
        return {
            "error": str(e),
            "success": False,
            "execution_time": time.time() - start_time
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
    
    # NEW: Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("agent_communication")
    
    try:
        # NEW: Get context components
        context_service = await get_context_service(user_id, conversation_id)
        memory_manager = await get_memory_manager(user_id, conversation_id)
        
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
        
        # NEW: Get relevant context from context service
        comprehensive_context = await context_service.get_context(
            input_text=message,
            use_vector_search=True
        )
        
        # NEW: Get relevant memories
        relevant_memories = await memory_manager.search_memories(
            query_text=message,
            limit=3,
            use_vector=True
        )
        
        # Format memories
        memory_text = ""
        for memory in relevant_memories:
            if hasattr(memory, 'content'):
                memory_text += f"- {memory.content[:200]}...\n"
            elif isinstance(memory, dict) and 'content' in memory:
                memory_text += f"- {memory['content'][:200]}...\n"
        
        # Format the message to include source agent information and context
        formatted_message = f"""
        Message from {source_agent_type.replace('_', ' ').title()}:
        
        {message}
        
        Relevant context:
        - Current narrative stage: {comprehensive_context.get('narrative_stage', {}).get('name', 'Unknown')}
        - Current location: {comprehensive_context.get('current_location', 'Unknown')}
        
        Relevant memories:
        {memory_text}
        
        Please respond with your thoughts and analysis based on your expertise as {target_agent_type.replace('_', ' ').title()}.
        """
        
        try:
            # Send the message to the target agent
            with trace(workflow_name="AgentCommunication", group_id=f"user_{user_id}"):
                response = await Runner.run(
                    target_agent,
                    formatted_message,
                    context=target_context
                )
            
            # NEW: Track token usage
            if hasattr(response, 'raw_responses') and response.raw_responses:
                for resp in response.raw_responses:
                    if hasattr(resp, 'usage'):
                        performance_monitor.record_token_usage(resp.usage.total_tokens)
            
            # NEW: Store this communication as a memory
            await memory_manager.add_memory(
                content=f"Communication from {source_agent_type} to {target_agent_type}: {message[:100]}...",
                memory_type="agent_communication",
                importance=0.4,
                tags=[source_agent_type, target_agent_type, "communication"],
                metadata={
                    "source_agent": source_agent_type,
                    "target_agent": target_agent_type,
                    "full_message": message
                }
            )
            
            # Return the response
            return {
                "source_agent": source_agent_type,
                "target_agent": target_agent_type,
                "original_message": message,
                "response": response.final_output,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in agent communication: {e}", exc_info=True)
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
    # NEW: Initialize performance monitoring
    performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
    timer_id = performance_monitor.start_timer("get_agent_recommendations")
    
    try:
        # NEW: Initialize context components
        context_service = await get_context_service(user_id, conversation_id)
        memory_manager = await get_memory_manager(user_id, conversation_id)
        
        # Get comprehensive context
        comprehensive_context = await context_service.get_context(
            input_text="story state recommendations",
            use_vector_search=True
        )
        
        # Initialize specialized agents
        specialized_agents = initialize_specialized_agents()
        
        # Get current story state
        director_agent, director_context = await initialize_story_director(user_id, conversation_id)
        
        # NEW: Use context service for better state information
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
        
        # NEW: Track token usage
        if hasattr(story_state, 'raw_responses') and story_state.raw_responses:
            for resp in story_state.raw_responses:
                if hasattr(resp, 'usage'):
                    performance_monitor.record_token_usage(resp.usage.total_tokens)
        
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
            
            # NEW: Include relevant memories in agent prompts
            relevant_memories = await memory_manager.search_memories(
                query_text=f"{agent_type} recommendation",
                limit=2,
                use_vector=True
            )
            
            memory_text = ""
            for memory in relevant_memories:
                if hasattr(memory, 'content'):
                    memory_text += f"- {memory.content[:150]}...\n"
                elif isinstance(memory, dict) and 'content' in memory:
                    memory_text += f"- {memory['content'][:150]}...\n"
            
            # Create recommendation prompt for each agent
            agent_prompt = f"""
            Based on the current story state:
            
            {story_state.final_output}
            
            Relevant memories:
            {memory_text}
            
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
        
        # NEW: Store this as a memory
        await memory_manager.add_memory(
            content=f"Generated agent recommendations for story state",
            memory_type="recommendation_generation",
            importance=0.6,
            tags=["recommendations", "story_state"],
            metadata={
                "agents": list(recommendations.keys()),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Return compiled recommendations
        return {
            "story_state": story_state.final_output,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": performance_monitor.get_metrics()
        }
    except Exception as e:
        logger.error(f"Error getting agent recommendations: {e}", exc_info=True)
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    finally:
        # Stop the timer
        if 'timer_id' in locals() and 'performance_monitor' in locals():
            performance_monitor.stop_timer(timer_id)
