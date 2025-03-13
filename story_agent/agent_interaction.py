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

logger = logging.getLogger(__name__)

# ----- Orchestration Functions -----

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
    
    try:
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
        
        # Run tasks in parallel to improve performance
        conflict_task = asyncio.create_task(
            analyze_conflict(conflict_id, conflict_context)
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
            
            # Recent events for context
            cursor.execute("""
                SELECT entry_text
                FROM PlayerJournal
                WHERE user_id = %s AND conversation_id = %s
                ORDER BY timestamp DESC
                LIMIT 3
            """, (user_id, conversation_id))
            
            recent_events = [row[0] for row in cursor.fetchall()]
            
        finally:
            cursor.close()
            conn.close()
        
        # Wait for conflict analysis to complete
        conflict_analysis = await conflict_task
        
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
                "director_metrics": director_context.metrics if hasattr(director_context, "metrics") else {}
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
    
    try:
        # Initialize specialized agents
        specialized_agents = initialize_specialized_agents()
        
        # Initialize contexts for each agent
        director_context = StoryDirectorContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Prepare tasks to run in parallel
        tasks = []
        
        # 1. Get active conflicts
        async def get_conflicts():
            if director_context.conflict_manager:
                return await director_context.conflict_manager.get_active_conflicts()
            return []
        
        tasks.append(asyncio.create_task(get_conflicts()))
        
        # 2. Get key NPCs
        async def get_npcs():
            from story_agent.tools import get_key_npcs
            
            class MockContext:
                def __init__(self, real_context):
                    self.context = real_context
            
            mock_ctx = MockContext(director_context)
            return await get_key_npcs(mock_ctx, limit=5)
        
        tasks.append(asyncio.create_task(get_npcs()))
        
        # 3. Get current narrative stage
        from logic.narrative_progression import get_current_narrative_stage
        
        tasks.append(asyncio.create_task(get_current_narrative_stage(user_id, conversation_id)))
        
        # 4. Get player stats
        async def get_player_stats():
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT corruption, confidence, willpower, obedience, dependency, lust
                    FROM PlayerStats
                    WHERE user_id = %s AND conversation_id = %s AND player_name = 'Chase'
                """, (user_id, conversation_id))
                
                row = cursor.fetchone()
                
                if not row:
                    return {}
                
                return {
                    "corruption": row[0],
                    "confidence": row[1],
                    "willpower": row[2],
                    "obedience": row[3],
                    "dependency": row[4],
                    "lust": row[5]
                }
            finally:
                cursor.close()
                conn.close()
        
        tasks.append(asyncio.create_task(get_player_stats()))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        conflicts = results[0]
        npcs = results[1]
        narrative_stage = results[2]
        player_stats = results[3]
        
        # Now run specialized agents in sequence, each building on the previous
        
        # 1. First, analyze the most important conflict if any
        if conflicts:
            main_conflict = conflicts[0]  # Most important conflict
            conflict_context = ConflictAnalystContext(
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            conflict_analysis = await analyze_conflict(main_conflict["conflict_id"], conflict_context)
        else:
            conflict_analysis = {"analysis": "No active conflicts found."}
        
        # 2. Generate appropriate narrative element
        element_type = story_context.get("requested_element_type", "")
        if not element_type:
            # Choose appropriate element type based on narrative stage
            if narrative_stage:
                stage_name = narrative_stage.name
                if stage_name in ["Innocent Beginning", "First Doubts"]:
                    element_type = "subtle_manipulation"
                elif stage_name in ["Creeping Realization"]:
                    element_type = "revelation"
                else:
                    element_type = "explicit_control"
            else:
                element_type = "general"
        
        narrative_context = NarrativeCrafterContext(
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        npc_names = [npc["npc_name"] for npc in npcs] if npcs else ["Mistress"]
        
        narrative_element = await generate_narrative_element(
            element_type,
            {
                "npc_names": npc_names,
                "narrative_stage": narrative_stage.name if narrative_stage else "Innocent Beginning",
                "player_stats": player_stats,
                "conflict_analysis": conflict_analysis["analysis"],
                "requested_focus": story_context.get("narrative_focus", "")
            },
            narrative_context
        )
        
        # 3. Finally, use the Story Director to integrate everything
        director_agent, _ = await initialize_story_director(user_id, conversation_id)
        
        integration_prompt = f"""
        Create a compelling story beat that integrates the following elements:
        
        CURRENT NARRATIVE STAGE: {narrative_stage.name if narrative_stage else "Innocent Beginning"}
        
        CONFLICT ANALYSIS: 
        {conflict_analysis["analysis"]}
        
        NARRATIVE ELEMENT:
        {narrative_element["content"]}
        
        PLAYER STATS:
        {json.dumps(player_stats, indent=2)}
        
        KEY NPCs:
        {", ".join(npc_names)}
        
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
        
        # Return the final comprehensive story beat
        return {
            "story_beat": story_beat.final_output,
            "narrative_stage": narrative_stage.name if narrative_stage else "Innocent Beginning",
            "element_type": element_type,
            "conflict_analysis": conflict_analysis,
            "narrative_element": narrative_element,
            "execution_time": execution_time,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error generating comprehensive story beat: {e}", exc_info=True)
        return {
            "error": str(e),
            "success": False,
            "execution_time": time.time() - start_time
        }

# ----- Agent Communication Interface -----

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
    
    # Format the message to include source agent information
    formatted_message = f"""
    Message from {source_agent_type.replace('_', ' ').title()}:
    
    {message}
    
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

# ----- API Interface Functions -----

async def get_agent_recommendations(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Get recommendations from all specialized agents regarding the current story state.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary with recommendations from each agent
    """
    # Initialize specialized agents
    specialized_agents = initialize_specialized_agents()
    
    # Get current story state
    director_agent, director_context = await initialize_story_director(user_id, conversation_id)
    
    story_state_prompt = """
    Analyze the current state of the story and provide a comprehensive overview of:
    1. The current narrative stage
    2. Active conflicts and their status
    3. Key NPC relationships
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
        
        # Create recommendation prompt for each agent
        agent_prompt = f"""
        Based on the current story state:
        
        {story_state.final_output}
        
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
    
    # Return compiled recommendations
    return {
        "story_state": story_state.final_output,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    }
