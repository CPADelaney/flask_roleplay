# nyx/nyx_task_integration.py

"""
Integration of creative task and activity recommendation agents into the Nyx workflow.

This module integrates the OpenAI Agents-based task generation and activity recommendation
systems with Nyx's narrative workflow.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json

# Updated imports for Agents SDK
from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field

# Import the task agent and recommendation agent
from story_agent.creative_task_agent import femdom_task_agent, CreativeTask
from story_agent.activity_recommender import activity_recommender_agent, ActivityRecommendations, ActivityRecommendation

logger = logging.getLogger(__name__)

class NarrativeResponse(BaseModel):
    """Structured output for Nyx's narrative responses"""
    narrative: str = Field(..., description="The main narrative response as Nyx")
    tension_level: int = Field(0, description="Current narrative tension level (0-10)")
    generate_image: bool = Field(False, description="Whether an image should be generated for this scene")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    environment_description: Optional[str] = Field(None, description="Updated environment description if changed")
    time_advancement: bool = Field(False, description="Whether time should advance after this interaction")
    

class NyxTaskIntegration:
    """Integrates task and activity agents with Nyx's workflow using the OpenAI Agents SDK"""
    
    _instances = {}
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'NyxTaskIntegration':
        """
        Get or create a NyxTaskIntegration instance for the specified user and conversation.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            NyxTaskIntegration instance
        """
        # Create a unique key for this user/conversation combination
        key = f"{user_id}:{conversation_id}"
        
        # Check if an instance already exists for this key
        if key not in cls._instances:
            # Create a new instance if none exists
            cls._instances[key] = cls()
            
        return cls._instances[key]
    
    
    def __init__(self):
        # No need to instantiate the agents - we'll use the pre-defined ones from the modules
        pass
    
    @function_tool
    async def generate_creative_task(
        self,
        ctx,
        npc_id: int,
        scenario_id: str,
        intensity_level: Optional[int] = None,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a creative task for the current scenario using the agent-based approach.
        
        Args:
            npc_id: ID of the NPC giving the task
            scenario_id: Current scenario ID
            intensity_level: Optional override for task intensity (1-5)
            user_id: User ID (extracted from context if not provided)
            conversation_id: Conversation ID (extracted from context if not provided)
        """
        try:
            # Extract context if not provided
            if user_id is None:
                user_id = ctx.context.user_id
            if conversation_id is None:
                conversation_id = ctx.context.conversation_id
            
            # Create the prompt for the task agent
            user_prompt = (
                f"Generate a creative task for NPC ID {npc_id} in scenario {scenario_id}.\n"
                f"user_id={user_id}, conversation_id={conversation_id}"
            )
            
            # If intensity level is specified, add it to the prompt
            if intensity_level is not None:
                user_prompt += f", intensity_level={intensity_level}"
            
            # Run the agent to generate a task
            result = await Runner.run(
                starting_agent=femdom_task_agent,
                input=user_prompt
            )
            
            # The result.final_output will be a CreativeTask object
            task = result.final_output
            
            # Override intensity if specified
            if intensity_level is not None:
                task.difficulty = max(1, min(5, intensity_level))
            
            # Format for return
            return {
                "success": True,
                "task": {
                    "title": task.title,
                    "description": task.description,
                    "duration": task.duration,
                    "difficulty": task.difficulty,
                    "required_items": task.required_items,
                    "success_criteria": task.success_criteria,
                    "reward_type": task.reward_type,
                    "npc_involvement": task.npc_involvement,
                    "task_type": task.task_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating creative task: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @function_tool
    async def recommend_activities(
        self,
        ctx,
        scenario_id: str,
        npc_ids: List[int],
        num_recommendations: int = 2,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get activity recommendations for the current scene using the agent-based approach.
        
        Args:
            scenario_id: Current scenario ID
            npc_ids: List of present NPC IDs
            num_recommendations: Number of recommendations to return (default 2)
            user_id: User ID (extracted from context if not provided)
            conversation_id: Conversation ID (extracted from context if not provided)
        """
        try:
            # Extract context if not provided
            if user_id is None:
                user_id = ctx.context.user_id
            if conversation_id is None:
                conversation_id = ctx.context.conversation_id
            
            # Call the activity recommender function from the module
            # This already handles the Agent interaction
            recommendations = await recommend_activities(
                user_id=user_id,
                conversation_id=conversation_id,
                scenario_id=scenario_id,
                npc_ids=npc_ids,
                num_recommendations=num_recommendations
            )
            
            # Extract the recommendations list from the returned object
            formatted_recommendations = []
            for rec in recommendations.recommendations:
                formatted_recommendations.append({
                    "activity_name": rec.activity_name,
                    "confidence_score": rec.confidence_score,
                    "reasoning": rec.reasoning,
                    "participating_npcs": rec.participating_npcs,
                    "estimated_duration": rec.estimated_duration,
                    "prerequisites": rec.prerequisites,
                    "expected_outcomes": rec.expected_outcomes
                })
            
            return {
                "success": True,
                "recommendations": formatted_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting activity recommendations: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @function_tool
    async def enhance_narrative_with_task(
        self,
        ctx,
        narrative_response: NarrativeResponse,
        task: Dict[str, Any]
    ) -> NarrativeResponse:
        # We do a local import to avoid circular import at runtime:
        # Add task information to narrative
        task_narrative = (
            f"\n\n{task['npc_involvement']}\n"
            f"Task: {task['title']}\n"
            f"{task['description']}\n"
            f"Duration: {task['duration']}\n"
            f"Success Criteria: {task['success_criteria']}\n"
            f"Reward: {task['reward_type']}"
        )
        
        narrative_response.narrative += task_narrative
        
        # Adjust tension level based on task difficulty
        narrative_response.tension_level = max(
            narrative_response.tension_level,
            task['difficulty'] * 2  # Scale 1-5 to 1-10
        )
        
        return narrative_response
    
    @function_tool
    async def enhance_narrative_with_activities(
        self,
        ctx,
        narrative_response: NarrativeResponse,
        recommendations: List[Dict[str, Any]]
    ) -> NarrativeResponse:
        """
        Enhance Nyx's narrative response with activity recommendations.
        
        Args:
            narrative_response: Original narrative response
            recommendations: List of activity recommendations
        """
        # Add activity recommendations to narrative
        activities_narrative = "\n\nSuggested activities:"
        
        for i, rec in enumerate(recommendations, 1):
            if rec["activity_name"].lower() == "none":
                activities_narrative += f"\n{i}. Continue with current activity"
            else:
                activities_narrative += (
                    f"\n{i}. {rec['activity_name']}"
                    f"\n   Duration: {rec['estimated_duration']}"
                    f"\n   Participants: {', '.join(rec['participating_npcs'])}"
                    f"\n   Reasoning: {rec['reasoning']}"
                )
        
        narrative_response.narrative += activities_narrative
        
        return narrative_response

    @function_tool
    async def run_activity_agent_directly(
        self,
        ctx,
        user_prompt: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the activity recommendation agent directly with a custom prompt.
        Useful for debugging or custom scenarios.
        
        Args:
            user_prompt: The prompt to send to the agent
            user_id: User ID (extracted from context if not provided)
            conversation_id: Conversation ID (extracted from context if not provided)
        """
        try:
            # Extract context if not provided
            if user_id is None:
                user_id = ctx.context.user_id
            if conversation_id is None:
                conversation_id = ctx.context.conversation_id
                
            # Add user and conversation IDs to the prompt
            full_prompt = f"{user_prompt}\nuser_id={user_id}, conversation_id={conversation_id}"
            
            # Run the agent
            result = await Runner.run(
                starting_agent=activity_recommender_agent,
                input=full_prompt
            )
            
            # Return the formatted recommendations
            recommendations = result.final_output
            return {
                "success": True,
                "recommendations": [rec.dict() for rec in recommendations.recommendations]
            }
        except Exception as e:
            logger.error(f"Error running activity agent directly: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @function_tool
    async def run_task_agent_directly(
        self,
        ctx,
        user_prompt: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the task generation agent directly with a custom prompt.
        Useful for debugging or custom scenarios.
        
        Args:
            user_prompt: The prompt to send to the agent
            user_id: User ID (extracted from context if not provided)
            conversation_id: Conversation ID (extracted from context if not provided)
        """
        try:
            # Extract context if not provided
            if user_id is None:
                user_id = ctx.context.user_id
            if conversation_id is None:
                conversation_id = ctx.context.conversation_id
                
            # Add user and conversation IDs to the prompt
            full_prompt = f"{user_prompt}\nuser_id={user_id}, conversation_id={conversation_id}"
            
            # Run the agent
            result = await Runner.run(
                starting_agent=femdom_task_agent,
                input=full_prompt
            )
            
            # Return the task data
            task = result.final_output
            return {
                "success": True,
                "task": task.dict()
            }
        except Exception as e:
            logger.error(f"Error running task agent directly: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
