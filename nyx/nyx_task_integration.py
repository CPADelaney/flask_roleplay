"""
Integration of creative task and activity recommendation agents into the Nyx workflow.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from story_agents.creative_task_agent import CreativeTaskGenerator, CreativeTask
from story_agents.activity_recommender import ActivityRecommender, ActivityRecommendation
from agents import function_tool
from nyx.nyx_agent_sdk import NarrativeResponse

logger = logging.getLogger(__name__)

class NyxTaskIntegration:
    """Integrates task and activity agents with Nyx's workflow"""
    
    def __init__(self):
        self.task_generator = CreativeTaskGenerator()
        self.activity_recommender = ActivityRecommender()
    
    @function_tool
    async def generate_creative_task(
        self,
        ctx,
        npc_id: str,
        scenario_id: str,
        intensity_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a creative task for the current scenario.
        
        Args:
            npc_id: ID of the NPC giving the task
            scenario_id: Current scenario ID
            intensity_level: Optional override for task intensity (1-5)
        """
        try:
            # Generate task
            task = self.task_generator.generate_task(npc_id, scenario_id)
            
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
            logger.error(f"Error generating creative task: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @function_tool
    async def recommend_activities(
        self,
        ctx,
        scenario_id: str,
        npc_ids: List[str],
        available_activities: List[Dict],
        num_recommendations: int = 2
    ) -> Dict[str, Any]:
        """
        Get activity recommendations for the current scene.
        
        Args:
            scenario_id: Current scenario ID
            npc_ids: List of present NPC IDs
            available_activities: List of available activities
            num_recommendations: Number of recommendations to return (default 2)
        """
        try:
            # Get recommendations
            recommendations = self.activity_recommender.recommend_activities(
                scenario_id,
                npc_ids,
                available_activities,
                num_recommendations
            )
            
            # Format for return
            formatted_recommendations = []
            for rec in recommendations:
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
            logger.error(f"Error getting activity recommendations: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @function_tool
    async def enhance_narrative_with_task(
        self,
        ctx,
        narrative_response: NarrativeResponse,
        task: CreativeTask
    ) -> NarrativeResponse:
        """
        Enhance Nyx's narrative response with task information.
        
        Args:
            narrative_response: Original narrative response
            task: Generated creative task
        """
        # Add task information to narrative
        task_narrative = (
            f"\n\n{task.npc_involvement}\n"
            f"Task: {task.title}\n"
            f"{task.description}\n"
            f"Duration: {task.duration}\n"
            f"Success Criteria: {task.success_criteria}\n"
            f"Reward: {task.reward_type}"
        )
        
        narrative_response.narrative += task_narrative
        
        # Adjust tension level based on task difficulty
        narrative_response.tension_level = max(
            narrative_response.tension_level,
            task.difficulty * 2  # Scale 1-5 to 1-10
        )
        
        return narrative_response
    
    @function_tool
    async def enhance_narrative_with_activities(
        self,
        ctx,
        narrative_response: NarrativeResponse,
        recommendations: List[ActivityRecommendation]
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
            if rec.activity_name.lower() == "none":
                activities_narrative += f"\n{i}. Continue with current activity"
            else:
                activities_narrative += (
                    f"\n{i}. {rec.activity_name}"
                    f"\n   Duration: {rec.estimated_duration}"
                    f"\n   Participants: {', '.join(rec.participating_npcs)}"
                    f"\n   Reasoning: {rec.reasoning}"
                )
        
        narrative_response.narrative += activities_narrative
        
        return narrative_response

# Example usage in Nyx's workflow:
"""
# In nyx_agent.py or similar:

async def process_user_input(user_id: int, conversation_id: int, user_input: str) -> Dict[str, Any]:
    # Initialize integration
    task_integration = NyxTaskIntegration()
    
    # Generate base narrative response
    narrative_response = await generate_base_response(user_input)
    
    # If appropriate, generate task
    if should_generate_task(context):
        task_result = await task_integration.generate_creative_task(
            npc_id=active_npc_id,
            scenario_id=current_scenario_id
        )
        if task_result["success"]:
            narrative_response = await task_integration.enhance_narrative_with_task(
                narrative_response,
                task_result["task"]
            )
    
    # If appropriate, recommend activities
    if should_recommend_activities(context):
        activity_result = await task_integration.recommend_activities(
            scenario_id=current_scenario_id,
            npc_ids=present_npc_ids,
            available_activities=get_available_activities()
        )
        if activity_result["success"]:
            narrative_response = await task_integration.enhance_narrative_with_activities(
                narrative_response,
                activity_result["recommendations"]
            )
    
    return narrative_response
""" 