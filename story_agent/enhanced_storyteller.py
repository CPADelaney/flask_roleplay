# story_agent/enhanced_storyteller.py

"""
Enhanced Storyteller Agent with Nyx Governance integration.

This version of the Storyteller Agent integrates with the central Nyx governance system
to ensure proper oversight and control over the storytelling process.
"""

import logging
import json
import asyncio
import time
import os
import asyncpg
from datetime import datetime

from agents import Agent, Runner, function_tool, handoff
from pydantic import BaseModel, Field

# Import base StorytellerAgent to extend
from story_agent.storyteller_agent import StorytellerAgent, NarrativeContext, NarrativeResponse

# Import governance integration
from nyx.enhanced_integration import (
    get_central_governance,
    process_story_beat_with_governance,
    broadcast_event_with_governance
)

from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

# Import other existing components
from logic.aggregator import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text
from logic.universal_updater import apply_universal_updates_async
from logic.gpt_image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from logic.time_cycle import get_current_time, should_advance_time, nightly_maintenance
from utils.performance import PerformanceTracker
from logic.conflict_system.conflict_integration import ConflictSystemIntegration

# Configuration
DB_DSN = os.getenv("DB_DSN")

logger = logging.getLogger(__name__)


class GovernedStorytellerAgent(StorytellerAgent):
    """
    Enhanced Storyteller Agent with Nyx governance integration.
    
    This class extends the base StorytellerAgent with governance functionality
    to ensure proper oversight and control over the storytelling process.
    """
    
    def __init__(self):
        # Initialize base class
        super().__init__()
        
        # Add governance-specific tools to the agent
        self.agent = Agent(
            name="EnhancedStoryManager",
            instructions="""
            You coordinate the storytelling process for a roleplaying game with subtle femdom elements
            under the governance of Nyx.
            
            Process player input, generate appropriate responses, and manage game state updates.
            
            Your job is to:
            1. Handle player input and determine the appropriate processing
            2. Coordinate NPC responses to player actions with governance oversight
            3. Manage time advancement based on player activities
            4. Generate narrative responses through the Narrator
            5. Update the game state through the Universal Updater
            6. Ensure all actions comply with Nyx's governance directives
            
            Maintain a balance between player agency and subtle narrative guidance.
            """,
            tools=[
                function_tool(self.get_aggregated_context),
                function_tool(self.process_npc_responses),
                function_tool(self.process_time_advancement),
                function_tool(self.generate_narrative_response),
                function_tool(self.apply_universal_updates),
                function_tool(self.generate_image_if_needed),
                function_tool(self.store_message),
                function_tool(self.check_governance_permission),  # New governance tool
                function_tool(self.report_action_to_governance)   # New governance tool
            ]
        )
    
    async def check_governance_permission(
        self,
        ctx,
        action_type: str,
        action_details: dict,
        agent_type: str = "storyteller"
    ) -> dict:
        """
        Check if an action is permitted by the governance system.
        
        Args:
            action_type: Type of action
            action_details: Details of the action
            agent_type: Type of agent (defaults to "storyteller")
            
        Returns:
            Dictionary with permission check results
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Get the governance system
            governance = await get_central_governance(user_id, conversation_id)
            
            # Check permission
            result = await governance.check_action_permission(
                agent_type=agent_type,
                agent_id=f"storyteller_{conversation_id}",
                action_type=action_type,
                action_details=action_details
            )
            
            return result
        except Exception as e:
            logger.error(f"Error checking governance permission: {e}")
            # Default to approved if there's an error
            return {
                "approved": True,
                "directive_applied": False,
                "reasoning": f"Error checking permission: {e}"
            }
    
    async def report_action_to_governance(
        self,
        ctx,
        action_type: str,
        action_description: str,
        result: dict,
        agent_type: str = "storyteller"
    ) -> dict:
        """
        Report an action and its result to the governance system.
        
        Args:
            action_type: Type of action
            action_description: Description of the action
            result: Result of the action
            agent_type: Type of agent (defaults to "storyteller")
            
        Returns:
            Dictionary with reporting results
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Get the governance system
            governance = await get_central_governance(user_id, conversation_id)
            
            # Report action
            report_result = await governance.process_agent_action_report(
                agent_type=agent_type,
                agent_id=f"storyteller_{conversation_id}",
                action={
                    "type": action_type,
                    "description": action_description
                },
                result=result
            )
            
            return report_result
        except Exception as e:
            logger.error(f"Error reporting action to governance: {e}")
            # Return basic success if there's an error
            return {
                "reported": True,
                "error": str(e)
            }
    
    async def process_npc_responses(self, ctx, user_input, activity_type="conversation", location=None):
        """
        Process NPC responses to player input with governance oversight.
        
        Args:
            user_input: Player's input text
            activity_type: Type of activity the player is performing
            location: Current location (optional)
            
        Returns:
            List of NPC responses
        """
        # Check permission with governance system
        permission = await self.check_governance_permission(
            ctx,
            action_type="process_npc_responses",
            action_details={
                "user_input": user_input,
                "activity_type": activity_type,
                "location": location
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"NPC responses not approved: {permission.get('reasoning')}")
            return []  # Return empty list if not approved
        
        # If there's an override action, use it
        if permission.get("override_action"):
            override = permission.get("override_action")
            user_input = override.get("user_input", user_input)
            activity_type = override.get("activity_type", activity_type)
            location = override.get("location", location)
        
        # Call original implementation
        npc_responses = await super().process_npc_responses(ctx, user_input, activity_type, location)
        
        # Report the action
        await self.report_action_to_governance(
            ctx,
            action_type="process_npc_responses",
            action_description=f"Processed NPC responses for input: {user_input[:50]}...",
            result={
                "response_count": len(npc_responses),
                "location": location
            }
        )
        
        return npc_responses
    
    async def process_time_advancement(self, ctx, activity_type="conversation", confirm_advance=False):
        """
        Process time advancement with governance oversight.
        
        Args:
            activity_type: Type of activity
            confirm_advance: Whether to confirm time advancement
            
        Returns:
            Time advancement results
        """
        # Check permission with governance system
        permission = await self.check_governance_permission(
            ctx,
            action_type="process_time_advancement",
            action_details={
                "activity_type": activity_type,
                "confirm_advance": confirm_advance
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Time advancement not approved: {permission.get('reasoning')}")
            # Return minimal result with no advancement
            return {
                "time_advanced": False,
                "would_advance": False,
                "periods": 0,
                "current_time": await self.get_current_game_time(ctx),
                "confirm_needed": False,
                "governance_blocked": True,
                "reason": permission.get("reasoning", "Not approved by governance")
            }
        
        # If there's an override action, use it
        if permission.get("override_action"):
            override = permission.get("override_action")
            activity_type = override.get("activity_type", activity_type)
            confirm_advance = override.get("confirm_advance", confirm_advance)
        
        # Call original implementation
        time_result = await super().process_time_advancement(ctx, activity_type, confirm_advance)
        
        # Report the action
        await self.report_action_to_governance(
            ctx,
            action_type="process_time_advancement",
            action_description=f"Processed time advancement for activity: {activity_type}",
            result={
                "time_advanced": time_result.time_advanced,
                "would_advance": time_result.would_advance,
                "periods": time_result.periods,
                "current_time": time_result.current_time
            }
        )
        
        return time_result
    
    async def generate_narrative_response(self, ctx, user_input, aggregator_data, npc_responses=None, time_result=None):
        """
        Generate a narrative response with governance oversight.
        
        Args:
            user_input: Player's input text
            aggregator_data: Aggregated game context
            npc_responses: List of NPC responses
            time_result: Time advancement results
            
        Returns:
            Narrative response
        """
        # Check permission with governance system
        permission = await self.check_governance_permission(
            ctx,
            action_type="generate_narrative_response",
            action_details={
                "user_input": user_input,
                "has_npc_responses": bool(npc_responses),
                "time_advanced": time_result.time_advanced if time_result else False
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Narrative response not approved: {permission.get('reasoning')}")
            # Create a minimal narrative response
            return NarrativeResponse(
                message=f"The system is currently unavailable: {permission.get('reasoning', 'Not approved by governance')}",
                generate_image=False,
                tension_level=0
            )
        
        # If there's an override or modification, apply it
        if permission.get("action_modifications"):
            mods = permission.get("action_modifications")
            if "input_modifiers" in mods:
                input_mods = mods["input_modifiers"]
                user_input = input_mods.get("user_input", user_input)
        
        # Call original implementation
        narrative_response = await super().generate_narrative_response(
            ctx, user_input, aggregator_data, npc_responses, time_result
        )
        
        # Report the action
        await self.report_action_to_governance(
            ctx,
            action_type="generate_narrative_response",
            action_description=f"Generated narrative response for input: {user_input[:50]}...",
            result={
                "message_length": len(narrative_response.message),
                "generate_image": narrative_response.generate_image,
                "tension_level": narrative_response.tension_level
            }
        )
        
        return narrative_response
    
    async def apply_universal_updates(self, ctx, narrative_response, additional_data=None):
        """
        Apply universal updates with governance oversight.
        
        Args:
            narrative_response: Narrative response
            additional_data: Additional data
            
        Returns:
            Update results
        """
        # Check permission with governance system
        permission = await self.check_governance_permission(
            ctx,
            action_type="apply_universal_updates",
            action_details={
                "narrative_length": len(narrative_response.message),
                "has_additional_data": bool(additional_data)
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Universal updates not approved: {permission.get('reasoning')}")
            return {
                "success": False,
                "governance_blocked": True,
                "reason": permission.get("reasoning", "Not approved by governance")
            }
        
        # Call original implementation
        update_result = await super().apply_universal_updates(ctx, narrative_response, additional_data)
        
        # Report the action
        await self.report_action_to_governance(
            ctx,
            action_type="apply_universal_updates",
            action_description="Applied universal updates based on narrative response",
            result=update_result
        )
        
        return update_result
    
    async def generate_image_if_needed(self, ctx, narrative_response):
        """
        Generate an image if needed with governance oversight.
        
        Args:
            narrative_response: Narrative response
            
        Returns:
            Image generation results
        """
        # If image generation not requested, don't check permission
        if not narrative_response.generate_image:
            return {"generated": False}
        
        # Check permission with governance system
        permission = await self.check_governance_permission(
            ctx,
            action_type="generate_image",
            action_details={
                "narrative_length": len(narrative_response.message),
                "image_prompt": narrative_response.image_prompt
            }
        )
        
        if not permission["approved"]:
            logger.warning(f"Image generation not approved: {permission.get('reasoning')}")
            return {
                "generated": False,
                "governance_blocked": True,
                "reason": permission.get("reasoning", "Not approved by governance")
            }
        
        # Call original implementation
        image_result = await super().generate_image_if_needed(ctx, narrative_response)
        
        # Report the action
        await self.report_action_to_governance(
            ctx,
            action_type="generate_image",
            action_description="Generated image for narrative response",
            result={
                "generated": image_result.get("generated", False),
                "has_url": "image_url" in image_result
            }
        )
        
        return image_result
    
    async def process_story_beat(self, user_id, conversation_id, user_input, data=None):
        """
        Process a complete story beat with governance oversight.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            user_input: Player's input text
            data: Additional data
            
        Returns:
            Complete response with narrative and updates
        """
        tracker = PerformanceTracker("story_beat")
        tracker.start_phase("initialization")
        
        # Set up context
        context = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "data": data or {}
        }
        
        tracker.end_phase()
        
        try:
            # 0. Get the governance system first
            tracker.start_phase("get_governance")
            governance = await get_central_governance(user_id, conversation_id)
            tracker.end_phase()
            
            # 1. Register with governance if needed
            tracker.start_phase("register_with_governance")
            if "storyteller" not in governance.registered_agents:
                await governance.register_agent(
                    agent_type="storyteller",
                    agent_instance=self,
                    agent_id=f"storyteller_{conversation_id}"
                )
            tracker.end_phase()
            
            # 2. Check high-level permission for this story beat
            tracker.start_phase("check_governance_permission")
            permission = await governance.check_action_permission(
                agent_type="storyteller",
                agent_id=f"storyteller_{conversation_id}",
                action_type="process_story_beat",
                action_details={
                    "user_input": user_input,
                    "has_data": bool(data)
                }
            )
            
            if not permission["approved"]:
                logger.warning(f"Story beat not approved: {permission.get('reasoning')}")
                return {
                    "error": permission.get("reasoning", "Not approved by governance"),
                    "message": "The system is currently unavailable.",
                    "governance_blocked": True
                }
            tracker.end_phase()
            
            # 3. Store user message
            tracker.start_phase("store_user_message")
            await self.store_message(None, "user", user_input)
            tracker.end_phase()
            
            # 4. Get aggregated context
            tracker.start_phase("get_context")
            aggregator_data = await self.get_aggregated_context(None, conversation_id)
            tracker.end_phase()
            
            # 5. Process NPC responses
            tracker.start_phase("npc_responses")
            current_location = aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown")
            npc_responses = await self.process_npc_responses(None, user_input, "conversation", current_location)
            tracker.end_phase()
            
            # 6. Process time advancement
            tracker.start_phase("time_advancement")
            confirm_advance = data.get("confirm_time_advance", False) if data else False
            time_result = await self.process_time_advancement(None, "conversation", confirm_advance)
            tracker.end_phase()
            
            # 7. Generate narrative response
            tracker.start_phase("narrative_response")
            narrative_response = await self.generate_narrative_response(
                None,
                user_input,
                aggregator_data,
                npc_responses,
                time_result
            )
            tracker.end_phase()
            
            # 8. Store assistant message
            tracker.start_phase("store_assistant_message")
            await self.store_message(None, "Nyx", narrative_response.message)
            tracker.end_phase()
            
            # 9. Apply universal updates
            tracker.start_phase("universal_updates")
            update_result = await self.apply_universal_updates(None, narrative_response)
            tracker.end_phase()
            
            # 10. Generate image if needed
            tracker.start_phase("image_generation")
            image_result = await self.generate_image_if_needed(None, narrative_response)
            tracker.end_phase()
            
            # 11. Report completion to governance
            tracker.start_phase("report_to_governance")
            await governance.process_agent_action_report(
                agent_type="storyteller",
                agent_id=f"storyteller_{conversation_id}",
                action={
                    "type": "process_story_beat",
                    "description": f"Processed story beat for input: {user_input[:50]}..."
                },
                result={
                    "npc_responses": len(npc_responses),
                    "time_advanced": time_result.time_advanced if hasattr(time_result, "time_advanced") else False,
                    "image_generated": image_result.get("generated", False) if image_result else False,
                    "narrative_length": len(narrative_response.message)
                }
            )
            tracker.end_phase()
            
            # 12. Build and return the final response
            tracker.start_phase("build_response")
            response = {
                "message": narrative_response.message,
                "tension_level": narrative_response.tension_level,
                "time_result": time_result.dict() if hasattr(time_result, "dict") else time_result,
                "confirm_needed": time_result.would_advance and not confirm_advance if hasattr(time_result, "would_advance") else False,
                "npc_responses": [resp.dict() for resp in npc_responses],
                "performance_metrics": tracker.get_metrics(),
                "governance_approved": True
            }
            
            if hasattr(narrative_response, "environment_update") and narrative_response.environment_update:
                response["environment_update"] = narrative_response.environment_update
            
            if image_result and image_result.get("generated"):
                response["image"] = {
                    "image_url": image_result["image_url"],
                    "prompt_used": image_result.get("prompt_used", "")
                }
                
            tracker.end_phase()
            
            return response
            
        except Exception as e:
            if tracker.current_phase:
                tracker.end_phase()
            
            logger.exception("[process_story_beat] Error")
            
            return {
                "error": str(e),
                "performance": tracker.get_metrics()
            }


# Helper function to get the governed storyteller
_governed_storyteller_instance = None

def get_governed_storyteller():
    """
    Get (or create) the governed storyteller agent.
    
    Returns:
        The governed storyteller agent
    """
    global _governed_storyteller_instance
    
    if _governed_storyteller_instance is None:
        _governed_storyteller_instance = GovernedStorytellerAgent()
    
    return _governed_storyteller_instance
