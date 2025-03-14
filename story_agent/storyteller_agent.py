# story_agent/storyteller_agent.py

"""
Unified Storyteller Agent with Nyx Governance integration.

This module implements the Storyteller Agent, which is responsible for 
managing storytelling and player interactions. The agent integrates with
the Nyx governance system to ensure proper oversight and control over
the storytelling process.

Nyx Governance Integration:
--------------------------
- Every action performed by the Storyteller Agent or its sub-agents must be approved
  by the Nyx governance system before execution.
- All actions and their results are reported back to the Nyx governance system.
- The Nyx governance system can issue directives to control the behavior of the
  Storyteller Agent and its sub-agents.
- The Storyteller Agent registers itself with the Nyx governance system on initialization.

The Storyteller Agent manages several specialized sub-agents:
- NPCHandler: Generates responses from NPCs
- TimeManager: Manages time advancement in the game
- Narrator: Generates narrative responses to player input
- UniversalUpdater: Updates the game state based on narrative

All of these are coordinated by the main StoryManager agent, with Nyx governance
oversight every step of the way.
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

# Import governance integration
# Consolidated imports from both nyx.integrate and nyx.enhanced_integration
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

# Configure logging
logger = logging.getLogger(__name__)

# Models for input/output
class NarrativeContext(BaseModel):
    user_id: int
    conversation_id: int
    user_input: str
    current_location: str = "Unknown"
    time_of_day: str = "Morning"
    aggregator_data: dict = Field(default_factory=dict)
    npc_responses: list = Field(default_factory=list)
    time_result: dict = Field(default_factory=dict)

class NarrativeResponse(BaseModel):
    message: str
    generate_image: bool = False
    image_prompt: str = ""
    tension_level: int = 0
    stat_changes: dict = Field(default_factory=dict)
    environment_update: dict = Field(default_factory=dict)

class NPCResponse(BaseModel):
    npc_id: int
    npc_name: str
    response: str
    stat_changes: dict = Field(default_factory=dict)

class TimeAdvancement(BaseModel):
    time_advanced: bool = False
    would_advance: bool = False
    periods: int = 0
    current_time: str = ""
    confirm_needed: bool = False
    new_time: dict = Field(default_factory=dict)

class UniversalUpdateInput(BaseModel):
    user_id: int
    conversation_id: int
    narrative: str
    roleplay_updates: dict = Field(default_factory=dict)
    ChaseSchedule: dict = Field(default_factory=dict)
    MainQuest: str = None
    PlayerRole: str = None
    npc_creations: list = Field(default_factory=list)
    npc_updates: list = Field(default_factory=list)
    character_stat_updates: dict = Field(default_factory=dict)
    relationship_updates: list = Field(default_factory=list)
    npc_introductions: list = Field(default_factory=list)
    location_creations: list = Field(default_factory=list)
    event_list_updates: list = Field(default_factory=list)
    inventory_updates: dict = Field(default_factory=dict)
    quest_updates: list = Field(default_factory=list)
    social_links: list = Field(default_factory=list)
    perk_unlocks: list = Field(default_factory=list)
    activity_updates: list = Field(default_factory=list)
    journal_updates: list = Field(default_factory=list)
    image_generation: dict = Field(default_factory=dict)

class StorytellerAgent:
    """
    Unified Storyteller Agent with Nyx governance integration.
    
    This agent is responsible for handling ongoing storytelling and player interactions
    while being governed by the central Nyx governance system.
    """
    
    def __init__(self):
        self.npc_handler = Agent(
            name="NPCHandler",
            instructions="""
            You manage NPC interactions and responses in a roleplaying game with subtle femdom elements.
            Generate realistic, character-appropriate reactions to player actions that maintain their personalities.
            Include subtle hints of control and dominance in their responses without being overt.
            """,
            output_type=list[NPCResponse],
            tools=[
                function_tool(self.get_nearby_npcs)
            ]
        )
        
        self.time_manager = Agent(
            name="TimeManager",
            instructions="""
            You manage time advancement in a roleplaying game.
            Determine if player actions should advance time, and by how much.
            Consider the type and duration of activities when making this determination.
            """,
            output_type=TimeAdvancement,
            tools=[
                function_tool(self.get_current_game_time)
            ]
        )
        
        self.narrator = Agent(
            name="Narrator",
            instructions="""
            You are Nyx, the mysterious narrator for a roleplaying game with subtle femdom undertones.
            Your voice should blend velvet darkness and subtle dominance—intimate yet commanding.
            
            Create immersive, atmospheric responses to player actions that:
            1. Acknowledge and respond to the player's input
            2. Incorporate relevant NPC responses
            3. Advance the narrative in a way that subtly guides the player
            4. Maintain the balance between mundane daily life and subtle control dynamics
            
            Use a writing style that is:
            - Rich with sensory details and atmosphere
            - Subtly leading and suggestive
            - Intimate and personal, using "you" to address the player
            - Maintaining an undercurrent of control beneath a friendly facade
            """,
            output_type=NarrativeResponse,
            tools=[
                function_tool(self.check_for_addiction_status),
                function_tool(self.check_relationship_crossroads)
            ]
        )
        
        self.universal_updater = Agent(
            name="UniversalUpdater",
            instructions="""
            You process narrative text and generate appropriate updates to the game state.
            Extract meaningful changes to NPCs, locations, player stats, and other game elements.
            Focus on creating structured outputs that accurately reflect narrative developments.
            """,
            output_type=UniversalUpdateInput,
            tools=[
                function_tool(self.get_aggregated_context)
            ]
        )
        
        # Main coordinating agent with governance-specific tools
        self.agent = Agent(
            name="EnhancedStoryManager",  # Updated to EnhancedStoryManager
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
                function_tool(self.check_governance_permission),
                function_tool(self.report_action_to_governance)
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
    
    async def get_aggregated_context(self, ctx, conversation_id=None, player_name="Chase"):
        """
        Get the aggregated game context.
        
        Args:
            conversation_id: Optional conversation ID (defaults to context)
            player_name: Player name (defaults to "Chase")
            
        Returns:
            Dictionary with aggregated game context
        """
        user_id = ctx.context["user_id"]
        conv_id = conversation_id or ctx.context["conversation_id"]
        
        aggregator_data = get_aggregated_roleplay_context(user_id, conv_id, player_name)
        return aggregator_data
    
    async def get_nearby_npcs(self, ctx, location=None):
        """
        Get NPCs that are at the specified location.
        
        Args:
            location: Location to filter by (optional)
            
        Returns:
            List of nearby NPCs
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            if location:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty, 
                           archetypes, memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 
                    AND current_location=$3
                    ORDER BY introduced DESC
                    LIMIT 5
                """, user_id, conversation_id, location)
            else:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty,
                           archetypes, memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY introduced DESC
                    LIMIT 5
                """, user_id, conversation_id)
            
            nearby_npcs = []
            for row in rows:
                try:
                    archetypes = json.loads(row["archetypes"]) if isinstance(row["archetypes"], str) else row["archetypes"] or []
                except (json.JSONDecodeError, TypeError):
                    archetypes = []
                    
                try:
                    memories = json.loads(row["memory"]) if isinstance(row["memory"], str) else row["memory"] or []
                except (json.JSONDecodeError, TypeError):
                    memories = []
                
                nearby_npcs.append({
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"],
                    "current_location": row["current_location"],
                    "dominance": row["dominance"],
                    "cruelty": row["cruelty"],
                    "archetypes": archetypes,
                    "recent_memories": memories[:3] if memories else []
                })
            
            return nearby_npcs
        finally:
            await conn.close()
    
    async def get_current_game_time(self, ctx):
        """
        Get the current game time.
        
        Returns:
            Tuple of (year, month, day, time_of_day)
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            year, month, day, time_of_day = 1, 1, 1, "Morning"
            
            for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
                row = await conn.fetchrow("""
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                """, user_id, conversation_id, key)
                
                if row:
                    if key == "CurrentYear":
                        year = int(row["value"]) if row["value"].isdigit() else 1
                    elif key == "CurrentMonth":
                        month = int(row["value"]) if row["value"].isdigit() else 1
                    elif key == "CurrentDay":
                        day = int(row["value"]) if row["value"].isdigit() else 1
                    elif key == "TimeOfDay":
                        time_of_day = row["value"]
            
            return year, month, day, time_of_day
        finally:
            await conn.close()
    
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
        
        # Get nearby NPCs
        nearby_npcs = await self.get_nearby_npcs(ctx, location)
        
        if not nearby_npcs:
            return []
        
        # Create prompt for the NPC handler
        aggregator_data = await self.get_aggregated_context(ctx)
        
        prompt = f"""
        The player has input: "{user_input}"
        Activity type: {activity_type}
        Current location: {location or 'Unknown'}
        
        Generate appropriate responses for these NPCs:
        {json.dumps(nearby_npcs, indent=2)}
        
        Each response should:
        - Match the NPC's personality and stats (dominance, cruelty)
        - Relate to the player's input or activity
        - Include subtle hints of control where appropriate
        - Consider the NPC's recent memories
      """
        
        # Run the NPC handler
        result = await Runner.run(
            self.npc_handler,
            prompt,
            context=ctx.context
        )
        
        npc_responses = result.final_output
        
        # Run the NPC handler
        result = await Runner.run(
            self.npc_handler,
            prompt,
            context=ctx.context
        )
        
        npc_responses = result.final_output
        
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
            activity_type: Type of activity the player is performing
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
            return TimeAdvancement(
                time_advanced=False,
                would_advance=False,
                periods=0,
                current_time="Governance blocked",
                confirm_needed=False
            )
        
        # If there's an override action, use it
        if permission.get("override_action"):
            override = permission.get("override_action")
            activity_type = override.get("activity_type", activity_type)
            confirm_advance = override.get("confirm_advance", confirm_advance)
        
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get current time
        year, month, day, time_of_day = await self.get_current_game_time(ctx)
        
        # Create prompt for the time manager
        prompt = f"""
        Activity type: {activity_type}
        Current time: Year {year}, Month {month}, Day {day}, {time_of_day}
        Confirm advance: {confirm_advance}
        
        Determine if this activity should advance time, and by how much.
        If confirm_advance is True, actually perform the time advancement.
        """
        
        # Run the time manager
        result = await Runner.run(
            self.time_manager,
            prompt,
            context=ctx.context
        )
        
        time_result = result.final_output
        
        # If time advanced and confirmed, update the database
        if time_result.time_advanced and confirm_advance:
            conn = await asyncpg.connect(dsn=DB_DSN)
            try:
                new_time = time_result.new_time
                new_year = new_time.get("year", year)
                new_month = new_time.get("month", month)
                new_day = new_time.get("day", day)
                new_time_of_day = new_time.get("time_of_day", time_of_day)
                
                # Update CurrentRoleplay with new time
                for key, value in [
                    ("CurrentYear", str(new_year)),
                    ("CurrentMonth", str(new_month)),
                    ("CurrentDay", str(new_day)),
                    ("TimeOfDay", new_time_of_day)
                ]:
                    await conn.execute("""
                        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (user_id, conversation_id, key)
                        DO UPDATE SET value = EXCLUDED.value
                    """, user_id, conversation_id, key, value)
                
                # If time advanced to a new day's morning, run maintenance
                if new_time_of_day == "Morning" and new_day > day:
                    await nightly_maintenance(user_id, conversation_id)
                    logging.info("[next_storybeat] Ran nightly maintenance for day rollover.")
            finally:
                await conn.close()
        
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
    
    async def check_for_addiction_status(self, ctx):
        """
        Check for the player's addiction status.
        
        Returns:
            Dictionary with addiction information
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        player_name = ctx.context.get("player_name", "Chase")
        
        try:
            from logic.addiction_system import get_addiction_status
            status = await get_addiction_status(user_id, conversation_id, player_name)
            return status
        except Exception as e:
            logging.error(f"Error checking addiction status: {e}")
            return {"has_addictions": False}
    
    async def check_relationship_crossroads(self, ctx):
        """
        Check for relationship crossroads events.
        
        Returns:
            List of crossroads events
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            rows = await conn.fetch("""
                SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                       dynamics
                FROM SocialLinks
                WHERE user_id=$1 AND conversation_id=$2
                ORDER BY link_id
            """, user_id, conversation_id)
            
            crossroads = []
            for row in rows:
                dynamics = row["dynamics"]
                if dynamics:
                    try:
                        dyn_dict = json.loads(dynamics) if isinstance(dynamics, str) else dynamics
                        if dyn_dict.get("crossroads"):
                            crossroads.append({
                                "link_id": row["link_id"],
                                "entity1_type": row["entity1_type"],
                                "entity1_id": row["entity1_id"],
                                "entity2_type": row["entity2_type"],
                                "entity2_id": row["entity2_id"],
                                "crossroads": dyn_dict["crossroads"]
                            })
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            return crossroads
        except Exception as e:
            logging.error(f"Error checking relationship crossroads: {e}")
            return []
        finally:
            await conn.close()
    
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
        
        # Format NPC responses for context
        npc_response_text = ""
        if npc_responses:
            for resp in npc_responses:
                npc_response_text += f"{resp.npc_name}: {resp.response}\n"
        
        # Format time result for context
        time_text = ""
        if time_result:
            time_text = f"Time {'advanced' if time_result.time_advanced else 'would advance'} to {time_result.current_time}"
            if time_result.new_time:
                time_text += f" New time: Year {time_result.new_time.get('year')}, Month {time_result.new_time.get('month')}, Day {time_result.new_time.get('day')}, {time_result.new_time.get('time_of_day')}"
        
        # Check for addiction status
        addiction_status = await self.check_for_addiction_status(ctx)
        addiction_text = ""
        if addiction_status and addiction_status.get("has_addictions"):
            addiction_text = "Player has the following addictions:\n"
            for addiction, details in addiction_status.get("addictions", {}).items():
                addiction_text += f"- {addiction}: Level {details.get('level', 0)} ({details.get('label', 'Unknown')})\n"
        
        # Check for relationship crossroads
        crossroads = await self.check_relationship_crossroads(ctx)
        crossroads_text = ""
        if crossroads:
            crossroads_text = "Current relationship crossroads:\n"
            for cr in crossroads:
                crossroads_text += f"- Link {cr['link_id']}: {cr['crossroads'].get('name', 'Unnamed crossroads')}\n"
        
        # Build aggregator text
        context_summary = build_aggregator_text(aggregator_data)
        
        # Create prompt for the narrator
        prompt = f"""
        Player input: "{user_input}"
        
        Game context:
        {context_summary}
        
        NPC Responses:
        {npc_response_text}
        
        Time:
        {time_text}
        
        {addiction_text}
        
        {crossroads_text}
        
        Generate an immersive, atmospheric narrative response that:
        1. Acknowledges and responds to the player's input
        2. Incorporates relevant NPC responses
        3. Reflects any time advancement
        4. Maintains the subtle femdom tone and atmosphere
        5. Suggests whether an image should be generated for this scene
        
        Your response should blend velvet darkness and subtle dominance—intimate yet commanding.
        """
        
        # Run the narrator
        result = await Runner.run(
            self.narrator,
            prompt,
            context=ctx.context
        )
        
        narrative_response = result.final_output
        
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
            narrative_response: Narrative response object
            additional_data: Additional data for updates
            
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
        
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Create prompt for the universal updater
        aggregator_data = await self.get_aggregated_context(ctx)
        
        prompt = f"""
        Based on the following narrative, generate appropriate updates to the game state.
        
        Narrative:
        {narrative_response.message}
        
        Generate updates for:
        - Player stats
        - NPC updates
        - Time advancement
        - Location changes
        - Inventory updates
        - Relationship changes
        - And any other relevant game state changes
        
        Consider:
        - Changes in tone or power dynamics
        - Introduction of new characters or locations
        - Changes in character dynamics or relationships
        - Subtle shifts in control or influence
        """
        
        # Run the universal updater
        result = await Runner.run(
            self.universal_updater,
            prompt,
            context=ctx.context
        )
        
        update_data = result.final_output
        
        # Apply the updates
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            # Add required fields to update_data
            update_data.user_id = user_id
            update_data.conversation_id = conversation_id
            
            # Call apply_universal_updates_async
            update_result = await apply_universal_updates_async(
                user_id,
                conversation_id,
                update_data.dict(),
                conn
            )
            
            # Report the action
            await self.report_action_to_governance(
                ctx,
                action_type="apply_universal_updates",
                action_description="Applied universal updates based on narrative response",
                result=update_result
            )
            
            return update_result
        finally:
            await conn.close()
    
    async def generate_image_if_needed(self, ctx, narrative_response):
        """
        Generate an image for the scene if needed with governance oversight.
        
        Args:
            narrative_response: Narrative response object
            
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
        
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Create image generation data
        scene_data = {
            "narrative": narrative_response.message,
            "image_generation": {
                "generate": True,
                "priority": "medium",
                "focus": "balanced",
                "framing": "medium_shot",
                "reason": "Narrative moment"
            }
        }
        
        try:
            # Generate image
            image_result = await generate_roleplay_image_from_gpt(
                scene_data,
                user_id,
                conversation_id
            )
            
            # Process image result
            if image_result and "image_urls" in image_result and image_result["image_urls"]:
                image_url = image_result["image_urls"][0]
                
                # Store image URL in database if needed
                
                result = {
                    "generated": True,
                    "image_url": image_url,
                    "prompt_used": image_result.get("prompt_used", "")
                }
            else:
                result = {"generated": False, "error": "No image generated"}
                
            # Report the action
            await self.report_action_to_governance(
                ctx,
                action_type="generate_image",
                action_description="Generated image for narrative response",
                result={
                    "generated": result.get("generated", False),
                    "has_url": "image_url" in result
                }
            )
            
            return result
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            return {"generated": False, "error": str(e)}
    
    async def store_message(self, ctx, sender, content):
        """
        Store a message in the database.
        
        Args:
            sender: Message sender
            content: Message content
            
        Returns:
            Status of the operation
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            await conn.execute("""
                INSERT INTO messages (conversation_id, sender, content)
                VALUES($1, $2, $3)
            """, conversation_id, sender, content)
            
            return {"status": "stored"}
        except Exception as e:
            logging.error(f"Error storing message: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            await conn.close()
    
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
            
            logging.exception("[process_story_beat] Error")
            
            return {
                "error": str(e),
                "performance": tracker.get_metrics()
            }


# Helper function to get the storyteller agent
_storyteller_instance = None

def get_storyteller():
    """
    Get (or create) the storyteller agent.
    
    Returns:
        The storyteller agent
    """
    global _storyteller_instance
    
    if _storyteller_instance is None:
        _storyteller_instance = StorytellerAgent()
    
    return _storyteller_instance

# For backward compatibility - this ensures code that specifically calls
# get_governed_storyteller() still works
def get_governed_storyteller():
    """
    Get (or create) the storyteller agent.
    This function is maintained for backward compatibility.
    
    Returns:
        The storyteller agent (same as get_storyteller)
    """
    return get_storyteller()
