# logic/npc_agents/npc_agent.py (Updated)

"""
Core NPC agent class that manages individual NPC behavior with memory capabilities.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

from db.connection import get_db_connection
from .decision_engine import NPCDecisionEngine
from .environment_perception import (
    fetch_environment_data,
    is_significant_action,
    execute_npc_action
)

# Memory system imports
from memory.wrapper import MemorySystem
from memory.core import Memory, MemoryType, MemorySignificance
from memory.masks import ProgressiveRevealManager

logger = logging.getLogger(__name__)


class NPCAgent:
    """
    Independent AI agent controlling a single NPC's behavior.
    Now enhanced with memory capabilities.

    Responsibilities:
    - Perceive environment (with memory-informed context)
    - Make decisions based on personality, current context, and memory
    - Execute chosen actions
    - Form and utilize memories with advanced cognitive features
    - Manage mask (presented vs. true personality)
    - Process emotional states and reactions
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        """
        Initialize an NPCAgent for a single NPC.

        Args:
            npc_id: The ID of the NPC
            user_id: The player or user ID
            conversation_id: The conversation/scene ID
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.decision_engine = NPCDecisionEngine(npc_id, user_id, conversation_id)
        
        # Lazy-loaded memory components
        self._memory_system = None
        self._mask_manager = None
        self.current_emotional_state = None
        self.last_perception = None
    
    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system
    
    async def _get_mask_manager(self):
        """Lazy-load the mask manager."""
        if self._mask_manager is None:
            self._mask_manager = ProgressiveRevealManager(self.user_id, self.conversation_id)
        return self._mask_manager

    async def perceive_environment(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the NPC's perception of the current environment & context.
        Now enhanced with memory-based perception.

        Args:
            current_context: Dictionary that may contain location/time or relevant info

        Returns:
            A dictionary containing:
              - environment (location, time_of_day, etc.)
              - relevant_memories
              - relationships
              - emotional_state
              - mask
              - timestamp
        """
        # Fetch basic environment data
        environment_data = await fetch_environment_data(
            self.user_id,
            self.conversation_id,
            current_context
        )

        # Enhance perception with memory system
        memory_system = await self._get_memory_system()
        
        # Retrieve relevant memories based on current context
        memory_result = await memory_system.recall(
            entity_type="npc",
            entity_id=self.npc_id,
            context=current_context,
            limit=7  # More memories for richer context
        )
        relevant_memories = memory_result.get("memories", [])
        
        # Check for flashback potential
        flashback = None
        if "text" in current_context and random.random() < 0.15:  # 15% chance of flashback
            flashback = await memory_system.npc_flashback(
                npc_id=self.npc_id, 
                context=current_context.get("text", "")
            )
        
        # Get current emotional state
        emotional_state = await memory_system.get_npc_emotion(self.npc_id)
        self.current_emotional_state = emotional_state
        
        # Get mask information (true vs. presented personality)
        mask_info = await memory_system.get_npc_mask(self.npc_id)
        
        # Get relationship data
        relationship_data = await self._fetch_relationships()
        
        # Combine into a single perception dictionary
        perception = {
            "environment": environment_data,
            "relevant_memories": relevant_memories,
            "flashback": flashback,
            "relationships": relationship_data,
            "emotional_state": emotional_state,
            "mask": mask_info,
            "timestamp": datetime.now().isoformat()
        }

        # Store the current perception
        self.last_perception = perception

        logger.debug("NPCAgent %s perceived environment with %d relevant memories", 
                     self.npc_id, len(relevant_memories))
        
        return perception

    async def make_decision(self,
                           perception: Optional[Dict[str, Any]] = None,
                           available_actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Decide which action to take based on current perception and available actions.
        Now enhanced with memory-driven decision making.

        Args:
            perception: The NPC's current perception dictionary
            available_actions: A list of possible actions the NPC could choose from

        Returns:
            A dictionary describing the chosen action (type, description, target, etc.)
        """
        if perception is None:
            # If no provided perception, use the last known or fetch a fresh one
            if self.last_perception is None:
                logger.debug("No prior perception found, fetching fresh environment data.")
                perception = await self.perceive_environment({})
            else:
                perception = self.last_perception

        # Pass the enriched perception with memories to the decision engine
        chosen_action = await self.decision_engine.decide(perception, available_actions)
        logger.debug("NPCAgent %s decided on action: %s", self.npc_id, chosen_action)
        
        # Check if this decision should trigger a mask slippage
        should_slip = False
        mask_info = perception.get("mask", {})
        
        # Calculate chance of mask slippage based on:
        # - NPC stats (higher dominance/cruelty means less control)
        # - Mask integrity (lower integrity = higher chance of slippage)
        # - Current emotional intensity
        
        # Get basic stats
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT dominance, cruelty
                FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            row = cursor.fetchone()
            if row:
                dominance, cruelty = row
                
                # Higher dominance/cruelty with lower mask integrity increases slip chance
                mask_integrity = mask_info.get("integrity", 100)
                slip_chance = (dominance + cruelty) / 200  # 0.0 to 1.0
                slip_chance *= (100 - mask_integrity) / 100  # Factor in mask integrity
                
                # Also factor in emotional intensity
                current_emotion = perception.get("emotional_state", {}).get("current_emotion", {})
                emotion_intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
                slip_chance *= (1 + emotion_intensity)
                
                # Decision to slip
                should_slip = random.random() < slip_chance
                
                if should_slip:
                    # Generate mask slippage
                    memory_system = await self._get_memory_system()
                    trigger = f"deciding to {chosen_action.get('description', 'act')}"
                    
                    # Emotional triggers are more likely to cause slips
                    if current_emotion.get("primary", {}).get("name") in ["anger", "fear", "excitement"]:
                        trigger = f"feeling {current_emotion.get('primary', {}).get('name')} while " + trigger
                        
                    slip_result = await memory_system.reveal_npc_trait(
                        npc_id=self.npc_id,
                        trigger=trigger
                    )
                    
                    # Add mask slippage information to action
                    chosen_action["mask_slippage"] = slip_result
        
        return chosen_action

    async def execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chosen action in the game world.
        Now records the action in memory.

        Args:
            action: Dictionary describing the action
            context: Additional contextual information for the execution

        Returns:
            A dictionary describing the result (e.g. outcome, emotional impact).
        """
        # Execute the action using existing code
        result = await execute_npc_action(
            self.npc_id,
            self.user_id,
            self.conversation_id,
            action,
            context
        )
        logger.debug("NPCAgent %s executed action '%s', got result: %s", self.npc_id, action, result)

        # Record the action in memory if significant
        if is_significant_action(action, result):
            memory_system = await self._get_memory_system()
            
            # Format memory text
            memory_text = f"I {action.get('description', 'did something')} which resulted in {result.get('outcome', 'something happening')}"
            
            # Record in memory system
            memory_result = await memory_system.remember(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                importance="medium",  # Default importance
                emotional=True,  # Analyze emotional content
                tags=["action", action.get("type", "unknown")]
            )
            
            # Update emotional state based on the action's impact
            emotional_impact = result.get("emotional_impact", 0)
            if abs(emotional_impact) >= 2:
                # Map emotional impact to an emotion
                if emotional_impact > 2:
                    emotion = "joy"
                    intensity = min(1.0, abs(emotional_impact) / 5.0)
                elif emotional_impact > 0:
                    emotion = "satisfaction"
                    intensity = min(0.7, abs(emotional_impact) / 5.0)
                elif emotional_impact < -2:
                    emotion = "anger"
                    intensity = min(1.0, abs(emotional_impact) / 5.0)
                else:
                    emotion = "sadness"
                    intensity = min(0.7, abs(emotional_impact) / 5.0)
                
                # Update emotional state
                await memory_system.update_npc_emotion(
                    npc_id=self.npc_id,
                    emotion=emotion,
                    intensity=intensity
                )

        return result

    async def process_player_action(self, player_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a player action and generate an NPC response.
        Now enhanced with memory of the interaction.

        Steps:
          1) Update the NPC's perception based on the player action context
          2) Remember the player's action
          3) Decide on a response action
          4) Execute that action
          5) Update relationships and record a memory of the interaction

        Args:
            player_action: A dict describing the player's action

        Returns:
            A dict: { "npc_id":..., "action":..., "result":... }
        """
        # Incorporate the player's action into the environment context
        context = {
            "player_action": player_action,
            "text": player_action.get("description", ""),
            "description": f"Player {player_action.get('description', 'did something')}"
        }

        # Step 1: Refresh or update our environment perception
        perception = await self.perceive_environment(context)
        
        # Step 2: Remember the player's action
        memory_system = await self._get_memory_system()
        
        # Format memory text for what player did
        player_memory_text = f"The player {player_action.get('description', 'did something')}"
        
        # Record the player's action from NPC's perspective
        memory_kwargs = {
            "tags": ["player_action", player_action.get("type", "unknown")]
        }
        
        # Adjust importance based on action type
        if player_action.get("type") in ["attack", "command", "insult", "seduce"]:
            memory_kwargs["importance"] = "high"
            
        # Remember the player's action
        await memory_system.remember(
            entity_type="npc", 
            entity_id=self.npc_id,
            memory_text=player_memory_text,
            **memory_kwargs
        )
        
        # Check if the player action triggers specific emotions
        player_action_type = player_action.get("type", "").lower()
        if player_action_type == "insult":
            # Insult might trigger anger or sadness
            if perception.get("mask", {}).get("integrity", 100) < 50:
                # Low mask integrity - more likely to show anger
                await memory_system.update_npc_emotion(self.npc_id, "anger", 0.7)
            else:
                # High mask integrity - more likely to mask true feelings
                await memory_system.update_npc_emotion(self.npc_id, "sadness", 0.5)
        elif player_action_type == "command" or player_action_type == "dominate":
            # Check if NPC has dominant traits hidden under the mask
            hidden_traits = perception.get("mask", {}).get("hidden_traits", {})
            if "dominant" in hidden_traits or "controlling" in hidden_traits:
                # Dominant NPC being dominated - might trigger anger or resentment
                await memory_system.update_npc_emotion(self.npc_id, "anger", 0.6)
                
                # Higher chance of mask slippage
                if random.random() < 0.3:
                    await memory_system.reveal_npc_trait(
                        npc_id=self.npc_id,
                        trigger=f"being commanded by player to {player_action.get('description', 'do something')}"
                    )
        
        # Step 3: Decide how to respond
        response_action = await self.make_decision(perception)

        # Step 4: Execute the chosen response
        result = await self.execute_action(response_action, context)

        # Step 5: Remember the interaction
        interaction_memory_text = (
            f"When the player {player_action.get('description','did something')}, "
            f"I responded by {response_action.get('description','doing something')}"
        )
        
        await memory_system.remember(
            entity_type="npc",
            entity_id=self.npc_id,
            memory_text=interaction_memory_text,
            importance="medium",
            tags=["interaction", "player", player_action.get("type", "unknown")]
        )
        
        # Step 6: Update relationships if relevant
        if player_action.get("type") == "talk":
            # Potentially update relationship here
            pass

        logger.debug("NPCAgent %s processed player action '%s': result=%s", 
                    self.npc_id, player_action, result)

        return {
            "npc_id": self.npc_id,
            "action": response_action,
            "result": result
        }

    async def perform_scheduled_activity(self) -> Optional[Dict[str, Any]]:
        """
        Perform the activity scheduled for this NPC at the current time of day.
        Now enhanced with memory of routines.

        Returns:
            A dict like {"npc_id":..., "action":..., "result":...}
            or None if no scheduled activity found or error occurs.
        """
        from db.connection import get_db_connection
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                # 1) Load timeOfDay & CurrentDay
                time_of_day = self._fetch_current_time_of_day(cursor)
                day_name = self._fetch_current_day_name(cursor)

                # 2) Retrieve NPC's schedule
                sched = self._fetch_npc_schedule(cursor)
                if not sched or day_name not in sched or time_of_day not in sched[day_name]:
                    logger.debug("No schedule found for NPC %s on day='%s' time='%s'.", 
                                self.npc_id, day_name, time_of_day)
                    return None

                activity_desc = sched[day_name][time_of_day]
                action = {
                    "type": "scheduled",
                    "description": activity_desc,
                    "target": "environment",
                    "stats_influenced": {}
                }

            # 3) Execute
            context = {
                "day": day_name,
                "time": time_of_day,
                "location": "scheduled_location"
            }
            result = await self.execute_action(action, context)

            # 4) Record a memory for routine using memory system
            memory_system = await self._get_memory_system()
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=f"I did '{activity_desc}' as scheduled for {day_name} {time_of_day}",
                importance="low",  # Low significance for routine activities
                tags=["routine", "scheduled", day_name.lower(), time_of_day.lower()]
            )

            logger.debug("NPCAgent %s performed scheduled activity: %s => %s", 
                        self.npc_id, activity_desc, result)
            
            return {
                "npc_id": self.npc_id,
                "action": action,
                "result": result
            }
        except Exception as e:
            logger.error("Error in perform_scheduled_activity for NPC %s: %s", self.npc_id, e)
            return None
            
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run periodic maintenance tasks on the NPC's memory system.
        
        Returns:
            Results of maintenance operations
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Run memory maintenance
            return await memory_system.maintain(
                entity_type="npc",
                entity_id=self.npc_id
            )
        except Exception as e:
            logger.error(f"Error running memory maintenance for NPC {self.npc_id}: {e}")
            return {"error": str(e)}

    async def get_beliefs_about_player(self) -> List[Dict[str, Any]]:
        """
        Get the NPC's beliefs about the player based on past interactions.
        Useful for generating dialog and response planning.
        
        Returns:
            List of beliefs with confidence levels
        """
        try:
            memory_system = await self._get_memory_system()
            beliefs = await memory_system.get_beliefs(
                entity_type="npc", 
                entity_id=self.npc_id,
                topic="player"
            )
            return beliefs
        except Exception as e:
            logger.error(f"Error getting beliefs about player for NPC {self.npc_id}: {e}")
            return []
            
    async def _fetch_relationships(self) -> Dict[str, Any]:
        """Get NPC's relationships with other entities."""
        relationships = {}
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            # Query all links from NPC to other entities
            cursor.execute("""
                SELECT entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE entity1_type = 'npc'
                  AND entity1_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            
            rows = cursor.fetchall()
            for entity_type, entity_id, link_type, link_level in rows:
                entity_name = "Unknown"
                
                if entity_type == "npc":
                    # Fetch NPC name
                    cursor.execute("""
                        SELECT npc_name
                        FROM NPCStats
                        WHERE npc_id = %s
                          AND user_id = %s
                          AND conversation_id = %s
                    """, (entity_id, self.user_id, self.conversation_id))
                    name_row = cursor.fetchone()
                    if name_row:
                        entity_name = name_row[0]
                elif entity_type == "player":
                    entity_name = "Player"
                
                relationships[entity_type] = {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "link_type": link_type,
                    "link_level": link_level
                }
        
        return relationships

    # ------------------------------------------------------------------
    # Internal helper methods for scheduling/time (unchanged from original)
    # ------------------------------------------------------------------

    def _fetch_current_time_of_day(self, cursor) -> str:
        """
        Helper to fetch the current time of day (e.g. 'Morning') from DB.
        Fallback is 'Morning' if unavailable.
        """
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s
              AND conversation_id=%s
              AND key='TimeOfDay'
        """, (self.user_id, self.conversation_id))
        row = cursor.fetchone()
        return row[0] if row else "Morning"

    def _fetch_current_day_name(self, cursor) -> str:
        """
        Helper to find the current day name from DB, e.g. 'Monday'.
        Fallback is 'Monday' if not found.
        """
        # 1) fetch numeric current_day
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s
              AND conversation_id=%s
              AND key='CurrentDay'
        """, (self.user_id, self.conversation_id))
        row = cursor.fetchone()
        current_day_num = int(row[0]) if (row and str(row[0]).isdigit()) else 1

        # 2) fetch day names
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s
              AND conversation_id=%s
              AND key='CalendarNames'
        """, (self.user_id, self.conversation_id))
        row2 = cursor.fetchone()
        if row2 and row2[0]:
            try:
                calendar_data = json.loads(row2[0])
                day_names = calendar_data.get("days", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            except Exception:
                day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        else:
            day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        day_index = (current_day_num - 1) % len(day_names)
        return day_names[day_index]

    def _fetch_npc_schedule(self, cursor) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Helper to load the schedule dict from NPCStats.schedule
        e.g. { "Monday": {"Morning":"desc", ...}, ... }
        """
        cursor.execute("""
            SELECT schedule
            FROM NPCStats
            WHERE npc_id=%s
              AND user_id=%s
              AND conversation_id=%s
        """, (self.npc_id, self.user_id, self.conversation_id))
        sched_row = cursor.fetchone()
        if not sched_row or not sched_row[0]:
            return None

        try:
            # schedule might be a string (JSON) or JSONB
            if isinstance(sched_row[0], str):
                return json.loads(sched_row[0])
            return sched_row[0]
        except Exception:
            logger.error("Invalid schedule data for NPC %s", self.npc_id)
            return None

# Need to import random for mask slippage chance
import random
