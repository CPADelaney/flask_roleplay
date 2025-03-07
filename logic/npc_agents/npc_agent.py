# logic/npc_agents/npc_agent.py

"""
Core NPC agent class that manages individual NPC behavior with enhanced memory capabilities.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .enhanced_memory_manager import EnhancedNPCMemoryManager
from .decision_engine import NPCDecisionEngine
from .relationship_manager import NPCRelationshipManager
from .environment_perception import (
    fetch_environment_data,
    is_significant_action,
    execute_npc_action
)

logger = logging.getLogger(__name__)


class NPCAgent:
    """
    Independent AI agent controlling a single NPC's behavior.

    Responsibilities:
    - Perceive environment (location, time, relevant memories, relationships)
    - Make decisions (via NPCDecisionEngine) based on perception + available actions
    - Execute chosen actions (with optional memory recording if action is significant)
    - Track player interactions and update relationships accordingly
    - Handle scheduled/routine activities (e.g., from the NPC's daily schedule)
    - Form and utilize memories with advanced cognitive features
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

        # Initialize enhanced memory manager instead of basic one
        self.memory_manager = EnhancedNPCMemoryManager(npc_id, user_id, conversation_id)
        self.decision_engine = NPCDecisionEngine(npc_id, user_id, conversation_id)
        self.relationship_manager = NPCRelationshipManager(npc_id, user_id, conversation_id)

        self.last_perception: Optional[Dict[str, Any]] = None
        self.current_emotional_state: Optional[Dict[str, Any]] = None

    async def perceive_environment(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the NPC's perception of the current environment & context.

        Args:
            current_context: Dictionary that may contain location/time or relevant info

        Returns:
            A dictionary containing:
              - environment (location, time_of_day, etc.)
              - relevant_memories
              - relationships
              - emotional_state
              - timestamp
        """
        # Fetch environment data
        environment_data = await fetch_environment_data(
            self.user_id,
            self.conversation_id,
            current_context
        )

        # Retrieve relevant memories using enhanced memory system
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(
            context=current_context
        )

        # Update or retrieve relationship data
        relationship_data = await self.relationship_manager.update_relationships(current_context)
        
        # Get current emotional state
        emotional_state = await self.memory_manager.get_emotional_state()
        self.current_emotional_state = emotional_state

        # Combine into a single perception dictionary
        perception = {
            "environment": environment_data,
            "relevant_memories": relevant_memories,
            "relationships": relationship_data,
            "emotional_state": emotional_state,
            "timestamp": datetime.now().isoformat()
        }

        # Store the current perception
        self.last_perception = perception

        logger.debug("NPCAgent %s perceived environment: %s", self.npc_id, perception)
        return perception

    async def make_decision(self,
                           perception: Optional[Dict[str, Any]] = None,
                           available_actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Decide which action to take based on current perception and available actions.

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

        # We pass the perception to the decision engine which now has access to enhanced memories
        chosen_action = await self.decision_engine.decide(perception, available_actions)
        logger.debug("NPCAgent %s decided on action: %s", self.npc_id, chosen_action)
        
        # Check if this decision requires a mask slippage
        should_slip = False
        
        # More dominant and cruel NPCs with deteriorating masks are more likely to slip
        if "dominance" in perception.get("basic_stats", {}) and "cruelty" in perception.get("basic_stats", {}):
            dominance = perception["basic_stats"]["dominance"]
            cruelty = perception["basic_stats"]["cruelty"]
            
            # Get mask information
            try:
                mask_info = await self.memory_manager.get_npc_mask()
                mask_integrity = mask_info.get("integrity", 100)
                
                # Higher dominance/cruelty with lower mask integrity increases slip chance
                slip_chance = (dominance + cruelty) / 200  # 0.0 to 1.0
                slip_chance *= (100 - mask_integrity) / 100  # Factor in mask integrity
                
                # Also factor in emotional intensity
                if self.current_emotional_state:
                    current_emotion = self.current_emotional_state.get("current_emotion", {})
                    emotion_intensity = current_emotion.get("intensity", 0)
                    slip_chance *= (1 + emotion_intensity)
                
                # Decision to slip
                should_slip = random.random() < slip_chance
                
                if should_slip:
                    # Generate mask slippage
                    slip_result = await self.memory_manager.generate_mask_slippage(
                        trigger=f"deciding to {chosen_action.get('description', 'act')}"
                    )
                    
                    # Record the slip as a memory
                    if "description" in slip_result:
                        await self.memory_manager.add_memory(
                            memory_text=f"My mask slipped: {slip_result['description']}",
                            memory_type="reflection",
                            significance=4,  # High significance
                            tags=["mask_slip", "true_nature"]
                        )
            except Exception as e:
                logger.error(f"Error checking for mask slippage: {e}")
        
        return chosen_action

    async def execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chosen action in the game world.

        Args:
            action: Dictionary describing the action
            context: Additional contextual information for the execution

        Returns:
            A dictionary describing the result (e.g. outcome, emotional impact).
        """
        result = await execute_npc_action(
            self.npc_id,
            self.user_id,
            self.conversation_id,
            action,
            context
        )
        logger.debug("NPCAgent %s executed action '%s', got result: %s", self.npc_id, action, result)

        # If it's a significant action, record it in memory
        if is_significant_action(action, result):
            memory_text = f"I {action.get('description', 'did something')} => {result.get('outcome', '')}"
            
            # Map emotional_impact to emotional_intensity (0-100 scale)
            emotional_impact = result.get("emotional_impact", 0)
            emotional_intensity = abs(emotional_impact) * 20  # Scale from -5...5 to 0...100
            
            # Add memory with emotional information
            await self.memory_manager.add_memory(
                memory_text=memory_text,
                memory_type="action",
                significance=action.get("significance", 3),
                emotional_intensity=emotional_intensity,
                tags=["action", action.get("type", "unknown")]
            )
            
            # Update emotional state based on the action's impact
            if emotional_impact != 0:
                # Map emotional impact to an emotion
                emotion = "neutral"
                if emotional_impact > 2:
                    emotion = "joy"
                elif emotional_impact > 0:
                    emotion = "satisfaction"
                elif emotional_impact < -2:
                    emotion = "anger"
                elif emotional_impact < 0:
                    emotion = "sadness"
                
                # Normalize intensity to 0-1 scale
                intensity = min(1.0, abs(emotional_impact) / 5.0)
                
                # Update emotional state
                await self.memory_manager.update_emotional_state(
                    primary_emotion=emotion,
                    intensity=intensity,
                    trigger=f"Action: {action.get('description', 'unknown action')}"
                )

        return result

    async def process_player_action(self, player_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a player action and generate an NPC response.

        Steps:
          1) Update the NPC's perception based on the new context
          2) Decide on a response action
          3) Execute that action
          4) Optionally update relationships and record a memory of the interaction

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

        # Step 2: Decide how to respond
        response_action = await self.make_decision(perception)

        # Step 3: Execute the chosen response
        result = await self.execute_action(response_action, context)

        # Step 4: Relationship updates if relevant
        if player_action.get("type") == "talk":
            await self.relationship_manager.update_relationship_from_interaction(
                "player",  # entity type
                self.user_id,
                player_action,
                response_action
            )

        # Record an interaction memory with enhanced emotional processing
        memory_text = (
            f"Player {player_action.get('description','???')} -> I responded by "
            f"{response_action.get('description','???')}"
        )
        
        # Extract emotional data from the result
        emotional_impact = result.get("emotional_impact", 0)
        emotional_intensity = abs(emotional_impact) * 20  # Scale to 0-100
        
        # Add the interaction memory
        await self.memory_manager.add_memory(
            memory_text=memory_text,
            memory_type="interaction",
            significance=3,
            emotional_intensity=emotional_intensity,
            tags=["interaction", "player", player_action.get("type", "unknown")]
        )
        
        # Update emotional state
        if emotional_impact != 0:
            # Map the impact to an emotion
            emotion = "neutral"
            if emotional_impact > 2:
                emotion = "joy"
            elif emotional_impact > 0:
                emotion = "satisfaction"
            elif emotional_impact < -2:
                emotion = "anger"
            elif emotional_impact < 0:
                emotion = "sadness"
                
            # Calculate intensity (0-1 scale)
            intensity = min(1.0, abs(emotional_impact) / 5.0)
            
            # Update emotional state
            await self.memory_manager.update_emotional_state(
                primary_emotion=emotion,
                intensity=intensity,
                trigger=f"Player action: {player_action.get('description', 'unknown')}"
            )

        logger.debug("NPCAgent %s processed player action '%s': result=%s", self.npc_id, player_action, result)

        return {
            "npc_id": self.npc_id,
            "action": response_action,
            "result": result
        }

    async def perform_scheduled_activity(self) -> Optional[Dict[str, Any]]:
        """
        Perform the activity scheduled for this NPC at the current time of day.
        Returns an optional dict with the action & result.

        Steps:
          1) Load current time/day from DB
          2) Find appropriate schedule entry
          3) Execute the scheduled action
          4) Record a memory of the routine

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
                    logger.debug("No schedule found for NPC %s on day='%s' time='%s'.", self.npc_id, day_name, time_of_day)
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

            # 4) Record a memory for routine using enhanced memory system
            await self.memory_manager.add_memory(
                memory_text=f"I did '{activity_desc}' as scheduled for {day_name} {time_of_day}",
                memory_type="routine",
                significance=2,  # Low-medium significance for routine activities
                tags=["routine", "scheduled", day_name.lower(), time_of_day.lower()]
            )

            logger.debug("NPCAgent %s performed scheduled activity: %s => %s", self.npc_id, activity_desc, result)
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
            # Run memory maintenance
            return await self.memory_manager.run_maintenance()
        except Exception as e:
            logger.error(f"Error running memory maintenance for NPC {self.npc_id}: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal helper methods for scheduling/time
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
