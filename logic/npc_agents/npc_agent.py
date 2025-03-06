# logic/npc_agents/npc_agent.py

"""
Core NPC agent class that manages individual NPC behavior
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .memory_manager import NPCMemoryManager
from .decision_engine import NPCDecisionEngine
from .relationship_manager import NPCRelationshipManager
from .environment_perception import (
    fetch_environment_data,
    is_significant_action,
    execute_npc_action
)

class NPCAgent:
    """Independent AI agent controlling a single NPC's behavior"""

    def __init__(self, npc_id, user_id, conversation_id):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

        self.memory_manager = NPCMemoryManager(npc_id, user_id, conversation_id)
        self.decision_engine = NPCDecisionEngine(npc_id, user_id, conversation_id)
        self.relationship_manager = NPCRelationshipManager(npc_id, user_id, conversation_id)

        self.last_perception = None

    async def perceive_environment(self, current_context):
        """Update NPC's perception of the current environment and context"""
        # Get environment data (location, entities, time)
        environment_data = await fetch_environment_data(
            self.user_id,
            self.conversation_id,
            current_context
        )

        # Retrieve relevant memories
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(current_context)

        # Update relationship awareness
        relationship_data = await self.relationship_manager.update_relationships(current_context)

        # Combine
        perception = {
            "environment": environment_data,
            "relevant_memories": relevant_memories,
            "relationships": relationship_data,
            "timestamp": datetime.now().isoformat()
        }

        # Store the current perception
        await self.store_perception(perception)
        self.last_perception = perception
        return perception

    async def make_decision(self, perception=None, available_actions=None):
        """Decide what action to take based on current perceptions"""
        if perception is None:
            if self.last_perception is None:
                # If no prior perception, get a fresh one
                perception = await self.perceive_environment({})
            else:
                perception = self.last_perception

        return await self.decision_engine.decide(perception, available_actions)

    async def execute_action(self, action, context):
        """Execute the chosen action in the game world"""
        result = await execute_npc_action(
            self.npc_id,
            self.user_id,
            self.conversation_id,
            action,
            context
        )

        # Record in memory if significant
        if is_significant_action(action, result):
            await self.memory_manager.add_memory(
                f"I {action['description']} with result: {result['outcome']}",
                memory_type="action",
                significance=action.get("significance", 3),
                emotional_valence=result.get("emotional_impact", 0)
            )

        return result

    async def store_perception(self, perception):
        """Store current perception in the agent state"""
        conn = None
        try:
            from db.connection import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if we have a state record
            cursor.execute("""
                SELECT 1
                FROM NPCAgentState
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            exists = cursor.fetchone() is not None

            # Keep only essential parts
            compact_perception = {
                "environment": perception["environment"],
                "timestamp": perception["timestamp"]
            }

            if exists:
                cursor.execute("""
                    UPDATE NPCAgentState
                    SET current_state = %s,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE npc_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """, (
                    json.dumps(compact_perception),
                    self.npc_id,
                    self.user_id,
                    self.conversation_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO NPCAgentState
                        (npc_id, user_id, conversation_id, current_state, last_updated)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    self.npc_id,
                    self.user_id,
                    self.conversation_id,
                    json.dumps(compact_perception)
                ))
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Error storing perception: {e}")
        finally:
            if conn:
                conn.close()

    async def process_player_action(self, player_action):
        """Process a player action and generate an appropriate response"""
        # Update perception based on player action
        context = {
            "player_action": player_action,
            "description": f"Player {player_action['description']}"
        }
        perception = await self.perceive_environment(context)

        # Decide on a response
        response_action = await self.make_decision(perception)

        # Execute
        result = await self.execute_action(response_action, context)

        # Update relationships if relevant
        if player_action["type"] == "talk":
            await self.relationship_manager.update_relationship_from_interaction(
                "player", self.user_id, player_action, response_action
            )

        # Create a memory of this interaction
        await self.memory_manager.add_memory(
            f"Player {player_action['description']} and I responded by {response_action['description']}",
            memory_type="interaction",
            significance=3,
            emotional_valence=result.get("emotional_impact", 0) / 10
        )

        return {
            "npc_id": self.npc_id,
            "action": response_action,
            "result": result
        }

    async def perform_scheduled_activity(self):
        """Perform the activity scheduled for the current time"""
        conn = None
        try:
            from db.connection import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()

            # Get current time from DB
            cursor.execute("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND key = 'TimeOfDay'
            """, (self.user_id, self.conversation_id))
            time_row = cursor.fetchone()
            time_of_day = time_row[0] if time_row else "Morning"

            # Current day
            cursor.execute("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND key = 'CurrentDay'
            """, (self.user_id, self.conversation_id))
            day_row = cursor.fetchone()
            current_day = int(day_row[0]) if day_row and day_row[0].isdigit() else 1

            # Day names
            cursor.execute("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND key = 'CalendarNames'
            """, (self.user_id, self.conversation_id))
            names_row = cursor.fetchone()
            if names_row and names_row[0]:
                try:
                    calendar_data = json.loads(names_row[0])
                    day_names = calendar_data.get("days", [
                        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
                    ])
                except:
                    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            else:
                day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

            day_index = (current_day - 1) % len(day_names)
            day_name = day_names[day_index]

            # NPC's schedule
            cursor.execute("""
                SELECT schedule
                FROM NPCStats
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            sched_row = cursor.fetchone()
            if not sched_row or not sched_row[0]:
                return None

            try:
                schedule = json.loads(sched_row[0]) if isinstance(sched_row[0], str) else sched_row[0]
            except:
                schedule = {}

            # Get current activity
            if day_name not in schedule or time_of_day not in schedule[day_name]:
                return None

            activity_desc = schedule[day_name][time_of_day]
            action = {
                "type": "scheduled",
                "description": activity_desc,
                "target": "environment",
                "stats_influenced": {}
            }

            # Execute
            context = {
                "day": day_name,
                "time": time_of_day,
                "location": "scheduled_location"
            }
            result = await self.execute_action(action, context)

            # Create a memory
            await self.memory_manager.add_memory(
                f"I {activity_desc} as scheduled for {day_name} {time_of_day}",
                memory_type="routine",
                significance=1,
                emotional_valence=0
            )
            return {
                "npc_id": self.npc_id,
                "action": action,
                "result": result
            }
        except Exception as e:
            logging.error(f"Error in perform_scheduled_activity: {e}")
            return None
        finally:
            if conn:
                conn.close()
