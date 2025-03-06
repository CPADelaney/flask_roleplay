# logic/npc_agents/agent_system.py

"""
Main system that integrates NPC agents with the game loop
"""

import logging
from typing import List, Dict, Any, Optional
from db.connection import get_db_connection
from .npc_agent import NPCAgent
from .agent_coordinator import NPCAgentCoordinator

class NPCAgentSystem:
    """Main system that integrates individual NPC agents with the game loop"""

    def __init__(self, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.coordinator = NPCAgentCoordinator(user_id, conversation_id)
        self.npc_agents = {}
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize agents for all NPCs in the conversation"""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT npc_id
                FROM NPCStats
                WHERE user_id=%s
                  AND conversation_id=%s
            """, (self.user_id, self.conversation_id))
            rows = cursor.fetchall()
            for row in rows:
                npc_id = row[0]
                self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        finally:
            cursor.close()
            conn.close()

    async def handle_player_action(self, player_action, context=None):
        """Handle a player action and determine NPC responses"""
        context = context or {}

        # Which NPCs are affected by the player action?
        affected_npcs = await self.determine_affected_npcs(player_action, context)
        if not affected_npcs:
            return {"npc_responses": []}

        # If only one NPC, handle direct
        if len(affected_npcs) == 1:
            return await self.handle_single_npc_interaction(affected_npcs[0], player_action, context)

        # If multiple, handle group
        return await self.handle_group_npc_interaction(affected_npcs, player_action, context)

    async def determine_affected_npcs(self, player_action, context):
        """Determine which NPCs are affected by a player action"""
        target_npc_id = player_action.get("target_npc_id")
        if target_npc_id:
            return [target_npc_id]

        location = player_action.get("target_location", context.get("location"))
        if not location:
            # Try to get from CurrentRoleplay
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND key = 'CurrentLocation'
                """, (self.user_id, self.conversation_id))
                row = cursor.fetchone()
                if row:
                    location = row[0]
            except Exception as e:
                logging.error(f"Error getting CurrentLocation: {e}")
            finally:
                conn.close()

        if location:
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT npc_id
                    FROM NPCStats
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND current_location = %s
                      AND introduced = TRUE
                """, (self.user_id, self.conversation_id, location))
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logging.error(f"Error getting NPCs in location: {e}")
                return []
            finally:
                conn.close()

        # If no location but action is "talk", find recently active NPCs
        if player_action.get("type") == "talk":
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT npc_id
                    FROM NPCAgentState
                    WHERE user_id = %s
                      AND conversation_id = %s
                    ORDER BY last_updated DESC
                    LIMIT 3
                """, (self.user_id, self.conversation_id))
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logging.error(f"Error getting recently active NPCs: {e}")
                return []
            finally:
                conn.close()

        return []

    async def handle_single_npc_interaction(self, npc_id, player_action, context):
        """Handle a player action directed at a single NPC"""
        if npc_id not in self.npc_agents:
            self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)

        response = await self.npc_agents[npc_id].process_player_action(player_action)
        return {"npc_responses": [response]}

    async def handle_group_npc_interaction(self, npc_ids, player_action, context):
        """Handle a player action directed at multiple NPCs"""
        return await self.coordinator.handle_player_action(player_action, context, npc_ids)

    async def process_npc_scheduled_activities(self):
        """Process scheduled activities for all NPCs"""
        npc_responses = []
        for npc_id, agent in self.npc_agents.items():
            response = await agent.perform_scheduled_activity()
            if response:
                npc_responses.append(response)
        return {"npc_responses": npc_responses}

    async def get_npc_name(self, npc_id):
        """Get the name of an NPC by ID"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_name
                FROM NPCStats
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (npc_id, self.user_id, self.conversation_id))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return f"NPC_{npc_id}"
        except Exception as e:
            logging.error(f"Error getting NPC name: {e}")
            return f"NPC_{npc_id}"
        finally:
            conn.close()
