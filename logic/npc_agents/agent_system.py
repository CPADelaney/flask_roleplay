# logic/npc_agents/agent_system.py

"""
Main system that integrates NPC agents with the game loop, optimized for clarity and maintainability.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from db.connection import get_db_connection
from .npc_agent import NPCAgent
from .agent_coordinator import NPCAgentCoordinator

logger = logging.getLogger(__name__)


class NPCAgentSystem:
    """
    Main system that integrates individual NPC agents with the game loop.

    Responsibilities:
    - Load and store a reference to each NPC agent (NPCAgent).
    - Provide methods to handle player actions directed at NPC(s),
      determining which NPCs are affected, and dispatching to single
      or group interaction handlers.
    - Process scheduled activities for all NPCs.
    """

    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the agent system for a specific user & conversation.

        Args:
            user_id: The ID of the user/player
            conversation_id: The ID of the current conversation/scene
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.coordinator = NPCAgentCoordinator(user_id, conversation_id)
        self.npc_agents: Dict[int, NPCAgent] = {}
        self.initialize_agents()

    def initialize_agents(self) -> None:
        """
        Initialize NPCAgent objects for all NPCs in the conversation.
        """
        logger.info("Initializing NPC agents for user=%s, conversation=%s", self.user_id, self.conversation_id)

        query = """
            SELECT npc_id
            FROM NPCStats
            WHERE user_id=%s
              AND conversation_id=%s
        """
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, (self.user_id, self.conversation_id))
            rows = cursor.fetchall()

            for row in rows:
                npc_id = row[0]
                self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)

        logger.info("Loaded %d NPC agents", len(self.npc_agents))

    async def handle_player_action(self, player_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a player action and determine NPC responses.

        Depending on how many NPCs are affected (one vs many), this
        method will delegate to the appropriate single/group logic.

        Args:
            player_action: A dictionary describing what the player is doing or saying
            context: Optional additional context (like current location/time)

        Returns:
            A dictionary { "npc_responses": [...] } or for multi-npc, a different structure
        """
        if context is None:
            context = {}

        # Determine which NPCs are affected by the action
        affected_npcs = await self.determine_affected_npcs(player_action, context)
        if not affected_npcs:
            logger.debug("No NPCs were affected by this action: %s", player_action)
            return {"npc_responses": []}

        # Single NPC path
        if len(affected_npcs) == 1:
            npc_id = affected_npcs[0]
            return await self.handle_single_npc_interaction(npc_id, player_action, context)

        # Multiple NPCs => group logic
        return await self.handle_group_npc_interaction(affected_npcs, player_action, context)

    async def determine_affected_npcs(self, player_action: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """
        Figure out which NPCs are affected by a given player action.

        Prioritizes:
        1. A 'target_npc_id' in the player_action
        2. NPCs in the specified location
        3. If action is "talk", fallback to the last 3 recently active NPCs

        Returns:
            A list of NPC IDs that are relevant to this action.
        """
        target_npc_id = player_action.get("target_npc_id")
        if target_npc_id:
            return [target_npc_id]

        location = player_action.get("target_location", context.get("location"))
        if not location:
            # Attempt to get location from CurrentRoleplay table
            location = self._fetch_current_location()
            if not location:
                logger.debug("No location found in context or CurrentRoleplay; can't determine affected NPCs.")

        # If we have a location, get all introduced NPCs there
        if location:
            with get_db_connection() as conn, conn.cursor() as cursor:
                try:
                    cursor.execute("""
                        SELECT npc_id
                        FROM NPCStats
                        WHERE user_id = %s
                          AND conversation_id = %s
                          AND current_location = %s
                          AND introduced = TRUE
                    """, (self.user_id, self.conversation_id, location))
                    npc_ids = [row[0] for row in cursor.fetchall()]
                    if npc_ids:
                        return npc_ids
                except Exception as e:
                    logger.error("Error getting NPCs in location '%s': %s", location, e)

        # Fallback: if the action is "talk", get the last 3 recently active NPCs
        if player_action.get("type") == "talk":
            with get_db_connection() as conn, conn.cursor() as cursor:
                try:
                    cursor.execute("""
                        SELECT DISTINCT npc_id
                        FROM NPCAgentState
                        WHERE user_id = %s
                          AND conversation_id = %s
                        ORDER BY last_updated DESC
                        LIMIT 3
                    """, (self.user_id, self.conversation_id))
                    npc_ids = [row[0] for row in cursor.fetchall()]
                    return npc_ids
                except Exception as e:
                    logger.error("Error getting recently active NPCs: %s", e)

        return []

    async def handle_single_npc_interaction(self, npc_id: int, player_action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a player action directed at a single NPC.

        Args:
            npc_id: The ID of the targeted NPC
            player_action: The player's action
            context: Additional context

        Returns:
            A dictionary with the NPC's response in npc_responses
        """
        logger.info("Handling single NPC interaction with npc_id=%s", npc_id)

        if npc_id not in self.npc_agents:
            self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)

        agent = self.npc_agents[npc_id]
        response = await agent.process_player_action(player_action)
        return {"npc_responses": [response]}

    async def handle_group_npc_interaction(self, npc_ids: List[int], player_action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs, delegating to the coordinator.

        Args:
            npc_ids: List of NPC IDs that are all affected
            player_action: The player's action
            context: Additional context

        Returns:
            A dictionary possibly containing "npc_responses"
        """
        logger.info("Handling group NPC interaction: %s", npc_ids)
        return await self.coordinator.handle_player_action(player_action, context, npc_ids)

    async def process_npc_scheduled_activities(self) -> Dict[str, Any]:
        """
        Process scheduled activities for all NPCs.

        This method calls each NPCAgent's `perform_scheduled_activity()` asynchronously, 
        gathering any responses that might be generated.

        Returns:
            A dictionary of the form {"npc_responses": [...]}, where each element is
            what the NPC "did" or any relevant message.
        """
        logger.info("Processing scheduled activities for %d NPCs", len(self.npc_agents))

        tasks = [agent.perform_scheduled_activity() for agent in self.npc_agents.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None or exception results
        npc_responses = []
        for res in results:
            if isinstance(res, Exception):
                logger.error("NPC scheduled activity failed: %s", res)
            elif res:
                npc_responses.append(res)

        return {"npc_responses": npc_responses}

    async def get_npc_name(self, npc_id: int) -> str:
        """
        Get the name of an NPC by ID.

        Args:
            npc_id: The ID of the NPC

        Returns:
            The NPC's name or a fallback in case of error
        """
        query = """
            SELECT npc_name
            FROM NPCStats
            WHERE npc_id = %s
              AND user_id = %s
              AND conversation_id = %s
        """
        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, (npc_id, self.user_id, self.conversation_id))
                row = cursor.fetchone()
                if row:
                    return row[0]
                return f"NPC_{npc_id}"
            except Exception as e:
                logger.error("Error getting NPC name for npc_id=%s: %s", npc_id, e)
                return f"NPC_{npc_id}"

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _fetch_current_location(self) -> Optional[str]:
        """
        Attempt to retrieve the current location from the CurrentRoleplay table.

        Returns:
            The current location string, or None if not found or on error.
        """
        logger.debug("Fetching current location from CurrentRoleplay")

        query = """
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id = %s
              AND conversation_id = %s
              AND key = 'CurrentLocation'
        """
        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, (self.user_id, self.conversation_id))
                row = cursor.fetchone()
                return row[0] if row else None
            except Exception as e:
                logger.error("Error getting CurrentLocation: %s", e)
                return None
