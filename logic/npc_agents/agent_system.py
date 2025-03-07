# logic/npc_agents/agent_system.py

"""
Main system that integrates NPC agents with the game loop, enhanced with memory integration.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from db.connection import get_db_connection
from .npc_agent import NPCAgent
from .agent_coordinator import NPCAgentCoordinator
from memory.wrapper import MemorySystem

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
    - Coordinate memory-related operations across NPCs.
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
        self._memory_system = None
        self.initialize_agents()

    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system

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
        Handle a player action and determine NPC responses with memory integration.

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

        # Get memory system for creating player action memories
        memory_system = await self._get_memory_system()

        # Create a memory of this action from the player's perspective
        player_memory_text = f"I {player_action.get('description', 'did something')}"
        
        # Add memory with appropriate tags
        await memory_system.remember(
            entity_type="player",
            entity_id=self.user_id,
            memory_text=player_memory_text,
            importance="medium",
            tags=["player_action", player_action.get("type", "unknown")]
        )

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
        Figure out which NPCs are affected by a given player action with improved memory awareness.

        Prioritizes:
        1. A 'target_npc_id' in the player_action
        2. NPCs in the specified location
        3. If action is "talk", fallback to the last 3 recently active NPCs
        4. NPCs with high relevance to the context based on memories

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
        location_npcs = []
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
                    location_npcs = [row[0] for row in cursor.fetchall()]
                except Exception as e:
                    logger.error("Error getting NPCs in location '%s': %s", location, e)

        if location_npcs:
            # If we have action text, we can prioritize NPCs by memory relevance
            if "description" in player_action and player_action.get("description"):
                try:
                    # Use the memory system to find NPCs most relevant to this context
                    memory_system = await self._get_memory_system()
                    
                    # Create a relevance score for each NPC based on memories
                    relevant_npcs = []
                    for npc_id in location_npcs:
                        # Get NPC memories relevant to this action
                        memory_result = await memory_system.recall(
                            entity_type="npc",
                            entity_id=npc_id,
                            query=player_action.get("description", ""),
                            limit=3
                        )
                        
                        memories = memory_result.get("memories", [])
                        # Calculate relevance based on memory count and significance
                        relevance = len(memories)
                        for memory in memories:
                            # Add significance to relevance
                            if "significance" in memory:
                                relevance += memory["significance"]
                        
                        relevant_npcs.append((npc_id, relevance))
                    
                    # Sort by relevance, highest first
                    relevant_npcs.sort(key=lambda x: x[1], reverse=True)
                    
                    # If we have relevant NPCs, prioritize them
                    if relevant_npcs and relevant_npcs[0][1] > 0:
                        # Get up to 3 most relevant NPCs
                        return [npc_id for npc_id, _ in relevant_npcs[:3]]
                except Exception as e:
                    logger.error("Error calculating NPC relevance: %s", e)
            
            # Default to location-based NPCs if we can't determine relevance
            return location_npcs

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
        Handle a player action directed at a single NPC with enhanced memory integration.

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

        # Get memory-related context to enhance interaction
        memory_system = await self._get_memory_system()
        
        # Enhance context with memory-related information
        enhanced_context = context.copy()
        
        # Add emotional state
        try:
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            enhanced_context["npc_emotional_state"] = emotional_state
        except Exception as e:
            logger.error(f"Error getting NPC emotional state: {e}")
        
        # Add mask information
        try:
            mask_info = await memory_system.get_npc_mask(npc_id)
            enhanced_context["npc_mask"] = mask_info
        except Exception as e:
            logger.error(f"Error getting NPC mask: {e}")
        
        # Add beliefs about the player
        try:
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id,
                topic="player"
            )
            enhanced_context["npc_beliefs_about_player"] = beliefs
        except Exception as e:
            logger.error(f"Error getting NPC beliefs: {e}")
        
        # Check if there are recent flashbacks for context
        try:
            flashback = await memory_system.npc_flashback(npc_id, player_action.get("description", ""))
            if flashback:
                enhanced_context["triggered_flashback"] = flashback
        except Exception as e:
            logger.error(f"Error checking for flashbacks: {e}")

        # Process the player action with the enhanced context
        agent = self.npc_agents[npc_id]
        response = await agent.process_player_action(player_action)
        
        # Create a memory of this interaction from the player's perspective
        try:
            # Get NPC name for better memory context
            npc_name = await self.get_npc_name(npc_id)
            result = response.get("result", {})
            
            player_memory_text = f"I {player_action.get('description', 'did something')} to {npc_name} and they {result.get('outcome', 'responded')}"
            
            await memory_system.remember(
                entity_type="player",
                entity_id=self.user_id,
                memory_text=player_memory_text,
                importance="medium",
                tags=["npc_interaction", player_action.get("type", "unknown")]
            )
        except Exception as e:
            logger.error(f"Error creating player memory: {e}")
        
        return {"npc_responses": [response]}

    async def handle_group_npc_interaction(self, npc_ids: List[int], player_action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs, delegating to the coordinator with memory integration.

        Args:
            npc_ids: List of NPC IDs that are all affected
            player_action: The player's action
            context: Additional context

        Returns:
            A dictionary possibly containing "npc_responses"
        """
        logger.info("Handling group NPC interaction: %s", npc_ids)
        
        # Get memory system for group interaction context
        memory_system = await self._get_memory_system()
        
        # Enhance context with group history
        enhanced_context = context.copy()
        enhanced_context["is_group_interaction"] = True
        
        # Get memories of previous group interactions
        group_context = {
            "participants": npc_ids,
            "type": "group_interaction"
        }
        
        # Add to the context a memory of the last group interaction if available
        player_group_memories = await memory_system.recall(
            entity_type="player",
            entity_id=self.user_id,
            query="group interaction",
            context=group_context,
            limit=1
        )
        
        if player_group_memories.get("memories"):
            enhanced_context["previous_group_interaction"] = player_group_memories["memories"][0]
            
        # Get the result from the coordinator
        result = await self.coordinator.handle_player_action(player_action, enhanced_context, npc_ids)
        
        # Create a memory of this group interaction for the player
        try:
            # Get NPC names for better memory content
            npc_names = []
            for npc_id in npc_ids:
                npc_name = await self.get_npc_name(npc_id)
                npc_names.append(npc_name)
                
            npc_list = ", ".join(npc_names)
            
            player_memory_text = f"I {player_action.get('description', 'interacted with')} a group including {npc_list}"
            
            await memory_system.remember(
                entity_type="player",
                entity_id=self.user_id,
                memory_text=player_memory_text,
                importance="medium",
                tags=["group_interaction", player_action.get("type", "unknown")]
            )
        except Exception as e:
            logger.error(f"Error creating player group memory: {e}")
        
        return result

    async def process_npc_scheduled_activities(self) -> Dict[str, Any]:
        """
        Process scheduled activities for all NPCs with memory integration.

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
        
        # Create memories for the player of notable NPC activities
        memory_system = await self._get_memory_system()
        
        for response in npc_responses:
            # Only create memories for significant or visible activities
            result = response.get("result", {})
            action = response.get("action", {})
            npc_id = response.get("npc_id")
            
            if npc_id and action and result.get("outcome"):
                # Check if this is a significant activity worth remembering
                significance = self._determine_activity_significance(action, result)
                
                if significance > 0:
                    try:
                        # Get NPC name for better memory
                        npc_name = await self.get_npc_name(npc_id)
                        
                        # Create player memory of observing the activity
                        player_memory_text = f"I observed {npc_name} {action.get('description', 'doing something')}"
                        
                        importance = "medium" if significance > 1 else "low"
                        
                        await memory_system.remember(
                            entity_type="player",
                            entity_id=self.user_id,
                            memory_text=player_memory_text,
                            importance=importance,
                            tags=["npc_observation", "scheduled_activity"]
                        )
                    except Exception as e:
                        logger.error(f"Error creating player memory of NPC activity: {e}")

        return {"npc_responses": npc_responses}
    
    def _determine_activity_significance(self, action: Dict[str, Any], result: Dict[str, Any]) -> int:
        """
        Determine how significant an NPC activity is for player memory formation.
        
        Returns:
            0: Not worth remembering
            1: Minor significance
            2: Moderate significance
            3: High significance
        """
        # Check if this action would be visible to the player
        action_type = action.get("type", "unknown")
        outcome = result.get("outcome", "")
        emotional_impact = result.get("emotional_impact", 0)
        
        # Hidden or purely internal activities aren't remembered
        if action_type == "think" or action_type == "plan":
            return 0
            
        # High emotional impact actions are more memorable
        if abs(emotional_impact) > 2:
            return 3
            
        # Certain action types are more memorable
        if action_type in ["talk", "command", "mock", "emotional_outburst"]:
            return 2
            
        # Actions that involve visible change
        if "visibly" in outcome or "noticeably" in outcome:
            return 2
            
        # Default for standard activities
        return 1

    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance tasks on all NPCs' memory systems.
        
        Returns:
            Results of maintenance operations
        """
        results = {}
        
        try:
            memory_system = await self._get_memory_system()
            
            # Run player memory maintenance
            player_result = await memory_system.maintain(
                entity_type="player",
                entity_id=self.user_id
            )
            
            results["player_maintenance"] = player_result
            
            # Run maintenance for each NPC
            npc_results = {}
            for npc_id, agent in self.npc_agents.items():
                try:
                    npc_result = await agent.run_memory_maintenance()
                    npc_results[npc_id] = npc_result
                except Exception as e:
                    logger.error(f"Error in memory maintenance for NPC {npc_id}: {e}")
                    npc_results[npc_id] = {"error": str(e)}
            
            results["npc_maintenance"] = npc_results
            
            # Run maintenance for the DM (Nyx) memory
            try:
                nyx_result = await memory_system.maintain(
                    entity_type="nyx",
                    entity_id=0
                )
                results["nyx_maintenance"] = nyx_result
            except Exception as e:
                logger.error(f"Error in Nyx memory maintenance: {e}")
                results["nyx_maintenance"] = {"error": str(e)}
                
            return results
        except Exception as e:
            logger.error(f"Error in system-wide memory maintenance: {e}")
            return {"error": str(e)}

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
    
    async def get_all_npc_beliefs_about_player(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get all NPCs' beliefs about the player.
        Useful for understanding how the player is perceived.
        
        Returns:
            Dictionary mapping NPC IDs to lists of beliefs
        """
        results = {}
        memory_system = await self._get_memory_system()
        
        for npc_id, agent in self.npc_agents.items():
            try:
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic="player"
                )
                
                # Only include NPCs that have formed beliefs
                if beliefs:
                    results[npc_id] = beliefs
            except Exception as e:
                logger.error(f"Error getting beliefs for NPC {npc_id}: {e}")
        
        return results
    
    async def get_player_beliefs_about_npcs(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get player's beliefs about each NPC.
        
        Returns:
            Dictionary mapping NPC IDs to lists of beliefs
        """
        results = {}
        memory_system = await self._get_memory_system()
        
        for npc_id in self.npc_agents:
            try:
                # Format topic for this specific NPC
                topic = f"npc_{npc_id}"
                
                beliefs = await memory_system.get_beliefs(
                    entity_type="player",
                    entity_id=self.user_id,
                    topic=topic
                )
                
                # Only include NPCs that the player has formed beliefs about
                if beliefs:
                    results[npc_id] = beliefs
            except Exception as e:
                logger.error(f"Error getting player beliefs about NPC {npc_id}: {e}")
        
        return results

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
