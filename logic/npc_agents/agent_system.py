# logic/npc_agents/agent_system.py

"""
Main system that integrates NPC agents with the game loop, enhanced with memory integration.
"""

import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta

from db.connection import get_db_connection
from .npc_agent import NPCAgent
from .agent_coordinator import NPCAgentCoordinator
from memory.wrapper import MemorySystem

logger = logging.getLogger(__name__)

class NPCSystemError(Exception):
    """Error in NPC system operations."""
    pass
    
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
        # Track when memory maintenance was last run
        self._last_memory_maintenance = datetime.now() - timedelta(hours=1)  # Run on first init
        # Track NPC emotional states to detect significant changes
        self._npc_emotional_states = {}
        # Track last flashback times to prevent too-frequent occurrences
        self._last_flashback_times = {}
        self.initialize_agents()

        self._setup_memory_maintenance_schedule()

    def _setup_memory_maintenance_schedule(self):
        """Schedule periodic memory maintenance for all NPCs."""
        async def run_maintenance_cycle():
            while True:
                try:
                    # Run maintenance every 15 minutes
                    await asyncio.sleep(900)  # 15 minutes in seconds
                    
                    # Log maintenance start
                    logger.info("Starting scheduled memory maintenance cycle")
                    
                    # Run maintenance
                    results = await self.run_memory_maintenance()
                    
                    # Log completion
                    logger.info(f"Completed memory maintenance: {results}")
                    
                except Exception as e:
                    logger.error(f"Error in scheduled memory maintenance: {e}")
        
        # Start the maintenance task
        asyncio.create_task(run_maintenance_cycle())

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
        
        # Determine emotional content based on action type
        action_type = player_action.get("type", "unknown")
        is_emotional = action_type in ["express_emotion", "shout", "cry", "laugh", "threaten"]
        
        # Add memory with appropriate tags
        await memory_system.remember(
            entity_type="player",
            entity_id=self.user_id,
            memory_text=player_memory_text,
            importance="medium",
            emotional=is_emotional,
            tags=["player_action", action_type]
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
                        
                        # Check for beliefs that might make this NPC more interested
                        beliefs = await memory_system.get_beliefs(
                            entity_type="npc",
                            entity_id=npc_id,
                            topic="player"
                        )
                        
                        # Increase relevance if NPC has beliefs about the player
                        belief_relevance = 0
                        for belief in beliefs:
                            if any(term in player_action.get("description", "").lower() 
                                  for term in belief.get("belief", "").lower().split()):
                                belief_relevance += belief.get("confidence", 0.5) * 2
                        
                        relevance += belief_relevance
                        
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

    async def batch_update_npcs(
        self,
        npc_ids: List[int],
        update_type: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update multiple NPCs in a single batch operation for better performance."""
        results = {
            "success_count": 0,
            "error_count": 0,
            "details": {}
        }
        
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)
        
        # Process different update types
        if update_type == "location_change":
            # Batch location update
            new_location = update_data.get("new_location")
            if not new_location:
                return {"error": "No location specified"}
                
            async with db_transaction() as conn:
                # Update all NPCs in a single query
                query = """
                    UPDATE NPCStats
                    SET current_location = $1
                    WHERE npc_id = ANY($2)
                    AND user_id = $3
                    AND conversation_id = $4
                    RETURNING npc_id
                """
                rows = await conn.fetch(
                    query, 
                    new_location, 
                    npc_ids, 
                    self.user_id, 
                    self.conversation_id
                )
                
                results["success_count"] = len(rows)
                results["updated_npcs"] = [r["npc_id"] for r in rows]
                
        elif update_type == "emotional_update":
            # Batch emotional state update
            emotion = update_data.get("emotion")
            intensity = update_data.get("intensity", 0.5)
            
            if not emotion:
                return {"error": "No emotion specified"}
                
            # Get memory system
            memory_system = await self._get_memory_system()
            
            # Process in smaller batches for better control
            batch_size = 5
            for i in range(0, len(npc_ids), batch_size):
                batch = npc_ids[i:i+batch_size]
                
                # Process each NPC in batch
                batch_tasks = []
                for npc_id in batch:
                    task = memory_system.update_npc_emotion(
                        npc_id=npc_id,
                        emotion=emotion,
                        intensity=intensity
                    )
                    batch_tasks.append(task)
                    
                # Execute batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for npc_id, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        results["error_count"] += 1
                        results["details"][npc_id] = {"error": str(result)}
                    else:
                        results["success_count"] += 1
                        results["details"][npc_id] = {"success": True}
        
        # Add other update types as needed
        
        return results

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
            
            # Store current emotional state to track changes
            self._npc_emotional_states[npc_id] = emotional_state
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
            # Only check for flashback if enough time has passed since the last one
            last_flashback_time = self._last_flashback_times.get(npc_id, datetime.now() - timedelta(hours=1))
            time_since_last = (datetime.now() - last_flashback_time).total_seconds()
            
            # Don't do flashbacks too frequently - at most every 10 minutes
            if time_since_last > 600:
                # Roll for flashback - 15% chance
                if random.random() < 0.15:
                    flashback = await memory_system.npc_flashback(npc_id, player_action.get("description", ""))
                    if flashback:
                        enhanced_context["triggered_flashback"] = flashback
                        # Record this flashback time
                        self._last_flashback_times[npc_id] = datetime.now()
        except Exception as e:
            logger.error(f"Error checking for flashbacks: {e}")
        
        # Check for memories relevant to this interaction
        try:
            relevant_memories = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query=player_action.get("description", "interaction with player"),
                context={"action_type": player_action.get("type", "unknown")},
                limit=3
            )
            
            if relevant_memories and "memories" in relevant_memories:
                enhanced_context["relevant_memories"] = relevant_memories["memories"]
        except Exception as e:
            logger.error(f"Error fetching relevant memories: {e}")

        # Process the player action with the enhanced context
        agent = self.npc_agents[npc_id]
        response = await agent.process_player_action(player_action, enhanced_context)
        
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
        
        # Check for emotion changes and potentially update the NPC's emotional state
        try:
            # Determine if the interaction likely caused an emotional response
            if self._should_update_emotion(player_action, response):
                # Generate a new emotional state based on the interaction
                new_emotion = await self._determine_new_emotion(
                    npc_id, 
                    player_action,
                    response,
                    emotional_state
                )
                
                if new_emotion:
                    # Update the NPC's emotional state
                    await memory_system.update_npc_emotion(
                        npc_id=npc_id,
                        emotion=new_emotion["name"],
                        intensity=new_emotion["intensity"]
                    )
        except Exception as e:
            logger.error(f"Error updating NPC emotional state: {e}")
        
        # Check for mask slippage
        try:
            # Determine if the interaction might cause mask slippage
            if self._should_check_mask_slippage(player_action, response, mask_info):
                # Calculate slippage probability
                slippage_chance = self._calculate_mask_slippage_chance(
                    player_action,
                    response,
                    mask_info
                )
                
                # Roll for slippage
                if random.random() < slippage_chance:
                    # Generate mask slippage
                    await memory_system.reveal_npc_trait(
                        npc_id=npc_id,
                        trigger=player_action.get("description", "interaction with player")
                    )
        except Exception as e:
            logger.error(f"Error processing mask slippage: {e}")
        
        # Update or create beliefs based on this interaction
        try:
            self._process_belief_updates(
                npc_id,
                player_action,
                response,
                beliefs
            )
        except Exception as e:
            logger.error(f"Error updating beliefs: {e}")
        
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
            
        # Add emotional states for all NPCs
        emotional_states = {}
        for npc_id in npc_ids:
            try:
                emotional_state = await memory_system.get_npc_emotion(npc_id)
                if emotional_state:
                    emotional_states[npc_id] = emotional_state
                    # Update stored state
                    self._npc_emotional_states[npc_id] = emotional_state
            except Exception as e:
                logger.error(f"Error getting emotional state for NPC {npc_id}: {e}")
        
        enhanced_context["emotional_states"] = emotional_states
        
        # Add mask information for all NPCs
        mask_states = {}
        for npc_id in npc_ids:
            try:
                mask_info = await memory_system.get_npc_mask(npc_id)
                if mask_info:
                    mask_states[npc_id] = mask_info
            except Exception as e:
                logger.error(f"Error getting mask info for NPC {npc_id}: {e}")
        
        enhanced_context["mask_states"] = mask_states
        
        # Add beliefs about the player for all NPCs
        npc_beliefs = {}
        for npc_id in npc_ids:
            try:
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic="player"
                )
                if beliefs:
                    npc_beliefs[npc_id] = beliefs
            except Exception as e:
                logger.error(f"Error getting beliefs for NPC {npc_id}: {e}")
        
        enhanced_context["npc_beliefs"] = npc_beliefs
        
        # Check for flashbacks in the group setting
        flashbacks = {}
        for npc_id in npc_ids:
            try:
                # Only check for flashback if enough time has passed since the last one
                last_flashback_time = self._last_flashback_times.get(npc_id, datetime.now() - timedelta(hours=1))
                time_since_last = (datetime.now() - last_flashback_time).total_seconds()
                
                # Don't do flashbacks too frequently and only 10% chance in group settings
                if time_since_last > 600 and random.random() < 0.1:
                    flashback = await memory_system.npc_flashback(npc_id, player_action.get("description", ""))
                    if flashback:
                        flashbacks[npc_id] = flashback
                        # Record this flashback time
                        self._last_flashback_times[npc_id] = datetime.now()
            except Exception as e:
                logger.error(f"Error checking flashback for NPC {npc_id}: {e}")
        
        enhanced_context["flashbacks"] = flashbacks
        
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
        
        # Check for consequential emotional changes and mask slippages from the group interaction
        for npc_id in npc_ids:
            try:
                # Process emotional changes from group interactions
                # Find the NPC's response
                npc_response = None
                for response in result.get("npc_responses", []):
                    if response.get("npc_id") == npc_id:
                        npc_response = response
                        break
                
                if npc_response:
                    # Check for emotional changes
                    if self._should_update_emotion(player_action, npc_response):
                        new_emotion = await self._determine_new_emotion(
                            npc_id, 
                            player_action,
                            npc_response,
                            emotional_states.get(npc_id)
                        )
                        
                        if new_emotion:
                            await memory_system.update_npc_emotion(
                                npc_id=npc_id,
                                emotion=new_emotion["name"],
                                intensity=new_emotion["intensity"]
                            )
                    
                    # Check for mask slippage
                    if self._should_check_mask_slippage(player_action, npc_response, mask_states.get(npc_id)):
                        slippage_chance = self._calculate_mask_slippage_chance(
                            player_action,
                            npc_response,
                            mask_states.get(npc_id)
                        )
                        
                        # Roll for slippage - slightly higher chance in group settings
                        if random.random() < (slippage_chance * 1.2):
                            await memory_system.reveal_npc_trait(
                                npc_id=npc_id,
                                trigger=f"group interaction about {player_action.get('description', 'something')}"
                            )
            except Exception as e:
                logger.error(f"Error processing group interaction effects for NPC {npc_id}: {e}")
        
        return result

    async def process_npc_scheduled_activities(self) -> Dict[str, Any]:
        """
        Process scheduled activities for all NPCs using the agent system.
        Optimized for better performance with many NPCs through batching.
        """
        logger.info("Processing scheduled activities")
        
        try:
            # Get current time information for context
            year, month, day, time_of_day = await self.get_current_game_time()
            
            # Create base context for all NPCs
            base_context = {
                "year": year,
                "month": month,
                "day": day,
                "time_of_day": time_of_day,
                "activity_type": "scheduled"
            }
            
            # Get all NPCs with their current locations - batch query for performance
            npc_data = await self._fetch_all_npc_data_for_activities()
            
            # Count total NPCs to process
            total_npcs = len(npc_data)
            if total_npcs == 0:
                return {"npc_responses": [], "count": 0}
                    
            logger.info(f"Processing scheduled activities for {total_npcs} NPCs")
            
            # For very large NPC counts, process in batches rather than all at once
            batch_size = 20  # Adjust based on system capabilities
            npc_responses = []
            
            # Process in batches
            for i in range(0, total_npcs, batch_size):
                batch = list(npc_data.items())[i:i+batch_size]
                
                # Create tasks for this batch
                batch_tasks = []
                for npc_id, data in batch:
                    batch_tasks.append(
                        self._process_single_npc_activity(npc_id, data, base_context)
                    )
                
                # Run batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process batch results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing scheduled activity: {result}")
                    elif result:  # Skip None results
                        npc_responses.append(result)
                    
                # If we have multiple batches, add a small delay between them to reduce system load
                if i + batch_size < total_npcs:
                    await asyncio.sleep(0.1)
                
            # After all NPCs processed, do agent system coordination
            try:
                # Call the internal coordination method instead of recursively calling this method
                agent_responses = await self._process_coordination_activities(base_context)
            except Exception as e:
                logger.error(f"Error in agent system coordination: {e}")
                agent_responses = []
            
            # Combined results from individual processing and agent system
            combined_results = {
                "npc_responses": npc_responses,
                "agent_system_responses": agent_responses,
                "count": len(npc_responses) + len(agent_responses)
            }
            
            # Add summary statistics
            combined_results["stats"] = {
                "total_npcs": total_npcs,
                "successful_activities": len(npc_responses),
                "time_of_day": time_of_day,
                "processing_time": None  # Could add timing info here
            }
            
            return combined_results
                
        except Exception as e:
            error_msg = f"Error processing NPC scheduled activities: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)

    async def _process_coordination_activities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process NPC coordination activities - handling group interactions and influences.
        This separates the coordination logic from the main activity processing to avoid recursion.
        
        Args:
            context: Base context for activities
            
        Returns:
            List of coordination activity responses
        """
        logger.info("Processing NPC coordination activities")
        
        try:
            # Get NPCs that are in the same location (for potential group activities)
            location_groups = await self._get_npcs_by_location()
            
            # Process each location group for potential interactions
            coordination_responses = []
            
            for location, npc_ids in location_groups.items():
                # Only process groups with multiple NPCs
                if len(npc_ids) < 2:
                    continue
                    
                # Find dominant NPCs in each location who might initiate group activities
                dominant_npcs = await self._find_dominant_npcs(npc_ids)
                
                for dom_npc_id in dominant_npcs:
                    # Create group context
                    group_context = context.copy()
                    group_context["location"] = location
                    group_context["group_members"] = npc_ids
                    group_context["initiator_id"] = dom_npc_id
                    
                    # Check if this group should interact
                    if await self._should_group_interact(dom_npc_id, npc_ids, group_context):
                        # Use coordinator to handle the group interaction
                        group_result = await self.coordinator.make_group_decisions(
                            npc_ids, 
                            group_context
                        )
                        
                        if group_result:
                            coordination_responses.append({
                                "type": "group_interaction",
                                "location": location,
                                "initiator": dom_npc_id,
                                "participants": npc_ids,
                                "result": group_result
                            })
            
            return coordination_responses
            
        except Exception as e:
            logger.error(f"Error in coordination activities: {e}")
            return []
            
    async def _find_dominant_npcs(self, npc_ids: List[int]) -> List[int]:
        """Find NPCs with high dominance that might initiate group activities."""
        dominant_npcs = []
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT npc_id 
                FROM NPCStats
                WHERE npc_id = ANY(%s)
                AND dominance > 65
                ORDER BY dominance DESC
            """, (npc_ids,))
            
            dominant_npcs = [row[0] for row in cursor.fetchall()]
        
        return dominant_npcs
    
    async def _should_group_interact(self, initiator_id: int, group_members: List[int], context: Dict[str, Any]) -> bool:
        """Determine if a group interaction should occur based on social dynamics."""
        # Base chance - 30%
        interaction_chance = 0.3
        
        # Check time of day - certain times are more social
        time_of_day = context.get("time_of_day", "")
        if time_of_day == "evening":
            interaction_chance += 0.2  # More group interactions in evening
        
        # Check if initiator has high dominance and cruelty (femdom context)
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT dominance, cruelty
                FROM NPCStats
                WHERE npc_id = %s
            """, (initiator_id,))
            
            row = cursor.fetchone()
            if row:
                dominance, cruelty = row
                
                # Highly dominant NPCs more likely to initiate group dynamics
                if dominance > 80:
                    interaction_chance += 0.2
                    
                # Cruel dominants enjoy group discipline scenes
                if cruelty > 70:
                    interaction_chance += 0.15
        
        # Random roll against the calculated chance
        return random.random() < interaction_chance
    
    async def _get_npcs_by_location(self) -> Dict[str, List[int]]:
        """Group NPCs by their current location."""
        location_groups = {}
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT npc_id, current_location
                FROM NPCStats
                WHERE user_id = %s
                AND conversation_id = %s
                AND current_location IS NOT NULL
            """, (self.user_id, self.conversation_id))
            
            for row in cursor.fetchall():
                npc_id, location = row
                
                if location not in location_groups:
                    location_groups[location] = []
                    
                location_groups[location].append(npc_id)
        
        return location_groups
    
    async def _fetch_all_npc_data_for_activities(self) -> Dict[int, Dict[str, Any]]:
        """
        Batch fetch NPC data needed for scheduled activities.
        Returns {npc_id: {data}} dictionary.
        """
        npc_data = {}
        
        try:
            # Use connection pool for better performance
            async with self.connection_pool.acquire() as conn:
                # Efficient batch query
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, schedule, 
                           dominance, cruelty, introduced
                    FROM NPCStats 
                    WHERE user_id=$1 AND conversation_id=$2
                """, self.user_id, self.conversation_id)
                
                for row in rows:
                    npc_id = row["npc_id"]
                    
                    # Parse JSON fields with error handling
                    schedule = None
                    if row["schedule"]:
                        try:
                            if isinstance(row["schedule"], str):
                                schedule = json.loads(row["schedule"])
                            else:
                                schedule = row["schedule"]
                        except json.JSONDecodeError:
                            schedule = {}
                    
                    npc_data[npc_id] = {
                        "name": row["npc_name"],
                        "location": row["current_location"],
                        "schedule": schedule,
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"],
                        "introduced": row["introduced"]
                    }
                    
            return npc_data
        except Exception as e:
            logger.error(f"Error fetching NPC data for activities: {e}")
            return {}
        
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
        Run comprehensive maintenance tasks on all NPCs' memory systems.
        This includes consolidation, decay, schema formation, and belief updates.
        
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
                    npc_result = await self._run_comprehensive_npc_maintenance(npc_id)
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

    async def _run_comprehensive_npc_maintenance(self, npc_id: int) -> Dict[str, Any]:
        """
        Run comprehensive memory maintenance for a single NPC.
        This includes basic maintenance, belief formation, schema detection,
        emotional decay, and mask evolution.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Results of maintenance operations
        """
        results = {}
        memory_system = await self._get_memory_system()
        
        # Basic memory maintenance (consolidation, decay, archiving)
        try:
            basic_result = await memory_system.maintain(
                entity_type="npc",
                entity_id=npc_id
            )
            results["basic_maintenance"] = basic_result
        except Exception as e:
            logger.error(f"Error in basic maintenance for NPC {npc_id}: {e}")
            results["basic_maintenance"] = {"error": str(e)}
        
        # Schema detection - find patterns in memories
        try:
            schema_result = await memory_system.generate_schemas(
                entity_type="npc",
                entity_id=npc_id
            )
            results["schema_generation"] = schema_result
        except Exception as e:
            logger.error(f"Error in schema generation for NPC {npc_id}: {e}")
            results["schema_generation"] = {"error": str(e)}
        
        # Belief updates - refine confidence in beliefs based on recent experiences
        try:
            # Get all beliefs
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id
            )
            
            belief_updates = 0
            for belief in beliefs:
                belief_id = belief.get("id")
                if belief_id and random.random() < 0.3:  # Only process some beliefs each cycle
                    # Slightly adjust confidence based on memory evidence
                    confidence_change = random.uniform(-0.05, 0.05)
                    
                    # Ensure confidence remains in valid range
                    new_confidence = max(0.1, min(0.95, belief.get("confidence", 0.5) + confidence_change))
                    
                    # Update the belief
                    await memory_system.update_belief_confidence(
                        entity_type="npc",
                        entity_id=npc_id,
                        belief_id=belief_id,
                        new_confidence=new_confidence,
                        reason="Regular belief reassessment"
                    )
                    
                    belief_updates += 1
                    
            results["belief_updates"] = belief_updates
        except Exception as e:
            logger.error(f"Error updating beliefs for NPC {npc_id}: {e}")
            results["belief_updates"] = {"error": str(e)}
        
        # Process emotional decay - emotions naturally fade over time
        try:
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            if emotional_state and "current_emotion" in emotional_state:
                current = emotional_state["current_emotion"]
                emotion_name = current.get("primary")
                intensity = current.get("intensity", 0.0)
                
                # Non-neutral emotions naturally decay over time
                if emotion_name != "neutral" and intensity > 0.3:
                    # Calculate decay - stronger emotions decay faster
                    decay_amount = 0.1 if intensity > 0.7 else 0.05
                    
                    # Reduce intensity
                    new_intensity = max(0.1, intensity - decay_amount)
                    
                    # If intensity is very low, revert to neutral
                    if new_intensity < 0.25:
                        emotion_name = "neutral"
                        new_intensity = 0.1
                    
                    # Update emotional state
                    await memory_system.update_npc_emotion(
                        npc_id=npc_id,
                        emotion=emotion_name,
                        intensity=new_intensity
                    )
                    
                    results["emotional_decay"] = {
                        "old_emotion": emotion_name,
                        "old_intensity": intensity,
                        "new_intensity": new_intensity,
                        "decayed": True
                    }
                else:
                    results["emotional_decay"] = {"decayed": False}
            else:
                results["emotional_decay"] = {"state_found": False}
        except Exception as e:
            logger.error(f"Error in emotional decay for NPC {npc_id}: {e}")
            results["emotional_decay"] = {"error": str(e)}
        
        # Mask evolution - random chance for mask integrity changes
        try:
            # Masks very slowly rebuild over time for most NPCs
            mask_info = await memory_system.get_npc_mask(npc_id)
            
            if mask_info:
                integrity = mask_info.get("integrity", 100)
                
                # Only process masks that aren't perfect or completely broken
                if 0 < integrity < 100:
                    # 80% chance mask slowly rebuilds, 20% chance it weakens further
                    if random.random() < 0.8:
                        # Small increase in integrity
                        new_integrity = min(100, integrity + random.uniform(0.2, 1.0))
                        
                        # We can't directly modify mask integrity, but we can create
                        # a small rebuilding effect by not triggering slippage events
                        results["mask_evolution"] = {
                            "old_integrity": integrity,
                            "estimated_new_integrity": new_integrity,
                            "direction": "rebuilding"
                        }
                    else:
                        # Random chance for slight mask deterioration
                        # Generate a very minor mask slip
                        await memory_system.reveal_npc_trait(
                            npc_id=npc_id,
                            trigger="introspection",
                            severity=1  # Minimal severity
                        )
                        
                        results["mask_evolution"] = {
                            "old_integrity": integrity,
                            "direction": "weakening",
                            "slip_generated": True
                        }
                else:
                    results["mask_evolution"] = {"no_change": True}
            else:
                results["mask_evolution"] = {"mask_not_found": True}
        except Exception as e:
            logger.error(f"Error in mask evolution for NPC {npc_id}: {e}")
            results["mask_evolution"] = {"error": str(e)}
        
        return results

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
    
    async def generate_npc_flashback(self, npc_id: int, context_text: str) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC based on specific context.
        This can be used to create dramatic moments in the narrative.
        
        Args:
            npc_id: ID of the NPC
            context_text: Text context that might trigger the flashback
            
        Returns:
            Flashback information or None if no flashback was generated
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Explicitly generate a flashback
            flashback = await memory_system.npc_flashback(npc_id, context_text)
            
            if flashback:
                # Record this flashback time
                self._last_flashback_times[npc_id] = datetime.now()
                
                # Update emotional state based on flashback
                # Flashbacks often cause emotional responses
                emotional_state = await memory_system.get_npc_emotion(npc_id)
                
                if emotional_state:
                    # Determine appropriate emotional response
                    # Most flashbacks are negative/traumatic
                    emotion = "fear"
                    intensity = 0.7
                    
                    # If flashback text suggests other emotions, adjust accordingly
                    text = flashback.get("text", "").lower()
                    if "happy" in text or "joy" in text or "good" in text:
                        emotion = "joy"
                    elif "anger" in text or "angr" in text or "rage" in text:
                        emotion = "anger"
                    elif "sad" in text or "sorrow" in text:
                        emotion = "sadness"
                    
                    # Update emotional state
                    await memory_system.update_npc_emotion(
                        npc_id=npc_id,
                        emotion=emotion,
                        intensity=intensity
                    )
            
            return flashback
        except Exception as e:
            logger.error(f"Error generating flashback for NPC {npc_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Helper methods for emotion and mask processing
    # ------------------------------------------------------------------
    
    def _should_update_emotion(self, player_action: Dict[str, Any], npc_response: Dict[str, Any]) -> bool:
        """
        Determine if an NPC's emotional state should be updated based on the interaction.
        
        Args:
            player_action: The player's action
            npc_response: The NPC's response
            
        Returns:
            True if the emotion should be updated, False otherwise
        """
        # Check player action type for emotional impact
        action_type = player_action.get("type", "").lower()
        emotional_action = action_type in [
            "express_emotion", "flirt", "threaten", "comfort", "insult", 
            "praise", "mock", "support", "challenge", "provoke"
        ]
        
        # Check response for emotional impact
        result = npc_response.get("result", {})
        emotional_impact = result.get("emotional_impact", 0)
        
        # Update if action is emotional or impact is significant
        return emotional_action or abs(emotional_impact) >= 2
    
    async def _determine_new_emotion(
        self, 
        npc_id: int, 
        player_action: Dict[str, Any], 
        npc_response: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, str]]:
        """
        Determine a new emotional state for an NPC based on interaction.
        
        Args:
            npc_id: ID of the NPC
            player_action: The player's action
            npc_response: The NPC's response
            current_state: Current emotional state (optional)
            
        Returns:
            New emotion information or None if no change
        """
        # Extract relevant information
        action_type = player_action.get("type", "").lower()
        action_desc = player_action.get("description", "").lower()
        result = npc_response.get("result", {})
        emotional_impact = result.get("emotional_impact", 0)
        
        # Default new emotion (only used if change needed)
        new_emotion = {
            "name": "neutral",
            "intensity": 0.3
        }
        
        # Determine emotion based on action type
        if action_type == "praise" or action_type == "comfort" or action_type == "support":
            new_emotion["name"] = "joy"
            new_emotion["intensity"] = 0.6
        elif action_type == "insult" or action_type == "mock":
            # Check for personality traits that might affect response
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT dominance, cruelty
                        FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (npc_id, self.user_id, self.conversation_id))
                    row = cursor.fetchone()
                    
                    if row:
                        dominance, cruelty = row
                        
                        # High dominance and cruelty NPCs are more likely to respond with anger
                        if dominance > 60 or cruelty > 60:
                            new_emotion["name"] = "anger"
                        else:
                            new_emotion["name"] = "sadness"
                        
                        new_emotion["intensity"] = min(0.8, 0.5 + (max(dominance, cruelty) / 100))
                    else:
                        # Default to sadness for most NPCs
                        new_emotion["name"] = "sadness"
                        new_emotion["intensity"] = 0.6
            except Exception as e:
                logger.error(f"Error getting NPC stats for emotion determination: {e}")
                # Default
                new_emotion["name"] = "sadness"
                new_emotion["intensity"] = 0.6
        elif action_type == "threaten":
            new_emotion["name"] = "fear"
            new_emotion["intensity"] = 0.7
        elif action_type == "flirt":
            # Determine response based on relationship and prior beliefs
            try:
                memory_system = await self._get_memory_system()
                
                # Get beliefs about player
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic="player"
                )
                
                # Determine response based on beliefs
                positive_belief = False
                for belief in beliefs:
                    belief_text = belief.get("belief", "").lower()
                    if ("attract" in belief_text or "like" in belief_text or 
                        "interest" in belief_text) and belief.get("confidence", 0) > 0.5:
                        positive_belief = True
                        break
                
                if positive_belief:
                    new_emotion["name"] = "joy"
                    new_emotion["intensity"] = 0.7
                else:
                    # Default to mild surprise
                    new_emotion["name"] = "surprise"
                    new_emotion["intensity"] = 0.5
            except Exception as e:
                logger.error(f"Error determining flirt response: {e}")
                # Default
                new_emotion["name"] = "surprise"
                new_emotion["intensity"] = 0.5
        
        # Use emotional impact from response if more significant than action type
        if abs(emotional_impact) >= 3:
            # Strong emotional impact overrides default
            if emotional_impact > 3:
                new_emotion["name"] = "joy"
                new_emotion["intensity"] = 0.7
            elif emotional_impact > 0:
                new_emotion["name"] = "joy"
                new_emotion["intensity"] = 0.5
            elif emotional_impact < -3:
                # For strong negative impact, randomize between anger and sadness
                new_emotion["name"] = "anger" if random.random() < 0.5 else "sadness"
                new_emotion["intensity"] = 0.7
            elif emotional_impact < 0:
                new_emotion["name"] = "sadness"
                new_emotion["intensity"] = 0.5
        
        # Check if this is a significant change from current state
        if current_state and "current_emotion" in current_state:
            current = current_state["current_emotion"]
            current_name = current.get("primary")
            current_intensity = current.get("intensity", 0)
            
            # Only update if significant change
            if (current_name == new_emotion["name"] and 
                abs(current_intensity - new_emotion["intensity"]) < 0.2):
                return None
        
        return new_emotion
    
    def _should_check_mask_slippage(
        self, 
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        mask_info: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Determine if mask slippage should be checked based on the interaction.
        
        Args:
            player_action: The player's action
            npc_response: The NPC's response
            mask_info: The NPC's mask information
            
        Returns:
            True if mask slippage should be checked, False otherwise
        """
        # If no mask or perfect integrity, no need to check
        if not mask_info or mask_info.get("integrity", 100) >= 95:
            return False
        
        # Check for challenging actions that might stress the mask
        action_type = player_action.get("type", "").lower()
        challenging_action = action_type in [
            "threaten", "challenge", "accuse", "provoke", "mock", "insult"
        ]
        
        # Check for emotional response that might weaken the mask
        result = npc_response.get("result", {})
        emotional_impact = abs(result.get("emotional_impact", 0))
        
        # Check if mask is already compromised
        compromised_mask = mask_info.get("integrity", 100) < 70
        
        # Check for slippage if action is challenging, impact is high, or mask is compromised
        return challenging_action or emotional_impact > 2 or compromised_mask
    
    def _calculate_mask_slippage_chance(
        self,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        mask_info: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate the probability of mask slippage based on interaction.
        
        Args:
            player_action: The player's action
            npc_response: The NPC's response
            mask_info: The NPC's mask information
            
        Returns:
            Probability of mask slippage (0.0-1.0)
        """
        if not mask_info:
            return 0.0
        
        # Base chance based on mask integrity
        # Lower integrity = higher chance
        integrity = mask_info.get("integrity", 100)
        base_chance = max(0.0, (100 - integrity) / 200)  # 0.0 to 0.5
        
        # Increase chance for challenging actions
        action_type = player_action.get("type", "").lower()
        challenging_action = action_type in [
            "threaten", "challenge", "accuse", "provoke", "mock", "insult"
        ]
        
        if challenging_action:
            base_chance += 0.1
        
        # Increase chance based on emotional impact
        result = npc_response.get("result", {})
        emotional_impact = abs(result.get("emotional_impact", 0))
        
        impact_factor = min(0.3, emotional_impact * 0.05)
        base_chance += impact_factor
        
        # Cap the maximum chance
        return min(0.75, base_chance)
    
    async def _process_belief_updates(
        self,
        npc_id: int,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        existing_beliefs: List[Dict[str, Any]]
    ) -> None:
        """
        Process belief updates based on player-NPC interaction.
        
        Args:
            npc_id: ID of the NPC
            player_action: The player's action
            npc_response: The NPC's response
            existing_beliefs: Existing beliefs about the player
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Extract information from action and response
            action_type = player_action.get("type", "").lower()
            action_desc = player_action.get("description", "").lower()
            result = npc_response.get("result", {})
            outcome = result.get("outcome", "").lower()
            emotional_impact = result.get("emotional_impact", 0)
            
            # Determine if this interaction should update or form beliefs
            should_update = False
            
            # Actions that are likely to form/update beliefs
            belief_forming_actions = [
                "threaten", "praise", "insult", "support", "betray", "help",
                "challenge", "defend", "protect", "attack", "share", "confide",
                "flirt", "reject"
            ]
            
            if action_type in belief_forming_actions:
                should_update = True
            
            # High emotional impact interactions are also belief-forming
            if abs(emotional_impact) >= 3:
                should_update = True
            
            if should_update:
                # Determine positive or negative belief direction
                positive = False
                if action_type in ["praise", "support", "help", "defend", "protect", "share", "confide"]:
                    positive = True
                elif emotional_impact > 2:
                    positive = True
                
                # Find relevant existing beliefs to update
                belief_to_update = None
                
                for belief in existing_beliefs:
                    belief_text = belief.get("belief", "").lower()
                    
                    # Match positive beliefs
                    if positive and ("trust" in belief_text or "friend" in belief_text or 
                                    "ally" in belief_text or "good" in belief_text):
                        belief_to_update = belief
                        break
                    # Match negative beliefs
                    elif not positive and ("distrust" in belief_text or "enemy" in belief_text or 
                                         "threat" in belief_text or "danger" in belief_text):
                        belief_to_update = belief
                        break
                
                if belief_to_update:
                    # Update confidence in existing belief
                    belief_id = belief_to_update.get("id")
                    old_confidence = belief_to_update.get("confidence", 0.5)
                    
                    # Determine confidence adjustment
                    adjustment = 0.1  # Default adjustment
                    
                    # Stronger adjustment for more impactful interactions
                    if abs(emotional_impact) > 3:
                        adjustment = 0.15
                    
                    # Apply adjustment (increase for consistent, decrease for contradictory)
                    if (positive and "good" in belief_to_update.get("belief", "").lower()) or\
                       (not positive and "bad" in belief_to_update.get("belief", "").lower()):
                        # Consistent with existing belief
                        new_confidence = min(0.95, old_confidence + adjustment)
                    else:
                        # Contradicts existing belief
                        new_confidence = max(0.1, old_confidence - adjustment)
                    
                    await memory_system.update_belief_confidence(
                        entity_type="npc",
                        entity_id=npc_id,
                        belief_id=belief_id,
                        new_confidence=new_confidence,
                        reason=f"Based on player's {action_type} action"
                    )
                else:
                    # Create new belief
                    belief_text = ""
                    
                    if positive:
                        belief_text = "The player is someone I can trust or rely on."
                        if action_type == "flirt":
                            belief_text = "The player is showing romantic interest in me."
                        elif action_type == "help" or action_type == "protect":
                            belief_text = "The player is helpful and protects others."
                    else:
                        belief_text = "The player might be dangerous or untrustworthy."
                        if action_type == "threaten":
                            belief_text = "The player makes threats and could be dangerous."
                        elif action_type == "insult" or action_type == "mock":
                            belief_text = "The player is disrespectful and cruel."
                    
                    # Only create new belief if it seems significant enough
                    if belief_text:
                        await memory_system.create_belief(
                            entity_type="npc",
                            entity_id=npc_id,
                            belief_text=belief_text,
                            confidence=0.6  # Initial confidence
                        )
        except Exception as e:
            logger.error(f"Error processing belief updates: {e}")

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
