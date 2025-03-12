# nyx/integrate.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
import asyncpg

from nyx.nyx_agent import NyxAgent
from logic.npc_agents.agent_coordinator import NPCAgentCoordinator
from logic.npc_agents.agent_system import NPCAgentSystem
from db.connection import get_db_connection

logger = logging.getLogger(__name__)

class GameEventManager:
    """Manages events that both Nyx and NPCs should be aware of"""
    
    def __init__(self, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.nyx_agent = NyxAgent(user_id, conversation_id)
        self.npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
    
    async def broadcast_event(self, event_type, event_data):
        """Broadcast event to both Nyx and NPCs"""
        logger.info(f"Broadcasting event {event_type} to Nyx and NPCs")
        
        # Tell Nyx about the event
        await self.nyx_agent.process_game_event(event_type, event_data)
        
        # Tell NPCs about the event
        affected_npcs = event_data.get("affected_npcs")
        if not affected_npcs:
            # If no specific NPCs mentioned, determine who would know
            affected_npcs = await self._determine_aware_npcs(event_type, event_data)
        
        if affected_npcs:
            await self.npc_coordinator.batch_update_npcs(
                affected_npcs,
                "event_update",
                {"event_type": event_type, "event_data": event_data}
            )
            
            logger.info(f"Event {event_type} broadcast to {len(affected_npcs)} NPCs")
        else:
            logger.info(f"No NPCs affected by event {event_type}")
            
        return {
            "event_type": event_type,
            "nyx_notified": True,
            "npcs_notified": len(affected_npcs) if affected_npcs else 0,
            "aware_npcs": affected_npcs
        }
    
    async def _determine_aware_npcs(self, event_type, event_data):
        """
        Determine which NPCs would be aware of an event based on:
        - NPCs in the same location
        - NPCs who have a relationship with involved entities
        - NPCs with relevant traits (perceptive, curious, etc.)
        """
        aware_npcs = []
        
        # Get current location from event_data
        location = event_data.get("location")
        if not location:
            # Try to get current location from game state
            try:
                async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow("""
                            SELECT value FROM CurrentRoleplay 
                            WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                        """, self.user_id, self.conversation_id)
                        
                        if row:
                            location = row["value"]
            except Exception as e:
                logger.error(f"Error getting location for event awareness: {e}")
        
        # If we have a location, get NPCs there
        if location:
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = %s AND conversation_id = %s AND current_location = %s
                    """, (self.user_id, self.conversation_id, location))
                    
                    for row in cursor.fetchall():
                        aware_npcs.append(row[0])
            except Exception as e:
                logger.error(f"Error getting NPCs in location: {e}")
        
        # Add NPCs with high perception or specific traits that would notice
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                # Query NPCs with perceptive traits
                cursor.execute("""
                    SELECT npc_id, personality_traits 
                    FROM NPCStats
                    WHERE user_id = %s AND conversation_id = %s
                """, (self.user_id, self.conversation_id))
                
                for row in cursor.fetchall():
                    npc_id = row[0]
                    traits = row[1]
                    
                    # Skip if already aware
                    if npc_id in aware_npcs:
                        continue
                    
                    # Parse traits
                    if isinstance(traits, str):
                        try:
                            traits_list = json.loads(traits)
                        except json.JSONDecodeError:
                            traits_list = []
                    else:
                        traits_list = traits or []
                    
                    # Check for perceptive traits
                    perceptive_traits = ["observant", "perceptive", "vigilant", "alert", "curious"]
                    if any(trait in traits_list for trait in perceptive_traits):
                        aware_npcs.append(npc_id)
        except Exception as e:
            logger.error(f"Error getting perceptive NPCs: {e}")
        
        # Add NPCs with relationships to involved entities
        involved_entities = event_data.get("involved_entities", [])
        for entity in involved_entities:
            if entity.get("type") == "npc":
                try:
                    entity_id = entity.get("id")
                    if entity_id:
                        with get_db_connection() as conn, conn.cursor() as cursor:
                            cursor.execute("""
                                SELECT entity1_id FROM SocialLinks
                                WHERE entity2_type = 'npc' AND entity2_id = %s AND link_level > 60
                                  AND user_id = %s AND conversation_id = %s
                            """, (entity_id, self.user_id, self.conversation_id))
                            
                            for row in cursor.fetchall():
                                related_npc = row[0]
                                if related_npc not in aware_npcs:
                                    aware_npcs.append(related_npc)
                except Exception as e:
                    logger.error(f"Error getting related NPCs: {e}")
        
        # For certain event types, add NPCs with specific interests
        if event_type in ["scene_change", "ritual", "player_status_change"]:
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = %s AND conversation_id = %s AND scheming_level > 7
                    """, (self.user_id, self.conversation_id))
                    
                    for row in cursor.fetchall():
                        scheming_npc = row[0]
                        if scheming_npc not in aware_npcs:
                            aware_npcs.append(scheming_npc)
            except Exception as e:
                logger.error(f"Error getting scheming NPCs: {e}")
        
        return aware_npcs


class NyxNPCIntegrationManager:
    """Manages integration between Nyx and NPC agents"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.nyx_agent = NyxAgent(user_id, conversation_id)
        self.npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
        self.npc_system = None  # Lazy-loaded
    
    async def get_npc_system(self):
        """Lazy-load the NPC system"""
        if self.npc_system is None:
            # Create a connection pool (you'd need to adjust this based on your actual implementation)
            pool = await asyncpg.create_pool(dsn=get_db_connection())
            self.npc_system = NPCAgentSystem(self.user_id, self.conversation_id, pool)
            await self.npc_system.initialize_agents()
        return self.npc_system
    
    async def process_user_input(self, user_input: str, context: Dict[str, Any]):
        """Process user input through both systems in an integrated way"""
        logger.info(f"Processing user input through Nyx and NPC integration")
        
        # First get Nyx's response
        nyx_response = await self.nyx_agent.process_input(user_input, context)
        
        # Extract NPC guidance from Nyx's response
        npc_guidance = self._extract_npc_guidance(nyx_response)
        
        # Add Nyx's guidance to context
        npc_context = context.copy()
        npc_context["nyx_guidance"] = npc_guidance
        
        # Get NPC responses with Nyx's guidance
        npc_responses = []
        if "responding_npcs" in npc_guidance and npc_guidance["responding_npcs"]:
            # Convert to player action for NPC system
            player_action = {
                "description": user_input,
                "type": "talk"
            }
            
            # Process through NPC system
            npc_system = await self.get_npc_system()
            npc_result = await npc_system.handle_player_action(
                player_action,
                npc_context
            )
            
            npc_responses = npc_result.get("npc_responses", [])
        
        # Combine responses
        combined_response = self._combine_responses(nyx_response, npc_responses)
        return combined_response
    
    def _extract_npc_guidance(self, nyx_response):
        """
        Extract guidance for NPCs from Nyx's response.
        
        This includes:
        - Which NPCs should respond
        - How they should respond (tone, content)
        - Any specific instructions for behavior
        """
        guidance = {
            "responding_npcs": [],
            "tone_guidance": {},
            "content_guidance": {},
            "nyx_expectations": {}
        }
        
        # Extract NPC guidance if it exists
        npc_guidance = nyx_response.get("npc_guidance")
        if npc_guidance:
            # Direct inclusion if structure matches
            return npc_guidance
        
        # If no explicit guidance, try to infer from response
        response_text = nyx_response.get("text", "")
        
        # Check for explicit NPC mentions
        npc_mentions = []
        
        # Get all NPCs for this conversation
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE user_id = %s AND conversation_id = %s
                """, (self.user_id, self.conversation_id))
                
                for row in cursor.fetchall():
                    npc_id, npc_name = row
                    # Check if NPC is mentioned by name
                    if npc_name in response_text:
                        npc_mentions.append(npc_id)
                        # Simple extraction of tone guidance
                        if "angrily" in response_text or "angry" in response_text:
                            guidance["tone_guidance"][npc_id] = "angry"
                        elif "surprised" in response_text or "shock" in response_text:
                            guidance["tone_guidance"][npc_id] = "surprised"
                        elif "nervously" in response_text or "nervous" in response_text:
                            guidance["tone_guidance"][npc_id] = "nervous"
        except Exception as e:
            logger.error(f"Error extracting NPC mentions: {e}")
        
        # If NPCs were explicitly mentioned, they should respond
        if npc_mentions:
            guidance["responding_npcs"] = npc_mentions
        
        # If no NPCs were explicitly mentioned, check for location-based cues
        if not guidance["responding_npcs"] and "location" in nyx_response:
            location = nyx_response.get("location")
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = %s AND conversation_id = %s AND current_location = %s
                    """, (self.user_id, self.conversation_id, location))
                    
                    for row in cursor.fetchall():
                        guidance["responding_npcs"].append(row[0])
            except Exception as e:
                logger.error(f"Error getting location-based NPCs: {e}")
        
        # Limit to max 3 responding NPCs to avoid overwhelming responses
        if len(guidance["responding_npcs"]) > 3:
            guidance["responding_npcs"] = guidance["responding_npcs"][:3]
        
        return guidance
    
    def _combine_responses(self, nyx_response, npc_responses):
        """
        Combine Nyx's response with NPC responses into a coherent output.
        
        Args:
            nyx_response: Response from Nyx
            npc_responses: List of NPC responses
            
        Returns:
            Combined response
        """
        combined = {
            "text": nyx_response.get("text", ""),
            "npc_responses": npc_responses,
            "generate_image": nyx_response.get("generate_image", False),
            "tension_level": nyx_response.get("tension_level", 0)
        }
        
        # Process any special fields from Nyx response
        for key in ["metadata", "location", "time_advancement", "environment_update"]:
            if key in nyx_response:
                combined[key] = nyx_response[key]
        
        # Add NPC dialogue snippets if available
        if npc_responses:
            npc_dialogue = []
            for response in npc_responses:
                npc_id = response.get("npc_id")
                
                # Get NPC name
                npc_name = "Unknown NPC"
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT npc_name FROM NPCStats
                            WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                        """, (npc_id, self.user_id, self.conversation_id))
                        
                        row = cursor.fetchone()
                        if row:
                            npc_name = row[0]
                except Exception as e:
                    logger.error(f"Error getting NPC name: {e}")
                
                # Add formatted dialogue
                action = response.get("action", {})
                result = response.get("result", {})
                
                if isinstance(result, dict):
                    dialogue = f"{npc_name}: {result.get('outcome', '')}"
                    npc_dialogue.append(dialogue)
            
            # Add NPC dialogue to combined response
            combined["npc_dialogue"] = npc_dialogue
        
        return combined
    
    async def transition_scene_with_npcs(
        self,
        user_id: int,
        conversation_id: int,
        new_location: str,
        transition_context: Dict[str, Any]
    ):
        """Manage scene transition with NPC movement"""
        logger.info(f"Transitioning scene to {new_location} with NPCs")
        
        # First, handle Nyx's scene transition
        # Assuming Nyx has a transition_scene method to handle scene transitions
        scene_result = await self.nyx_agent.process_input(
            f"We're now moving to {new_location}",
            {"transition_to": new_location, **transition_context}
        )
        
        # Get NPCs that should move to the new location
        npcs_to_move = transition_context.get("accompanying_npcs", [])
        
        if not npcs_to_move:
            # If not specified, check if current location is known
            old_location = transition_context.get("current_location")
            if old_location:
                # Get NPCs at current location (they should move with the player)
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT npc_id FROM NPCStats
                            WHERE user_id = %s AND conversation_id = %s AND current_location = %s
                        """, (user_id, conversation_id, old_location))
                        
                        for row in cursor.fetchall():
                            npcs_to_move.append(row[0])
                except Exception as e:
                    logger.error(f"Error getting NPCs at location: {e}")
        
        # Move NPCs to the new location
        if npcs_to_move:
            await self.npc_coordinator.batch_update_npcs(
                npcs_to_move,
                "location_change",
                {"new_location": new_location}
            )
            
            # Generate emotions and memory for NPCs about the location change
            await self.npc_coordinator.batch_update_npcs(
                npcs_to_move,
                "memory_update",
                {"memory_text": f"I moved to {new_location} with the player", 
                "tags": ["location_change"]}
            )
            
            logger.info(f"Moved {len(npcs_to_move)} NPCs to {new_location}")
        
        # Enhance the scene result with NPC information
        scene_result["npcs_moved"] = npcs_to_move
        scene_result["new_location"] = new_location
        
        return scene_result
    
    async def process_group_decision_with_nyx(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make group decisions with Nyx's guidance.
        
        Args:
            npc_ids: IDs of NPCs involved in the decision
            shared_context: Shared context for the group decision
            
        Returns:
            Result of the group decision process
        """
        logger.info(f"Making group decision with Nyx's guidance for {len(npc_ids)} NPCs")
        
        # First, get Nyx's guidance for the scene
        nyx_input = f"I need guidance for a group interaction with the following NPCs: {npc_ids}"
        nyx_response = await self.nyx_agent.process_input(nyx_input, shared_context)
        
        # Extract guidance
        nyx_guidance = nyx_response.get("text", "")
        
        # Parse for helpful instructions
        try:
            # Simple parsing: look for sections about each NPC
            npc_guidance = {}
            for npc_id in npc_ids:
                # Try to get NPC name
                npc_name = f"NPC_{npc_id}"
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (npc_id, self.user_id, self.conversation_id))
                    
                    row = cursor.fetchone()
                    if row:
                        npc_name = row[0]
                
                # Look for guidance about this NPC
                if npc_name in nyx_guidance:
                    start_idx = nyx_guidance.find(npc_name)
                    end_idx = nyx_guidance.find("\n\n", start_idx)
                    if end_idx == -1:
                        end_idx = len(nyx_guidance)
                    
                    npc_section = nyx_guidance[start_idx:end_idx]
                    npc_guidance[npc_id] = npc_section
        except Exception as e:
            logger.error(f"Error parsing Nyx guidance: {e}")
            npc_guidance = {}
        
        # Add Nyx's guidance to context
        enhanced_context = shared_context.copy()
        enhanced_context["nyx_guidance"] = nyx_guidance
        enhanced_context["npc_specific_guidance"] = npc_guidance
        
        # Make group decisions with Nyx's guidance
        result = await self.npc_coordinator.make_group_decisions(
            npc_ids,
            enhanced_context
        )
        
        # Add Nyx's guidance to result
        result["nyx_guidance"] = nyx_guidance
        result["npc_specific_guidance"] = npc_guidance
        
        return result


async def run_joint_memory_maintenance(user_id: int, conversation_id: int):
    """Maintain both Nyx and NPC memories with coordination"""
    logger.info(f"Running joint memory maintenance for user {user_id}, conversation {conversation_id}")
    
    # Run Nyx memory maintenance
    nyx_agent = NyxAgent(user_id, conversation_id)
    nyx_result = await nyx_agent.run_memory_maintenance()
    
    # Run NPC memory maintenance
    npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
    npc_ids = await npc_coordinator.load_agents()
    
    # Create reflections based on Nyx's insights
    if "reflections" in nyx_result:
        for reflection in nyx_result["reflections"]:
            await npc_coordinator.batch_update_npcs(
                npc_ids,
                "belief_update",
                {"belief_text": f"Nyx believes: {reflection}",
                "confidence": 0.8,
                "topic": "nyx_perspective"}
            )
    
    # Perform NPC memory maintenance
    npc_maintenance_results = {}
    batch_size = 5  # Process NPCs in smaller batches
    
    for i in range(0, len(npc_ids), batch_size):
        batch = npc_ids[i:i+batch_size]
        
        batch_results = {}
        for npc_id in batch:
            try:
                # Get the NPC agent
                if npc_id in npc_coordinator.active_agents:
                    agent = npc_coordinator.active_agents[npc_id]
                    
                    # Run memory maintenance
                    result = await agent.run_memory_maintenance()
                    batch_results[npc_id] = result
            except Exception as e:
                logger.error(f"Error in memory maintenance for NPC {npc_id}: {e}")
                batch_results[npc_id] = {"error": str(e)}
        
        # Save batch results
        npc_maintenance_results.update(batch_results)
        
        # Small delay between batches
        if i + batch_size < len(npc_ids):
            await asyncio.sleep(0.1)
    
    # Share important NPC memories with Nyx
    for npc_id, result in npc_maintenance_results.items():
        if "important_memories" in result:
            for memory in result.get("important_memories", []):
                try:
                    # Get NPC name
                    npc_name = f"NPC_{npc_id}"
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT npc_name FROM NPCStats
                            WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                        """, (npc_id, user_id, conversation_id))
                        
                        row = cursor.fetchone()
                        if row:
                            npc_name = row[0]
                    
                    # Share with Nyx's memory system
                    await nyx_agent.memory_system.add_memory(
                        memory_text=f"{npc_name}: {memory.get('text', '')}",
                        memory_type="observation",
                        memory_scope="game",
                        significance=memory.get("significance", 5),
                        tags=["npc_memory", f"npc_{npc_id}"],
                        metadata={"source_npc": npc_id, "npc_name": npc_name}
                    )
                except Exception as e:
                    logger.error(f"Error sharing NPC memory with Nyx: {e}")
    
    return {
        "nyx_maintenance": nyx_result,
        "npc_maintenance": npc_maintenance_results,
        "npcs_processed": len(npc_ids),
        "shared_memories": sum(1 for r in npc_maintenance_results.values() 
                            for m in r.get("important_memories", []))
    }
