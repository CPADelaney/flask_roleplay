# nyx/integrate.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
import asyncpg

from logic.npc_agents.agent_system import NPCAgentSystem
from db.connection import get_db_connection

logger = logging.getLogger(__name__)

class GameEventManager:
    """Manages events that both Nyx and NPCs should be aware of"""

    player_action = {
        "type": "talk",
        "description": user_input
    }
    
    def __init__(self, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.nyx_agent_sdk = NyxAgent(user_id, conversation_id)
        self.npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
    
    async def broadcast_event(self, event_type, event_data):
        """Broadcast event to both Nyx and NPCs"""
        logger.info(f"Broadcasting event {event_type} to Nyx and NPCs")
        
        # Tell Nyx about the event
        await self.nyx_agent_sdk.process_game_event(event_type, event_data)
        
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

    def get_nyx_agent(user_id, conversation_id):
        from nyx.nyx_agent_sdk import NyxAgent
        return NyxAgent(user_id, conversation_id)
        
    def get_npc_coordinator(user_id, conversation_id):
        from logic.npc_agents.agent_coordinator import NPCAgentCoordinator
        return NPCAgentCoordinator(user_id, conversation_id)
    
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
        self.nyx_agent_sdk = NyxAgent(user_id, conversation_id)
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
    
    def get_npc_coordinator(user_id, conversation_id):
        """Lazy-load NPCAgentCoordinator to avoid circular imports."""
        from logic.npc_agents.agent_coordinator import NPCAgentCoordinator
        return NPCAgentCoordinator(user_id, conversation_id)

    async def orchestrate_scene(
        self,
        location: str,
        player_action: str = None,
        involved_npcs: List[int] = None
    ) -> Dict[str, Any]:
        """
        Have Nyx orchestrate an entire scene, directing all NPCs.
        """
        # Gather context for the scene
        context = await self._gather_scene_context(location, player_action, involved_npcs)
        
        # Get Nyx's scene directive
        scene_directive = await self.make_scene_decision(context)
        
        # Have NPCs act according to the directive
        npc_responses = await self._execute_npc_directives(scene_directive)
        
        # Create a coherent scene narrative
        scene_narrative = await self.nyx_agent_sdk.create_scene_narrative(
            scene_directive, npc_responses, context
        )
        
        return {
            "narrative": scene_narrative,
            "npc_responses": npc_responses,
            "location": location
        }

    async def orchestrate_scene(
        self,
        location: str,
        player_action: str = None,
        involved_npcs: List[int] = None
    ) -> Dict[str, Any]:
        """
        Have Nyx orchestrate an entire scene, directing all NPCs.
        """
        # Gather context for the scene
        context = await self._gather_scene_context(location, player_action, involved_npcs)
        
        # Get Nyx's scene directive
        scene_directive = await self.make_scene_decision(context)
        
        # Have NPCs act according to the directive
        npc_responses = await self._execute_npc_directives(scene_directive)
        
        # Create a coherent scene narrative
        scene_narrative = await self.nyx_agent_sdk.create_scene_narrative(
            scene_directive, npc_responses, context
        )
        
        return {
            "narrative": scene_narrative,
            "npc_responses": npc_responses,
            "location": location
        }
    
    def _extract_npc_guidance(self, nyx_response):
        """
        Extract comprehensive guidance for NPCs from Nyx's response.
        
        This includes:
        - Which NPCs should respond
        - How they should respond (tone, content, emotions)
        - Any specific instructions for behavior
        - Conflicts or tensions to highlight
        """
        guidance = {
            "responding_npcs": [],
            "tone_guidance": {},
            "content_guidance": {},
            "emotion_guidance": {},  # Added emotional guidance
            "conflict_guidance": {},  # Added conflict guidance
            "nyx_expectations": {}
        }
        
        # Extract explicit NPC guidance if it exists
        npc_guidance = nyx_response.get("npc_guidance")
        if npc_guidance:
            # Direct inclusion if structure matches
            return npc_guidance
        
        # If no explicit guidance, try to infer from response
        response_text = nyx_response.get("text", "")
        
        # Check for explicit NPC mentions
        npc_mentions = []
        emotion_indicators = {
            "angry": ["angrily", "fuming", "enraged", "angry"],
            "surprised": ["surprised", "shocked", "startled", "astonished"],
            "nervous": ["nervously", "anxiously", "hesitantly", "timidly"],
            "pleased": ["pleased", "happily", "cheerfully", "gladly"],
            "suspicious": ["suspiciously", "warily", "cautiously", "skeptically"]
        }
        
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
                        
                        # Extract sentence containing NPC name for context
                        sentences = response_text.split('.')
                        for sentence in sentences:
                            if npc_name in sentence:
                                # Check for emotional indicators
                                for emotion, indicators in emotion_indicators.items():
                                    if any(indicator in sentence.lower() for indicator in indicators):
                                        guidance["tone_guidance"][npc_id] = emotion
                                        guidance["emotion_guidance"][npc_id] = emotion
                                        break
                                
                                # Check for content guidance
                                if "says" in sentence or "tells" in sentence or "asks" in sentence:
                                    # Extract what they might say
                                    content_start = sentence.find('"')
                                    content_end = sentence.rfind('"')
                                    if content_start != -1 and content_end != -1 and content_end > content_start:
                                        guidance["content_guidance"][npc_id] = sentence[content_start+1:content_end]
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
        
        # Look for conflict cues
        conflict_keywords = ["tension", "conflict", "disagree", "argue", "fight", "dispute"]
        if any(keyword in response_text.lower() for keyword in conflict_keywords):
            # Extract NPCs involved in conflict
            for npc_id in guidance["responding_npcs"]:
                if npc_id in guidance["tone_guidance"] and guidance["tone_guidance"][npc_id] in ["angry", "suspicious"]:
                    guidance["conflict_guidance"][npc_id] = True
        
        # Limit to max 3 responding NPCs to avoid overwhelming responses
        if len(guidance["responding_npcs"]) > 3:
            guidance["responding_npcs"] = guidance["responding_npcs"][:3]
        
        return guidance

     async def _store_significant_npc_reactions(self, npc_responses):
        """Store significant NPC reactions for future context"""
        if not npc_responses:
            return
            
        for response in npc_responses:
            npc_id = response.get("npc_id")
            if not npc_id:
                continue
                
            result = response.get("result", {})
            emotional_impact = result.get("emotional_impact", 0)
            
            # Only store significant reactions
            if abs(emotional_impact) >= 2:
                try:
                    # Get NPC name
                    npc_name = "Unknown NPC"
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT npc_name FROM NPCStats
                            WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                        """, (npc_id, self.user_id, self.conversation_id))
                        
                        row = cursor.fetchone()
                        if row:
                            npc_name = row[0]
                    
                    # Store in Nyx's memory system
                    memory_text = f"{npc_name} reacted with {result.get('outcome', 'a strong response')}"
                    
                    await self.nyx_agent_sdk.memory_system.add_memory(
                        memory_text=memory_text,
                        memory_type="observation",
                        memory_scope="game",
                        significance=min(abs(emotional_impact) + 3, 10),  # Convert to significance scale
                        tags=["npc_reaction", f"npc_{npc_id}"],
                        metadata={"npc_id": npc_id, "npc_name": npc_name, "emotional_impact": emotional_impact}
                    )
                except Exception as e:
                    logger.error(f"Error storing NPC reaction: {e}")   

    async def transition_scene_with_npcs(
        self,
        user_id: int,
        conversation_id: int,
        new_location: str,
        transition_context: Dict[str, Any]
    ):
        """Manage scene transition with enhanced NPC movement and reactions"""
        logger.info(f"Transitioning scene to {new_location} with NPCs")
        
        # Create location context for Nyx
        location_context = await self._get_location_context(new_location)
        enhanced_context = {**transition_context, "location_details": location_context}
        
        # First, handle Nyx's scene transition
        scene_result = await self.nyx_agent_sdk.process_user_input(
            f"We're now moving to {new_location}",
            {"transition_to": new_location, **enhanced_context}
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
        
        # Additional context from Nyx's response
        npc_transition_guidance = scene_result.get("npc_transition_guidance", {})
        
        # Move NPCs to the new location
        if npcs_to_move:
            # Get NPC reactions to this location if available from Nyx
            npc_location_reactions = {}
            for npc_id in npcs_to_move:
                if str(npc_id) in npc_transition_guidance:
                    npc_location_reactions[npc_id] = npc_transition_guidance[str(npc_id)]
            
            # Move the NPCs
            await self.npc_coordinator.batch_update_npcs(
                npcs_to_move,
                "location_change",
                {"new_location": new_location}
            )
            
            # Handle emotional responses to location if provided
            for npc_id, reaction in npc_location_reactions.items():
                if "emotion" in reaction:
                    await self.npc_coordinator.batch_update_npcs(
                        [npc_id],
                        "emotional_update",
                        {"emotion": reaction["emotion"], "intensity": reaction.get("intensity", 0.5)}
                    )
            
            # Generate memories for NPCs about the location change
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
        scene_result["npc_reactions"] = npc_location_reactions if 'npc_location_reactions' in locals() else {}
        
        return scene_result
    
    async def _get_location_context(self, location: str) -> Dict[str, Any]:
        """Get detailed location context for better scene descriptions"""
        location_context = {"name": location}
        
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    # Get location details
                    row = await conn.fetchrow("""
                        SELECT description, properties 
                        FROM Locations
                        WHERE name = $1
                    """, location)
                    
                    if row:
                        location_context["description"] = row["description"]
                        if row["properties"]:
                            location_context["properties"] = json.loads(row["properties"])
                    
                    # Get NPCs already at this location
                    rows = await conn.fetch("""
                        SELECT npc_id, npc_name, personality_traits
                        FROM NPCStats
                        WHERE current_location = $1
                          AND user_id = $2
                          AND conversation_id = $3
                    """, location, self.user_id, self.conversation_id)
                    
                    if rows:
                        location_context["resident_npcs"] = [
                            {
                                "npc_id": row["npc_id"],
                                "npc_name": row["npc_name"],
                                "traits": row["personality_traits"] if row["personality_traits"] else []
                            }
                            for row in rows
                        ]
        except Exception as e:
            logger.error(f"Error getting location context: {e}")
        
        return location_context
    
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
        nyx_response = await self.nyx_agent_sdk.process_user_input(self.user_id, self.conversation_id, nyx_input, shared_context)
        
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
        """Maintain both Nyx and NPC memories with enhanced coordination"""
        logger.info(f"Running joint memory maintenance for user {user_id}, conversation {conversation_id}")
        
        # Run Nyx memory maintenance
        nyx_agent_sdk = NyxAgent(user_id, conversation_id)
        nyx_result = await nyx_agent_sdk.run_memory_maintenance()
        
        # Run NPC memory maintenance
        npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
        npc_ids = await npc_coordinator.load_agents()
        
        # Create reflections based on Nyx's insights
        if "reflections" in nyx_result:
            for reflection in nyx_result["reflections"]:
                # Determine which NPCs should be aware of this reflection
                aware_npcs = await _determine_reflection_aware_npcs(
                    user_id, conversation_id, npc_ids, reflection
                )
                
                # Share with relevant NPCs
                if aware_npcs:
                    await npc_coordinator.batch_update_npcs(
                        aware_npcs,
                        "belief_update",
                        {"belief_text": f"Nyx believes: {reflection}",
                        "confidence": 0.8,
                        "topic": "nyx_perspective"}
                    )
        
        # Perform NPC memory maintenance
        npc_maintenance_results = {}
        batch_size = 5  # Process NPCs in smaller batches
        
        # With proper batch processing:
        npc_responses = []
        if "responding_npcs" in npc_guidance and npc_guidance["responding_npcs"]:
            # Convert to player action for NPC system
            player_action = {
                "description": user_input,
                "type": "talk"
            }
            
            # Process through NPC system
            npc_system = await self.get_npc_system()
            
            # Process NPCs in batches
            for i in range(0, len(npc_guidance["responding_npcs"]), batch_size):
                batch = npc_guidance["responding_npcs"][i:i+batch_size]
                batch_context = npc_context.copy()
                batch_context["batch_npcs"] = batch
                
                batch_result = await npc_system.handle_player_action(
                    player_action,
                    batch_context
                )
                
                if "npc_responses" in batch_result:
                    npc_responses.extend(batch_result["npc_responses"])
            
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
                        
                        # Determine significance based on memory significance
                        memory_significance = memory.get("significance", 5)
                        # Only share very significant memories
                        if memory_significance >= 7:
                            # Share with Nyx's memory system
                            await nyx_agent_sdk.memory_system.add_memory(
                                memory_text=f"{npc_name}: {memory.get('text', '')}",
                                memory_type="observation",
                                memory_scope="game",
                                significance=memory.get("significance", 5),
                                tags=["npc_memory", f"npc_{npc_id}"],
                                metadata={"source_npc": npc_id, "npc_name": npc_name}
                            )
                    except Exception as e:
                        logger.error(f"Error sharing NPC memory with Nyx: {e}")
        
        # Identify potential cross-NPC relationships or conflicts
        relationship_insights = await _identify_npc_relationships(user_id, conversation_id, npc_ids)
        if relationship_insights:
            # Create a reflection for Nyx about NPC relationships
            relationship_text = "NPC Relationships: " + relationship_insights
            await nyx_agent_sdk.memory_system.add_memory(
                memory_text=relationship_text,
                memory_type="reflection",
                memory_scope="game",
                significance=6,
                tags=["npc_relationships", "group_dynamics"],
                metadata={"maintenance_generated": True}
            )
        
        return {
            "nyx_maintenance": nyx_result,
            "npc_maintenance": npc_maintenance_results,
            "npcs_processed": len(npc_ids),
            "shared_memories": sum(1 for r in npc_maintenance_results.values() 
                                for m in r.get("important_memories", []) if m.get("significance", 0) >= 7),
            "relationship_insights": bool(relationship_insights)
        }

    async def _determine_reflection_aware_npcs(user_id, conversation_id, npc_ids, reflection):
        """Determine which NPCs would be aware of Nyx's reflection"""
        aware_npcs = []
        
        # Check for perceptive NPCs
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                # Query NPCs with perceptive traits
                cursor.execute("""
                    SELECT npc_id, personality_traits, scheming_level, dominance
                    FROM NPCStats
                    WHERE user_id = %s AND conversation_id = %s AND npc_id = ANY(%s)
                """, (user_id, conversation_id, npc_ids))
                
                for row in cursor.fetchall():
                    npc_id = row[0]
                    traits = row[1] or []
                    scheming_level = row[2] or 0
                    dominance = row[3] or 50
                    
                    # Parse traits if needed
                    if isinstance(traits, str):
                        try:
                            traits_list = json.loads(traits)
                        except json.JSONDecodeError:
                            traits_list = []
                    else:
                        traits_list = traits
                    
                    # Check for perceptive traits
                    perceptive_traits = ["observant", "perceptive", "vigilant", "alert", "curious"]
                    has_perceptive_trait = any(trait in traits_list for trait in perceptive_traits)
                    
                    # High schemers should be aware
                    is_high_schemer = scheming_level >= 7
                    
                    # Highly dominant NPCs are more likely to be aware
                    is_dominant = dominance >= 75
                    
                    # Check for specifics in the reflection that might trigger awareness
                    reflection_topics = ["power", "control", "submission", "dominance"]
                    has_relevant_topic = any(topic in reflection.lower() for topic in reflection_topics)
                    
                    # Determine if this NPC should be aware
                    if is_high_schemer or (has_perceptive_trait and (is_dominant or has_relevant_topic)):
                        aware_npcs.append(npc_id)
        except Exception as e:
            logger.error(f"Error determining reflection-aware NPCs: {e}")
        
        return aware_npcs

    async def _identify_npc_relationships(user_id, conversation_id, npc_ids):
        """Identify relationships and potential conflicts between NPCs"""
        if len(npc_ids) < 2:
            return ""
        
        insights = []
        
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                # Get all social links between NPCs
                placeholders = ','.join(['%s'] * len(npc_ids))
                cursor.execute(f"""
                    SELECT entity1_id, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND entity1_type = 'npc'
                      AND entity2_type = 'npc'
                      AND entity1_id IN ({placeholders})
                      AND entity2_id IN ({placeholders})
                """, (user_id, conversation_id, *npc_ids, *npc_ids))
                
                links = cursor.fetchall()
                
                # Get NPC names
                cursor.execute(f"""
                    SELECT npc_id, npc_name
                    FROM NPCStats
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND npc_id IN ({placeholders})
                """, (user_id, conversation_id, *npc_ids))
                
                npc_names = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Process links to find insights
                for e1_id, e2_id, link_type, link_level in links:
                    e1_name = npc_names.get(e1_id, f"NPC_{e1_id}")
                    e2_name = npc_names.get(e2_id, f"NPC_{e2_id}")
                    
                    if link_type == "rival" and link_level < 30:
                        insights.append(f"{e1_name} and {e2_name} have a strong rivalry.")
                    elif link_type == "ally" and link_level > 70:
                        insights.append(f"{e1_name} and {e2_name} are close allies.")
                    elif link_type == "co_conspirator" and link_level > 80:
                        insights.append(f"{e1_name} and {e2_name} appear to be co-conspirators.")
        except Exception as e:
            logger.error(f"Error identifying NPC relationships: {e}")
        
        return " ".join(insights)

    async def _gather_npc_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather important NPC states to inform Nyx's understanding"""
        npc_context = {}
        
        try:
            # Get current location from context
            location = context.get("location")
            if not location:
                # Try to get from game state
                async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow("""
                            SELECT value FROM CurrentRoleplay 
                            WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                        """, self.user_id, self.conversation_id)
                        
                        if row:
                            location = row["value"]
            
            if location:
                # Get NPCs at this location with their states
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT n.npc_id, n.npc_name, n.dominance, n.cruelty, 
                               n.personality_traits, n.schedule
                        FROM NPCStats n
                        WHERE n.user_id = %s 
                          AND n.conversation_id = %s 
                          AND n.current_location = %s
                    """, (self.user_id, self.conversation_id, location))
                    
                    for row in cursor.fetchall():
                        npc_id, npc_name = row[0], row[1]
                        
                        # Get emotional state for NPC
                        memory_system = await self._get_memory_system(npc_id)
                        emotional_state = await memory_system.get_npc_emotion(npc_id)
                        
                        npc_context[npc_id] = {
                            "npc_name": npc_name,
                            "dominance": row[2],
                            "cruelty": row[3],
                            "personality_traits": row[4],
                            "emotional_state": emotional_state
                        }
        except Exception as e:
            logger.error(f"Error gathering NPC context: {e}")
        
        return npc_context
    
    async def _get_memory_system(self, npc_id):
        """Get memory system for an NPC"""
        from memory.wrapper import MemorySystem
        
        try:
            # First check if NPC has an active agent
            coordinator = self._get_npc_coordinator()
            if npc_id in coordinator.active_agents:
                agent = coordinator.active_agents[npc_id]
                if hasattr(agent, "context") and hasattr(agent.context, "memory_system"):
                    return agent.context.memory_system
            
            # Otherwise create a new memory system instance
            return await MemorySystem.get_instance(
                self.user_id, 
                self.conversation_id,
                entity_type="npc",
                entity_id=npc_id
            )
        except Exception as e:
            logger.error(f"Error getting memory system for NPC {npc_id}: {e}")
            # Fallback to creating a new memory system
            return await MemorySystem.get_instance(
                self.user_id, 
                self.conversation_id
            )

    async def approve_group_interaction(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Approve or modify a requested group interaction.
        
        Args:
            request: Dictionary with npc_ids, context, and other details
            
        Returns:
            Dictionary with approval status and any modifications
        """
        # Default to approved
        result = {
            "approved": True,
            "reason": "Approved by Nyx"
        }
        
        try:
            # Get narrative context
            narrative_data = await self._get_current_narrative_context()
            
            # Check if interaction aligns with narrative goals
            if narrative_data.get("active_arcs"):
                current_arc = narrative_data["active_arcs"][0]
                
                # Extract needed NPCs for current arc
                arc_npcs = current_arc.get("required_npcs", [])
                requested_npcs = request.get("npc_ids", [])
                
                # If this interaction doesn't include NPCs needed for the current arc
                # and those NPCs are supposed to be in the current location
                # Consider rejecting or modifying
                if arc_npcs and not any(npc_id in requested_npcs for npc_id in arc_npcs):
                    current_location = request.get("context", {}).get("location")
                    
                    # Get where the required NPCs should be
                    required_npc_locations = await self._get_npc_locations(arc_npcs)
                    
                    # If required NPCs should be here but aren't included
                    if any(required_npc_locations.get(npc_id) == current_location for npc_id in arc_npcs):
                        # Modify instead of reject - add the required NPCs
                        modified_npcs = requested_npcs.copy()
                        for npc_id in arc_npcs:
                            if required_npc_locations.get(npc_id) == current_location and npc_id not in modified_npcs:
                                modified_npcs.append(npc_id)
                        
                        # Return modified context
                        modified_context = request.get("context", {}).copy()
                        modified_context["modified_by_nyx"] = True
                        modified_context["nyx_guidance"] = f"Ensure {self._get_npc_names(arc_npcs)} are involved to advance the current narrative arc."
                        
                        result["approved"] = True
                        result["modified_context"] = modified_context
                        result["modified_npc_ids"] = modified_npcs
                        result["reason"] = "Modified by Nyx to include required NPCs for narrative progression"
            
            # Add Nyx's current guidance
            result["nyx_guidance"] = await self._generate_interaction_guidance(
                request.get("npc_ids", []), 
                request.get("context", {})
            )
            
            return result
        except Exception as e:
            logger.error(f"Error in Nyx approval: {e}")
            return result  # Default to approved

    async def _get_current_narrative_context(self) -> Dict[str, Any]:
        """Get current narrative context from database."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT value FROM CurrentRoleplay 
                        WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
                    """, self.user_id, self.conversation_id)
                    
                    if row and row["value"]:
                        return json.loads(row["value"])
            return {}
        except Exception as e:
            logger.error(f"Error getting narrative context: {e}")
            return {}
            
    async def _get_npc_locations(self, npc_ids: List[int]) -> Dict[int, str]:
        """Get current locations for a list of NPCs."""
        locations = {}
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    for npc_id in npc_ids:
                        row = await conn.fetchrow("""
                            SELECT current_location FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """, self.user_id, self.conversation_id, npc_id)
                        
                        if row:
                            locations[npc_id] = row["current_location"]
            return locations
        except Exception as e:
            logger.error(f"Error getting NPC locations: {e}")
            return locations
            
    async def _get_npc_names(self, npc_ids: List[int]) -> str:
        """Get names of NPCs as a formatted string."""
        names = []
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    for npc_id in npc_ids:
                        row = await conn.fetchrow("""
                            SELECT npc_name FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """, self.user_id, self.conversation_id, npc_id)
                        
                        if row:
                            names.append(row["npc_name"])
            return ", ".join(names)
        except Exception as e:
            logger.error(f"Error getting NPC names: {e}")
            return ", ".join([f"NPC_{npc_id}" for npc_id in npc_ids])
        
    async def process_user_input(self, user_input: str, context: Dict[str, Any]):
        # Get Nyx's response with enhanced NPC awareness
        nyx_response = await self.nyx_agent_sdk.process_user_input(
            self.user_id, 
            self.conversation_id,
            user_input, 
            enhanced_context
        )
        
        # Extract NPC guidance from Nyx's response
        npc_guidance = self._extract_npc_guidance(nyx_response)
        
        # Add Nyx's guidance to context for NPCs
        npc_context = context.copy()
        npc_context["nyx_guidance"] = npc_guidance
        
        # Get NPC responses with Nyx's guidance
        npc_system = await self.get_npc_system()
        npc_responses = []
        
        # Only process if there are NPCs to respond
        if "responding_npcs" in npc_guidance and npc_guidance["responding_npcs"]:
            # Create player action from user input
            player_action = {
                "description": user_input,
                "type": "talk"
            }
            
            # Process NPCs in batches
            batch_size = 3
            for i in range(0, len(npc_guidance["responding_npcs"]), batch_size):
                batch = npc_guidance["responding_npcs"][i:i+batch_size]
                batch_context = npc_context.copy()
                batch_context["batch_npcs"] = batch
                
                batch_result = await npc_system.handle_player_action(
                    player_action,
                    batch_context
                )
                
                if "npc_responses" in batch_result:
                    npc_responses.extend(batch_result["npc_responses"])
                
                # Small delay between batches
                if i + batch_size < len(npc_guidance["responding_npcs"]):
                    await asyncio.sleep(0.1)
        
        # Combine responses
        combined_response = self._combine_responses(nyx_response, npc_responses)
        
        # Store important NPC reactions for future Nyx context
        await self._store_significant_npc_reactions(npc_responses)
        
        return combined_response
