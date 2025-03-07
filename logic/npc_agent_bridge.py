# logic/npc_agent_bridge.py

"""
Bridge module to integrate IntegratedNPCSystem with the existing NPCAgentSystem.
This provides compatibility and a smooth transition path.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional

from logic.fully_integrated_npc_system import IntegratedNPCSystem
from logic.npc_agents import NPCAgentSystem  # Your existing NPC system

class NPCSystemBridge:
    """
    Bridge class that provides a unified interface between the legacy NPCAgentSystem
    and the new IntegratedNPCSystem.
    
    This allows for gradual migration to the new system while maintaining compatibility
    with existing code.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize both systems and provide unified access.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize both systems
        self.integrated_system = IntegratedNPCSystem(user_id, conversation_id)
        self.agent_system = NPCAgentSystem(user_id, conversation_id)
        
        logging.info(f"Initialized NPCSystemBridge for user={user_id}, conversation={conversation_id}")
    
    async def handle_player_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a player action using both systems for a smooth transition.
        Uses the agent system for responses but records memories and updates
        relationships with the integrated system.
        
        Args:
            action: Player action data
            context: Context information
            
        Returns:
            Response data including NPC responses
        """
        # Step 1: Use agent system for responses (compatibility)
        agent_responses = await self.agent_system.handle_player_action(action, context)
        
        # Step 2: Enhance with integrated system features
        enhanced_responses = []
        
        for response in agent_responses.get("npc_responses", []):
            npc_id = response.get("npc_id")
            if not npc_id:
                enhanced_responses.append(response)
                continue
                
            # Record memory of this interaction
            memory_text = f"Player action: {action.get('description', 'did something')}. I responded with: {response.get('action', {}).get('description', 'a response')}"
            await self.integrated_system.record_memory_event(npc_id, memory_text)
            
            # Determine interaction type
            interaction_type = "extended_conversation"
            if "no" in action.get("description", "").lower() or "won't" in action.get("description", "").lower():
                interaction_type = "defiant_response"
            elif "yes" in action.get("description", "").lower() or "okay" in action.get("description", "").lower():
                interaction_type = "submissive_response"
            
            # Update relationship based on interaction (simplified)
            try:
                # Get NPC and player relationship
                links = await self.get_relationship_links(npc_id)
                if links:
                    link_id = links[0].get("link_id")
                    
                    # Different dimension changes based on interaction type
                    dimension_changes = {}
                    if interaction_type == "defiant_response":
                        dimension_changes = {"tension": +5, "respect": -2}
                    elif interaction_type == "submissive_response":
                        dimension_changes = {"control": +5, "dependency": +3}
                    elif interaction_type == "extended_conversation":
                        dimension_changes = {"trust": +2}
                    
                    if dimension_changes:
                        await self.integrated_system.update_relationship_dimensions(
                            link_id, dimension_changes, 
                            f"Response to player action: {action.get('description', '')[:30]}"
                        )
            except Exception as e:
                logging.warning(f"Error updating relationship: {e}")
            
            # Add the enhanced response
            enhanced_responses.append(response)
        
        # Step 3: Return enhanced response data
        return {
            **agent_responses,
            "npc_responses": enhanced_responses,
            "enhanced": True
        }
    
    async def get_relationship_links(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get relationship links between an NPC and the player.
        
        Args:
            npc_id: NPC ID
            
        Returns:
            List of relationship link data
        """
        from db.connection import get_db_connection
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT link_id, link_type, link_level
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s
                AND ((entity1_type='player' AND entity1_id=%s AND entity2_type='npc' AND entity2_id=%s)
                OR (entity1_type='npc' AND entity1_id=%s AND entity2_type='player' AND entity2_id=%s))
            """, (
                self.user_id, self.conversation_id,
                self.user_id, npc_id,
                npc_id, self.user_id
            ))
            
            links = []
            for row in cursor.fetchall():
                link_id, link_type, link_level = row
                links.append({
                    "link_id": link_id,
                    "link_type": link_type,
                    "link_level": link_level
                })
            
            return links
        finally:
            cursor.close()
            conn.close()
    
    async def create_new_npc(self, environment_desc: str, day_names: List[str] = None) -> int:
        """
        Create a new NPC using the integrated system.
        
        Args:
            environment_desc: Environment description
            day_names: List of day names
            
        Returns:
            NPC ID
        """
        if day_names is None:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
        return await self.integrated_system.create_new_npc(environment_desc, day_names)
    
    async def get_current_game_time(self):
        """
        Get the current game time using the integrated system.
        
        Returns:
            Tuple of (year, month, day, time_of_day)
        """
        return await self.integrated_system.get_current_game_time()
    
    async def advance_time_with_activity(self, activity_type: str) -> Dict[str, Any]:
        """
        Advance time based on an activity using the integrated system.
        
        Args:
            activity_type: Type of activity
            
        Returns:
            Dictionary with time advancement results
        """
        return await self.integrated_system.advance_time_with_activity(activity_type)
    
    async def process_player_activity(self, player_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a player's activity using the integrated system.
        
        Args:
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Dictionary with processing results
        """
        return await self.integrated_system.process_player_activity(player_input, context)
    
    async def handle_npc_interaction(self, 
                                   npc_id: int, 
                                   interaction_type: str,
                                   player_input: str,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a complete interaction between player and NPC using the integrated system.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Dictionary with interaction results
        """
        return await self.integrated_system.handle_npc_interaction(
            npc_id, interaction_type, player_input, context
        )
    
    async def check_for_relationship_events(self) -> List[Dict[str, Any]]:
        """
        Check for significant relationship events using the integrated system.
        
        Returns:
            List of event dictionaries
        """
        return await self.integrated_system.check_for_relationship_events()
    
    async def generate_multi_npc_scene(self, 
                                    npc_ids: List[int], 
                                    location: str = None,
                                    include_player: bool = True) -> Dict[str, Any]:
        """
        Generate a scene with multiple NPCs using the integrated system.
        
        Args:
            npc_ids: List of NPC IDs to include
            location: Location for the scene
            include_player: Whether to include the player
            
        Returns:
            Scene information
        """
        return await self.integrated_system.generate_multi_npc_scene(
            npc_ids, location, include_player
        )
    
    async def generate_overheard_conversation(self, 
                                           npc_ids: List[int],
                                           topic: str = None,
                                           about_player: bool = False) -> Dict[str, Any]:
        """
        Generate a conversation between NPCs that the player can overhear.
        
        Args:
            npc_ids: List of NPC IDs to include
            topic: Topic of conversation
            about_player: Whether the conversation is about the player
            
        Returns:
            Conversation information
        """
        return await self.integrated_system.generate_overheard_conversation(
            npc_ids, topic, about_player
        )
