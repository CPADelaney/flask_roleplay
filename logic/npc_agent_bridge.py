# logic/npc_agent_bridge.py

"""
Bridge module to integrate IntegratedNPCSystem with the existing NPCAgentSystem.
This provides compatibility and a smooth transition path.
"""

import logging
import asyncio
import asyncpg
from typing import Dict, List, Any, Optional

from logic.fully_integrated_npc_system import IntegratedNPCSystem
from npcs.npc_agent_system import NPCAgentSystem  # Your existing NPC system
from db.connection import get_db_connection_context
from logic.dynamic_relationships import OptimizedRelationshipManager

from agents import Agent, function_tool, Runner, ModelSettings, trace

class NPCSystemBridge:
    """
    Bridge class that provides a unified interface between the legacy NPCAgentSystem
    and the new OpenAI Agents SDK-based system.
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
        
        # Flag to determine whether to use legacy or new system
        self.use_new_system = False
        
        logging.info(f"Initialized NPCSystemBridge for user={user_id}, conversation={conversation_id}")
    
    async def handle_player_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a player action using the appropriate system.
        
        Args:
            action: Player action data
            context: Context information
            
        Returns:
            Response data including NPC responses
        """
        if self.use_new_system:
            # Use the new OpenAI Agents SDK-based system
            return await self.integrated_system.process_player_activity(
                f"The player {action.get('description', 'did something')}", 
                context
            )
        else:
            # Use the legacy system with enhancement from integrated system
            agent_responses = await self.agent_system.handle_player_action(action, context)
            
            # Enhance with integrated system features
            enhanced_responses = []
            
            for response in agent_responses.get("npc_responses", []):
                npc_id = response.get("npc_id")
                if not npc_id:
                    enhanced_responses.append(response)
                    continue
                    
                # Record memory of this interaction
                memory_text = f"Player action: {action.get('description', 'did something')}. I responded with: {response.get('action', {}).get('description', 'a response')}"
                await self.integrated_system.record_memory_event(npc_id, memory_text)
                
                # Add enhanced response
                enhanced_responses.append(response)
            
            # Return enhanced response data
            return {
                **agent_responses,
                "npc_responses": enhanced_responses,
                "enhanced": True
            }
    
    async def get_relationship_links(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get relationship links between an NPC and the player using the new dynamic relationships system.
        
        Args:
            npc_id: NPC ID
            
        Returns:
            List of relationship link data
        """
        links = []
        try:
            # Use the new relationship manager
            manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
            
            # Get the relationship state between the NPC and player
            state = await manager.get_relationship_state(
                'npc', npc_id, 'player', self.user_id
            )
            
            # Get a summary of the relationship
            summary = state.to_summary()
            
            # Convert to the expected format
            # Map the new multi-dimensional system to the old link format for compatibility
            
            # Calculate an overall "link level" from dimensions
            # This is a weighted average of positive dimensions
            positive_dimensions = {
                'trust': state.dimensions.trust,
                'respect': state.dimensions.respect,
                'affection': state.dimensions.affection,
                'intimacy': state.dimensions.intimacy,
                'fascination': state.dimensions.fascination
            }
            
            # Calculate weighted average (0-100 scale)
            total_positive = sum(max(0, v) for v in positive_dimensions.values())
            avg_positive = total_positive / len(positive_dimensions) if positive_dimensions else 0
            
            # Determine link type based on dominant dimensions
            link_type = self._determine_link_type(state.dimensions)
            
            # Create link data in the expected format
            link_data = {
                "link_id": state.link_id,
                "link_type": link_type,
                "link_level": int(avg_positive),  # Convert to int for compatibility
                # Additional data from the new system
                "dimensions": summary['dimensions'],
                "patterns": summary.get('patterns', []),
                "archetypes": summary.get('archetypes', []),
                "momentum": summary.get('momentum_magnitude', 0),
                "duration_days": summary.get('duration_days', 0)
            }
            
            links.append(link_data)
            
            return links
            
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logging.error(f"DB Error getting relationship links: {db_err}", exc_info=True)
            return []
        except Exception as e:
            logging.error(f"Error getting relationship links: {e}", exc_info=True)
            return []
    
    def _determine_link_type(self, dimensions) -> str:
        """
        Determine the link type based on relationship dimensions.
        
        Args:
            dimensions: RelationshipDimensions object
            
        Returns:
            String representing the link type
        """
        # Analyze dimensions to determine the dominant relationship type
        
        # Check for hostile relationships
        if dimensions.trust < -30 or dimensions.respect < -30:
            return "hostile"
        
        # Check for dominant/submissive dynamics
        if dimensions.influence > 50:
            return "dominant"
        elif dimensions.influence < -50:
            return "submissive"
        
        # Check for romantic relationships
        if dimensions.affection > 60 and dimensions.intimacy > 50:
            return "romantic"
        
        # Check for friendly relationships
        if dimensions.affection > 30 and dimensions.trust > 30:
            return "friendly"
        
        # Check for professional relationships
        if dimensions.respect > 50 and dimensions.trust > 30:
            return "professional"
        
        # Check for dependent relationships
        if dimensions.dependence > 60:
            return "dependent"
        
        # Default to neutral
        return "neutral"
    
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
