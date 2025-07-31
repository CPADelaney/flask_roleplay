# logic/lore/core/system.py

"""
Main Lore System class that integrates all components.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..config.settings import config
from .narrative import narrative_progression, NarrativeStage, NarrativeError
from logic.dynamic_relationships import (
    OptimizedRelationshipManager, 
    process_relationship_interaction_tool,
    get_relationship_summary_tool
)
from ..utils.cache import invalidate_cache_pattern

logger = logging.getLogger(__name__)

class LoreError(Exception):
    """Custom exception for lore system errors."""
    pass

class LoreSystem:
    """Main class that integrates all lore system components."""
    
    def __init__(self):
        """Initialize the lore system."""
        self.narrative = narrative_progression
        self.social_links = social_links_manager
    
    async def get_current_state(
        self,
        user_id: int,
        conversation_id: int
    ) -> Dict[str, Any]:
        """
        Get the current state of the lore system for a user.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Dictionary containing current narrative stage and social links
            
        Raises:
            LoreError: If state retrieval fails
        """
        try:
            # Get current narrative stage
            current_stage = await self.narrative.get_current_stage(user_id, conversation_id)
            
            # Get all social links for the player
            player_links = await self.social_links.get_entity_links(
                user_id, conversation_id,
                "player", 1  # Assuming player ID is 1
            )
            
            # Get stage events
            stage_events = await self.narrative.get_stage_events(current_stage)
            
            return {
                "narrative_stage": {
                    "name": current_stage.name,
                    "description": current_stage.description,
                    "required_corruption": current_stage.required_corruption,
                    "required_dependency": current_stage.required_dependency,
                    "events": stage_events
                },
                "social_links": [
                    {
                        "link_id": link.link_id,
                        "link_type": link.link_type,
                        "link_level": link.link_level,
                        "entity_type": (
                            link.entity2_type if link.entity1_type == "player"
                            else link.entity1_type
                        ),
                        "entity_id": (
                            link.entity2_id if link.entity1_type == "player"
                            else link.entity1_id
                        ),
                        "dimensions": link.dimensions,
                        "last_updated": link.last_updated.isoformat()
                    }
                    for link in player_links
                ]
            }
            
        except (NarrativeError, SocialLinkError) as e:
            logger.error(f"Failed to get current state: {e}")
            raise LoreError(f"Failed to retrieve current state: {str(e)}")
    
    async def update_relationship(self, user_id: int, conversation_id: int,
                                entity_type: str, entity_id: int,
                                interaction: Dict[str, Any] = None,
                                dimension_changes: Dict[str, float] = None) -> Dict[str, Any]:
        """Update a relationship using the new dynamic system."""
        manager = OptimizedRelationshipManager(user_id, conversation_id)
        
        if interaction:
            # Process as interaction
            result = await manager.process_interaction(
                entity1_type="player",
                entity1_id=1,
                entity2_type=entity_type,
                entity2_id=entity_id,
                interaction=interaction
            )
        elif dimension_changes:
            # Direct dimension update
            state = await manager.get_relationship_state(
                entity1_type="player",
                entity1_id=1,
                entity2_type=entity_type,
                entity2_id=entity_id
            )
            
            for dim, change in dimension_changes.items():
                if hasattr(state.dimensions, dim):
                    current = getattr(state.dimensions, dim)
                    setattr(state.dimensions, dim, current + change)
            
            state.dimensions.clamp()
            await manager._queue_update(state)
            await manager._flush_updates()
            
            result = {"success": True, "changes": dimension_changes}
        
        return result
    
    async def get_available_events(
        self,
        user_id: int,
        conversation_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get all available events for the current narrative stage.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            List of available events
            
        Raises:
            LoreError: If event retrieval fails
        """
        try:
            current_stage = await self.narrative.get_current_stage(
                user_id, conversation_id
            )
            return await self.narrative.get_stage_events(current_stage)
            
        except NarrativeError as e:
            logger.error(f"Failed to get available events: {e}")
            raise LoreError(f"Failed to retrieve available events: {str(e)}")
    
    async def get_relationship_network(
        self,
        user_id: int,
        conversation_id: int,
        entity_type: str,
        entity_id: int
    ) -> Dict[str, Any]:
        """
        Get the relationship network for an entity.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            entity_type: Type of entity
            entity_id: ID of entity
            
        Returns:
            Dictionary containing relationship network
            
        Raises:
            LoreError: If network retrieval fails
        """
        try:
            # Get all links for the entity
            links = await self.social_links.get_entity_links(
                user_id, conversation_id,
                entity_type, entity_id
            )
            
            # Build network
            network = {
                "entity": {
                    "type": entity_type,
                    "id": entity_id
                },
                "relationships": []
            }
            
            for link in links:
                # Determine the other entity
                other_type = (
                    link.entity2_type if link.entity1_type == entity_type
                    else link.entity1_type
                )
                other_id = (
                    link.entity2_id if link.entity1_type == entity_type
                    else link.entity1_id
                )
                
                network["relationships"].append({
                    "entity": {
                        "type": other_type,
                        "id": other_id
                    },
                    "link_type": link.link_type,
                    "link_level": link.link_level,
                    "dimensions": link.dimensions,
                    "last_updated": link.last_updated.isoformat()
                })
            
            return network
            
        except SocialLinkError as e:
            logger.error(f"Failed to get relationship network: {e}")
            raise LoreError(f"Failed to retrieve relationship network: {str(e)}")

# Create global lore system instance
lore_system = LoreSystem()
