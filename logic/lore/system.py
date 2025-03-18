"""
Main Lore System class that integrates all components.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..config.settings import config
from .narrative import narrative_progression, NarrativeStage, NarrativeError
from .social_links import social_links_manager, SocialLink, SocialLinkError
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
    
    def get_current_state(
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
            current_stage = self.narrative.get_current_stage(user_id, conversation_id)
            
            # Get all social links for the player
            player_links = self.social_links.get_entity_links(
                user_id, conversation_id,
                "player", 1  # Assuming player ID is 1
            )
            
            # Get stage events
            stage_events = self.narrative.get_stage_events(current_stage)
            
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
    
    def update_social_link(
        self,
        user_id: int,
        conversation_id: int,
        entity_type: str,
        entity_id: int,
        link_level: Optional[int] = None,
        link_type: Optional[str] = None,
        dimensions: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Update a social link and check for narrative progression.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            entity_type: Type of entity to update link with
            entity_id: ID of entity to update link with
            link_level: New link level
            link_type: New link type
            dimensions: New relationship dimensions
            
        Returns:
            Dictionary containing updated link and any narrative changes
            
        Raises:
            LoreError: If update fails
        """
        try:
            # Get or create the social link
            link = self.social_links.get_social_link(
                user_id, conversation_id,
                "player", 1,  # Assuming player ID is 1
                entity_type, entity_id
            )
            
            if not link:
                link = self.social_links.create_social_link(
                    user_id, conversation_id,
                    "player", 1,
                    entity_type, entity_id
                )
            
            # Update the link
            updated_link = self.social_links.update_social_link(
                link,
                link_level=link_level,
                link_type=link_type,
                dimensions=dimensions
            )
            
            # Check for narrative progression
            new_stage = self.narrative.check_for_stage_transition(
                user_id, conversation_id
            )
            
            result = {
                "social_link": {
                    "link_id": updated_link.link_id,
                    "link_type": updated_link.link_type,
                    "link_level": updated_link.link_level,
                    "dimensions": updated_link.dimensions,
                    "last_updated": updated_link.last_updated.isoformat()
                }
            }
            
            if new_stage:
                # Apply the stage transition
                self.narrative.apply_stage_transition(
                    user_id, conversation_id, new_stage
                )
                
                # Get stage events
                stage_events = self.narrative.get_stage_events(new_stage)
                
                result["narrative_change"] = {
                    "old_stage": self.narrative.get_current_stage(
                        user_id, conversation_id
                    ).name,
                    "new_stage": {
                        "name": new_stage.name,
                        "description": new_stage.description,
                        "events": stage_events
                    }
                }
            
            return result
            
        except (NarrativeError, SocialLinkError) as e:
            logger.error(f"Failed to update social link: {e}")
            raise LoreError(f"Failed to update social link: {str(e)}")
    
    def get_available_events(
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
            current_stage = self.narrative.get_current_stage(
                user_id, conversation_id
            )
            return self.narrative.get_stage_events(current_stage)
            
        except NarrativeError as e:
            logger.error(f"Failed to get available events: {e}")
            raise LoreError(f"Failed to retrieve available events: {str(e)}")
    
    def get_relationship_network(
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
            links = self.social_links.get_entity_links(
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