# logic/lore/core/social_links.py

"""
Social Links System - Compatibility wrapper for the new dynamic relationship system.
This module provides backward compatibility for code that expects the old social links interface.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class SocialLink:
    """Represents a social link between entities (compatibility wrapper)."""
    link_id: int
    user_id: int
    conversation_id: int
    link_type: str
    link_level: int
    link_history: List[Dict[str, Any]]
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    dimensions: Dict[str, int]
    last_updated: datetime

class SocialLinkError(Exception):
    """Custom exception for social link-related errors."""
    pass

class SocialLinksManager:
    """
    Compatibility wrapper that translates old social links API to the new relationship system.
    """
    
    def __init__(self):
        """Initialize the social links manager."""
        self._managers = {}  # Cache of relationship managers by (user_id, conversation_id)
    
    async def _get_manager(self, user_id: int, conversation_id: int):
        """Get or create a relationship manager for the given user/conversation."""
        key = (user_id, conversation_id)
        if key not in self._managers:
            from logic.dynamic_relationships import OptimizedRelationshipManager
            self._managers[key] = OptimizedRelationshipManager(user_id, conversation_id)
        return self._managers[key]
    
    def _convert_state_to_link(
        self,
        state,
        user_id: int,
        conversation_id: int
    ) -> SocialLink:
        """Convert a RelationshipState to a SocialLink for compatibility."""
        # Map new dimensions to old link_level (0-100 scale)
        avg_positive = (
            max(0, state.dimensions.trust) + 
            max(0, state.dimensions.respect) + 
            max(0, state.dimensions.affection)
        ) / 3
        link_level = int(avg_positive)
        
        # Determine link type based on dimensions
        if state.dimensions.affection > 60:
            link_type = "romantic"
        elif state.dimensions.trust > 70 and state.dimensions.respect > 70:
            link_type = "close_friend"
        elif state.dimensions.trust > 50:
            link_type = "friend"
        elif state.dimensions.trust < -30:
            link_type = "enemy"
        elif state.dimensions.respect < -30:
            link_type = "rival"
        else:
            link_type = "neutral"
        
        # Convert dimensions to integer dict
        dimensions = {
            "trust": int(state.dimensions.trust),
            "respect": int(state.dimensions.respect),
            "affection": int(state.dimensions.affection),
            "fascination": int(state.dimensions.fascination),
            "influence": int(state.dimensions.influence),
            "dependence": int(state.dimensions.dependence),
            "intimacy": int(state.dimensions.intimacy)
        }
        
        # Build history from snapshots
        history = []
        for snapshot in state.history.significant_snapshots:
            history.append({
                "timestamp": snapshot['timestamp'].isoformat(),
                "dimensions": snapshot.get('dimensions', {}),
                "diff": snapshot.get('diff', {})
            })
        
        return SocialLink(
            link_id=state.link_id or 0,
            user_id=user_id,
            conversation_id=conversation_id,
            link_type=link_type,
            link_level=link_level,
            link_history=history,
            entity1_type=state.entity1_type,
            entity1_id=state.entity1_id,
            entity2_type=state.entity2_type,
            entity2_id=state.entity2_id,
            dimensions=dimensions,
            last_updated=state.last_interaction
        )
    
    def validate_link_type(self, link_type: str) -> bool:
        """Validate a social link type."""
        valid_types = ["neutral", "friend", "close_friend", "romantic", "rival", "enemy"]
        return link_type in valid_types
    
    def validate_link_level(self, link_level: int) -> bool:
        """Validate a social link level."""
        return 0 <= link_level <= 100
    
    def validate_dimensions(self, dimensions: Dict[str, int]) -> bool:
        """Validate relationship dimensions."""
        valid_dims = ["trust", "respect", "affection", "fascination", 
                     "influence", "dependence", "intimacy"]
        return all(
            dim in valid_dims and -100 <= value <= 100
            for dim, value in dimensions.items()
        )
    
    async def get_social_link(
        self,
        user_id: int,
        conversation_id: int,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int
    ) -> Optional[SocialLink]:
        """
        Get a social link between two entities.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            
        Returns:
            Social link if found, None otherwise
            
        Raises:
            SocialLinkError: If link retrieval fails
        """
        try:
            manager = await self._get_manager(user_id, conversation_id)
            state = await manager.get_relationship_state(
                entity1_type, entity1_id,
                entity2_type, entity2_id
            )
            
            if state and state.link_id:
                return self._convert_state_to_link(state, user_id, conversation_id)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get social link: {e}")
            raise SocialLinkError(f"Failed to retrieve social link: {str(e)}")
    
    async def create_social_link(
        self,
        user_id: int,
        conversation_id: int,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int,
        link_type: str = "neutral",
        link_level: int = 0,
        dimensions: Optional[Dict[str, int]] = None
    ) -> SocialLink:
        """
        Create a new social link.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            link_type: Type of link
            link_level: Initial link level
            dimensions: Optional relationship dimensions
            
        Returns:
            Created social link
            
        Raises:
            SocialLinkError: If link creation fails
        """
        if not self.validate_link_type(link_type):
            raise SocialLinkError(f"Invalid link type: {link_type}")
        
        if not self.validate_link_level(link_level):
            raise SocialLinkError(f"Invalid link level: {link_level}")
        
        if dimensions and not self.validate_dimensions(dimensions):
            raise SocialLinkError(f"Invalid dimensions: {dimensions}")
        
        try:
            manager = await self._get_manager(user_id, conversation_id)
            
            # Get or create the relationship
            state = await manager.get_relationship_state(
                entity1_type, entity1_id,
                entity2_type, entity2_id
            )
            
            # Set initial dimensions based on link_type and link_level
            if dimensions:
                for dim, value in dimensions.items():
                    if hasattr(state.dimensions, dim):
                        setattr(state.dimensions, dim, float(value))
            else:
                # Set defaults based on link_type
                if link_type == "friend":
                    state.dimensions.trust = 50
                    state.dimensions.affection = 40
                elif link_type == "close_friend":
                    state.dimensions.trust = 70
                    state.dimensions.affection = 60
                    state.dimensions.respect = 70
                elif link_type == "romantic":
                    state.dimensions.trust = 60
                    state.dimensions.affection = 70
                    state.dimensions.intimacy = 50
                elif link_type == "rival":
                    state.dimensions.respect = 50
                    state.dimensions.affection = -20
                elif link_type == "enemy":
                    state.dimensions.trust = -50
                    state.dimensions.respect = -30
                    state.dimensions.affection = -40
            
            state.dimensions.clamp()
            await manager._queue_update(state)
            await manager._flush_updates()
            
            return self._convert_state_to_link(state, user_id, conversation_id)
            
        except Exception as e:
            logger.error(f"Failed to create social link: {e}")
            raise SocialLinkError(f"Failed to create social link: {str(e)}")
    
    async def update_social_link(
        self,
        link: SocialLink,
        link_level: Optional[int] = None,
        link_type: Optional[str] = None,
        dimensions: Optional[Dict[str, int]] = None
    ) -> SocialLink:
        """
        Update an existing social link.
        
        Args:
            link: Social link to update
            link_level: New link level
            link_type: New link type
            dimensions: New relationship dimensions
            
        Returns:
            Updated social link
            
        Raises:
            SocialLinkError: If link update fails
        """
        if link_level is not None and not self.validate_link_level(link_level):
            raise SocialLinkError(f"Invalid link level: {link_level}")
        
        if link_type is not None and not self.validate_link_type(link_type):
            raise SocialLinkError(f"Invalid link type: {link_type}")
        
        if dimensions is not None and not self.validate_dimensions(dimensions):
            raise SocialLinkError(f"Invalid dimensions: {dimensions}")
        
        try:
            manager = await self._get_manager(link.user_id, link.conversation_id)
            
            state = await manager.get_relationship_state(
                link.entity1_type, link.entity1_id,
                link.entity2_type, link.entity2_id
            )
            
            # Apply dimension updates
            if dimensions:
                for dim, value in dimensions.items():
                    if hasattr(state.dimensions, dim):
                        setattr(state.dimensions, dim, float(value))
            
            # Apply link_level update (affects trust/affection)
            if link_level is not None:
                # Map link_level to dimension changes
                state.dimensions.trust = link_level - 20  # 0-100 -> -20 to 80
                state.dimensions.affection = link_level - 30  # 0-100 -> -30 to 70
            
            state.dimensions.clamp()
            await manager._queue_update(state)
            await manager._flush_updates()
            
            return self._convert_state_to_link(state, link.user_id, link.conversation_id)
            
        except Exception as e:
            logger.error(f"Failed to update social link: {e}")
            raise SocialLinkError(f"Failed to update social link: {str(e)}")
    
    async def get_entity_links(
        self,
        user_id: int,
        conversation_id: int,
        entity_type: str,
        entity_id: int
    ) -> List[SocialLink]:
        """
        Get all social links for an entity.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            entity_type: Type of entity
            entity_id: ID of entity
            
        Returns:
            List of social links
            
        Raises:
            SocialLinkError: If link retrieval fails
        """
        try:
            from db.connection import get_db_connection_context
            manager = await self._get_manager(user_id, conversation_id)
            
            links = []
            
            async with get_db_connection_context() as conn:
                # Find all relationships for this entity
                rows = await conn.fetch("""
                    SELECT entity1_type, entity1_id, entity2_type, entity2_id
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (
                        (entity1_type = $3 AND entity1_id = $4)
                        OR (entity2_type = $3 AND entity2_id = $4)
                    )
                """, user_id, conversation_id, entity_type, entity_id)
                
                for row in rows:
                    state = await manager.get_relationship_state(
                        row['entity1_type'], row['entity1_id'],
                        row['entity2_type'], row['entity2_id']
                    )
                    
                    if state and state.link_id:
                        links.append(self._convert_state_to_link(state, user_id, conversation_id))
            
            return links
            
        except Exception as e:
            logger.error(f"Failed to get entity links: {e}")
            raise SocialLinkError(f"Failed to retrieve entity links: {str(e)}")

# Create global social links manager instance
social_links_manager = SocialLinksManager()
