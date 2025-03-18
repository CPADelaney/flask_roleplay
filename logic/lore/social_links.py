"""
Social Links System - Manages relationships and social dynamics between entities in the lore system.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from .utils.db import get_db_connection, execute_query
from .utils.cache import get_cache, set_cache, delete_cache

logger = logging.getLogger(__name__)

@dataclass
class SocialLink:
    """Represents a social link between entities."""
    link_id: int
    link_type: str
    link_level: int
    link_history: List[Dict[str, Any]]
    dimensions: Dict[str, float]

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
    """Manages social links and relationship dynamics."""
    
    def __init__(self):
        """Initialize the social links manager."""
        self.link_types = config.SOCIAL_LINK_TYPES
        self.dimensions = config.RELATIONSHIP_DIMENSIONS
    
    def validate_link_type(self, link_type: str) -> bool:
        """
        Validate a social link type.
        
        Args:
            link_type: Link type to validate
            
        Returns:
            True if valid, False otherwise
        """
        return link_type in self.link_types
    
    def validate_link_level(self, link_level: int) -> bool:
        """
        Validate a social link level.
        
        Args:
            link_level: Link level to validate
            
        Returns:
            True if valid, False otherwise
        """
        return config.MIN_LINK_LEVEL <= link_level <= config.MAX_LINK_LEVEL
    
    def validate_dimensions(self, dimensions: Dict[str, int]) -> bool:
        """
        Validate relationship dimensions.
        
        Args:
            dimensions: Dictionary of dimension values
            
        Returns:
            True if valid, False otherwise
        """
        return all(
            dim in self.dimensions and 
            config.MIN_LINK_LEVEL <= value <= config.MAX_LINK_LEVEL
            for dim, value in dimensions.items()
        )
    
    def get_social_link(
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
        cache_key = f"lore:social_link:{user_id}:{conversation_id}:{entity1_type}:{entity1_id}:{entity2_type}:{entity2_id}"
        
        # Try to get from cache first
        cached_link = get_cached_value(cache_key)
        if cached_link:
            return SocialLink(**cached_link)
        
        try:
            query = """
                SELECT link_id, link_type, link_level, link_history,
                       entity1_type, entity1_id, entity2_type, entity2_id,
                       dimensions, last_updated
                FROM SocialLinks
                WHERE user_id = %(user_id)s
                AND conversation_id = %(conversation_id)s
                AND entity1_type = %(entity1_type)s
                AND entity1_id = %(entity1_id)s
                AND entity2_type = %(entity2_type)s
                AND entity2_id = %(entity2_id)s
            """
            result = execute_query(query, {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "entity1_type": entity1_type,
                "entity1_id": entity1_id,
                "entity2_type": entity2_type,
                "entity2_id": entity2_id
            })
            
            if not result:
                return None
            
            row = result[0]
            link = SocialLink(
                link_id=row[0],
                link_type=row[1],
                link_level=row[2],
                link_history=row[3],
                entity1_type=row[4],
                entity1_id=row[5],
                entity2_type=row[6],
                entity2_id=row[7],
                dimensions=row[8],
                last_updated=row[9]
            )
            
            # Cache the result
            set_cached_value(cache_key, link.__dict__)
            return link
            
        except DatabaseError as e:
            logger.error(f"Failed to get social link: {e}")
            raise SocialLinkError(f"Failed to retrieve social link: {str(e)}")
    
    def create_social_link(
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
            raise SocialLinkError(config.ERROR_MESSAGES["invalid_link_type"])
        
        if not self.validate_link_level(link_level):
            raise SocialLinkError(
                config.ERROR_MESSAGES["invalid_link_level"].format(
                    min=config.MIN_LINK_LEVEL,
                    max=config.MAX_LINK_LEVEL
                )
            )
        
        dimensions = dimensions or {dim: 0 for dim in self.dimensions}
        if not self.validate_dimensions(dimensions):
            raise SocialLinkError(config.ERROR_MESSAGES["invalid_dimension"])
        
        try:
            query = """
                INSERT INTO SocialLinks
                (user_id, conversation_id, entity1_type, entity1_id,
                 entity2_type, entity2_id, link_type, link_level,
                 link_history, dimensions, last_updated)
                VALUES
                (%(user_id)s, %(conversation_id)s, %(entity1_type)s, %(entity1_id)s,
                 %(entity2_type)s, %(entity2_id)s, %(link_type)s, %(link_level)s,
                 %(link_history)s, %(dimensions)s, %(last_updated)s)
                RETURNING link_id
            """
            result = execute_query(query, {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "entity1_type": entity1_type,
                "entity1_id": entity1_id,
                "entity2_type": entity2_type,
                "entity2_id": entity2_id,
                "link_type": link_type,
                "link_level": link_level,
                "link_history": [],
                "dimensions": dimensions,
                "last_updated": datetime.utcnow()
            })
            
            link_id = result[0][0]
            return self.get_social_link(
                user_id, conversation_id,
                entity1_type, entity1_id,
                entity2_type, entity2_id
            )
            
        except DatabaseError as e:
            logger.error(f"Failed to create social link: {e}")
            raise SocialLinkError(f"Failed to create social link: {str(e)}")
    
    def update_social_link(
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
            raise SocialLinkError(
                config.ERROR_MESSAGES["invalid_link_level"].format(
                    min=config.MIN_LINK_LEVEL,
                    max=config.MAX_LINK_LEVEL
                )
            )
        
        if link_type is not None and not self.validate_link_type(link_type):
            raise SocialLinkError(config.ERROR_MESSAGES["invalid_link_type"])
        
        if dimensions is not None and not self.validate_dimensions(dimensions):
            raise SocialLinkError(config.ERROR_MESSAGES["invalid_dimension"])
        
        try:
            # Record the change in history
            history_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "link_level": link_level or link.link_level,
                "link_type": link_type or link.link_type,
                "dimensions": dimensions or link.dimensions
            }
            
            query = """
                UPDATE SocialLinks
                SET link_level = COALESCE(%(link_level)s, link_level),
                    link_type = COALESCE(%(link_type)s, link_type),
                    dimensions = COALESCE(%(dimensions)s, dimensions),
                    link_history = link_history || %(history_entry)s,
                    last_updated = %(last_updated)s
                WHERE link_id = %(link_id)s
            """
            execute_query(query, {
                "link_id": link.link_id,
                "link_level": link_level,
                "link_type": link_type,
                "dimensions": dimensions,
                "history_entry": history_entry,
                "last_updated": datetime.utcnow()
            })
            
            # Invalidate cache
            cache_key = f"lore:social_link:{link.entity1_type}:{link.entity1_id}:{link.entity2_type}:{link.entity2_id}"
            invalidate_cache_pattern(cache_key)
            
            return self.get_social_link(
                link.user_id, link.conversation_id,
                link.entity1_type, link.entity1_id,
                link.entity2_type, link.entity2_id
            )
            
        except DatabaseError as e:
            logger.error(f"Failed to update social link: {e}")
            raise SocialLinkError(f"Failed to update social link: {str(e)}")
    
    def get_entity_links(
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
            query = """
                SELECT link_id, link_type, link_level, link_history,
                       entity1_type, entity1_id, entity2_type, entity2_id,
                       dimensions, last_updated
                FROM SocialLinks
                WHERE user_id = %(user_id)s
                AND conversation_id = %(conversation_id)s
                AND (
                    (entity1_type = %(entity_type)s AND entity1_id = %(entity_id)s)
                    OR (entity2_type = %(entity_type)s AND entity2_id = %(entity_id)s)
                )
            """
            results = execute_query(query, {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "entity_type": entity_type,
                "entity_id": entity_id
            })
            
            return [
                SocialLink(
                    link_id=row[0],
                    link_type=row[1],
                    link_level=row[2],
                    link_history=row[3],
                    entity1_type=row[4],
                    entity1_id=row[5],
                    entity2_type=row[6],
                    entity2_id=row[7],
                    dimensions=row[8],
                    last_updated=row[9]
                )
                for row in results
            ]
            
        except DatabaseError as e:
            logger.error(f"Failed to get entity links: {e}")
            raise SocialLinkError(f"Failed to retrieve entity links: {str(e)}")

# Create global social links manager instance
social_links_manager = SocialLinksManager() 