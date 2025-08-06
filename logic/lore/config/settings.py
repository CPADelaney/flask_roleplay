"""
Configuration settings for the Lore System.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class LoreConfig:
    """Main configuration class for the Lore System."""
    
    # Database settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000
    
    # Narrative progression settings
    NARRATIVE_STAGES: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "Innocent Beginning",
                "description": "Initial stage with unaware player and subtle NPC control",
                "required_corruption": 0,
                "required_dependency": 0,
            },
            {
                "name": "First Doubts",
                "description": "Player notices inconsistencies in NPC behavior",
                "required_corruption": 20,
                "required_dependency": 15,
            },
            {
                "name": "Creeping Realization",
                "description": "Clear moments of awareness about manipulation",
                "required_corruption": 40,
                "required_dependency": 35,
            },
            {
                "name": "Veil Thinning",
                "description": "NPCs openly manipulate and control the player",
                "required_corruption": 60,
                "required_dependency": 55,
            },
            {
                "name": "Full Revelation",
                "description": "Complete acknowledgment of position and relationships",
                "required_corruption": 80,
                "required_dependency": 75,
            },
        ]
    )

    # Social links settings
    SOCIAL_LINK_TYPES: List[str] = field(
        default_factory=lambda: [
            "neutral",
            "friendly",
            "hostile",
            "romantic",
            "mentor",
            "student",
            "rival",
            "ally",
        ]
    )
    
    MAX_LINK_LEVEL: int = 100
    MIN_LINK_LEVEL: int = -100
    
    # Relationship dimensions
    RELATIONSHIP_DIMENSIONS: List[str] = field(
        default_factory=lambda: [
            "trust",
            "respect",
            "closeness",
            "dominance",
            "cruelty",
            "intensity",
        ]
    )

    # Cache keys
    CACHE_KEYS: Dict[str, str] = field(
        default_factory=lambda: {
            "narrative_stage": "lore:narrative_stage:{user_id}:{conversation_id}",
            "social_links": "lore:social_links:{user_id}:{conversation_id}",
            "npc_stats": "lore:npc_stats:{user_id}:{conversation_id}:{npc_id}",
            "player_stats": "lore:player_stats:{user_id}:{conversation_id}",
        }
    )

    # Error messages
    ERROR_MESSAGES: Dict[str, str] = field(
        default_factory=lambda: {
            "db_connection": "Failed to establish database connection",
            "cache_error": "Cache operation failed",
            "invalid_link_type": "Invalid social link type",
            "invalid_link_level": "Link level must be between {min} and {max}",
            "npc_not_found": "NPC with ID {npc_id} not found",
            "invalid_dimension": "Invalid relationship dimension",
        }
    )

# Create a global instance
config = LoreConfig() 
