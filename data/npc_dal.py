# data/npc_dal.py

"""
NPC Data Access Layer

This module provides a unified interface for accessing NPC data from the database.
It consolidates duplicate data access methods from various modules into a single,
well-structured class.
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import random

# Database connection
from db.connection import get_db_connection_context

# Caching support
from utils.caching import cache_result, invalidate_cache, get_cached_result

logger = logging.getLogger(__name__)

class NPCDataAccessError(Exception):
    """Base exception for NPC data access errors"""
    pass

class NPCNotFoundError(NPCDataAccessError):
    """Exception raised when an NPC cannot be found"""
    pass

class NPCDataAccess:
    """
    Data Access Layer for NPC-related database operations.
    
    This class provides methods for retrieving, creating, and updating NPC data,
    consolidating the data access methods from multiple modules into a single interface.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the NPC data access layer.
        
        Args:
            user_id: The user ID for context
            conversation_id: The conversation ID for context
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Cache metrics for monitoring
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'last_refresh': datetime.now()
        }
    
    async def get_npc_details(self, npc_id: Optional[int] = None, npc_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about an NPC by ID or name.
        
        Args:
            npc_id: Optional ID of the NPC
            npc_name: Optional name of the NPC
            
        Returns:
            Dictionary with NPC details
            
        Raises:
            NPCNotFoundError: If the NPC cannot be found
            NPCDataAccessError: For other data access errors
        """
        if not npc_id and not npc_name:
            raise ValueError("Either npc_id or npc_name must be provided")
        
        # Check cache first
        cache_key = f"npc_details:{self.user_id}:{self.conversation_id}:{npc_id or npc_name}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            self.cache_metrics['hits'] += 1
            return cached_result
        
        self.cache_metrics['misses'] += 1
        
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT npc_id, npc_name, introduced, archetypes, archetype_summary, 
                           archetype_extras_summary, physical_description, relationships,
                           dominance, cruelty, closeness, trust, respect, intensity,
                           hobbies, personality_traits, likes, dislikes, affiliations,
                           schedule, current_location, sex, age, memory, faction_affiliations
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """
                
                params = [self.user_id, self.conversation_id]
                
                if npc_id is not None:
                    query += " AND npc_id=$3"
                    params.append(npc_id)
                elif npc_name is not None:
                    query += " AND LOWER(npc_name)=LOWER($3)"
                    params.append(npc_name)
                
                query += " LIMIT 1"
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    error_msg = f"NPC not found: {npc_id or npc_name}"
                    logger.error(error_msg)
                    raise NPCNotFoundError(error_msg)
                
                # Build the NPC details with parsed JSON fields
                npc_details = {
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"],
                    "introduced": row["introduced"],
                    "archetypes": self._parse_json_field(row["archetypes"], []),
                    "archetype_summary": row["archetype_summary"],
                    "archetype_extras_summary": row["archetype_extras_summary"],
                    "physical_description": row["physical_description"],
                    "relationships": self._parse_json_field(row["relationships"], []),
                    "stats": {
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"],
                        "closeness": row["closeness"],
                        "trust": row["trust"],
                        "respect": row["respect"],
                        "intensity": row["intensity"]
                    },
                    "hobbies": self._parse_json_field(row["hobbies"], []),
                    "personality_traits": self._parse_json_field(row["personality_traits"], []),
                    "likes": self._parse_json_field(row["likes"], []),
                    "dislikes": self._parse_json_field(row["dislikes"], []),
                    "affiliations": self._parse_json_field(row["affiliations"], []),
                    "schedule": self._parse_json_field(row["schedule"], {}),
                    "current_location": row["current_location"],
                    "sex": row["sex"],
                    "age": row["age"],
                    "memories": self._parse_json_field(row["memory"], []),
                    "faction_affiliations": self._parse_json_field(row["faction_affiliations"], [])
                }
                
                # Store in cache
                cache_result(cache_key, npc_details, ttl_seconds=300)  # Cache for 5 minutes
                
                return npc_details
                
        except NPCNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error getting NPC details: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def get_npc_name(self, npc_id: int) -> str:
        """
        Get the name of an NPC by ID.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Name of the NPC
            
        Raises:
            NPCNotFoundError: If the NPC cannot be found
            NPCDataAccessError: For other data access errors
        """
        # Check cache first
        cache_key = f"npc_name:{self.user_id}:{self.conversation_id}:{npc_id}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            self.cache_metrics['hits'] += 1
            return cached_result
        
        self.cache_metrics['misses'] += 1
        
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT npc_name
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    LIMIT 1
                """
                
                row = await conn.fetchrow(query, self.user_id, self.conversation_id, npc_id)
                
                if not row:
                    error_msg = f"NPC not found: {npc_id}"
                    logger.error(error_msg)
                    raise NPCNotFoundError(error_msg)
                
                npc_name = row["npc_name"]
                
                # Store in cache
                cache_result(cache_key, npc_name, ttl_seconds=3600)  # Cache for 1 hour
                
                return npc_name
                
        except NPCNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error getting NPC name: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def get_npc_memories(self, npc_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories associated with an NPC.
        
        Args:
            npc_id: ID of the NPC
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory objects
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                # First check if the NPC has memories in the NPCStats table
                query = """
                    SELECT memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    LIMIT 1
                """
                
                row = await conn.fetchrow(query, self.user_id, self.conversation_id, npc_id)
                
                if row and row["memory"]:
                    memories = self._parse_json_field(row["memory"], [])
                    
                    # Format as memory objects if they're simple strings
                    if memories and isinstance(memories[0], str):
                        memories = [
                            {
                                "id": i,
                                "text": memory,
                                "type": "personal",
                                "significance": 5,
                                "emotional_intensity": 50
                            } for i, memory in enumerate(memories)
                        ]
                    
                    return memories[:limit]
                
                # Try the unified_memories table
                query = """
                    SELECT id, memory_text, memory_type, significance, emotional_intensity, 
                           tags, timestamp, is_emotional, importance
                    FROM unified_memories
                    WHERE entity_type='npc' AND entity_id=$1 
                      AND user_id=$2 AND conversation_id=$3 
                      AND status='active'
                    ORDER BY timestamp DESC
                    LIMIT $4
                """
                
                rows = await conn.fetch(query, npc_id, self.user_id, self.conversation_id, limit)
                
                memories = []
                for row in rows:
                    memory = {
                        "id": row["id"],
                        "text": row["memory_text"],
                        "type": row["memory_type"],
                        "significance": row["significance"],
                        "emotional_intensity": row["emotional_intensity"],
                        "tags": self._parse_json_field(row["tags"], []),
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                        "is_emotional": row["is_emotional"],
                        "importance": row["importance"]
                    }
                    memories.append(memory)
                
                # If still no memories, try the legacy NPCMemories table
                if not memories:
                    query = """
                        SELECT id, memory_text, emotional_intensity, significance, memory_type,
                               timestamp
                        FROM NPCMemories
                        WHERE npc_id=$1 AND status='active'
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """
                    
                    rows = await conn.fetch(query, npc_id, limit)
                    
                    for row in rows:
                        memory = {
                            "id": row["id"],
                            "text": row["memory_text"],
                            "type": row["memory_type"],
                            "significance": row["significance"],
                            "emotional_intensity": row["emotional_intensity"],
                            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                        }
                        memories.append(memory)
                
                return memories
                
        except Exception as e:
            error_msg = f"Error getting NPC memories: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def get_npc_relationships(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get all relationships for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of relationship objects
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT sl.link_id, sl.entity2_type, sl.entity2_id, sl.link_type, sl.link_level,
                           CASE WHEN sl.entity2_type = 'npc' THEN n.npc_name ELSE 'Chase' END as target_name
                    FROM SocialLinks sl
                    LEFT JOIN NPCStats n ON sl.entity2_type = 'npc' AND sl.entity2_id = n.npc_id 
                                 AND n.user_id = sl.user_id AND n.conversation_id = sl.conversation_id
                    WHERE sl.entity1_type = 'npc' 
                      AND sl.entity1_id = $1
                      AND sl.user_id = $2 
                      AND sl.conversation_id = $3
                """
                
                rows = await conn.fetch(query, npc_id, self.user_id, self.conversation_id)
                
                relationships = []
                for row in rows:
                    relationship = {
                        "link_id": row["link_id"],
                        "target_type": row["entity2_type"],
                        "target_id": row["entity2_id"],
                        "target_name": row["target_name"],
                        "link_type": row["link_type"],
                        "link_level": row["link_level"]
                    }
                    relationships.append(relationship)
                
                return relationships
                
        except Exception as e:
            error_msg = f"Error getting NPC relationships: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def get_relationship_details(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int
    ) -> Dict[str, Any]:
        """
        Get detailed information about a relationship between two entities.
        
        Args:
            entity1_type: Type of the first entity (e.g., "npc", "player")
            entity1_id: ID of the first entity
            entity2_type: Type of the second entity
            entity2_id: ID of the second entity
            
        Returns:
            Dictionary with relationship details or empty relationship if not found
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            # Check cache first
            cache_key = f"relationship:{self.user_id}:{self.conversation_id}:{entity1_type}:{entity1_id}:{entity2_type}:{entity2_id}"
            cached_result = get_cached_result(cache_key)
            if cached_result:
                self.cache_metrics['hits'] += 1
                return cached_result
            
            self.cache_metrics['misses'] += 1
            
            async with get_db_connection_context() as conn:
                # Try both orientations of the relationship
                for e1t, e1i, e2t, e2i in [(entity1_type, entity1_id, entity2_type, entity2_id),
                                       (entity2_type, entity2_id, entity1_type, entity1_id)]:
                    query = """
                        SELECT link_id, link_type, link_level, link_history, dynamics, 
                               group_interaction, relationship_stage, experienced_crossroads,
                               experienced_rituals
                        FROM SocialLinks
                        WHERE user_id=$1 AND conversation_id=$2
                          AND entity1_type=$3 AND entity1_id=$4
                          AND entity2_type=$5 AND entity2_id=$6
                        LIMIT 1
                    """
                    
                    row = await conn.fetchrow(query, self.user_id, self.conversation_id, e1t, e1i, e2t, e2i)
                    
                    if row:
                        # Process JSON fields
                        link_history = self._parse_json_field(row["link_history"], [])
                        dynamics = self._parse_json_field(row["dynamics"], {})
                        experienced_crossroads = self._parse_json_field(row["experienced_crossroads"], {})
                        experienced_rituals = self._parse_json_field(row["experienced_rituals"], {})
                        
                        relationship = {
                            "link_id": row["link_id"],
                            "entity1_type": e1t,
                            "entity1_id": e1i,
                            "entity2_type": e2t,
                            "entity2_id": e2i,
                            "link_type": row["link_type"],
                            "link_level": row["link_level"],
                            "link_history": link_history,
                            "dynamics": dynamics,
                            "group_interaction": row["group_interaction"],
                            "relationship_stage": row["relationship_stage"],
                            "experienced_crossroads": experienced_crossroads,
                            "experienced_rituals": experienced_rituals
                        }
                        
                        # Store in cache
                        cache_result(cache_key, relationship, ttl_seconds=300)
                        
                        return relationship
            
            # No relationship found, return a default empty relationship
            empty_relationship = {
                "entity1_type": entity1_type,
                "entity1_id": entity1_id,
                "entity2_type": entity2_type,
                "entity2_id": entity2_id,
                "link_type": "none",
                "link_level": 0,
                "link_history": [],
                "dynamics": {},
                "relationship_stage": "strangers",
                "experienced_crossroads": {},
                "experienced_rituals": {}
            }
            
            # Store empty relationship in cache
            cache_result(cache_key, empty_relationship, ttl_seconds=300)
            
            return empty_relationship
                
        except Exception as e:
            error_msg = f"Error getting relationship details: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def get_nearby_npcs(self, location: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get NPCs that are at a specific location or all NPCs if location not specified.
        
        Args:
            location: Location to filter by (optional)
            limit: Maximum number of NPCs to retrieve
            
        Returns:
            List of nearby NPCs
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                if location:
                    query = """
                        SELECT npc_id, npc_name, current_location, dominance, cruelty
                        FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 
                        AND current_location=$3
                        ORDER BY introduced DESC
                        LIMIT $4
                    """
                    
                    rows = await conn.fetch(query, self.user_id, self.conversation_id, location, limit)
                else:
                    query = """
                        SELECT npc_id, npc_name, current_location, dominance, cruelty
                        FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2
                        ORDER BY introduced DESC
                        LIMIT $3
                    """
                    
                    rows = await conn.fetch(query, self.user_id, self.conversation_id, limit)
                
                nearby_npcs = []
                for row in rows:
                    nearby_npcs.append({
                        "npc_id": row["npc_id"],
                        "npc_name": row["npc_name"],
                        "current_location": row["current_location"],
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"]
                    })
                
                return nearby_npcs
                
        except Exception as e:
            error_msg = f"Error getting nearby NPCs: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def get_available_npcs(self, include_introduced_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all available NPCs with their details.
        
        Args:
            include_introduced_only: Whether to include only introduced NPCs
            
        Returns:
            List of NPC data dictionaries
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT npc_id, npc_name, introduced, dominance, cruelty, closeness, trust,
                           respect, intensity, sex, current_location, faction_affiliations
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """
                
                if include_introduced_only:
                    query += " AND introduced=TRUE"
                
                query += " ORDER BY dominance DESC"
                
                rows = await conn.fetch(query, self.user_id, self.conversation_id)
                
                npcs = []
                for row in rows:
                    # Parse faction affiliations
                    faction_affiliations = self._parse_json_field(row["faction_affiliations"], [])
                    
                    # Build the NPC data dictionary
                    npc = {
                        "npc_id": row["npc_id"],
                        "npc_name": row["npc_name"],
                        "introduced": row["introduced"],
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"],
                        "closeness": row["closeness"],
                        "trust": row["trust"],
                        "respect": row["respect"],
                        "intensity": row["intensity"],
                        "sex": row["sex"],
                        "current_location": row["current_location"],
                        "faction_affiliations": faction_affiliations
                    }
                    
                    # Get relationship with player (async)
                    relationship = await self.get_relationship_details("npc", row["npc_id"], "player", 0)
                    npc["relationship_with_player"] = {
                        "link_type": relationship.get("link_type", "none"),
                        "link_level": relationship.get("link_level", 0)
                    }
                    
                    npcs.append(npc)
                
                return npcs
                
        except Exception as e:
            error_msg = f"Error getting available NPCs: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def add_npc_memory(self, npc_id: int, memory_text: str, memory_type: str = "observation", significance: int = 5, emotional_intensity: int = 50) -> int:
        """
        Add a memory to an NPC.
        
        Args:
            npc_id: ID of the NPC
            memory_text: Text of the memory
            memory_type: Type of the memory
            significance: Significance of the memory (1-10)
            emotional_intensity: Emotional intensity of the memory (0-100)
            
        Returns:
            ID of the created memory
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                # First, update the memory array in NPCStats
                query = """
                    SELECT memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                """
                
                row = await conn.fetchrow(query, self.user_id, self.conversation_id, npc_id)
                
                memories = []
                if row and row["memory"]:
                    memories = self._parse_json_field(row["memory"], [])
                
                memories.append(memory_text)
                
                update_query = """
                    UPDATE NPCStats
                    SET memory = $1
                    WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                """
                
                await conn.execute(update_query, json.dumps(memories), self.user_id, self.conversation_id, npc_id)
                
                # Also add to unified_memories for better compatibility
                insert_query = """
                    INSERT INTO unified_memories (
                        entity_type, entity_id, user_id, conversation_id,
                        memory_text, memory_type, significance, emotional_intensity,
                        status, timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """
                
                memory_id = await conn.fetchval(
                    insert_query,
                    "npc", npc_id, self.user_id, self.conversation_id,
                    memory_text, memory_type, significance, emotional_intensity,
                    "active", datetime.now()
                )
                
                # Invalidate cache for this NPC
                self._invalidate_npc_cache(npc_id)
                
                return memory_id
                
        except Exception as e:
            error_msg = f"Error adding NPC memory: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def update_npc_stats(self, npc_id: int, stats_changes: Dict[str, int]) -> Dict[str, Any]:
        """
        Update stats for an NPC.
        
        Args:
            npc_id: ID of the NPC
            stats_changes: Dictionary of stat changes (e.g., {"dominance": 5, "cruelty": -2})
            
        Returns:
            Dictionary with updated stats
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                # Get current stats
                query = """
                    SELECT dominance, cruelty, closeness, trust, respect, intensity
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                """
                
                row = await conn.fetchrow(query, self.user_id, self.conversation_id, npc_id)
                
                if not row:
                    raise NPCNotFoundError(f"NPC not found: {npc_id}")
                
                # Calculate new stats
                new_stats = {}
                for stat, value in row.items():
                    if stat in stats_changes:
                        new_stats[stat] = max(0, min(100, value + stats_changes[stat]))
                    else:
                        new_stats[stat] = value
                
                # Update stats
                update_query = """
                    UPDATE NPCStats
                    SET dominance=$1, cruelty=$2, closeness=$3, trust=$4, respect=$5, intensity=$6
                    WHERE user_id=$7 AND conversation_id=$8 AND npc_id=$9
                """
                
                await conn.execute(
                    update_query,
                    new_stats["dominance"], new_stats["cruelty"], new_stats["closeness"],
                    new_stats["trust"], new_stats["respect"], new_stats["intensity"],
                    self.user_id, self.conversation_id, npc_id
                )
                
                # Invalidate cache for this NPC
                self._invalidate_npc_cache(npc_id)
                
                return new_stats
                
        except NPCNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error updating NPC stats: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def update_npc_location(self, npc_id: int, new_location: str) -> bool:
        """
        Update the current location of an NPC.
        
        Args:
            npc_id: ID of the NPC
            new_location: New location for the NPC
            
        Returns:
            True if successful
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                query = """
                    UPDATE NPCStats
                    SET current_location=$1
                    WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                    RETURNING npc_id
                """
                
                result = await conn.fetchrow(query, new_location, self.user_id, self.conversation_id, npc_id)
                
                if not result:
                    raise NPCNotFoundError(f"NPC not found: {npc_id}")
                
                # Invalidate cache for this NPC
                self._invalidate_npc_cache(npc_id)
                
                return True
                
        except NPCNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error updating NPC location: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    def _parse_json_field(self, field_value, default_value=None):
        """
        Safely parse a JSON field from the database.
        
        Args:
            field_value: The field value to parse
            default_value: Default value to return if parsing fails
            
        Returns:
            Parsed JSON value or default value
        """
        if field_value is None:
            return default_value
        
        if not isinstance(field_value, str):
            return field_value
        
        try:
            return json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            return default_value
    
    def _invalidate_npc_cache(self, npc_id: int) -> None:
        """
        Invalidate cache entries for a specific NPC.
        
        Args:
            npc_id: ID of the NPC
        """
        # Invalidate NPC details cache
        cache_key = f"npc_details:{self.user_id}:{self.conversation_id}:{npc_id}"
        invalidate_cache(cache_key)
        
        # Invalidate NPC name cache
        cache_key = f"npc_name:{self.user_id}:{self.conversation_id}:{npc_id}"
        invalidate_cache(cache_key)
    
    async def get_npc_count(self) -> int:
        """
        Get the total count of NPCs for the current user/conversation.
        
        Returns:
            Total count of NPCs
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT COUNT(*) as count
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """
                
                row = await conn.fetchrow(query, self.user_id, self.conversation_id)
                
                return row["count"] if row else 0
                
        except Exception as e:
            error_msg = f"Error getting NPC count: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def get_npcs_by_faction(self, faction_id: int) -> List[Dict[str, Any]]:
        """
        Get all NPCs that belong to a specific faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            List of NPC data dictionaries
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        try:
            async with get_db_connection_context() as conn:
                # Query NPCs with faction affiliation
                query = """
                    SELECT npc_id, npc_name, introduced, dominance, cruelty, 
                           current_location, faction_affiliations
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """
                
                rows = await conn.fetch(query, self.user_id, self.conversation_id)
                
                # Filter NPCs by faction affiliation
                npcs_in_faction = []
                for row in rows:
                    # Parse faction affiliations
                    faction_affiliations = self._parse_json_field(row["faction_affiliations"], [])
                    
                    # Check if the NPC belongs to the faction
                    belongs_to_faction = False
                    faction_position = None
                    
                    for affiliation in faction_affiliations:
                        if isinstance(affiliation, dict) and affiliation.get("faction_id") == faction_id:
                            belongs_to_faction = True
                            faction_position = affiliation.get("position")
                            break
                    
                    if belongs_to_faction:
                        npcs_in_faction.append({
                            "npc_id": row["npc_id"],
                            "npc_name": row["npc_name"],
                            "introduced": row["introduced"],
                            "dominance": row["dominance"],
                            "cruelty": row["cruelty"],
                            "current_location": row["current_location"],
                            "faction_position": faction_position
                        })
                
                return npcs_in_faction
                
        except Exception as e:
            error_msg = f"Error getting NPCs by faction: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
    
    async def check_npc_exists(self, npc_id: Optional[int] = None, npc_name: Optional[str] = None) -> bool:
        """
        Check if an NPC exists by ID or name.
        
        Args:
            npc_id: Optional ID of the NPC
            npc_name: Optional name of the NPC
            
        Returns:
            True if the NPC exists, False otherwise
            
        Raises:
            NPCDataAccessError: For data access errors
        """
        if not npc_id and not npc_name:
            raise ValueError("Either npc_id or npc_name must be provided")
        
        try:
            async with get_db_connection_context() as conn:
                query = """
                    SELECT 1
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """
                
                params = [self.user_id, self.conversation_id]
                
                if npc_id is not None:
                    query += " AND npc_id=$3"
                    params.append(npc_id)
                elif npc_name is not None:
                    query += " AND LOWER(npc_name)=LOWER($3)"
                    params.append(npc_name)
                
                query += " LIMIT 1"
                
                row = await conn.fetchrow(query, *params)
                
                return row is not None
                
        except Exception as e:
            error_msg = f"Error checking if NPC exists: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise NPCDataAccessError(error_msg)
