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
# FIX: Import the specific cache instance we need
from utils.caching import NPC_CACHE # Import the MemoryCache instance

# Configure logger
# Ensure logger is properly configured at the application entry point
# For this module:
logger = logging.getLogger(__name__) # Use standard Python logging

class NPCDataAccessError(Exception):
    """Base exception for NPC data access errors"""
    pass

class NPCNotFoundError(NPCDataAccessError):
    """Exception raised when an NPC cannot be found"""
    pass

class NPCDataAccess:
    """
    Data Access Layer for NPC-related database operations.
    """

    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the NPC data access layer.
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        # Cache metrics are now handled within the MemoryCache instance (NPC_CACHE.stats())
        # self.cache_metrics = ... # Remove local cache metrics tracking

    async def get_npc_details(self, npc_id: Optional[int] = None, npc_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about an NPC by ID or name.
        """
        if not npc_id and not npc_name:
            raise ValueError("Either npc_id or npc_name must be provided")

        # Check cache first
        # Use a specific prefix for clarity if desired
        cache_key = f"npc:details:{self.user_id}:{self.conversation_id}:{npc_id or npc_name}"
        # FIX: Use NPC_CACHE.get()
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            # Logging handled by MemoryCache get method
            # self.cache_metrics['hits'] += 1
            return cached_result

        # Logging handled by MemoryCache get method
        # self.cache_metrics['misses'] += 1

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
                    # Use LOWER for case-insensitive matching consistently
                    query += " AND LOWER(npc_name)=LOWER($3)"
                    params.append(npc_name)

                query += " LIMIT 1"
                logger.debug(f"Executing query for NPC details: {query} with params: {params}")
                row = await conn.fetchrow(query, *params)

                if not row:
                    error_msg = f"NPC not found: user={self.user_id}, convo={self.conversation_id}, id/name={npc_id or npc_name}"
                    logger.warning(error_msg) # Use warning or error based on expected frequency
                    raise NPCNotFoundError(error_msg)

                # Build the NPC details with parsed JSON fields
                npc_details = {
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"],
                    "introduced": row["introduced"],
                    "archetypes": self._parse_json_field(row["archetypes"], default_value=[]),
                    "archetype_summary": row["archetype_summary"],
                    "archetype_extras_summary": row["archetype_extras_summary"],
                    "physical_description": row["physical_description"],
                    "relationships": self._parse_json_field(row["relationships"], default_value=[]),
                    "stats": {
                        "dominance": row["dominance"], "cruelty": row["cruelty"],
                        "closeness": row["closeness"], "trust": row["trust"],
                        "respect": row["respect"], "intensity": row["intensity"]
                    },
                    "hobbies": self._parse_json_field(row["hobbies"], default_value=[]),
                    "personality_traits": self._parse_json_field(row["personality_traits"], default_value=[]),
                    "likes": self._parse_json_field(row["likes"], default_value=[]),
                    "dislikes": self._parse_json_field(row["dislikes"], default_value=[]),
                    "affiliations": self._parse_json_field(row["affiliations"], default_value=[]),
                    "schedule": self._parse_json_field(row["schedule"], default_value={}),
                    "current_location": row["current_location"],
                    "sex": row["sex"],
                    "age": row["age"],
                    "memories": self._parse_json_field(row["memory"], default_value=[]),
                    "faction_affiliations": self._parse_json_field(row["faction_affiliations"], default_value=[])
                }

                # Store in cache
                # FIX: Use NPC_CACHE.set() with 'ttl' argument
                NPC_CACHE.set(cache_key, npc_details, ttl=300)  # Cache for 5 minutes

                return npc_details

        except NPCNotFoundError:
             raise # Re-raise specific error
        except Exception as e:
            error_msg = f"Error getting NPC details (key: {cache_key}): {str(e)}"
            logger.exception(error_msg) # Use exception to include traceback
            raise NPCDataAccessError(error_msg) from e

    async def get_npc_name(self, npc_id: int) -> str:
        """
        Get the name of an NPC by ID.
        """
        cache_key = f"npc:name:{self.user_id}:{self.conversation_id}:{npc_id}"
        # FIX: Use NPC_CACHE.get()
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            return cached_result

        try:
            async with get_db_connection_context() as conn:
                query = "SELECT npc_name FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3 LIMIT 1"
                logger.debug(f"Executing query for NPC name: {query} with params: {[self.user_id, self.conversation_id, npc_id]}")
                row = await conn.fetchrow(query, self.user_id, self.conversation_id, npc_id)

                if not row:
                    error_msg = f"NPC name not found: user={self.user_id}, convo={self.conversation_id}, id={npc_id}"
                    logger.warning(error_msg)
                    raise NPCNotFoundError(error_msg)

                npc_name = row["npc_name"]

                # Store in cache
                # FIX: Use NPC_CACHE.set() with 'ttl' argument
                NPC_CACHE.set(cache_key, npc_name, ttl=3600)  # Cache for 1 hour

                return npc_name

        except NPCNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error getting NPC name (key: {cache_key}): {str(e)}"
            logger.exception(error_msg)
            raise NPCDataAccessError(error_msg) from e
    
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
        self, entity1_type: str, entity1_id: int, entity2_type: str, entity2_id: int
    ) -> Dict[str, Any]:
        """
        Get detailed information about a relationship between two entities.
        Uses caching.
        """
        # Ensure consistent key order regardless of entity order input
        e1_key = f"{entity1_type}:{entity1_id}"
        e2_key = f"{entity2_type}:{entity2_id}"
        sorted_keys = sorted([e1_key, e2_key])
        cache_key = f"relationship:{self.user_id}:{self.conversation_id}:{sorted_keys[0]}:{sorted_keys[1]}"

        # FIX: Use NPC_CACHE.get() (or potentially a different cache instance like RELATIONSHIP_CACHE if defined)
        # Using NPC_CACHE here for consistency with the original code's import attempt.
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            return cached_result

        try:
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
                    logger.debug(f"Executing query for relationship details: {query} with params: {[self.user_id, self.conversation_id, e1t, e1i, e2t, e2i]}")
                    row = await conn.fetchrow(query, self.user_id, self.conversation_id, e1t, e1i, e2t, e2i)

                    if row:
                        # Process JSON fields
                        link_history = self._parse_json_field(row["link_history"], [])
                        dynamics = self._parse_json_field(row["dynamics"], {})
                        experienced_crossroads = self._parse_json_field(row["experienced_crossroads"], {})
                        experienced_rituals = self._parse_json_field(row["experienced_rituals"], {})

                        relationship = {
                            "link_id": row["link_id"],
                            "entity1_type": e1t, "entity1_id": e1i, # Store the orientation found
                            "entity2_type": e2t, "entity2_id": e2i,
                            "link_type": row["link_type"],
                            "link_level": row["link_level"],
                            "link_history": link_history,
                            "dynamics": dynamics,
                            "group_interaction": row["group_interaction"],
                            "relationship_stage": row["relationship_stage"],
                            "experienced_crossroads": experienced_crossroads,
                            "experienced_rituals": experienced_rituals
                        }

                        # FIX: Use NPC_CACHE.set() with ttl
                        NPC_CACHE.set(cache_key, relationship, ttl=300) # Cache for 5 min
                        return relationship

            # No relationship found, return a default empty structure
            logger.debug(f"No relationship found for key: {cache_key}. Returning default.")
            empty_relationship = {
                 # Store original requested orientation in the 'empty' result for clarity
                "entity1_type": entity1_type, "entity1_id": entity1_id,
                "entity2_type": entity2_type, "entity2_id": entity2_id,
                "link_type": "none", "link_level": 0, "link_history": [], "dynamics": {},
                "relationship_stage": "strangers", "experienced_crossroads": {}, "experienced_rituals": {}
            }

            # Cache the 'empty' result to avoid repeated DB lookups for non-existent relationships
            # FIX: Use NPC_CACHE.set() with ttl
            NPC_CACHE.set(cache_key, empty_relationship, ttl=300) # Cache 'not found' state briefly
            return empty_relationship

        except Exception as e:
            error_msg = f"Error getting relationship details (key: {cache_key}): {str(e)}"
            logger.exception(error_msg)
            raise NPCDataAccessError(error_msg) from e
    
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
        Add a memory to an NPC (updating both NPCStats and unified_memories).
        Invalidates relevant NPC cache entries.
        """
        # Implementation remains similar, but ensure cache invalidation at the end
        try:
            memory_id = -1 # Default invalid ID
            async with get_db_connection_context() as conn:
                # Transaction for atomicity
                 async with conn.transaction():
                    # 1. Update the memory array in NPCStats (if still used)
                    # Consider deprecating this if unified_memories is primary
                    query_select_mem = "SELECT memory FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3 FOR UPDATE"
                    row = await conn.fetchrow(query_select_mem, self.user_id, self.conversation_id, npc_id)

                    if row: # Only update if NPC exists in NPCStats
                         memories = self._parse_json_field(row["memory"], [])
                         # Avoid adding duplicates if possible, or just append
                         if memory_text not in memories: # Simple check
                             memories.append(memory_text)
                             query_update_mem = "UPDATE NPCStats SET memory = $1 WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4"
                             await conn.execute(query_update_mem, json.dumps(memories), self.user_id, self.conversation_id, npc_id)
                    else:
                         logger.warning(f"NPC {npc_id} not found in NPCStats while trying to add memory '{memory_text[:50]}...'. Skipping NPCStats update.")


                    # 2. Add to unified_memories (primary storage)
                    insert_query = """
                        INSERT INTO unified_memories (
                            entity_type, entity_id, user_id, conversation_id,
                            memory_text, memory_type, significance, emotional_intensity,
                            status, timestamp
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id
                    """
                    memory_id = await conn.fetchval(
                        insert_query,
                        "npc", npc_id, self.user_id, self.conversation_id,
                        memory_text, memory_type, significance, emotional_intensity,
                        "active", datetime.now()
                    )

            # Invalidate cache for this NPC after successful DB updates
            if memory_id > 0:
                 self._invalidate_npc_cache(npc_id)
                 logger.info(f"Added memory (ID: {memory_id}) for NPC {npc_id} and invalidated cache.")
            else:
                 logger.error(f"Failed to add memory for NPC {npc_id} (unified_memories insert failed?).")


            return memory_id

        except Exception as e:
            error_msg = f"Error adding NPC memory for NPC {npc_id}: {str(e)}"
            logger.exception(error_msg)
            raise NPCDataAccessError(error_msg) from e

    
    async def update_npc_stats(self, npc_id: int, stats_changes: Dict[str, int]) -> Dict[str, Any]:
        """
        Update stats for an NPC. Invalidates relevant cache.
        """
        # Implementation remains similar, but ensure cache invalidation
        try:
             new_stats = {}
             async with get_db_connection_context() as conn:
                 async with conn.transaction():
                    # Get current stats first, locking the row
                    query_select = """
                        SELECT dominance, cruelty, closeness, trust, respect, intensity
                        FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                        FOR UPDATE
                    """
                    row = await conn.fetchrow(query_select, self.user_id, self.conversation_id, npc_id)

                    if not row:
                        raise NPCNotFoundError(f"NPC not found for stat update: {npc_id}")

                    # Calculate new stats, clamping between 0 and 100
                    current_stats = dict(row)
                    for stat, change in stats_changes.items():
                        if stat in current_stats:
                            current_value = current_stats[stat] or 0 # Handle potential NULLs
                            new_value = max(0, min(100, current_value + change))
                            new_stats[stat] = new_value
                        else:
                            logger.warning(f"Attempted to update non-existent stat '{stat}' for NPC {npc_id}")

                    # Only update if there are valid stats to change
                    if new_stats:
                        # Build SET clause dynamically for provided stats
                        set_clauses = [f"{stat}=${i+4}" for i, stat in enumerate(new_stats.keys())]
                        params = list(new_stats.values()) + [self.user_id, self.conversation_id, npc_id]

                        update_query = f"""
                            UPDATE NPCStats
                            SET {', '.join(set_clauses)}
                            WHERE user_id=${len(params)-2} AND conversation_id=${len(params)-1} AND npc_id=${len(params)}
                        """
                        await conn.execute(update_query, *params)
                        logger.info(f"Updated stats for NPC {npc_id}: {stats_changes}. New values: {new_stats}")
                    else:
                        logger.warning(f"No valid stats provided for update on NPC {npc_id}. No changes made.")
                        # Return current stats if no changes made
                        new_stats = current_stats


             # Invalidate cache for this NPC after successful DB update
             self._invalidate_npc_cache(npc_id)

             # Return the final calculated stats (either updated or original if no changes)
             final_stats_to_return = {**current_stats, **new_stats}
             return final_stats_to_return

        except NPCNotFoundError:
             raise
        except Exception as e:
             error_msg = f"Error updating NPC stats for NPC {npc_id}: {str(e)}"
             logger.exception(error_msg)
             raise NPCDataAccessError(error_msg) from e

    
    async def update_npc_location(self, npc_id: int, new_location: str) -> bool:
        """
        Update the current location of an NPC. Invalidates relevant cache.
        """
        try:
             async with get_db_connection_context() as conn:
                 query = """
                     UPDATE NPCStats SET current_location=$1
                     WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                     RETURNING npc_id
                 """
                 result = await conn.fetchval(query, new_location, self.user_id, self.conversation_id, npc_id)

                 if result is None: # fetchval returns None if no row updated
                     raise NPCNotFoundError(f"NPC not found for location update: {npc_id}")

             # Invalidate cache for this NPC
             self._invalidate_npc_cache(npc_id)
             logger.info(f"Updated location for NPC {npc_id} to '{new_location}' and invalidated cache.")
             return True

        except NPCNotFoundError:
            raise
        except Exception as e:
             error_msg = f"Error updating NPC location for NPC {npc_id}: {str(e)}"
             logger.exception(error_msg)
             raise NPCDataAccessError(error_msg) from e


    def _parse_json_field(self, field_value: Optional[Union[str, Dict, List]], default_value: Any = None) -> Any:
        """Safely parse JSON field, handling potential errors and non-string inputs."""
        if field_value is None:
             # logger.debug("JSON field is None, returning default.")
             return default_value

        # If it's already parsed (e.g., asyncpg might do this), return directly
        if isinstance(field_value, (dict, list)):
            # logger.debug("JSON field already parsed, returning directly.")
            return field_value

        if isinstance(field_value, str):
            try:
                # Handle empty strings explicitly
                if not field_value.strip():
                    # logger.debug("JSON field is empty string, returning default.")
                    return default_value
                return json.loads(field_value)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse JSON field: '{field_value[:100]}...'. Error: {e}. Returning default value.")
                return default_value
        else:
            # Log if it's an unexpected type
            logger.warning(f"Unexpected type for JSON field: {type(field_value)}. Value: '{str(field_value)[:100]}...'. Returning default.")
            return default_value

    
    def _invalidate_npc_cache(self, npc_id: int) -> None:
        """Invalidate specific cache entries related to an NPC."""
        logger.debug(f"Invalidating cache entries for NPC ID: {npc_id}")

        # Invalidate NPC details cache
        details_key = f"npc:details:{self.user_id}:{self.conversation_id}:{npc_id}"
        # FIX: Use NPC_CACHE.delete()
        NPC_CACHE.delete(details_key)

        # Invalidate NPC name cache
        name_key = f"npc:name:{self.user_id}:{self.conversation_id}:{npc_id}"
        # FIX: Use NPC_CACHE.delete()
        NPC_CACHE.delete(name_key)

        # Add invalidation for other related keys if necessary
        # e.g., invalidate related relationship caches, location caches containing the NPC, etc.
        # Example: Invalidate relationship caches involving this NPC
        # This requires knowing potential relationship partners, which might be complex.
        # A simpler approach might be pattern deletion if keys are structured well,
        # but be cautious with performance implications of pattern matching.
        # relation_pattern = f"relationship:{self.user_id}:{self.conversation_id}:.*npc:{npc_id}.*"
        # removed_count = NPC_CACHE.remove_pattern(f"npc:{npc_id}") # Example pattern

        logger.debug(f"Completed cache invalidation for keys related to NPC ID: {npc_id}")
        
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
