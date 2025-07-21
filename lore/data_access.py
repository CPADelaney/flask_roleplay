# lore/data_access.py

"""
Lore Data Access Layer

This module provides standardized data access classes for all lore-related data.
It serves as the single point of interaction with the database for lore components.
"""

import logging
import json
import asyncio
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import asyncpg
from datetime import datetime

from db.connection import get_db_connection_context
from embedding.vector_store import generate_embedding, compute_similarity

logger = logging.getLogger(__name__)

class BaseDataAccess:
    """Base class for all data access components."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """
        Initialize the base data access component.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize the data access component.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.initialized:
            return True
            
        try:
            # Test database connection
            async with get_db_connection_context() as conn:
                await conn.fetchval("SELECT 1")
                self.initialized = True
                return True
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        # Nothing to clean up since we're using the shared pool
        pass
    
    async def add_user_conversation_filters(self, query: str, params: List[Any]) -> Tuple[str, List[Any]]:
        """
        Add user and conversation ID filters to a query.
        
        Args:
            query: The SQL query
            params: List of current parameters
            
        Returns:
            Tuple of modified query and parameters
        """
        modified_query = query
        modified_params = list(params)  # Create a copy
        
        # Add user_id filter if specified
        if self.user_id is not None:
            if "WHERE" in modified_query:
                modified_query += f" AND user_id = ${len(modified_params) + 1}"
            else:
                modified_query += f" WHERE user_id = ${len(modified_params) + 1}"
            modified_params.append(self.user_id)
            
        # Add conversation_id filter if specified
        if self.conversation_id is not None:
            if "WHERE" in modified_query:
                modified_query += f" AND conversation_id = ${len(modified_params) + 1}"
            else:
                modified_query += f" WHERE conversation_id = ${len(modified_params) + 1}"
            modified_params.append(self.conversation_id)
            
        return modified_query, modified_params


class NPCDataAccess(BaseDataAccess):
    """Data access for NPC-related data."""
    
    async def get_npc_details(self, npc_id: Optional[int] = None, 
                            npc_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get NPC details by ID or name.
        
        Args:
            npc_id: Optional NPC ID
            npc_name: Optional NPC name
            
        Returns:
            Dictionary with NPC details
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Build base query
            query = """
                SELECT * FROM NPCStats
            """
            params = []
            
            # Add conditions based on provided parameters
            conditions = []
            
            if npc_id is not None:
                conditions.append(f"npc_id = ${len(params) + 1}")
                params.append(npc_id)
            
            if npc_name is not None:
                conditions.append(f"npc_name = ${len(params) + 1}")
                params.append(npc_name)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Add limit
            query += " LIMIT 1"
            
            # Execute query
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, *params)
                    
                if not row:
                    return {}
                    
                npc_data = dict(row)
                    
                # Parse JSON fields
                for field in ["personality_traits", "archetypes", "affiliations"]:
                    if field in npc_data and isinstance(npc_data[field], str):
                        try:
                            npc_data[field] = json.loads(npc_data[field])
                        except json.JSONDecodeError:
                            npc_data[field] = []
                    
                return npc_data
                    
        except Exception as e:
            logger.error(f"Error getting NPC details: {e}")
            return {}
    
    async def get_npc_relationships(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get relationships for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of NPC relationships
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT r.*, n.npc_name as related_npc_name 
                FROM NPCRelationships r
                JOIN NPCStats n ON r.related_npc_id = n.npc_id
                WHERE r.npc_id = $1
            """
            params = [npc_id]
            
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Execute query
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
                    
        except Exception as e:
            logger.error(f"Error getting NPC relationships: {e}")
            return []
    
    async def get_npc_cultural_attributes(self, npc_id: int) -> Dict[str, Any]:
        """
        Get cultural attributes for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with cultural attributes
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get NPC details first
            npc_data = await self.get_npc_details(npc_id=npc_id)
            
            if not npc_data:
                return {}
                
            # Extract cultural information
            cultural_data = {
                "nationality": None,
                "faith": None,
                "languages": [],
                "primary_dialect": None,
                "dialect_features": {},
                "cultural_norms_followed": []
            }
            
            async with get_db_connection_context() as conn:
                # Get nationality
                query = """
                    SELECT n.* FROM Nations n
                    JOIN NPCAttributes a ON n.id = a.value_id
                    WHERE a.npc_id = $1 AND a.attribute_type = 'nationality'
                """
                nationality = await conn.fetchrow(query, npc_id)
                if nationality:
                    cultural_data["nationality"] = dict(nationality)
                
                # Get faith/religion
                query = """
                    SELECT ce.* FROM CulturalElements ce
                    JOIN NPCAttributes a ON ce.id = a.value_id
                    WHERE a.npc_id = $1 AND a.attribute_type = 'faith'
                    AND ce.element_type = 'religion'
                """
                faith = await conn.fetchrow(query, npc_id)
                if faith:
                    cultural_data["faith"] = dict(faith)
                
                # Get languages
                query = """
                    SELECT l.*, la.fluency 
                    FROM Languages l
                    JOIN NPCLanguageAbilities la ON l.id = la.language_id
                    WHERE la.npc_id = $1
                    ORDER BY la.fluency DESC
                """
                languages = await conn.fetch(query, npc_id)
                cultural_data["languages"] = [dict(lang) for lang in languages]
                
                # Get dialect
                if cultural_data["languages"]:
                    query = """
                        SELECT d.* FROM Dialects d
                        JOIN NPCAttributes a ON d.id = a.value_id
                        WHERE a.npc_id = $1 AND a.attribute_type = 'dialect'
                    """
                    dialect = await conn.fetchrow(query, npc_id)
                    if dialect:
                        cultural_data["primary_dialect"] = dict(dialect)
                        
                        # Get dialect features
                        query = """
                            SELECT feature_type, feature_value FROM DialectFeatures
                            WHERE dialect_id = $1
                        """
                        features = await conn.fetch(query, dialect["id"])
                        for feature in features:
                            feature_type = feature["feature_type"]
                            if feature_type not in cultural_data["dialect_features"]:
                                cultural_data["dialect_features"][feature_type] = []
                            cultural_data["dialect_features"][feature_type].append(
                                feature["feature_value"]
                            )
                
                # Get cultural norms
                query = """
                    SELECT * FROM CulturalNorms cn
                    JOIN NPCAttributes a ON cn.id = a.value_id
                    WHERE a.npc_id = $1 AND a.attribute_type = 'cultural_norm'
                """
                norms = await conn.fetch(query, npc_id)
                cultural_data["cultural_norms_followed"] = [dict(norm) for norm in norms]
            
            return cultural_data
            
        except Exception as e:
            logger.error(f"Error getting NPC cultural attributes: {e}")
            return {}
    
    async def get_npc_personality(self, npc_id: int) -> Dict[str, Any]:
        """
        Get personality information for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with personality information
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get NPC details first
            npc_data = await self.get_npc_details(npc_id=npc_id)
            
            if not npc_data:
                return {
                    "traits": {},
                    "dominance": 50,
                    "archetypes": [],
                    "alignment": "neutral"
                }
            
            # Create structured personality data
            result = {
                "traits": {},
                "dominance": npc_data.get("dominance", 50),
                "archetypes": [],
                "alignment": "neutral"
            }
            
            # Extract traits with their values
            traits = npc_data.get("personality_traits", {})
            if isinstance(traits, dict):
                result["traits"] = traits
            elif isinstance(traits, list):
                # Convert list format to dict format
                for trait in traits:
                    if isinstance(trait, dict) and "name" in trait and "value" in trait:
                        result["traits"][trait["name"]] = trait["value"]
                    elif isinstance(trait, str):
                        # Assume default strength if only trait name provided
                        result["traits"][trait] = 7
            
            # Get archetypes from archetype field or fallback to backstory
            archetypes = npc_data.get("archetypes", [])
            if isinstance(archetypes, list):
                result["archetypes"] = archetypes[:3]  # Limit to top 3
            elif isinstance(archetypes, dict) and "primary" in archetypes:
                # Handle structured archetype format
                result["archetypes"] = [archetypes["primary"]]
                if "secondary" in archetypes:
                    result["archetypes"].append(archetypes["secondary"])
                
            # Get alignment
            psychological_profile = npc_data.get("psychological_profile", {})
            if isinstance(psychological_profile, dict):
                result["alignment"] = psychological_profile.get("alignment", "neutral")
                
                # Add additional temperament data if available
                if "temperament" in psychological_profile:
                    result["temperament"] = psychological_profile["temperament"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting NPC personality: {e}")
            return {
                "traits": {},
                "dominance": 50,
                "archetypes": [],
                "alignment": "neutral"
            }


class LocationDataAccess(BaseDataAccess):
    """Data access for location-related data."""
    
    async def get_location_details(self, location_id: Optional[int] = None, 
                                 location_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get location details by ID or name.
        
        Args:
            location_id: Optional location ID
            location_name: Optional location name
            
        Returns:
            Dictionary with location details
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Build base query
            query = """
                SELECT * FROM Locations
            """
            params = []
            
            # Add conditions based on provided parameters
            conditions = []
            
            if location_id is not None:
                conditions.append(f"id = ${len(params) + 1}")
                params.append(location_id)
            
            if location_name is not None:
                conditions.append(f"location_name = ${len(params) + 1}")
                params.append(location_name)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Add limit
            query += " LIMIT 1"
            
            # Execute query
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    return {}
                    
                return dict(row)
                    
        except Exception as e:
            logger.error(f"Error getting location details: {e}")
            return {}
    
    async def get_location_by_name(self, location_name: str) -> Dict[str, Any]:
        """
        Get location details by name.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Location details
        """
        return await self.get_location_details(location_name=location_name)
    
    async def get_location_with_lore(self, location_id: int) -> Dict[str, Any]:
        """
        Get location with its associated lore.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Location with lore
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get location details
            location = await self.get_location_details(location_id=location_id)
            
            if not location:
                return {}
                
            # Get location lore
            query = """
                SELECT * FROM LocationLore
                WHERE location_id = $1
            """
            
            async with get_db_connection_context() as conn:
                lore = await conn.fetchrow(query, location_id)
                
                if lore:
                    lore_dict = dict(lore)
                    
                    # Combine location and lore
                    for key, value in lore_dict.items():
                        if key not in ["id", "user_id", "conversation_id", "location_id"]:
                            location[f"lore_{key}"] = value
                
                return location
                    
        except Exception as e:
            logger.error(f"Error getting location with lore: {e}")
            return {}
    
    async def get_cultural_context_for_location(self, location_id: int) -> Dict[str, Any]:
        """
        Get cultural context for a location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with cultural context
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT ce.* 
                FROM CulturalElements ce
                JOIN LoreConnections lc ON ce.id = lc.source_id
                WHERE lc.target_id = $1
                AND ce.user_id = $2 AND ce.conversation_id = $3
                AND lc.source_type = 'CulturalElements'
                AND lc.target_type = 'Locations'
                AND lc.connection_type = 'practiced_at'
            """
            
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    query, 
                    location_id, 
                    self.user_id, 
                    self.conversation_id
                )
                
                elements = [dict(row) for row in rows]
                
                # Also get religious elements
                religious_query = """
                    SELECT ce.* 
                    FROM CulturalElements ce
                    JOIN LoreConnections lc ON ce.id = lc.source_id
                    WHERE lc.target_id = $1
                    AND ce.user_id = $2 AND ce.conversation_id = $3
                    AND ce.element_type = 'religion'
                    AND lc.source_type = 'CulturalElements'
                    AND lc.target_type = 'Locations'
                    AND lc.connection_type = 'practiced_at'
                """
                
                religious_rows = await conn.fetch(
                    religious_query, 
                    location_id, 
                    self.user_id, 
                    self.conversation_id
                )
                
                religious_elements = [dict(row) for row in religious_rows]
                
                return {
                    "elements": elements,
                    "religious_elements": religious_elements,
                    "count": len(elements) + len(religious_elements)
                }
                    
        except Exception as e:
            logger.error(f"Error getting cultural context for location: {e}")
            return {"elements": [], "religious_elements": [], "count": 0}
    
    async def get_political_context_for_location(self, location_id: int) -> Dict[str, Any]:
        """
        Get political context for a location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with political context
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT f.*, lc.strength as influence_level, lc.description as influence_description 
                FROM Factions f
                JOIN LoreConnections lc ON f.id = lc.source_id
                WHERE lc.target_id = $1
                AND f.user_id = $2 AND f.conversation_id = $3
                AND lc.source_type = 'Factions'
                AND lc.target_type = 'Locations'
                AND (lc.connection_type = 'controls' OR lc.connection_type = 'influences')
            """
            
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    query, 
                    location_id, 
                    self.user_id, 
                    self.conversation_id
                )
                
                factions = [dict(row) for row in rows]
                
                return {
                    "ruling_factions": factions,
                    "count": len(factions)
                }
                    
        except Exception as e:
            logger.error(f"Error getting political context for location: {e}")
            return {"ruling_factions": [], "count": 0}
    
    async def get_environmental_conditions(self, location_id: int) -> Dict[str, Any]:
        """
        Get environmental conditions for a location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with environmental conditions
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT * FROM EnvironmentalConditions
                WHERE location_id = $1
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, location_id)
                
                if not row:
                    return {}
                    
                return dict(row)
                    
        except Exception as e:
            logger.error(f"Error getting environmental conditions: {e}")
            return {}
    
    async def get_comprehensive_location_context(self, location_id: int) -> Dict[str, Any]:
        """
        Get comprehensive context for a location including all related data.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with comprehensive location context
        """
        # Get basic location with lore
        location = await self.get_location_with_lore(location_id)
        
        # Get cultural context
        cultural = await self.get_cultural_context_for_location(location_id)
        
        # Get political context
        political = await self.get_political_context_for_location(location_id)
        
        # Get environmental conditions
        environment = await self.get_environmental_conditions(location_id)
        
        return {
            "location": location,
            "cultural_context": cultural,
            "political_context": political,
            "environmental_conditions": environment
        }


class FactionDataAccess(BaseDataAccess):
    """Data access for faction-related data."""
    
    async def get_faction_details(self, faction_id: int) -> Dict[str, Any]:
        """
        Get details about a faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            Faction details
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT * FROM Factions
                WHERE id = $1
            """
            params = [faction_id]
            
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Execute query
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    return {}
                    
                faction_data = dict(row)
                
                # Parse JSON arrays
                for field in ["values", "goals", "rivals", "allies"]:
                    if field in faction_data and isinstance(faction_data[field], str):
                        try:
                            faction_data[field] = json.loads(faction_data[field])
                        except json.JSONDecodeError:
                            faction_data[field] = []
                
                return faction_data
                    
        except Exception as e:
            logger.error(f"Error getting faction details: {e}")
            return {}
    
    async def get_faction_by_name(self, faction_name: str) -> Dict[str, Any]:
        """
        Get faction details by name.
        
        Args:
            faction_name: Name of the faction
            
        Returns:
            Faction details
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT * FROM Factions
                WHERE name = $1
            """
            params = [faction_name]
            
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Add limit
            query += " LIMIT 1"
            
            # Execute query
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    return {}
                    
                faction_data = dict(row)
                
                # Parse JSON arrays
                for field in ["values", "goals", "rivals", "allies"]:
                    if field in faction_data and isinstance(faction_data[field], str):
                        try:
                            faction_data[field] = json.loads(faction_data[field])
                        except json.JSONDecodeError:
                            faction_data[field] = []
                
                return faction_data
                    
        except Exception as e:
            logger.error(f"Error getting faction by name: {e}")
            return {}
    
    async def get_faction_relationships(self, faction_id: int) -> List[Dict[str, Any]]:
        """
        Get relationships for a faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            List of faction relationships
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT lc.*, f.name as related_faction_name, f.type as related_faction_type
                FROM LoreConnections lc
                JOIN Factions f ON lc.target_id = f.id
                WHERE lc.source_id = $1
                AND lc.source_type = 'Factions'
                AND lc.target_type = 'Factions'
            """
            params = [faction_id]
            
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Also get relationships where this faction is the target
            reverse_query = """
                SELECT lc.*, f.name as related_faction_name, f.type as related_faction_type,
                       'reverse' as direction
                FROM LoreConnections lc
                JOIN Factions f ON lc.source_id = f.id
                WHERE lc.target_id = $1
                AND lc.source_type = 'Factions'
                AND lc.target_type = 'Factions'
            """
            reverse_params = [faction_id]
            
            # Add user/conversation filters
            reverse_query, reverse_params = await self.add_user_conversation_filters(reverse_query, reverse_params)
            
            # Execute queries
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, *params)
                reverse_rows = await conn.fetch(reverse_query, *reverse_params)
                
                # Combine results
                all_relationships = [dict(row) for row in rows]
                all_relationships.extend([dict(row) for row in reverse_rows])
                
                return all_relationships
                    
        except Exception as e:
            logger.error(f"Error getting faction relationships: {e}")
            return []


class LoreKnowledgeAccess(BaseDataAccess):
    """Data access for lore knowledge and discovery."""
    
    async def add_lore_knowledge(self, entity_type: str, entity_id: int, 
                              lore_type: str, lore_id: int, 
                              knowledge_level: int, is_secret: bool = False) -> bool:
        """
        Record that an entity has knowledge of a piece of lore.
        
        Args:
            entity_type: Type of entity (e.g., "npc", "player")
            entity_id: ID of the entity
            lore_type: Type of lore (e.g., "Factions", "WorldLore")
            lore_id: ID of the lore
            knowledge_level: Level of knowledge (1-10)
            is_secret: Whether this knowledge is secret
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                INSERT INTO LoreKnowledge (
                    user_id, conversation_id, entity_type, entity_id,
                    lore_type, lore_id, knowledge_level, is_secret,
                    discovered_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (user_id, conversation_id, entity_type, entity_id, lore_type, lore_id)
                DO UPDATE SET
                    knowledge_level = GREATEST(LoreKnowledge.knowledge_level, EXCLUDED.knowledge_level),
                    is_secret = EXCLUDED.is_secret,
                    last_accessed = NOW()
            """
            
            async with get_db_connection_context() as conn:
                await conn.execute(
                    query,
                    self.user_id,
                    self.conversation_id,
                    entity_type,
                    entity_id,
                    lore_type,
                    lore_id,
                    knowledge_level,
                    is_secret
                )
                
                return True
                    
        except Exception as e:
            logger.error(f"Error adding lore knowledge: {e}")
            return False
    
    async def get_entity_knowledge(self, entity_type: str, entity_id: int) -> List[Dict[str, Any]]:
        """
        Get all lore known by an entity.
        
        Args:
            entity_type: Type of entity (e.g., "npc", "player")
            entity_id: ID of the entity
            
        Returns:
            List of known lore items
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            query = """
                SELECT lk.*, lk.id as knowledge_id
                FROM LoreKnowledge lk
                WHERE lk.entity_type = $1 AND lk.entity_id = $2
            """
            params = [entity_type, entity_id]
            
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Execute query
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(query, *params)
                
                knowledge_items = []
                
                # Get details for each knowledge item
                for row in rows:
                    knowledge = dict(row)
                    lore_type = knowledge["lore_type"]
                    lore_id = knowledge["lore_id"]
                    
                    # Get the lore details based on type
                    if lore_type == "Factions":
                        lore_query = "SELECT * FROM Factions WHERE id = $1"
                    elif lore_type == "WorldLore":
                        lore_query = "SELECT * FROM WorldLore WHERE id = $1"
                    elif lore_type == "CulturalElements":
                        lore_query = "SELECT * FROM CulturalElements WHERE id = $1"
                    elif lore_type == "HistoricalEvents":
                        lore_query = "SELECT * FROM HistoricalEvents WHERE id = $1"
                    elif lore_type == "LocationLore":
                        lore_query = "SELECT * FROM LocationLore WHERE location_id = $1"
                    else:
                        # Unknown lore type
                        continue
                    
                    # Get lore details
                    lore_row = await conn.fetchrow(lore_query, lore_id)
                    
                    if lore_row:
                        lore_data = dict(lore_row)
                        
                        # Combine knowledge and lore data
                        combined = {
                            "knowledge_id": knowledge["knowledge_id"],
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "lore_type": lore_type,
                            "lore_id": lore_id,
                            "knowledge_level": knowledge["knowledge_level"],
                            "is_secret": knowledge["is_secret"]
                        }
                        
                        # Add lore details
                        for key, value in lore_data.items():
                            if key not in ["id", "user_id", "conversation_id"]:
                                combined[f"lore_{key}"] = value
                        
                        knowledge_items.append(combined)
                
                return knowledge_items
                    
        except Exception as e:
            logger.error(f"Error getting entity knowledge: {e}")
            return []
    
    async def get_lore_by_id(self, lore_type: str, lore_id: int) -> Dict[str, Any]:
        """
        Get lore details by type and ID.
        
        Args:
            lore_type: Type of lore (e.g., "Factions", "WorldLore")
            lore_id: ID of the lore
            
        Returns:
            Lore details
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get the lore details based on type
            if lore_type == "Factions":
                query = "SELECT * FROM Factions WHERE id = $1"
            elif lore_type == "WorldLore":
                query = "SELECT * FROM WorldLore WHERE id = $1"
            elif lore_type == "CulturalElements":
                query = "SELECT * FROM CulturalElements WHERE id = $1"
            elif lore_type == "HistoricalEvents":
                query = "SELECT * FROM HistoricalEvents WHERE id = $1"
            elif lore_type == "LocationLore":
                query = "SELECT * FROM LocationLore WHERE location_id = $1"
            else:
                # Unknown lore type
                return {}
            
            params = [lore_id]
            
            # Add user/conversation filters
            query, params = await self.add_user_conversation_filters(query, params)
            
            # Execute query
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    return {}
                    
                lore_data = dict(row)
                
                # Add lore type
                lore_data["lore_type"] = lore_type
                
                return lore_data
                    
        except Exception as e:
            logger.error(f"Error getting lore by ID: {e}")
            return {}
    
    async def get_relevant_lore(self, query: str, min_relevance: float = 0.6, 
                              limit: int = 5, lore_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get lore relevant to a search query.
        
        Args:
            query: Search query text
            min_relevance: Minimum relevance score (0-1)
            limit: Maximum number of results
            lore_types: Optional list of lore types to include
            
        Returns:
            List of relevant lore items
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Generate embedding for the query
            query_embedding = await generate_embedding(query)
            
            # Types to search
            search_types = lore_types or ["Factions", "WorldLore", "CulturalElements", "HistoricalEvents", "LocationLore"]
            
            all_results = []
            
            async with get_db_connection_context() as conn:
                # Search each lore type
                for lore_type in search_types:
                    if lore_type == "Factions":
                        table = "Factions"
                        columns = "id, name, type, description"
                    elif lore_type == "WorldLore":
                        table = "WorldLore"
                        columns = "id, name, category, description"
                    elif lore_type == "CulturalElements":
                        table = "CulturalElements"
                        columns = "id, name, element_type as type, description"
                    elif lore_type == "HistoricalEvents":
                        table = "HistoricalEvents"
                        columns = "id, name, date_description, description"
                    elif lore_type == "LocationLore":
                        table = "LocationLore l JOIN Locations loc ON l.location_id = loc.id"
                        columns = "l.location_id as id, loc.location_name as name, 'location' as type, loc.description"
                    else:
                        # Unknown lore type
                        continue
                    
                    # Build query
                    search_query = f"""
                        SELECT {columns}, embedding
                        FROM {table}
                        WHERE user_id = $1 AND conversation_id = $2
                    """
                    
                    # Execute query
                    rows = await conn.fetch(search_query, self.user_id, self.conversation_id)
                    
                    # Calculate relevance for each item
                    for row in rows:
                        item = dict(row)
                        
                        # Skip if no embedding - FIXED: proper check for numpy array
                        if "embedding" not in item or item["embedding"] is None:
                            continue
                        
                        # Convert embedding to list if it's not already
                        embedding = item["embedding"]
                        if hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        
                        # Skip if embedding is empty
                        if not embedding or len(embedding) == 0:
                            continue
                            
                        # Calculate similarity
                        similarity = await compute_similarity(query_embedding, embedding)
                        
                        # Only include if above threshold
                        if similarity >= min_relevance:
                            # Add lore type and similarity
                            item["lore_type"] = lore_type
                            item["relevance"] = similarity
                            
                            # Remove embedding to reduce response size
                            if "embedding" in item:
                                del item["embedding"]
                            
                            all_results.append(item)
            
            # Sort by relevance and limit results
            all_results.sort(key=lambda x: x["relevance"], reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Error getting relevant lore: {e}")
            return []
    
    async def generate_available_lore_for_context(self, query_text: str, entity_type: str, 
                                               entity_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get lore known by an entity that is relevant to the current context.
        
        Args:
            query_text: The context query
            entity_type: Type of entity (e.g., "npc", "player")
            entity_id: ID of the entity
            limit: Maximum number of results
            
        Returns:
            List of relevant known lore
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get all knowledge for the entity
            entity_knowledge = await self.get_entity_knowledge(entity_type, entity_id)
            
            if not entity_knowledge:
                return []
                
            # Generate embedding for the query
            query_embedding = await generate_embedding(query_text)
            
            # Calculate relevance for each knowledge item
            results_with_relevance = []
            
            for knowledge in entity_knowledge:
                # Get the full lore item
                lore_type = knowledge["lore_type"]
                lore_id = knowledge["lore_id"]
                
                lore_item = await self.get_lore_by_id(lore_type, lore_id)
                
                if not lore_item:
                    continue
                    
                # Skip if no embedding - FIXED: proper check for numpy array
                if "embedding" not in lore_item or lore_item["embedding"] is None:
                    # Try to generate an embedding from the description
                    if "description" in lore_item:
                        try:
                            lore_item["embedding"] = await generate_embedding(lore_item["description"])
                        except Exception:
                            continue
                    else:
                        continue
                
                # Convert embedding to list if it's not already
                embedding = lore_item["embedding"]
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                # Skip if embedding is empty
                if not embedding or len(embedding) == 0:
                    continue
                
                # Calculate similarity
                similarity = await compute_similarity(query_embedding, embedding)
                
                # Add to results
                lore_with_knowledge = {
                    **lore_item,
                    "knowledge_level": knowledge["knowledge_level"],
                    "is_secret": knowledge["is_secret"],
                    "relevance": similarity
                }
                
                # Remove embedding to reduce response size
                if "embedding" in lore_with_knowledge:
                    del lore_with_knowledge["embedding"]
                
                results_with_relevance.append(lore_with_knowledge)
            
            # Sort by relevance and limit results
            results_with_relevance.sort(key=lambda x: x["relevance"], reverse=True)
            return results_with_relevance[:limit]
            
        except Exception as e:
            logger.error(f"Error generating available lore for context: {e}")
            return []
