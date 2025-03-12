# lore/lore_manager.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncpg

from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity 
from utils.caching import LoreCache

# Initialize cache for lore items
LORE_CACHE = LoreCache(max_size=500, ttl=3600)  # 1 hour TTL

class LoreManager:
    """
    Core manager for lore system, handling storage, retrieval, and integration
    with other game systems.
    """
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """
        Initialize the lore manager.
        
        Args:
            user_id: Optional user ID for user-specific lore (for campaign variations)
            conversation_id: Optional conversation ID for conversation-specific lore
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def get_connection_pool(self) -> asyncpg.Pool:
        """Get a connection pool for database operations."""
        return await asyncpg.create_pool(dsn=get_db_connection())
    
    async def add_world_lore(self, name: str, category: str, description: str, 
                           significance: int = 5, tags: List[str] = None) -> int:
        """
        Add a piece of world lore to the database.
        
        Args:
            name: Name of the lore element
            category: Category (creation_myth, cosmology, etc.)
            description: Full description of the lore
            significance: Importance from 1-10
            tags: List of tags for categorization
            
        Returns:
            ID of the created lore
        """
        tags = tags or []
        
        # Generate embedding for semantic search
        embedding = await generate_embedding(f"{name} {category} {description}")
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                lore_id = await conn.fetchval("""
                    INSERT INTO WorldLore (name, category, description, significance, embedding, tags)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, name, category, description, significance, embedding, tags)
                
                # Clear the relevant cache
                LORE_CACHE.invalidate_pattern("world_lore")
                
                return lore_id
    
    async def add_faction(self, name: str, faction_type: str, description: str, 
                         values: List[str], goals: List[str], **kwargs) -> int:
        """
        Add a faction to the database.
        
        Args:
            name: Name of the faction
            faction_type: Type of faction (political, religious, etc.)
            description: Description of the faction
            values: Core values of the faction
            goals: Goals of the faction
            **kwargs: Additional faction attributes
            
        Returns:
            ID of the created faction
        """
        # Generate embedding for semantic search
        embedding = await generate_embedding(f"{name} {faction_type} {description} {' '.join(values)} {' '.join(goals)}")
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Prepare the values dictionary with required fields
                insert_values = {
                    'name': name,
                    'type': faction_type,
                    'description': description,
                    'values': values,
                    'goals': goals,
                    'embedding': embedding
                }
                
                # Add optional fields if provided
                optional_fields = [
                    'headquarters', 'founding_story', 'rivals', 'allies', 'territory',
                    'resources', 'hierarchy_type', 'secret_knowledge', 'public_reputation',
                    'color_scheme', 'symbol_description'
                ]
                
                for field in optional_fields:
                    if field in kwargs:
                        insert_values[field] = kwargs[field]
                
                # Construct column names and placeholders for dynamic query
                columns = ', '.join(insert_values.keys())
                placeholders = ', '.join(f'${i+1}' for i in range(len(insert_values)))
                
                # Execute the insertion query
                faction_id = await conn.fetchval(f"""
                    INSERT INTO Factions ({columns})
                    VALUES ({placeholders})
                    RETURNING id
                """, *insert_values.values())
                
                # Clear the relevant cache
                LORE_CACHE.invalidate_pattern("faction")
                
                return faction_id
    
    async def add_cultural_element(self, name: str, element_type: str, description: str,
                                 practiced_by: List[str], significance: int = 5, **kwargs) -> int:
        """
        Add a cultural element to the database.
        
        Args:
            name: Name of the cultural element
            element_type: Type of element (tradition, custom, taboo, etc.)
            description: Description of the element
            practiced_by: List of factions/regions that practice this
            significance: Importance from 1-10
            **kwargs: Additional attributes
            
        Returns:
            ID of the created cultural element
        """
        # Generate embedding for semantic search
        embedding = await generate_embedding(f"{name} {element_type} {description} {' '.join(practiced_by)}")
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Required fields
                insert_values = {
                    'name': name,
                    'type': element_type,
                    'description': description,
                    'practiced_by': practiced_by,
                    'significance': significance,
                    'embedding': embedding
                }
                
                # Optional fields
                optional_fields = ['historical_origin', 'related_elements']
                
                for field in optional_fields:
                    if field in kwargs:
                        insert_values[field] = kwargs[field]
                
                # Construct dynamic query
                columns = ', '.join(insert_values.keys())
                placeholders = ', '.join(f'${i+1}' for i in range(len(insert_values)))
                
                # Execute query
                element_id = await conn.fetchval(f"""
                    INSERT INTO CulturalElements ({columns})
                    VALUES ({placeholders})
                    RETURNING id
                """, *insert_values.values())
                
                # Clear the relevant cache
                LORE_CACHE.invalidate_pattern("cultural")
                
                return element_id
    
    async def add_historical_event(self, name: str, description: str,
                                 date_description: str, significance: int = 5, **kwargs) -> int:
        """
        Add a historical event to the database.
        
        Args:
            name: Name of the historical event
            description: Description of what happened
            date_description: When it happened (textual description)
            significance: Importance from 1-10
            **kwargs: Additional attributes
            
        Returns:
            ID of the created historical event
        """
        # Generate embedding for semantic search
        embedding = await generate_embedding(f"{name} {date_description} {description}")
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Required fields
                insert_values = {
                    'name': name,
                    'description': description,
                    'date_description': date_description,
                    'significance': significance,
                    'embedding': embedding
                }
                
                # Optional fields
                optional_fields = [
                    'participating_factions', 'affected_locations', 'consequences',
                    'historical_figures', 'commemorated_by'
                ]
                
                for field in optional_fields:
                    if field in kwargs:
                        insert_values[field] = kwargs[field]
                
                # Construct dynamic query
                columns = ', '.join(insert_values.keys())
                placeholders = ', '.join(f'${i+1}' for i in range(len(insert_values)))
                
                # Execute query
                event_id = await conn.fetchval(f"""
                    INSERT INTO HistoricalEvents ({columns})
                    VALUES ({placeholders})
                    RETURNING id
                """, *insert_values.values())
                
                # Clear the relevant cache
                LORE_CACHE.invalidate_pattern("historical")
                
                return event_id
    
    async def add_geographic_region(self, name: str, description: str, **kwargs) -> int:
        """
        Add a geographic region to the database.
        
        Args:
            name: Name of the region
            description: Description of the region
            **kwargs: Additional attributes
            
        Returns:
            ID of the created region
        """
        # Generate embedding for semantic search
        embedding = await generate_embedding(f"{name} {description}")
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Required fields
                insert_values = {
                    'name': name,
                    'description': description,
                    'embedding': embedding
                }
                
                # Optional fields
                optional_fields = [
                    'climate', 'notable_features', 'main_settlements', 'cultural_traits',
                    'governing_faction', 'resources', 'strategic_importance', 'population_description'
                ]
                
                for field in optional_fields:
                    if field in kwargs:
                        insert_values[field] = kwargs[field]
                
                # Construct dynamic query
                columns = ', '.join(insert_values.keys())
                placeholders = ', '.join(f'${i+1}' for i in range(len(insert_values)))
                
                # Execute query
                region_id = await conn.fetchval(f"""
                    INSERT INTO GeographicRegions ({columns})
                    VALUES ({placeholders})
                    RETURNING id
                """, *insert_values.values())
                
                # Clear the relevant cache
                LORE_CACHE.invalidate_pattern("geographic")
                
                return region_id
    
    async def add_location_lore(self, location_id: int, **kwargs) -> bool:
        """
        Add lore to an existing location.
        
        Args:
            location_id: ID of the location
            **kwargs: Lore attributes to add
            
        Returns:
            Success status
        """
        if not kwargs:
            return False
        
        # Generate combined description for embedding
        combined_text = " ".join([str(v) for v in kwargs.values() if v])
        embedding = await generate_embedding(combined_text)
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if location exists
                location_exists = await conn.fetchval("""
                    SELECT 1 FROM Locations WHERE id = $1
                """, location_id)
                
                if not location_exists:
                    return False
                
                # Check if location lore exists
                lore_exists = await conn.fetchval("""
                    SELECT 1 FROM LocationLore WHERE location_id = $1
                """, location_id)
                
                # Add fields to insert_values
                insert_values = {'location_id': location_id, 'embedding': embedding}
                
                for field, value in kwargs.items():
                    if field in [
                        'founding_story', 'hidden_secrets', 'supernatural_phenomena',
                        'local_legends', 'historical_significance', 'associated_factions'
                    ]:
                        insert_values[field] = value
                
                if lore_exists:
                    # Update existing lore
                    set_clause = ', '.join(f"{k} = ${i+2}" for i, k in enumerate(insert_values.keys()) if k != 'location_id')
                    
                    await conn.execute(f"""
                        UPDATE LocationLore
                        SET {set_clause}
                        WHERE location_id = $1
                    """, location_id, *[insert_values[k] for k in insert_values.keys() if k != 'location_id'])
                else:
                    # Insert new lore
                    columns = ', '.join(insert_values.keys())
                    placeholders = ', '.join(f'${i+1}' for i in range(len(insert_values)))
                    
                    await conn.execute(f"""
                        INSERT INTO LocationLore ({columns})
                        VALUES ({placeholders})
                    """, *insert_values.values())
                
                # Clear the relevant cache
                LORE_CACHE.invalidate_pattern(f"location_{location_id}")
                
                return True
    
    async def add_lore_connection(self, source_type: str, source_id: int,
                                target_type: str, target_id: int,
                                connection_type: str, description: str = None,
                                strength: int = 5) -> int:
        """
        Add a connection between two lore elements.
        
        Args:
            source_type: Type of source element (WorldLore, Factions, etc.)
            source_id: ID of source element
            target_type: Type of target element
            target_id: ID of target element
            connection_type: Type of connection (influences, conflicts_with, etc.)
            description: Description of the connection
            strength: Strength of the connection (1-10)
            
        Returns:
            ID of the created connection
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    connection_id = await conn.fetchval("""
                        INSERT INTO LoreConnections (
                            source_type, source_id, target_type, target_id,
                            connection_type, description, strength
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (source_type, source_id, target_type, target_id, connection_type)
                        DO UPDATE SET description = $6, strength = $7
                        RETURNING id
                    """, source_type, source_id, target_type, target_id,
                        connection_type, description, strength)
                    
                    # Clear relevant caches
                    LORE_CACHE.invalidate_pattern(f"{source_type}_{source_id}")
                    LORE_CACHE.invalidate_pattern(f"{target_type}_{target_id}")
                    
                    return connection_id
                except Exception as e:
                    logging.error(f"Error adding lore connection: {e}")
                    return None
    
    async def add_lore_knowledge(self, entity_type: str, entity_id: int,
                               lore_type: str, lore_id: int,
                               knowledge_level: int = 5, is_secret: bool = False) -> int:
        """
        Record that an entity knows about a piece of lore.
        
        Args:
            entity_type: Type of entity (npc, player, faction)
            entity_id: ID of entity
            lore_type: Type of lore (WorldLore, Factions, etc.)
            lore_id: ID of lore
            knowledge_level: How much they know (1-10)
            is_secret: Whether this is secret knowledge
            
        Returns:
            ID of the created knowledge entry
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    knowledge_id = await conn.fetchval("""
                        INSERT INTO LoreKnowledge (
                            entity_type, entity_id, lore_type, lore_id,
                            knowledge_level, is_secret
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (entity_type, entity_id, lore_type, lore_id)
                        DO UPDATE SET knowledge_level = $5, is_secret = $6
                        RETURNING id
                    """, entity_type, entity_id, lore_type, lore_id,
                        knowledge_level, is_secret)
                    
                    # Clear relevant caches
                    LORE_CACHE.invalidate_pattern(f"knowledge_{entity_type}_{entity_id}")
                    
                    return knowledge_id
                except Exception as e:
                    logging.error(f"Error adding lore knowledge: {e}")
                    return None
    
    async def add_lore_discovery_opportunity(self, lore_type: str, lore_id: int,
                                         discovery_method: str, difficulty: int = 5,
                                         **kwargs) -> int:
        """
        Add an opportunity for players to discover lore.
        
        Args:
            lore_type: Type of lore (WorldLore, Factions, etc.)
            lore_id: ID of lore
            discovery_method: How it can be discovered
            difficulty: How difficult to discover (1-10)
            **kwargs: Additional attributes including location_id, event_id, or npc_id
            
        Returns:
            ID of the created discovery opportunity
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Required fields
                insert_values = {
                    'lore_type': lore_type,
                    'lore_id': lore_id,
                    'discovery_method': discovery_method,
                    'difficulty': difficulty
                }
                
                # Must include exactly one of location_id, event_id, or npc_id
                if 'location_id' in kwargs:
                    insert_values['location_id'] = kwargs['location_id']
                elif 'event_id' in kwargs:
                    insert_values['event_id'] = kwargs['event_id']
                elif 'npc_id' in kwargs:
                    insert_values['npc_id'] = kwargs['npc_id']
                else:
                    raise ValueError("Must provide exactly one of location_id, event_id, or npc_id")
                
                # Optional prerequisites
                if 'prerequisites' in kwargs:
                    insert_values['prerequisites'] = json.dumps(kwargs['prerequisites'])
                
                # Construct dynamic query
                columns = ', '.join(insert_values.keys())
                placeholders = ', '.join(f'${i+1}' for i in range(len(insert_values)))
                
                # Execute query
                opportunity_id = await conn.fetchval(f"""
                    INSERT INTO LoreDiscoveryOpportunities ({columns})
                    VALUES ({placeholders})
                    RETURNING id
                """, *insert_values.values())
                
                return opportunity_id
    
    async def get_lore_by_id(self, lore_type: str, lore_id: int) -> Dict[str, Any]:
        """
        Get a specific lore item by its ID and type.
        
        Args:
            lore_type: Type of lore (WorldLore, Factions, etc.)
            lore_id: ID of the lore
            
        Returns:
            Dictionary with lore data
        """
        # Check cache first
        cache_key = f"{lore_type}_{lore_id}"
        cached_lore = LORE_CACHE.get(cache_key)
        if cached_lore:
            return cached_lore
        
        # Map lore_type to table name
        table_map = {
            'WorldLore': 'WorldLore',
            'Factions': 'Factions',
            'CulturalElements': 'CulturalElements',
            'HistoricalEvents': 'HistoricalEvents',
            'GeographicRegions': 'GeographicRegions',
            'LocationLore': 'LocationLore'
        }
        
        table_name = table_map.get(lore_type)
        if not table_name:
            return None
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get the lore item
                query = f"SELECT * FROM {table_name} WHERE id = $1"
                if lore_type == 'LocationLore':
                    query = f"SELECT * FROM {table_name} WHERE location_id = $1"
                
                row = await conn.fetchrow(query, lore_id)
                
                if not row:
                    return None
                
                # Convert to dictionary
                lore_data = dict(row)
                
                # Remove embedding as it's not JSON serializable
                if 'embedding' in lore_data:
                    del lore_data['embedding']
                
                # Get connections to/from this lore item
                connections = await conn.fetch("""
                    SELECT * FROM LoreConnections
                    WHERE (source_type = $1 AND source_id = $2)
                       OR (target_type = $1 AND target_id = $2)
                """, lore_type, lore_id)
                
                lore_data['connections'] = [dict(conn) for conn in connections]
                
                # Cache the result
                LORE_CACHE.set(cache_key, lore_data)
                
                return lore_data
    
    async def get_entity_lore_knowledge(self, entity_type: str, entity_id: int) -> List[Dict[str, Any]]:
        """
        Get all lore known by a specific entity.
        
        Args:
            entity_type: Type of entity (npc, player, faction)
            entity_id: ID of entity
            
        Returns:
            List of lore items with knowledge levels
        """
        # Check cache first
        cache_key = f"knowledge_{entity_type}_{entity_id}"
        cached_knowledge = LORE_CACHE.get(cache_key)
        if cached_knowledge:
            return cached_knowledge
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get all knowledge entries for this entity
                rows = await conn.fetch("""
                    SELECT lk.lore_type, lk.lore_id, lk.knowledge_level, lk.is_secret
                    FROM LoreKnowledge lk
                    WHERE entity_type = $1 AND entity_id = $2
                """, entity_type, entity_id)
                
                if not rows:
                    return []
                
                # Create list to hold complete lore items
                knowledge_items = []
                
                # For each knowledge entry, get the referenced lore
                for row in rows:
                    lore_item = await self.get_lore_by_id(row['lore_type'], row['lore_id'])
                    
                    if lore_item:
                        # Add knowledge information
                        lore_item['knowledge_level'] = row['knowledge_level']
                        lore_item['is_secret'] = row['is_secret']
                        knowledge_items.append(lore_item)
                
                # Cache the result
                LORE_CACHE.set(cache_key, knowledge_items)
                
                return knowledge_items
    
    async def get_relevant_lore(self, query_text: str, lore_types: List[str] = None,
                              min_relevance: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get lore relevant to a given query using vector similarity search.
        
        Args:
            query_text: Text to search for
            lore_types: List of lore types to search in (or None for all)
            min_relevance: Minimum relevance score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of relevant lore items with relevance scores
        """
        # Generate embedding for the query
        query_embedding = await generate_embedding(query_text)
        
        # Define which tables to search
        all_tables = ['WorldLore', 'Factions', 'CulturalElements', 
                      'HistoricalEvents', 'GeographicRegions', 'LocationLore']
        tables_to_search = lore_types or all_tables
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                all_results = []
                
                # Search each table for relevant items
                for table in tables_to_search:
                    id_field = 'id'
                    if table == 'LocationLore':
                        id_field = 'location_id'
                    
                    # Perform vector similarity search
                    rows = await conn.fetch(f"""
                        SELECT {id_field} as id, '{table}' as lore_type, *,
                               1 - (embedding <=> $1) as relevance
                        FROM {table}
                        WHERE 1 - (embedding <=> $1) > $2
                        ORDER BY relevance DESC
                        LIMIT $3
                    """, query_embedding, min_relevance, limit)
                    
                    for row in rows:
                        result = dict(row)
                        # Remove embedding
                        if 'embedding' in result:
                            del result['embedding']
                        all_results.append(result)
                
                # Sort all results by relevance
                all_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
                
                # Return top results
                return all_results[:limit]
    
    async def discover_lore(self, lore_type: str, lore_id: int, entity_type: str, entity_id: int,
                          knowledge_level: int = 5) -> bool:
        """
        Record that an entity has discovered a piece of lore.
        
        Args:
            lore_type: Type of lore (WorldLore, Factions, etc.)
            lore_id: ID of lore
            entity_type: Type of entity (npc, player, faction)
            entity_id: ID of entity
            knowledge_level: How much they know (1-10)
            
        Returns:
            Success status
        """
        try:
            # Add to LoreKnowledge
            await self.add_lore_knowledge(
                entity_type, entity_id,
                lore_type, lore_id,
                knowledge_level
            )
            
            # If there's a discovery opportunity, mark it as discovered
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE LoreDiscoveryOpportunities
                        SET discovered = TRUE
                        WHERE lore_type = $1 AND lore_id = $2
                    """, lore_type, lore_id)
            
            return True
        except Exception as e:
            logging.error(f"Error recording lore discovery: {e}")
            return False
    
    async def get_faction_influences_region(self, faction_id: int, region_id: int) -> Dict[str, Any]:
        """
        Get details about how a faction influences a region.
        
        Args:
            faction_id: ID of the faction
            region_id: ID of the region
            
        Returns:
            Connection details or None
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Look for a connection of type 'influences'
                row = await conn.fetchrow("""
                    SELECT * FROM LoreConnections
                    WHERE source_type = 'Factions' AND source_id = $1
                      AND target_type = 'GeographicRegions' AND target_id = $2
                      AND connection_type = 'influences'
                """, faction_id, region_id)
                
                if not row:
                    return None
                
                return dict(row)
    
    async def get_faction_relationships(self, faction_id: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all relationships this faction has with other factions.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            Dictionary of relationships by type
        """
        # Check cache first
        cache_key = f"faction_relationships_{faction_id}"
        cached_relationships = LORE_CACHE.get(cache_key)
        if cached_relationships:
            return cached_relationships
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get all connections to/from this faction with other factions
                rows = await conn.fetch("""
                    SELECT * FROM LoreConnections
                    WHERE (source_type = 'Factions' AND source_id = $1 AND target_type = 'Factions')
                       OR (target_type = 'Factions' AND target_id = $1 AND source_type = 'Factions')
                """, faction_id)
                
                # Organize by connection type
                relationships = {}
                
                for row in rows:
                    row_dict = dict(row)
                    conn_type = row_dict['connection_type']
                    
                    if conn_type not in relationships:
                        relationships[conn_type] = []
                    
                    # Determine which faction is the other one
                    other_faction_id = None
                    if row_dict['source_type'] == 'Factions' and row_dict['source_id'] == faction_id:
                        other_faction_id = row_dict['target_id']
                    else:
                        other_faction_id = row_dict['source_id']
                    
                    # Get the other faction's name
                    other_faction = await conn.fetchrow("""
                        SELECT id, name, type FROM Factions WHERE id = $1
                    """, other_faction_id)
                    
                    if other_faction:
                        # Add relationship details
                        relationship = {
                            'faction_id': other_faction['id'],
                            'faction_name': other_faction['name'],
                            'faction_type': other_faction['type'],
                            'strength': row_dict['strength'],
                            'description': row_dict['description']
                        }
                        
                        relationships[conn_type].append(relationship)
                
                # Cache the result
                LORE_CACHE.set(cache_key, relationships)
                
                return relationships
    
    async def get_world_lore_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all world lore of a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of lore items
        """
        # Check cache first
        cache_key = f"world_lore_category_{category}"
        cached_lore = LORE_CACHE.get(cache_key)
        if cached_lore:
            return cached_lore
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM WorldLore
                    WHERE category = $1
                    ORDER BY significance DESC
                """, category)
                
                lore_items = []
                for row in rows:
                    item = dict(row)
                    if 'embedding' in item:
                        del item['embedding']
                    lore_items.append(item)
                
                # Cache the result
                LORE_CACHE.set(cache_key, lore_items)
                
                return lore_items
    
    async def update_npc_lore_knowledge(self, npc_id: int, cultural_background: str = None,
                                      faction_memberships: List[str] = None) -> bool:
        """
        Update an NPC's lore knowledge fields.
        
        Args:
            npc_id: ID of the NPC
            cultural_background: Cultural background (affects knowledge)
            faction_memberships: List of faction names the NPC belongs to
            
        Returns:
            Success status
        """
        updates = {}
        
        if cultural_background is not None:
            updates['cultural_background'] = cultural_background
        
        if faction_memberships is not None:
            updates['faction_memberships'] = faction_memberships
        
        if not updates:
            return False
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Construct set clause
                    set_clauses = []
                    values = []
                    
                    for i, (key, value) in enumerate(updates.items()):
                        set_clauses.append(f"{key} = ${i+2}")
                        values.append(value)
                    
                    set_clause = ', '.join(set_clauses)
                    
                    # Update NPC
                    await conn.execute(f"""
                        UPDATE NPCStats
                        SET {set_clause}
                        WHERE npc_id = $1
                    """, npc_id, *values)
                    
                    return True
                except Exception as e:
                    logging.error(f"Error updating NPC lore knowledge: {e}")
                    return False
    
    async def get_location_with_lore(self, location_id: int) -> Dict[str, Any]:
        """
        Get a location with its associated lore.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Location data with lore
        """
        # Check cache first
        cache_key = f"location_with_lore_{location_id}"
        cached_location = LORE_CACHE.get(cache_key)
        if cached_location:
            return cached_location
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get basic location info
                location = await conn.fetchrow("""
                    SELECT * FROM Locations
                    WHERE id = $1
                """, location_id)
                
                if not location:
                    return None
                
                # Get location lore
                lore = await conn.fetchrow("""
                    SELECT * FROM LocationLore
                    WHERE location_id = $1
                """, location_id)
                
                # Get connections
                connections = await conn.fetch("""
                    SELECT * FROM LoreConnections
                    WHERE (source_type = 'LocationLore' AND source_id = $1)
                       OR (target_type = 'LocationLore' AND target_id = $1)
                """, location_id)
                
                # Combine data
                result = dict(location)
                
                if lore:
                    lore_dict = dict(lore)
                    if 'embedding' in lore_dict:
                        del lore_dict['embedding']
                    
                    # Merge in lore fields
                    for key, value in lore_dict.items():
                        if key != 'location_id' and key != 'id':
                            result[f"lore_{key}"] = value
                
                if connections:
                    result['connections'] = [dict(conn) for conn in connections]
                
                # Cache the result
                LORE_CACHE.set(cache_key, result)
                
                return result
    
    async def generate_available_lore_for_context(self, context_text: str, entity_type: str = None,
                                                entity_id: int = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Generate available lore for a given context, optionally filtered by entity knowledge.
        
        Args:
            context_text: Context to find relevant lore for
            entity_type: Optional entity type to filter by knowledge
            entity_id: Optional entity ID to filter by knowledge
            limit: Maximum number of results
            
        Returns:
            List of relevant lore items
        """
        # Get relevant lore based on context
        relevant_lore = await self.get_relevant_lore(
            context_text,
            min_relevance=0.65,
            limit=limit * 2  # Get more than needed to allow for knowledge filtering
        )
        
        # If no entity filtering, return the top results
        if entity_type is None or entity_id is None:
            return relevant_lore[:limit]
        
        # Get entity's knowledge
        entity_knowledge = await self.get_entity_lore_knowledge(entity_type, entity_id)
        known_lore_ids = {(k['lore_type'], k['id']): k['knowledge_level'] for k in entity_knowledge}
        
        # Filter and organize lore
        filtered_lore = []
        
        for lore in relevant_lore:
            lore_type = lore['lore_type']
            lore_id = lore['id']
            
            # Check if entity knows this lore
            if (lore_type, lore_id) in known_lore_ids:
                # Entity knows this lore - adjust description based on knowledge level
                knowledge_level = known_lore_ids[(lore_type, lore_id)]
                
                # For low knowledge, provide less detailed information
                if knowledge_level < 4:
                    # Keep basic info but truncate detailed description
                    if 'description' in lore and len(lore['description']) > 100:
                        lore['description'] = lore['description'][:100] + "... [more knowledge needed]"
                
                lore['entity_knowledge_level'] = knowledge_level
                filtered_lore.append(lore)
            else:
                # Entity doesn't know this lore yet, but it's relevant
                # Include a hint but not the full details
                if 'description' in lore and len(lore['description']) > 50:
                    lore['description'] = lore['description'][:50] + "... [unknown lore]"
                
                lore['entity_knowledge_level'] = 0
                filtered_lore.append(lore)
        
        # Sort by knowledge level (known first) then relevance
        filtered_lore.sort(key=lambda x: (x.get('entity_knowledge_level', 0), x.get('relevance', 0)), reverse=True)
        
        return filtered_lore[:limit]
