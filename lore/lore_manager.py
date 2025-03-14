# lore/lore_manager.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncpg

from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity 
from utils.caching import LoreCache

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

# Initialize cache for lore items
LORE_CACHE = LoreCache(max_size=500, ttl=3600)  # 1 hour TTL

class LoreManager:
    """
    Core manager for lore system, handling storage, retrieval, and integration
    with other game systems with full Nyx governance integration.
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
        self.governor = None
        self.directive_handler = None
        
    async def initialize_governance(self):
        """Initialize governance connections and directive handler"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
            
        # Initialize directive handler if needed
        if not self.directive_handler:
            self.directive_handler = DirectiveHandler(
                self.user_id, 
                self.conversation_id, 
                AgentType.NARRATIVE_CRAFTER,
                "lore_manager"
            )
            
            # Register handlers for different directive types
            self.directive_handler.register_handler(DirectiveType.ACTION, self._handle_action_directive)
            self.directive_handler.register_handler(DirectiveType.PROHIBITION, self._handle_prohibition_directive)
            
            # Start background processing of directives
            await self.directive_handler.start_background_processing()
    
    async def _handle_action_directive(self, directive):
        """Handle action directives from Nyx"""
        instruction = directive.get("instruction", "")
        
        if "purge_lore" in instruction.lower():
            category = directive.get("category", "all")
            return await self._purge_lore_category(category)
            
        elif "update_lore_significance" in instruction.lower():
            lore_type = directive.get("lore_type")
            lore_id = directive.get("lore_id")
            new_significance = directive.get("significance")
            
            if lore_type and lore_id and new_significance is not None:
                return await self._update_lore_significance(lore_type, lore_id, new_significance)
        
        return {"status": "unknown_directive", "instruction": instruction}
    
    async def _handle_prohibition_directive(self, directive):
        """Handle prohibition directives from Nyx"""
        # Mark certain lore operations as prohibited
        prohibited = directive.get("prohibited_actions", [])
        
        # Store these in context for later checking
        self.prohibited_lore_actions = prohibited
        
        return {"status": "prohibition_registered", "prohibited": prohibited}
    
    async def _purge_lore_category(self, category: str) -> Dict[str, Any]:
        """Purge a category of lore from the database"""
        if category == "all":
            # Dangerous operation, should have governance permission
            tables = ["WorldLore", "Factions", "CulturalElements", "HistoricalEvents", 
                     "GeographicRegions", "LocationLore"]
            
            purged_counts = {}
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    for table in tables:
                        result = await conn.execute(f"DELETE FROM {table}")
                        purged_counts[table] = result
                        
            # Invalidate all cache
            LORE_CACHE.clear()
            
            return {"status": "purged_all", "counts": purged_counts}
        else:
            # Map category to table
            table_map = {
                "world": "WorldLore",
                "factions": "Factions",
                "cultural": "CulturalElements",
                "historical": "HistoricalEvents",
                "geographic": "GeographicRegions",
                "locations": "LocationLore"
            }
            
            if category not in table_map:
                return {"status": "error", "message": f"Unknown category: {category}"}
            
            table = table_map[category]
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    result = await conn.execute(f"DELETE FROM {table}")
            
            # Invalidate specific cache
            LORE_CACHE.invalidate_pattern(category)
            
            return {"status": f"purged_{category}", "count": result}
    
    async def _update_lore_significance(self, lore_type: str, lore_id: int, significance: int) -> Dict[str, Any]:
        """Update significance of a lore item"""
        # Map lore_type to table
        table_map = {
            'WorldLore': 'WorldLore',
            'Factions': 'Factions',
            'CulturalElements': 'CulturalElements',
            'HistoricalEvents': 'HistoricalEvents',
            'GeographicRegions': 'GeographicRegions',
            'LocationLore': 'LocationLore'
        }
        
        table = table_map.get(lore_type)
        if not table:
            return {"status": "error", "message": f"Unknown lore type: {lore_type}"}
        
        # Ensure significance is within bounds
        significance = max(1, min(10, significance))
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                await conn.execute(f"""
                    UPDATE {table}
                    SET significance = $1
                    WHERE id = $2
                """, significance, lore_id)
        
        # Invalidate cache
        LORE_CACHE.invalidate_pattern(f"{lore_type}_{lore_id}")
        
        return {
            "status": "updated",
            "lore_type": lore_type,
            "lore_id": lore_id,
            "new_significance": significance
        }
    
    async def get_connection_pool(self) -> asyncpg.Pool:
        """Get a connection pool for database operations."""
        return await asyncpg.create_pool(dsn=get_db_connection())
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_world_lore",
        action_description="Adding world lore: {name}",
        id_from_context=lambda ctx: "lore_manager"
    )
    async def add_world_lore(self, name: str, category: str, description: str, 
                           significance: int = 5, tags: List[str] = None) -> int:
        """
        Add a piece of world lore to the database with governance oversight.
        
        Args:
            name: Name of the lore element
            category: Category (creation_myth, cosmology, etc.)
            description: Full description of the lore
            significance: Importance from 1-10
            tags: List of tags for categorization
            
        Returns:
            ID of the created lore
        """
        # Check for prohibited actions through directive handler
        if hasattr(self, 'prohibited_lore_actions') and "add_world_lore" in self.prohibited_lore_actions:
            logging.warning("Adding world lore prohibited by directive")
            return -1
            
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
                
                # Log the creation to memory system through governance
                if self.governor:
                    memory_system = await self.governor.get_memory_system()
                    await memory_system.add_memory(
                        memory_text=f"Added world lore: {name} - Category: {category}",
                        memory_type="lore",
                        memory_scope="game",
                        significance=min(significance + 1, 10),  # Slightly more significant as a creation event
                        tags=["lore_creation", "world_lore", category] + tags
                    )
                
                return lore_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_faction",
        action_description="Adding faction: {name}",
        id_from_context=lambda ctx: "lore_manager"
    )
    async def add_faction(self, name: str, faction_type: str, description: str, 
                         values: List[str], goals: List[str], **kwargs) -> int:
        """
        Add a faction to the database with governance oversight.
        
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
        # Check for prohibited actions through directive handler
        if hasattr(self, 'prohibited_lore_actions') and "add_faction" in self.prohibited_lore_actions:
            logging.warning("Adding faction prohibited by directive")
            return -1
            
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
                
                # Log the creation to memory system through governance
                if self.governor:
                    memory_system = await self.governor.get_memory_system()
                    await memory_system.add_memory(
                        memory_text=f"Added faction: {name} ({faction_type})",
                        memory_type="lore",
                        memory_scope="game",
                        significance=6,
                        tags=["lore_creation", "faction", faction_type]
                    )
                
                return faction_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_cultural_element",
        action_description="Adding cultural element: {name}",
        id_from_context=lambda ctx: "lore_manager"
    )
    async def add_cultural_element(self, name: str, element_type: str, description: str,
                                 practiced_by: List[str], significance: int = 5, **kwargs) -> int:
        """
        Add a cultural element to the database with governance oversight.
        
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
        # Check for prohibited actions through directive handler
        if hasattr(self, 'prohibited_lore_actions') and "add_cultural_element" in self.prohibited_lore_actions:
            logging.warning("Adding cultural element prohibited by directive")
            return -1
            
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
                
                # Log the creation to memory system through governance
                if self.governor:
                    memory_system = await self.governor.get_memory_system()
                    await memory_system.add_memory(
                        memory_text=f"Added cultural element: {name} ({element_type})",
                        memory_type="lore",
                        memory_scope="game",
                        significance=significance,
                        tags=["lore_creation", "cultural", element_type]
                    )
                
                return element_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_historical_event",
        action_description="Adding historical event: {name}",
        id_from_context=lambda ctx: "lore_manager"
    )
    async def add_historical_event(self, name: str, description: str,
                                 date_description: str, significance: int = 5, **kwargs) -> int:
        """
        Add a historical event to the database with governance oversight.
        
        Args:
            name: Name of the historical event
            description: Description of what happened
            date_description: When it happened (textual description)
            significance: Importance from 1-10
            **kwargs: Additional attributes
            
        Returns:
            ID of the created historical event
        """
        # Check for prohibited actions through directive handler
        if hasattr(self, 'prohibited_lore_actions') and "add_historical_event" in self.prohibited_lore_actions:
            logging.warning("Adding historical event prohibited by directive")
            return -1
            
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
                
                # Log the creation to memory system through governance
                if self.governor:
                    memory_system = await self.governor.get_memory_system()
                    await memory_system.add_memory(
                        memory_text=f"Added historical event: {name} ({date_description})",
                        memory_type="lore",
                        memory_scope="game",
                        significance=significance,
                        tags=["lore_creation", "historical_event", "history"]
                    )
                
                return event_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_relevant_lore",
        action_description="Retrieving relevant lore for: {query_text}",
        id_from_context=lambda ctx: "lore_manager"
    )
    async def get_relevant_lore(self, query_text: str, lore_types: List[str] = None,
                              min_relevance: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get lore relevant to a given query using vector similarity search with governance oversight.
        
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
        
        # Check for prohibited actions through directive handler
        if hasattr(self, 'prohibited_lore_actions'):
            for table in list(tables_to_search):
                # Check if this specific table is prohibited
                if f"search_{table.lower()}" in self.prohibited_lore_actions:
                    tables_to_search.remove(table)
                    logging.warning(f"Search in {table} prohibited by directive")
            
            # Check if all search is prohibited
            if "search_all" in self.prohibited_lore_actions:
                logging.warning("All lore search prohibited by directive")
                return []
        
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="discover_lore",
        action_description="Recording lore discovery for {entity_type} {entity_id}",
        id_from_context=lambda ctx: "lore_manager"
    )
    async def discover_lore(self, lore_type: str, lore_id: int, entity_type: str, entity_id: int,
                          knowledge_level: int = 5) -> bool:
        """
        Record that an entity has discovered a piece of lore with governance oversight.
        
        Args:
            lore_type: Type of lore (WorldLore, Factions, etc.)
            lore_id: ID of lore
            entity_type: Type of entity (npc, player, faction)
            entity_id: ID of entity
            knowledge_level: How much they know (1-10)
            
        Returns:
            Success status
        """
        # Check for prohibited actions through directive handler
        if hasattr(self, 'prohibited_lore_actions') and "discover_lore" in self.prohibited_lore_actions:
            logging.warning("Lore discovery prohibited by directive")
            return False
            
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
            
            # Add memory of the discovery through governance
            if self.governor:
                # Get entity name
                entity_name = await self._get_entity_name(entity_type, entity_id)
                
                # Get lore name
                lore_data = await self.get_lore_by_id(lore_type, lore_id)
                lore_name = lore_data.get("name", f"Unknown {lore_type}")
                
                # Add memory
                memory_system = await self.governor.get_memory_system()
                await memory_system.add_memory(
                    memory_text=f"{entity_name} discovered lore: {lore_name}",
                    memory_type="observation",
                    memory_scope="game",
                    significance=knowledge_level,
                    tags=["lore_discovery", lore_type.lower(), entity_type],
                    metadata={
                        "lore_type": lore_type,
                        "lore_id": lore_id,
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "knowledge_level": knowledge_level
                    }
                )
            
            return True
        except Exception as e:
            logging.error(f"Error recording lore discovery: {e}")
            return False
    
    async def _get_entity_name(self, entity_type: str, entity_id: int) -> str:
        """Get the name of an entity for memory creation"""
        if entity_type == "npc":
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, entity_id, self.user_id, self.conversation_id)
                    
                    if row:
                        return row["npc_name"]
                    return f"NPC {entity_id}"
        elif entity_type == "player":
            return "Player"
        elif entity_type == "faction":
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT name FROM Factions
                        WHERE id = $1
                    """, entity_id)
                    
                    if row:
                        return row["name"]
                    return f"Faction {entity_id}"
        else:
            return f"{entity_type.capitalize()} {entity_id}"

    @with_governance_permission(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_lore_knowledge",
        id_from_context=lambda ctx: "lore_manager"
    )
    async def add_lore_knowledge(self, entity_type: str, entity_id: int,
                               lore_type: str, lore_id: int,
                               knowledge_level: int = 5, is_secret: bool = False) -> int:
        """
        Record that an entity knows about a piece of lore with governance permission.
        
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
                                                    
    async def register_with_governance(self):
        """Register with Nyx governance system"""
        await self.initialize_governance()
        
        # Register this manager with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_manager",
            agent_instance=self
        )
        
        # Issue a general directive for lore maintenance
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_manager",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Maintain world lore consistency and handle knowledge appropriately.",
                "scope": "global"
            },
            priority=3,  # Medium-low priority
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"LoreManager registered with Nyx governance system for user {self.user_id}, conversation {self.conversation_id}")
