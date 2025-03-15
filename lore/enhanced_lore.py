# lore/enhanced_lore.py

"""
Enhanced Lore System for Dynamic Femdom RPG

This module extends the existing DynamicLoreGenerator with comprehensive
systems for generating, evolving, and maintaining deep, consistent lore
in a dynamically generated femdom setting.
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner, trace
from agents.run_context import RunContextWrapper

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

# Database and embedding functionality
from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity
from utils.caching import LoreCache

# Import existing modules
from lore.lore_manager import LoreManager
from lore.lore_tools import (
    generate_foundation_lore,
    generate_factions,
    generate_cultural_elements,
    generate_historical_events,
    generate_locations,
    generate_quest_hooks
)

class LoreCache:
    """Unified cache system for all lore types"""
    
    def __init__(self):
        self.caches = {}
        self.default_max_size = 1000
        self.default_ttl = 7200  # 2 hour TTL
    
    def get_cache(self, cache_type):
        """Get or create a cache for the specified type"""
        if cache_type not in self.caches:
            from utils.caching import LoreCache as BaseLoreCache
            self.caches[cache_type] = BaseLoreCache(max_size=self.default_max_size, ttl=self.default_ttl)
        return self.caches[cache_type]
    
    def get(self, cache_type, key):
        """Get an item from the specified cache"""
        cache = self.get_cache(cache_type)
        return cache.get(key)
    
    def set(self, cache_type, key, value):
        """Set an item in the specified cache"""
        cache = self.get_cache(cache_type)
        cache.set(key, value)
    
    def invalidate_pattern(self, cache_type, pattern):
        """Invalidate keys matching a pattern in the specified cache"""
        cache = self.get_cache(cache_type)
        cache.invalidate_pattern(pattern)

# Create a single cache manager instance
CACHE_MANAGER = LoreCache()

class BaseLoreManager:
    """
    Enhanced base class for all lore management systems.
    Provides common functionality for governance registration,
    database access, authorization, and table initialization.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the base lore manager.
        
        Args:
            user_id: ID of the user
            conversation_id: ID of the conversation
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.governor = None
        self.initialized = False
        self.cache_prefix = self.__class__.__name__.lower()
    
    async def initialize_governance(self):
        """
        Initialize Nyx governance connection.
        
        Returns:
            The governance instance
        """
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    async def ensure_initialized(self):
        """
        Ensure governance is initialized and any necessary tables exist.
        Should be overridden by derived classes to include table initialization.
        """
        if not self.initialized:
            await self.initialize_governance()
            await self.initialize_tables()
            self.initialized = True
    
    async def register_with_governance(
        self, 
        agent_type: AgentType, 
        agent_id: str, 
        directive_text: str, 
        scope: str = "world_building",
        priority: DirectivePriority = DirectivePriority.MEDIUM
    ):
        """
        Register with Nyx governance system.
        
        Args:
            agent_type: Type of agent (from AgentType enum)
            agent_id: Unique ID for this agent
            directive_text: Text describing the directive
            scope: Scope of the directive
            priority: Priority level for the directive
        """
        await self.ensure_initialized()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=agent_type,
            agent_id=agent_id,
            agent_instance=self
        )
        
        # Issue a directive
        await self.governor.issue_directive(
            agent_type=agent_type,
            agent_id=agent_id,
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": directive_text,
                "scope": scope
            },
            priority=priority,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"{agent_id} registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
    
    async def check_permission(
        self, 
        agent_type: AgentType, 
        agent_id: str, 
        action_type: str, 
        action_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if an action is permitted by governance system.
        
        Args:
            agent_type: Type of agent
            agent_id: ID of the agent
            action_type: Type of action
            action_details: Details of the action
            
        Returns:
            Dictionary with permission result
        """
        await self.ensure_initialized()
        
        permission = await self.governor.check_action_permission(
            agent_type=agent_type,
            agent_id=agent_id,
            action_type=action_type,
            action_details=action_details
        )
        
        return permission
    
    async def report_action(
        self, 
        agent_type: AgentType, 
        agent_id: str, 
        action: Dict[str, Any], 
        result: Dict[str, Any]
    ):
        """
        Report an action to the governance system.
        
        Args:
            agent_type: Type of agent
            agent_id: ID of the agent
            action: Action details
            result: Result of the action
        """
        await self.ensure_initialized()
        
        await self.governor.process_agent_action_report(
            agent_type=agent_type,
            agent_id=agent_id,
            action=action,
            result=result
        )
    
    async def get_connection_pool(self):
        """
        Get a database connection pool.
        
        Returns:
            Database connection pool
        """
        return await self.lore_manager.get_connection_pool()
    
    def create_run_context(self, ctx=None):
        """
        Create a run context for agents.
        
        Args:
            ctx: Optional existing context
            
        Returns:
            RunContextWrapper instance
        """
        if ctx:
            return RunContextWrapper(context=ctx.context)
        else:
            return RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
    
    async def initialize_tables(self):
        """
        Initialize database tables.
        Should be overridden by derived classes.
        """
        pass
    
    async def initialize_tables_for_class(self, table_definitions: Dict[str, str]):
        """
        Initialize tables using a dictionary of table definitions.
        
        Args:
            table_definitions: Dictionary mapping table names to CREATE TABLE statements
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for table_name, create_statement in table_definitions.items():
                    # Check if table exists
                    table_exists = await conn.fetchval(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = '{table_name.lower()}'
                        );
                    """)
                    
                    if not table_exists:
                        # Create the table
                        await conn.execute(create_statement)
                        logging.info(f"{table_name} table created")
    
    async def generate_and_store_embedding(
        self, 
        text: str, 
        conn, 
        table_name: str, 
        id_field: str, 
        id_value: Any
    ):
        """
        Generate an embedding for text and store it in the database.
        
        Args:
            text: Text to generate embedding for
            conn: Database connection
            table_name: Name of the table
            id_field: Name of the ID field
            id_value: Value of the ID
        """
        # Generate embedding
        embedding = await generate_embedding(text)
        
        # Update the table
        await conn.execute(f"""
            UPDATE {table_name}
            SET embedding = $1
            WHERE {id_field} = $2
        """, embedding, id_value)
    
    def get_cache(self, key):
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None
        """
        cache_key = f"{self.cache_prefix}_{key}_{self.user_id}_{self.conversation_id}"
        return CACHE_MANAGER.get(self.cache_prefix, cache_key)
    
    def set_cache(self, key, value):
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_key = f"{self.cache_prefix}_{key}_{self.user_id}_{self.conversation_id}"
        CACHE_MANAGER.set(self.cache_prefix, cache_key, value)
    
    def invalidate_cache_pattern(self, pattern):
        """
        Invalidate cache keys matching a pattern.
        
        Args:
            pattern: Pattern to match
        """
        CACHE_MANAGER.invalidate_pattern(self.cache_prefix, pattern)

class MatriarchalPowerStructureFramework(BaseLoreManager):
    """
    Defines core principles for power dynamics in femdom settings,
    ensuring consistency across all generated lore.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.core_principles = self._initialize_core_principles()
        
    def _initialize_core_principles(self) -> Dict[str, Any]:
        """Initialize core principles for matriarchal power structures"""
        return {
            "power_dynamics": {
                "dominant_gender": "female",
                "power_expression": ["political", "economic", "religious", "domestic", "sexual"],
                "hierarchy_types": ["matrilineal", "matrifocal", "matriarchal", "gynocentric"],
                "masculine_roles": ["service", "support", "nurture", "protection", "resources"],
                "counter_dynamics": ["resistance movements", "historical shifts", "regional variations"]
            },
            "societal_norms": {
                "female_leadership": {"political", "religious", "economic", "familial", "military"},
                "female_property_rights": {"land ownership", "business ownership", "inheritance"},
                "male_status_markers": {"service quality", "obedience", "beauty", "utility", "devotion"},
                "relationship_structures": {"polygyny", "polyandry", "monoandry", "collective households"},
                "enforcement_mechanisms": {"social pressure", "legal restrictions", "physical punishment", "economic sanctions"}
            },
            "symbolic_representations": {
                "feminine_symbols": ["chalice", "circle", "spiral", "dome", "moon"],
                "masculine_symbols": ["kneeling figures", "chains", "collars", "restraints"],
                "cultural_motifs": ["female nurture", "female authority", "male submission", "service ethics"]
            }
        }
    
    async def apply_power_lens(self, foundation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply matriarchal lens to generated foundation lore
        
        Args:
            foundation_data: Original foundation lore
            
        Returns:
            Foundation lore transformed through matriarchal lens
        """
        # Transform generic social structures into matriarchal ones
        if "social_structure" in foundation_data:
            foundation_data["social_structure"] = self._transform_to_matriarchal(
                foundation_data["social_structure"]
            )
        
        # Transform cosmology to reflect feminine primacy
        if "cosmology" in foundation_data:
            foundation_data["cosmology"] = self._feminize_cosmology(
                foundation_data["cosmology"]
            )
        
        # Transform magic system to reflect gendered power dynamics
        if "magic_system" in foundation_data:
            foundation_data["magic_system"] = self._gender_magic_system(
                foundation_data["magic_system"]
            )
        
        # Ensure world history reflects matriarchal development
        if "world_history" in foundation_data:
            foundation_data["world_history"] = self._matriarchalize_history(
                foundation_data["world_history"]
            )
            
        # Ensure calendar system reflects feminine significance
        if "calendar_system" in foundation_data:
            foundation_data["calendar_system"] = self._feminize_calendar(
                foundation_data["calendar_system"]
            )
            
        return foundation_data
    
    def _transform_to_matriarchal(self, social_structure: str) -> str:
        """
        Transform generic social structure description to a matriarchal one
        
        Args:
            social_structure: Original social structure description
            
        Returns:
            Matriarchal social structure description
        """
        # This is a simplified version - in a full implementation, 
        # you would use an LLM to transform the text more intelligently
        
        principles = self.core_principles["power_dynamics"]
        norms = self.core_principles["societal_norms"]
        
        # Extract key elements from original structure
        has_monarchy = "monarchy" in social_structure.lower()
        has_aristocracy = "aristocracy" in social_structure.lower() or "noble" in social_structure.lower()
        has_democracy = "democracy" in social_structure.lower() or "republic" in social_structure.lower()
        has_theocracy = "theocracy" in social_structure.lower() or "religious" in social_structure.lower()
        has_tribal = "tribal" in social_structure.lower() or "clan" in social_structure.lower()
        
        # Create matriarchal version
        transformed = "This society is fundamentally matriarchal. "
        
        if has_monarchy:
            transformed += "The supreme ruler is always a Queen or Empress, with succession passed through the maternal line. "
        if has_aristocracy:
            transformed += "Noble titles and land are held predominantly by women, with men serving as consorts or stewards. "
        if has_democracy:
            transformed += "While there is a democratic process, only women may vote or hold significant office. Men may serve in supportive administrative roles. "
        if has_theocracy:
            transformed += "Religious authority is held exclusively by women, with male clergy serving in subordinate positions. "
        if has_tribal:
            transformed += "Clan and tribal leadership passes through the maternal line, with Matriarchs holding ultimate authority. "
            
        # Add details about male status and position
        transformed += "Men are valued for their service to women and society, with status determined by their usefulness, obedience, and beauty. "
        
        # Add societal norms
        transformed += "Female property ownership is absolute, with inheritance flowing through the maternal line. "
        transformed += "Cultural practices, laws, and social norms all reinforce the natural authority of women over men. "
        
        # Merge with original content where appropriate
        if len(social_structure) > 200:  # Only if there's substantial original content
            transformed += "While maintaining these fundamental matriarchal principles, the society also incorporates elements of its historical development: " + social_structure
            
        return transformed
    
    def generate_hierarchical_constraints(self) -> Dict[str, Any]:
        """
        Generate consistent rules about power hierarchies.
        
        Returns:
            Dictionary of hierarchical constraints to maintain across lore
        """
        hierarchy_types = self.core_principles["power_dynamics"]["hierarchy_types"]
        chosen_type = random.choice(hierarchy_types)
        
        constraints = {
            "dominant_hierarchy_type": chosen_type,
            "power_expressions": random.sample(self.core_principles["power_dynamics"]["power_expression"], 3),
            "masculine_roles": random.sample(self.core_principles["power_dynamics"]["masculine_roles"], 3),
            "leadership_domains": random.sample(list(self.core_principles["societal_norms"]["female_leadership"]), 3),
            "property_rights": random.sample(list(self.core_principles["societal_norms"]["female_property_rights"]), 2),
            "status_markers": random.sample(list(self.core_principles["societal_norms"]["male_status_markers"]), 3),
            "relationship_structure": random.choice(list(self.core_principles["societal_norms"]["relationship_structures"])),
            "enforcement_mechanisms": random.sample(list(self.core_principles["societal_norms"]["enforcement_mechanisms"]), 2)
        }
        
        if chosen_type == "matrilineal":
            constraints["description"] = "Descent and inheritance pass through the maternal line, with women controlling family resources."
        elif chosen_type == "matrifocal":
            constraints["description"] = "Women are the center of family life and decision-making, with men in peripheral roles."
        elif chosen_type == "matriarchal":
            constraints["description"] = "Women hold formal political, economic, and social power over men in all domains."
        elif chosen_type == "gynocentric":
            constraints["description"] = "Society and culture are centered on feminine needs, values, and perspectives."
        
        return constraints
    
    def generate_power_expressions(self) -> List[Dict[str, Any]]:
        """
        Generate specific expressions of female power and male submission
        
        Returns:
            List of power expression descriptions
        """
        expressions = []
        
        # Political expressions
        expressions.append({
            "domain": "political",
            "title": "Council of Matriarchs",
            "description": "The ruling council composed exclusively of senior women who make all major decisions for the community.",
            "male_role": "Advisors and administrators who carry out the Matriarchs' decisions without question."
        })
        
        # Economic expressions
        expressions.append({
            "domain": "economic",
            "title": "Female Property Ownership",
            "description": "All significant property, businesses, and resources are owned and controlled by women.",
            "male_role": "Men manage resources only as agents of their female relatives or superiors."
        })
        
        # Religious expressions
        expressions.append({
            "domain": "religious",
            "title": "Priestesshood",
            "description": "Religious authority vested in female clergy who interpret the will of the Goddesses.",
            "male_role": "Temple servants who handle mundane tasks and participate in rituals as directed."
        })
        
        # Domestic expressions
        expressions.append({
            "domain": "domestic",
            "title": "Household Governance",
            "description": "Women control the household, making all significant decisions about family life.",
            "male_role": "Men handle domestic labor and childcare under female direction."
        })
        
        # Sexual expressions
        expressions.append({
            "domain": "sexual",
            "title": "Female Sexual Agency",
            "description": "Women determine when, how, and with whom sexual activity occurs.",
            "male_role": "Men's sexuality is considered a resource to be managed and directed by women."
        })
        
        # Military expressions
        expressions.append({
            "domain": "military",
            "title": "Feminine Command",
            "description": "Military leadership is exclusively female, with generals and officers all being women.",
            "male_role": "Men serve as foot soldiers, following orders from their female superiors."
        })
        
        return expressions

class LoreDynamicsSystem(BaseLoreManager):
    """
    Consolidated system for evolving world lore, generating emergent events,
    expanding content, and managing how the world changes over time.
    Combines functionality from LoreEvolutionSystem, EmergentLoreSystem, and LoreExpansionSystem.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.faith_system = ReligionManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
    
    async def initialize_tables(self):
        """Ensure required tables exist"""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Ensure LoreChangeHistory table exists
                history_table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'lorechangehistory'
                    );
                """)
                
                if not history_table_exists:
                    await conn.execute("""
                        CREATE TABLE LoreChangeHistory (
                            id SERIAL PRIMARY KEY,
                            lore_type TEXT NOT NULL,
                            lore_id TEXT NOT NULL,
                            previous_description TEXT NOT NULL,
                            new_description TEXT NOT NULL,
                            change_reason TEXT NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                # Add any other required tables for tracking lore evolution
                
    #===========================================================================
    # CORE LORE EVOLUTION METHODS
    #===========================================================================
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore with event",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        # Ensure initialized
        await self.ensure_initialized()
        
        # Check permissions
        permission = await self.check_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="evolve_lore_with_event",
            action_details={"event_description": event_description}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore evolution not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # 1. Identify affected lore elements
        affected_elements = await self._identify_affected_lore(event_description)
        
        # 2. Generate modifications for each affected element
        updates = await self._generate_lore_updates(affected_elements, event_description)
        
        # 3. Apply updates to database
        await self._apply_lore_updates(updates)
        
        # 4. Generate potential new lore elements triggered by this event
        new_elements = await self._generate_consequential_lore(event_description, affected_elements)
        
        # 5. Report to governance system
        await self.report_action(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "evolve_lore_with_event",
                "description": f"Evolved lore with event: {event_description[:50]}"
            },
            result={
                "affected_elements": len(affected_elements),
                "new_elements_created": len(new_elements)
            }
        )
        
        return {
            "affected_elements": affected_elements,
            "updates": updates,
            "new_elements": new_elements
        }
    
    async def _identify_affected_lore(self, event_description: str) -> List[Dict[str, Any]]:
        """
        Identify lore elements that would be affected by the event
        
        Args:
            event_description: Description of the event
            
        Returns:
            List of affected lore elements with relevance scores
        """
        # Generate embedding for the event
        event_embedding = await generate_embedding(event_description)
        
        # Search all lore types for potentially affected elements
        affected_elements = []
        
        # List of lore types to check
        lore_types = [
            "WorldLore", 
            "Factions", 
            "CulturalElements", 
            "HistoricalEvents", 
            "GeographicRegions", 
            "LocationLore",
            "UrbanMyths",
            "LocalHistories",
            "Landmarks",
            "NotableFigures"
        ]
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for lore_type in lore_types:
                    try:
                        # Check if table exists
                        table_exists = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = '{lore_type.lower()}'
                            );
                        """)
                        
                        if not table_exists:
                            continue
                            
                        # Not all tables may have embedding - handle that case
                        has_embedding = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = '{lore_type.lower()}' AND column_name = 'embedding'
                            );
                        """)
                        
                        if not has_embedding:
                            continue
                        
                        # Perform similarity search
                        id_field = 'id'
                        if lore_type == 'LocationLore':
                            id_field = 'location_id'
                            
                        rows = await conn.fetch(f"""
                            SELECT {id_field} as id, name, description, 
                                   1 - (embedding <=> $1) as relevance
                            FROM {lore_type}
                            WHERE 1 - (embedding <=> $1) > 0.6
                            ORDER BY relevance DESC
                            LIMIT 5
                        """, event_embedding)
                        
                        for row in rows:
                            affected_elements.append({
                                'lore_type': lore_type,
                                'lore_id': row['id'],
                                'name': row['name'],
                                'description': row['description'],
                                'relevance': row['relevance']
                            })
                    except Exception as e:
                        logging.error(f"Error checking {lore_type} for affected elements: {e}")
        
        # Sort by relevance
        affected_elements.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Filter to most relevant (if too many)
        if len(affected_elements) > 15:
            affected_elements = affected_elements[:15]
            
        return affected_elements
    
    async def _generate_lore_updates(
        self, 
        affected_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> List[Dict[str, Any]]:
        """
        Generate specific updates for affected lore elements
        
        Args:
            affected_elements: List of affected lore elements
            event_description: Description of the event
            
        Returns:
            List of updates to apply
        """
        updates = []
        
        # Get current world state for context
        world_state = await self._fetch_world_state()
        
        # Calculate societal impact of the event
        societal_impact = await self._calculate_societal_impact(
            event_description, 
            world_state.get('stability_index', 8),
            world_state.get('power_hierarchy', {})
        )
        
        # Track relationship changes and power shifts for cascade effects
        relationship_changes = {}
        power_shifts = {}
        
        # Use an LLM to generate updates for each element
        for element in affected_elements:
            # Create the run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Get relationship network for this element
            related_elements = await self._fetch_related_elements(element.get('lore_id', ''))
            
            # Determine element's position in power hierarchy
            hierarchy_position = await self._get_hierarchy_position(element)
            
            # Get update history
            update_history = await self._fetch_element_update_history(element.get('lore_id', ''))
            
            # Build enhanced prompt for the LLM
            prompt = await self._build_enhanced_lore_prompt(
                element=element,
                event_description=event_description,
                societal_impact=societal_impact,
                related_elements=related_elements,
                hierarchy_position=hierarchy_position,
                update_history=update_history
            )
            
            # Create an agent for lore updates
            lore_update_agent = Agent(
                name="LoreUpdateAgent",
                instructions="You update existing lore elements based on narrative events while maintaining thematic consistency.",
                model="o3-mini"
            )
            
            # Get the response
            result = await Runner.run(lore_update_agent, prompt, context=run_ctx.context)
            response_text = result.final_output
            
            try:
                # Parse the response
                update_data = await self._parse_lore_update_response(response_text, element)
                
                # Add the update
                updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': update_data['new_description'],
                    'update_reason': update_data['update_reason'],
                    'impact_level': update_data['impact_level'],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'narrative_themes': update_data.get('narrative_themes', []),
                    'mood_shift': update_data.get('mood_shift', 'neutral')
                })
                
                # Calculate cascade effects
                if related_elements:
                    cascade_effects = await self._calculate_cascade_effects(
                        element, 
                        update_data, 
                        related_elements
                    )
                    
                    # Track relationship changes
                    for rel_id, change in cascade_effects.get('relationship_changes', {}).items():
                        if rel_id in relationship_changes:
                            relationship_changes[rel_id] += change
                        else:
                            relationship_changes[rel_id] = change
                    
                    # Track power shifts
                    for faction_id, shift in cascade_effects.get('power_shifts', {}).items():
                        if faction_id in power_shifts:
                            power_shifts[faction_id] += shift
                        else:
                            power_shifts[faction_id] = shift
                
            except Exception as e:
                logging.error(f"Failed to parse LLM response or update element: {e}")
                # Add fallback update if parsing fails
                updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': element['description'] + f"\n\nRecent developments: {event_description}",
                    'update_reason': "Event impact (automatic fallback)",
                    'impact_level': 3,
                    'timestamp': datetime.datetime.now().isoformat()
                })
        
        # Generate cascade updates for indirectly affected elements
        if relationship_changes or power_shifts:
            cascade_element_ids = [id for id, change in relationship_changes.items() if abs(change) > 0.5]
            if cascade_element_ids:
                cascade_elements = await self._fetch_elements_by_ids(cascade_element_ids)
                cascade_updates = await self._generate_cascade_updates(
                    cascade_elements,
                    event_description,
                    relationship_changes,
                    power_shifts
                )
                updates.extend(cascade_updates)
            
            # Update world power balance
            if power_shifts:
                await self._update_world_power_balance(power_shifts)
        
        return updates
    
    async def _apply_lore_updates(self, updates: List[Dict[str, Any]]) -> None:
        """
        Apply the generated updates to the database
        
        Args:
            updates: List of updates to apply
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for update in updates:
                    lore_type = update['lore_type']
                    lore_id = update['lore_id']
                    new_description = update['new_description']
                    
                    # Generate new embedding for the updated content
                    # Determine ID field name
                    id_field = 'id'
                    if lore_type == 'LocationLore':
                        id_field = 'location_id'
                    
                    try:
                        # Update the description first
                        await conn.execute(f"""
                            UPDATE {lore_type}
                            SET description = $1
                            WHERE {id_field} = $2
                        """, new_description, lore_id)
                        
                        # Generate and store new embedding
                        item_name = update.get('name', 'Unknown')
                        embedding_text = f"{item_name} {new_description}"
                        await self.generate_and_store_embedding(embedding_text, conn, lore_type, id_field, lore_id)
                        
                        # Record the update in a history table if it exists
                        history_table_exists = await conn.fetchval(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'lorechangehistory'
                            );
                        """)
                        
                        if history_table_exists:
                            await conn.execute("""
                                INSERT INTO LoreChangeHistory 
                                (lore_type, lore_id, previous_description, new_description, change_reason)
                                VALUES ($1, $2, $3, $4, $5)
                            """, lore_type, lore_id, update['old_description'], new_description, update['update_reason'])
                    except Exception as e:
                        logging.error(f"Error updating {lore_type} ID {lore_id}: {e}")
                        
                    # Clear relevant cache
                    LORE_CACHE.invalidate_pattern(f"{lore_type}_{lore_id}")
    
    async def _generate_consequential_lore(
        self, 
        event_description: str, 
        affected_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate new lore elements that might emerge as a consequence of the event
        
        Args:
            event_description: Description of the event
            affected_elements: List of affected lore elements
            
        Returns:
            List of new lore elements
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Extract names of affected elements for context
        affected_names = [f"{e['name']} ({e['lore_type']})" for e in affected_elements[:5]]
        affected_names_text = ", ".join(affected_names)
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following event and affected lore elements, generate 1-3 new lore elements that would emerge as consequences.
        
        EVENT:
        {event_description}
        
        AFFECTED ELEMENTS:
        {affected_names_text}
        
        Generate between 1 and 3 new lore elements of different types. Consider:
        - New urban myths that might emerge
        - Changes to local history or folklore
        - New traditions or cultural practices
        - New historical events that this triggers
        - New notable figures who rise to prominence
        - New organizations or factions that form
        
        For each new element, provide:
        1. "lore_type": The type of lore element (WorldLore, Factions, CulturalElements, UrbanMyths, etc.)
        2. "name": A name for the element
        3. "description": A detailed description
        4. "connection": How it relates to the event and other lore
        5. "significance": A number from 1-10 indicating importance
        
        Format your response as a JSON array containing these objects.
        IMPORTANT: Maintain the matriarchal/femdom power dynamics in all new lore.
        """
        
        # Create an agent for new lore generation
        lore_creation_agent = Agent(
            name="LoreCreationAgent",
            instructions="You create new lore elements that emerge from significant events.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(lore_creation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        # Parse the JSON response
        try:
            new_elements = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(new_elements, list):
                if isinstance(new_elements, dict):
                    new_elements = [new_elements]
                else:
                    new_elements = []
                    
            # Save the new lore elements to appropriate tables
            await self._save_new_lore_elements(new_elements, event_description)
                
            return new_elements
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for new lore: {response_text}")
            return []
    
    async def _save_new_lore_elements(
        self, 
        new_elements: List[Dict[str, Any]], 
        event_description: str
    ) -> None:
        """
        Save newly generated lore elements to appropriate tables
        
        Args:
            new_elements: List of new lore elements
            event_description: Original event description
        """
        for element in new_elements:
            lore_type = element.get('lore_type')
            name = element.get('name')
            description = element.get('description')
            significance = element.get('significance', 5)
            
            if not all([lore_type, name, description]):
                continue
                
            # Generate embedding for the new content
            embedding_text = f"{name} {description}"
            embedding = await generate_embedding(embedding_text)
            
            # Save based on type
            try:
                if lore_type == "WorldLore":
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category=element.get('category', 'event_consequence'),
                        description=description,
                        significance=significance,
                        tags=element.get('tags', ['event_consequence', 'new_lore'])
                    )
                elif lore_type == "Factions":
                    await self.lore_manager.add_faction(
                        name=name,
                        faction_type=element.get('faction_type', 'event_consequence'),
                        description=description,
                        values=element.get('values', ['power', 'authority']),
                        goals=element.get('goals', ['stability', 'influence']),
                        headquarters=element.get('headquarters'),
                        founding_story=f"Founded in response to: {event_description}"
                    )
                elif lore_type == "CulturalElements":
                    await self.lore_manager.add_cultural_element(
                        name=name,
                        element_type=element.get('element_type', 'tradition'),
                        description=description,
                        practiced_by=element.get('practiced_by', ['society']),
                        significance=significance,
                        historical_origin=f"Emerged from: {event_description}"
                    )
                elif lore_type == "HistoricalEvents":
                    await self.lore_manager.add_historical_event(
                        name=name,
                        description=description,
                        date_description=element.get('date_description', 'Recently'),
                        significance=significance,
                        participating_factions=element.get('participating_factions', []),
                        consequences=element.get('consequences', ['Still unfolding'])
                    )
                elif lore_type == "UrbanMyths":
                    # Fall back to world lore
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category='urban_myth',
                        description=description,
                        significance=significance,
                        tags=['urban_myth', 'event_consequence']
                    )
                elif lore_type == "NotableFigures":
                    # Fall back to world lore
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category='notable_figure',
                        description=description,
                        significance=significance,
                        tags=['notable_figure', 'event_consequence']
                    )
                else:
                    # Generic fallback for unknown types
                    await self.lore_manager.add_world_lore(
                        name=name,
                        category=lore_type.lower(),
                        description=description,
                        significance=significance,
                        tags=[lore_type.lower(), 'event_consequence']
                    )
            except Exception as e:
                logging.error(f"Error saving new {lore_type} '{name}': {e}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="mature_lore_over_time",
        action_description="Maturing lore over time",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def mature_lore_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """
        Natural evolution of lore over time, simulating how history and culture develop
        
        Args:
            days_passed: Number of in-game days that have passed
            
        Returns:
            Summary of maturation changes
        """
        # Check permissions with governance system
        await self.ensure_initialized()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="mature_lore_over_time",
            action_details={"days_passed": days_passed}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore maturation not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # Calculate probabilities based on days passed
        # More time = more changes
        base_probability = min(0.5, days_passed * 0.02)  # Cap at 50%
        
        maturation_summary = {
            "myth_evolution": [],
            "culture_development": [],
            "geopolitical_shifts": [],
            "reputation_changes": []
        }
        
        # 1. Urban Myths Evolution (highest chance of change)
        if random.random() < (base_probability * 1.5):
            myth_changes = await self._evolve_urban_myths()
            maturation_summary["myth_evolution"] = myth_changes
        
        # 2. Cultural Elements Development
        if random.random() < base_probability:
            culture_changes = await self._develop_cultural_elements()
            maturation_summary["culture_development"] = culture_changes
        
        # 3. Geopolitical Landscape Shifts (rarer)
        if random.random() < (base_probability * 0.7):
            geopolitical_changes = await self._shift_geopolitical_landscape()
            maturation_summary["geopolitical_shifts"] = geopolitical_changes
        
        # 4. Notable Figure Reputation Changes
        if random.random() < (base_probability * 1.2):
            reputation_changes = await self._evolve_notable_figures()
            maturation_summary["reputation_changes"] = reputation_changes
        
        # Report to governance system
        changes_count = sum(len(changes) for changes in maturation_summary.values())
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "mature_lore_over_time",
                "description": f"Matured lore over {days_passed} days"
            },
            result={
                "changes_applied": changes_count
            }
        )
        
        return {
            "days_passed": days_passed,
            "changes_applied": changes_count,
            "maturation_summary": maturation_summary
        }
    
    #===========================================================================
    # EMERGENT EVENT METHODS (From EmergentLoreSystem)
    #===========================================================================
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """
        Generate a random emergent event in the world with governance oversight.
        
        Returns:
            Event details and lore impact
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get current world state for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get active factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get some nations
                nations = await conn.fetch("""
                    SELECT id, name, government_type
                    FROM Nations
                    LIMIT 3
                """)
                
                # Get some locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Convert to lists
                faction_data = [dict(faction) for faction in factions]
                nation_data = [dict(nation) for nation in nations]
                location_data = [dict(location) for location in locations]
        
        # Determine event type
        event_types = [
            "political_shift", "military_conflict", "natural_disaster",
            "cultural_development", "technological_advancement", "religious_event",
            "economic_change", "diplomatic_incident"
        ]
        
        # Weight certain events based on available data
        weights = [1] * len(event_types)
        
        # More likely to generate faction-related events if we have factions
        if faction_data:
            faction_event_indices = [0, 1, 3, 6, 7]  # political, military, cultural, economic, diplomatic
            for idx in faction_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate nation-related events if we have nations
        if nation_data:
            nation_event_indices = [0, 1, 7]  # political, military, diplomatic
            for idx in nation_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # More likely to generate location-related events if we have locations
        if location_data:
            location_event_indices = [2, 3, 5]  # natural, cultural, religious
            for idx in location_event_indices:
                if idx < len(weights):
                    weights[idx] += 1
        
        # Select event type
        event_type = random.choices(event_types, weights=weights, k=1)[0]
        
        # Create a prompt based on event type
        prompt = self._create_event_prompt(event_type, faction_data, nation_data, location_data)
        
        # Create an agent for event generation
        event_agent = Agent(
            name="EmergentEventAgent",
            instructions="You create emergent world events for fantasy settings.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(event_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            event_data = json.loads(response_text)
            
            # Create a unified event description for lore evolution
            event_description = event_data.get("description", "")
            event_name = event_data.get("event_name", "Unnamed Event")
            
            # Use lore evolution functionality to evolve lore based on this event
            lore_updates = await self.evolve_lore_with_event(
                f"{event_name}: {event_description}"
            )
            
            # Add the lore updates to the event data
            event_data["lore_updates"] = lore_updates
            
            # Return the combined data
            return event_data
            
        except Exception as e:
            logging.error(f"Error generating emergent event: {e}")
            return {"error": str(e)}
    
    def _create_event_prompt(self, event_type, faction_data, nation_data, location_data):
        """
        Create a tailored prompt for generating a specific type of event
        
        Args:
            event_type: Type of event to generate
            faction_data: Faction context data
            nation_data: Nation context data
            location_data: Location context data
            
        Returns:
            Prompt string
        """
        if event_type == "political_shift":
            entities = faction_data if faction_data else nation_data
            prompt = f"""
            Generate a political shift event for this world:
            
            POTENTIAL ENTITIES INVOLVED:
            {json.dumps(entities, indent=2)}
            
            Create a significant political shift event that:
            1. Changes power dynamics in a meaningful way
            2. Is specific and detailed enough to impact the world
            3. Reinforces or challenges matriarchal power structures
            4. Could lead to interesting narrative developments
            
            Return a JSON object with:
            - event_name: Name of the event
            - event_type: "political_shift"
            - description: Detailed description
            - entities_involved: Array of entity names involved
            - instigator: The primary instigating entity
            - immediate_consequences: Array of immediate consequences
            - potential_long_term_effects: Array of potential long-term effects
            - affected_lore_categories: Array of lore categories this would affect
            """
        # Other event types handled similarly...
        # (Implementation for other event types would go here)
        else:  # diplomatic_incident (fallback)
            prompt = f"""
            Generate a diplomatic incident for this world:
            
            POTENTIAL NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            POTENTIAL FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a diplomatic incident that:
            1. Creates tension between two or more groups
            2. Stems from a specific action or misunderstanding
            3. Requires diplomatic resolution
            4. Highlights cultural differences or values
            
            Return a JSON object with:
            - event_name: Name of the diplomatic incident
            - event_type: "diplomatic_incident"
            - description: Detailed description
            - parties_involved: Array of involved parties
            - instigating_action: What triggered the incident
            - cultural_factors: Cultural factors at play
            - severity: Severity level (1-10)
            - potential_resolutions: Possible ways to resolve it
            - current_status: Current diplomatic status
            - affected_lore_categories: Array of lore categories this would affect
            """
            
        return prompt
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_cyclical_event",
        action_description="Generating cyclical seasonal event",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_cyclical_event(self, ctx, season: str = None, is_solstice: bool = False, is_equinox: bool = False) -> Dict[str, Any]:
        """
        Generate a cyclical seasonal event with governance oversight.
        
        Args:
            season: Optional season (spring, summer, fall, winter)
            is_solstice: Whether this is a solstice event
            is_equinox: Whether this is an equinox event
            
        Returns:
            Cyclical event details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Set default season if not provided
        if not season:
            seasons = ["spring", "summer", "fall", "winter"]
            season = random.choice(seasons)
            
        # Get relevant data for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get cultural elements
                cultural_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    WHERE type IN ('festival', 'tradition', 'ceremony', 'ritual')
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get religious practices
                religious_practices = await conn.fetch("""
                    SELECT name, practice_type, description, frequency
                    FROM ReligiousPractices
                    WHERE frequency LIKE '%annual%' OR frequency LIKE '%seasonal%'
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Get suitable locations
                locations = await conn.fetch("""
                    SELECT location_name, description
                    FROM Locations
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                # Convert to lists
                cultural_data = [dict(element) for element in cultural_elements]
                religious_data = [dict(practice) for practice in religious_practices]
                location_data = [dict(location) for location in locations]
        
        # Determine event type and special modifiers
        event_subtypes = ["festival", "ritual", "harvest", "hunt", "pilgrimage", "market", "ceremony"]
        
        if is_solstice:
            event_subtypes.extend(["celestial alignment", "divine manifestation", "magical surge"])
        if is_equinox:
            event_subtypes.extend(["balance ritual", "transition ceremony", "world renewal"])
            
        if season == "spring":
            event_subtypes.extend(["planting", "fertility rite", "renewal ceremony"])
        elif season == "summer":
            event_subtypes.extend(["sun celebration", "competition", "coming of age"])
        elif season == "fall":
            event_subtypes.extend(["harvest festival", "ancestor veneration", "preparation ritual"])
        elif season == "winter":
            event_subtypes.extend(["endurance trial", "darkness ritual", "shelter ceremony"])
            
        event_subtype = random.choice(event_subtypes)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a cyclical {season} {event_subtype} event for a matriarchal fantasy world.
        
        EXISTING CULTURAL ELEMENTS:
        {json.dumps(cultural_data, indent=2)}
        
        EXISTING RELIGIOUS PRACTICES:
        {json.dumps(religious_data, indent=2)}
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        SPECIAL MODIFIERS:
        Solstice Event: {"Yes" if is_solstice else "No"}
        Equinox Event: {"Yes" if is_equinox else "No"}
        
        Create a cyclical event that:
        1. Is tied to the {season} season specifically
        2. Reinforces matriarchal power and feminine divine connection
        3. Involves community participation with specific roles
        4. Has cultural and possibly religious significance
        
        Return a JSON object with:
        - event_name: Name of the cyclical event
        - event_type: "cyclical_event"
        - subtype: The specific type of event (festival, ritual, etc.)
        - season: The associated season
        - description: Detailed description
        - frequency: How often it occurs
        - duration: How long it lasts
        - locations: Where it's celebrated
        - primary_participants: Who leads or conducts the event
        - community_role: How the community participates
        - religious_significance: Any religious meaning
        - cultural_significance: Cultural meaning and importance
        - material_components: Required materials or preparations
        - traditions: Specific traditions associated with this event
        - gender_roles: How gender influences participation
        """
        
        # Create an agent for cyclical event generation
        cyclical_agent = Agent(
            name="CyclicalEventAgent",
            instructions="You create seasonal festivals and cyclical events for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(cyclical_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            event_data = json.loads(response_text)
            
            # Store this as a cultural element in the database
            try:
                # Prepare the data
                name = event_data.get("event_name", f"{season.capitalize()} {event_subtype.capitalize()}")
                description = event_data.get("description", "")
                
                cultural_description = description
                if "cultural_significance" in event_data:
                    cultural_description += f"\n\nCultural Significance: {event_data['cultural_significance']}"
                if "traditions" in event_data:
                    traditions = event_data.get("traditions", [])
                    if isinstance(traditions, list):
                        cultural_description += f"\n\nTraditions: {', '.join(traditions)}"
                    else:
                        cultural_description += f"\n\nTraditions: {traditions}"
                        
                event_data["stored_as_cultural_element"] = True
                
                # Add to CulturalElements
                element_id = await self.lore_manager.add_cultural_element(
                    name=name,
                    element_type=event_data.get("subtype", "festival"),
                    description=cultural_description,
                    practiced_by=event_data.get("locations", []),
                    significance=8,  # Cyclical events are quite significant
                    historical_origin=f"This {event_data.get('subtype', 'event')} developed as a {season} celebration."
                )
                
                event_data["cultural_element_id"] = element_id
                
                # If it has religious significance, also add as a religious practice
                if "religious_significance" in event_data and event_data.get("religious_significance"):
                    religious_description = description
                    religious_description += f"\n\nReligious Significance: {event_data['religious_significance']}"
                    
                    # Add to ReligiousPractices
                    practice_id = await self.faith_system.add_religious_practice(
                        run_ctx,
                        name=name,
                        practice_type=event_data.get("subtype", "festival"),
                        description=religious_description,
                        purpose="seasonal celebration",
                        frequency=f"Annual ({season})",
                        required_elements=event_data.get("material_components", [])
                    )
                    
                    event_data["religious_practice_id"] = practice_id
                    event_data["stored_as_religious_practice"] = True
            except Exception as e:
                logging.error(f"Error storing cyclical event: {e}")
                event_data["storage_error"] = str(e)
            
            return event_data
            
        except Exception as e:
            logging.error(f"Error generating cyclical event: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_nation",
        action_description="Generating additional nation for the world",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_additional_nation(self, ctx) -> Dict[str, Any]:
        """
        Generate an additional nation with governance oversight.
        
        Returns:
            New nation details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get existing nations for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check how many nations we already have
                nation_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM Nations
                """)
                
                # Get existing nations
                existing_nations = await conn.fetch("""
                    SELECT name, government_type, matriarchy_level, neighboring_nations
                    FROM Nations
                    LIMIT 10
                """)
                
                # Get pantheons for religious context
                pantheons = await conn.fetch("""
                    SELECT name, description
                    FROM Pantheons
                    LIMIT 3
                """)
                
                # Get some cultural elements for context
                cultural_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    ORDER BY RANDOM()
                    LIMIT 5
                """)
                
                # Convert to lists
                nation_data = [dict(nation) for nation in existing_nations]
                pantheon_data = [dict(pantheon) for pantheon in pantheons]
                cultural_data = [dict(element) for element in cultural_elements]
        
        # Determine if this should be a matriarchal or non-matriarchal nation
        # Keep a balance in the world
        matriarchal_nations = [n for n in nation_data if n.get("matriarchy_level", 0) >= 7]
        non_matriarchal_nations = [n for n in nation_data if n.get("matriarchy_level", 0) <= 3]
        
        # Default to a medium matriarchy level
        target_matriarchy_level = random.randint(4, 7)
        
        # If we have more matriarchal nations, make this one less matriarchal
        if len(matriarchal_nations) > len(non_matriarchal_nations) + 1:
            target_matriarchy_level = random.randint(1, 4)
        # If we have more non-matriarchal nations, make this one more matriarchal
        elif len(non_matriarchal_nations) > len(matriarchal_nations):
            target_matriarchy_level = random.randint(7, 10)
            
        # Create a prompt for the LLM
        prompt = f"""
        Generate a new nation for a fantasy world with matriarchal themes.
        
        EXISTING NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        RELIGIOUS CONTEXT:
        {json.dumps(pantheon_data, indent=2)}
        
        CULTURAL CONTEXT:
        {json.dumps(cultural_data, indent=2)}
        
        Create a nation that:
        1. Has a matriarchy level of approximately {target_matriarchy_level}/10
        2. Is distinct from existing nations
        3. Has interesting cultural and political traits
        4. Has believable relationships with existing nations
        5. Has at least one unique or interesting feature
        
        Return a JSON object with:
        - name: Name of the nation
        - government_type: Type of government
        - description: Detailed description
        - relative_power: Power level (1-10)
        - matriarchy_level: {target_matriarchy_level} (with slight variation possible)
        - population_scale: Scale of population
        - major_resources: Array of key resources
        - major_cities: Array of key cities/settlements
        - cultural_traits: Array of defining cultural traits
        - notable_features: Other notable features
        - neighboring_nations: Array of nations that border this one
        """
        
        # Create an agent for nation generation
        nation_agent = Agent(
            name="NationGenerationAgent",
            instructions="You create nations for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(nation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            nation_data = json.loads(response_text)
            
            # Add nation to database through the geopolitical manager
            nation_id = await self.geopolitical_manager.add_nation(
                run_ctx,
                name=nation_data.get("name", "New Nation"),
                government_type=nation_data.get("government_type", "unknown"),
                description=nation_data.get("description", ""),
                relative_power=nation_data.get("relative_power", 5),
                matriarchy_level=nation_data.get("matriarchy_level", target_matriarchy_level),
                population_scale=nation_data.get("population_scale"),
                major_resources=nation_data.get("major_resources", []),
                major_cities=nation_data.get("major_cities", []),
                cultural_traits=nation_data.get("cultural_traits", []),
                notable_features=nation_data.get("notable_features"),
                neighboring_nations=nation_data.get("neighboring_nations", [])
            )
            
            # Update relationships with neighboring nations
            for neighbor_name in nation_data.get("neighboring_nations", []):
                # Find neighbor in existing nations
                for existing in existing_nations:
                    if existing["name"] == neighbor_name:
                        # Add this nation to the neighbor's neighboring_nations
                        neighbor_neighbors = existing.get("neighboring_nations", [])
                        if not neighbor_neighbors:
                            neighbor_neighbors = []
                        
                        if nation_data.get("name") not in neighbor_neighbors:
                            neighbor_neighbors.append(nation_data.get("name"))
                            
                            # Update neighbor
                            await conn.execute("""
                                UPDATE Nations
                                SET neighboring_nations = $1
                                WHERE name = $2
                            """, neighbor_neighbors, neighbor_name)
            
            # Update with the ID
            nation_data["id"] = nation_id
            
            return nation_data
            
        except Exception as e:
            logging.error(f"Error generating additional nation: {e}")
            return {"error": str(e)}
            
    #===========================================================================
    # LORE EXPANSION METHODS (From LoreExpansionSystem)
    #===========================================================================
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_faction",
        action_description="Generating additional faction for the world",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_additional_faction(self, ctx, faction_type: str = None) -> Dict[str, Any]:
        """
        Generate an additional faction with governance oversight.
        
        Args:
            faction_type: Optional type of faction to generate
            
        Returns:
            New faction details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get existing factions for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get existing factions
                existing_factions = await conn.fetch("""
                    SELECT name, type, description, values, goals
                    FROM Factions
                    LIMIT 10
                """)
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, government_type, matriarchy_level
                    FROM Nations
                    LIMIT 5
                """)
                
                # Get some cultural elements for context
                cultural_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    ORDER BY RANDOM()
                    LIMIT 5
                """)
                
                # Convert to lists
                faction_data = [dict(faction) for faction in existing_factions]
                nation_data = [dict(nation) for nation in nations]
                cultural_data = [dict(element) for element in cultural_elements]
        
        # If faction_type not specified, select one that's underrepresented
        if not faction_type:
            faction_types = ["political", "religious", "criminal", "mercantile", "academic", "military", "social"]
            existing_types = [f.get("type", "").lower() for f in faction_data]
            
            # Count each type
            type_counts = {}
            for t in faction_types:
                type_counts[t] = sum(1 for et in existing_types if t in et)
            
            # Find underrepresented types
            min_count = min(type_counts.values()) if type_counts else 0
            underrepresented = [t for t, c in type_counts.items() if c == min_count]
            
            # Choose one randomly from underrepresented
            faction_type = random.choice(underrepresented if underrepresented else faction_types)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a new {faction_type} faction for a matriarchal fantasy world:
        
        EXISTING FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        CULTURAL CONTEXT:
        {json.dumps(cultural_data, indent=2)}
        
        Create a {faction_type} faction that:
        1. Is distinct from existing factions
        2. Has a clear role and purpose in the world
        3. Has interesting values and goals
        4. Reflects matriarchal power dynamics
        5. Has potential for interesting stories and conflicts
        
        Return a JSON object with:
        - name: Name of the faction
        - type: Type of faction (should include "{faction_type}")
        - description: Detailed description
        - values: Array of core values
        - goals: Array of primary goals
        - headquarters: Main location or headquarters
        - founding_story: How the faction was founded
        - rivals: Array of rival factions or groups
        - allies: Array of allied factions or groups
        - territory: Areas where they operate
        - resources: Key resources they control
        - hierarchy_type: Structure of leadership
        - secret_knowledge: Any secret knowledge they possess
        - public_reputation: How they're viewed publicly
        - color_scheme: Colors associated with them
        - symbol_description: Their emblem or symbol
        """
        
        # Create an agent for faction generation
        faction_agent = Agent(
            name="FactionGenerationAgent",
            instructions="You create factions for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(faction_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            faction_data = json.loads(response_text)
            
            # Add faction to database
            faction_id = await self.lore_manager.add_faction(
                name=faction_data.get("name", f"Unnamed {faction_type.capitalize()} Faction"),
                faction_type=faction_data.get("type", faction_type),
                description=faction_data.get("description", ""),
                values=faction_data.get("values", []),
                goals=faction_data.get("goals", []),
                headquarters=faction_data.get("headquarters"),
                founding_story=faction_data.get("founding_story"),
                rivals=faction_data.get("rivals", []),
                allies=faction_data.get("allies", []),
                territory=faction_data.get("territory"),
                resources=faction_data.get("resources", []),
                hierarchy_type=faction_data.get("hierarchy_type"),
                secret_knowledge=faction_data.get("secret_knowledge"),
                public_reputation=faction_data.get("public_reputation"),
                color_scheme=faction_data.get("color_scheme"),
                symbol_description=faction_data.get("symbol_description")
            )
            
            # Update with the ID
            faction_data["id"] = faction_id
            
            return faction_data
            
        except Exception as e:
            logging.error(f"Error generating additional faction: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_locations",
        action_description="Generating additional locations for the world",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_additional_locations(self, ctx, location_types: List[str] = None, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate additional locations with governance oversight.
        
        Args:
            location_types: Optional types of locations to generate
            count: Number of locations to generate
            
        Returns:
            List of new location details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Set default location types if not provided
        if not location_types:
            location_types = ["settlement", "wilderness", "landmark", "dungeon", "ruin"]
        
        # Get context for location generation
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get existing locations
                existing_locations = await conn.fetch("""
                    SELECT location_name, description
                    FROM Locations
                    LIMIT 10
                """)
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, government_type
                    FROM Nations
                    LIMIT 5
                """)
                
                # Get factions for context
                factions = await conn.fetch("""
                    SELECT name, type
                    FROM Factions
                    LIMIT 5
                """)
                
                # Convert to lists
                location_data = [dict(location) for location in existing_locations]
                nation_data = [dict(nation) for nation in nations]
                faction_data = [dict(faction) for faction in factions]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate {count} new locations for a matriarchal fantasy world.
        
        EXISTING LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        LOCATION TYPES TO INCLUDE:
        {json.dumps(location_types, indent=2)}
        
        Create {count} locations that:
        1. Are diverse in type and purpose
        2. Include detailed descriptions
        3. Have connections to factions, nations, or existing locations
        4. Reflect matriarchal power dynamics in their design or purpose
        5. Have potential for interesting stories and encounters
        
        Return a JSON array where each location has:
        - name: Name of the location
        - type: Type of location (should be one from the provided types)
        - description: Detailed description
        - controlling_faction: Which faction controls it (if any)
        - nation: Which nation it's in (if any)
        - notable_features: Array of notable features
        - hidden_secrets: Array of hidden secrets
        - strategic_importance: Any strategic importance
        - population: Description of population (if inhabited)
        - points_of_interest: Array of interesting places within
        """
        
        # Create an agent for location generation
        location_agent = Agent(
            name="LocationGenerationAgent",
            instructions="You create locations for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(location_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            locations = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(locations, list):
                if isinstance(locations, dict):
                    locations = [locations]
                else:
                    locations = []
            
            # Add each location
            created_locations = []
            for location in locations:
                try:
                    # Add to Locations table
                    location_id = await self.lore_manager.add_location(
                        location_name=location.get("name", "Unnamed Location"),
                        description=location.get("description", ""),
                        location_type=location.get("type", "unknown")
                    )
                    
                    # Add to LocationLore table
                    await self.lore_manager.add_location_lore(
                        location_id=location_id,
                        founding_story=f"This {location.get('type', 'location')} was established as {location.get('name')}.",
                        hidden_secrets=location.get("hidden_secrets", []),
                        local_legends=[],
                        historical_significance=location.get("strategic_importance", "")
                    )
                    
                    # Update the location with its ID
                    location["id"] = location_id
                    created_locations.append(location)
                    
                except Exception as e:
                    logging.error(f"Error adding location {location.get('name')}: {e}")
            
            return created_locations
            
        except Exception as e:
            logging.error(f"Error generating additional locations: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_cultural_elements",
        action_description="Generating additional cultural elements",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def generate_additional_cultural_elements(self, ctx, element_types: List[str] = None, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate additional cultural elements with governance oversight.
        
        Args:
            element_types: Optional types of elements to generate
            count: Number of elements to generate
            
        Returns:
            List of new cultural element details
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Set default element types if not provided
        if not element_types:
            element_types = ["tradition", "custom", "taboo", "holiday", "ceremony", "social norm", "art form"]
        
        # Get context for cultural element generation
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get existing cultural elements
                existing_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    LIMIT 10
                """)
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, cultural_traits
                    FROM Nations
                    LIMIT 5
                """)
                
                # Get factions for context
                factions = await conn.fetch("""
                    SELECT name, type, values
                    FROM Factions
                    LIMIT 5
                """)
                
                # Convert to lists
                element_data = [dict(element) for element in existing_elements]
                nation_data = [dict(nation) for nation in nations]
                faction_data = [dict(faction) for faction in factions]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate {count} new cultural elements for a matriarchal fantasy world.
        
        EXISTING CULTURAL ELEMENTS:
        {json.dumps(element_data, indent=2)}
        
        NATIONS AND THEIR CULTURAL TRAITS:
        {json.dumps(nation_data, indent=2)}
        
        FACTIONS AND THEIR VALUES:
        {json.dumps(faction_data, indent=2)}
        
        ELEMENT TYPES TO INCLUDE:
        {json.dumps(element_types, indent=2)}
        
        Create {count} cultural elements that:
        1. Are diverse in type and purpose
        2. Include detailed descriptions
        3. Clearly connect to groups that practice them
        4. Reflect matriarchal power dynamics
        5. Feel authentic to a richly developed culture
        
        Return a JSON array where each element has:
        - name: Name of the cultural element
        - type: Type of element (should be one from the provided types)
        - description: Detailed description
        - practiced_by: Array of groups who practice this element
        - historical_origin: How the element originated
        - significance: Importance from 1-10
        - related_elements: Any related cultural elements
        - gender_aspects: How gender influences this element
        """
        
        # Create an agent for cultural element generation
        cultural_agent = Agent(
            name="CulturalElementAgent",
            instructions="You create cultural elements for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(cultural_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            elements = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(elements, list):
                if isinstance(elements, dict):
                    elements = [elements]
                else:
                    elements = []
            
            # Add each cultural element
            created_elements = []
            for element in elements:
                try:
                    # Add to database using lore_manager
                    element_id = await self.lore_manager.add_cultural_element(
                        name=element.get("name", f"Unnamed {element.get('type', 'Custom')}"),
                        element_type=element.get("type", "custom"),
                        description=element.get("description", ""),
                        practiced_by=element.get("practiced_by", []),
                        significance=element.get("significance", 5),
                        historical_origin=element.get("historical_origin")
                    )
                    
                    # Update the element with its ID
                    element["id"] = element_id
                    created_elements.append(element)
                    
                except Exception as e:
                    logging.error(f"Error adding cultural element {element.get('name')}: {e}")
            
            return created_elements
            
        except Exception as e:
            logging.error(f"Error generating additional cultural elements: {e}")
            return []
    
    #===========================================================================
    # WORLD EVOLUTION METHODS (Combined functionality)
    #===========================================================================
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_world_over_time",
        action_description="Evolving world over time",
        id_from_context=lambda ctx: "lore_dynamics"
    )
    async def evolve_world_over_time(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
        """
        Evolve the world over time, simulating the passage of days or weeks.
        
        Args:
            days_passed: Number of days to simulate
            
        Returns:
            Dictionary of evolution results
        """
        # Create run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        evolution_results = {}
        
        # 1. Mature lore naturally over time
        try:
            maturation = await self.mature_lore_over_time(days_passed)
            evolution_results["lore_maturation"] = maturation
            logging.info(f"Matured lore over {days_passed} days with {maturation.get('changes_applied', 0)} changes")
        except Exception as e:
            logging.error(f"Error maturing lore: {e}")
            evolution_results["lore_maturation_error"] = str(e)
        
        # 2. Generate random emergent events based on time passed
        try:
            # Scale number of events with time passed, but randomize
            num_events = max(1, min(5, days_passed // 10))
            num_events = random.randint(1, num_events)
            
            emergent_events = []
            for _ in range(num_events):
                event = await self.generate_emergent_event(run_ctx)
                emergent_events.append(event)
            
            evolution_results["emergent_events"] = emergent_events
            logging.info(f"Generated {len(emergent_events)} emergent events")
        except Exception as e:
            logging.error(f"Error generating emergent events: {e}")
            evolution_results["emergent_events_error"] = str(e)
        
        # 3. Maybe add a new faction if enough time has passed
        if days_passed >= 30 and random.random() < 0.4:  # 40% chance if 30+ days
            try:
                new_faction = await self.generate_additional_faction(run_ctx)
                evolution_results["new_faction"] = new_faction
                logging.info(f"Generated new faction: {new_faction.get('name', 'Unknown')}")
            except Exception as e:
                logging.error(f"Error generating new faction: {e}")
                evolution_results["new_faction_error"] = str(e)
        
        # 4. Maybe add a new nation if enough time has passed
        if days_passed >= 60 and random.random() < 0.3:  # 30% chance if 60+ days
            try:
                new_nation = await self.generate_additional_nation(run_ctx)
                evolution_results["new_nation"] = new_nation
                logging.info(f"Generated new nation: {new_nation.get('name', 'Unknown')}")
            except Exception as e:
                logging.error(f"Error generating new nation: {e}")
                evolution_results["new_nation_error"] = str(e)
        
        # 5. Generate new locations at a regular pace
        if days_passed >= 15 and random.random() < 0.5:  # 50% chance if 15+ days
            try:
                location_count = min(3, 1 + days_passed // 30)  # More time = more locations
                new_locations = await self.generate_additional_locations(run_ctx, count=location_count)
                evolution_results["new_locations"] = new_locations
                logging.info(f"Generated {len(new_locations)} new locations")
            except Exception as e:
                logging.error(f"Error generating new locations: {e}")
                evolution_results["new_locations_error"] = str(e)
                
        # 6. Generate new cultural elements as time passes
        if days_passed >= 20 and random.random() < 0.6:  # 60% chance if 20+ days
            try:
                element_count = min(3, 1 + days_passed // 25)  # More time = more elements
                new_elements = await self.generate_additional_cultural_elements(run_ctx, count=element_count)
                evolution_results["new_cultural_elements"] = new_elements
                logging.info(f"Generated {len(new_elements)} new cultural elements")
            except Exception as e:
                logging.error(f"Error generating new cultural elements: {e}")
                evolution_results["new_cultural_elements_error"] = str(e)
                
        # 7. Maybe generate a seasonal event if appropriate
        if random.random() < 0.7:  # 70% chance regardless of time
            try:
                # Choose a random season weighted by days passed (assumes 90 days per season)
                season_progress = (days_passed % 360) // 90
                seasons = ["winter", "spring", "summer", "fall"]
                season = seasons[season_progress]
                
                # Determine if it's a special day
                is_solstice = False
                is_equinox = False
                
                if season in ["summer", "winter"] and random.random() < 0.2:
                    is_solstice = True
                elif season in ["spring", "fall"] and random.random() < 0.2:
                    is_equinox = True
                
                seasonal_event = await self.generate_cyclical_event(run_ctx, season=season, 
                                                                is_solstice=is_solstice, 
                                                                is_equinox=is_equinox)
                evolution_results["seasonal_event"] = seasonal_event
                logging.info(f"Generated seasonal event: {seasonal_event.get('event_name', 'Unknown')}")
            except Exception as e:
                logging.error(f"Error generating seasonal event: {e}")
                evolution_results["seasonal_event_error"] = str(e)
        
        return evolution_results
    
    #===========================================================================
    # SUPPORT METHODS - Reused by multiple functions
    #===========================================================================
    
    async def _fetch_world_state(self) -> Dict[str, Any]:
        """Fetch current world state from database"""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Example query - adjust based on your schema
                world_state = await conn.fetchrow("""
                    SELECT * FROM WorldState 
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                
                if world_state:
                    return dict(world_state)
                else:
                    # Return default values if no world state found
                    return {
                        'stability_index': 8,
                        'narrative_tone': 'dramatic',
                        'power_dynamics': 'strict_hierarchy',
                        'power_hierarchy': {}
                    }
    
    async def _fetch_related_elements(self, lore_id: str) -> List[Dict[str, Any]]:
        """Fetch elements related to the given lore ID"""
        if not lore_id:
            return []
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Example query - adjust based on your schema
                    related = await conn.fetch("""
                        SELECT e.lore_id, e.name, e.lore_type, r.relationship_type, r.relationship_strength 
                        FROM LoreElements e
                        JOIN LoreRelationships r ON e.lore_id = r.target_id
                        WHERE r.source_id = $1
                    """, lore_id)
                    
                    return [dict(rel) for rel in related]
                except Exception as e:
                    logging.error(f"Error fetching related elements for {lore_id}: {e}")
                    return []
    
    async def _get_hierarchy_position(self, element: Dict[str, Any]) -> int:
        """Determine element's position in the power hierarchy"""
        # Implementation would calculate or retrieve hierarchy position
        # For now, return a default based on lore_type
        if element.get('lore_type', '') == 'character':
            # Check if character has a stored hierarchy value
            if 'hierarchy_position' in element:
                return element['hierarchy_position']
            else:
                # Default based on name keywords
                name = element.get('name', '').lower()
                if any(title in name for title in ['queen', 'empress', 'matriarch', 'high', 'supreme']):
                    return 1
                elif any(title in name for title in ['princess', 'duchess', 'lady', 'noble']):
                    return 3
                elif any(title in name for title in ['advisor', 'minister', 'council']):
                    return 5
                else:
                    return 8
        elif element.get('lore_type', '') == 'faction':
            # Check faction importance
            if 'importance' in element:
                return max(1, 10 - element['importance'])
            else:
                return 4  # Default for factions
        elif element.get('lore_type', '') == 'location':
            # Check location significance
            if 'significance' in element:
                return max(1, 10 - element['significance'])
            else:
                return 6  # Default for locations
        else:
            return 5  # Default middle position
    
    async def _fetch_element_update_history(self, lore_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent update history for an element"""
        if not lore_id:
            return []
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Example query - adjust based on your schema
                    history = await conn.fetch("""
                        SELECT timestamp, update_reason
                        FROM LoreUpdates
                        WHERE lore_id = $1
                        ORDER BY timestamp DESC
                        LIMIT $2
                    """, lore_id, limit)
                    
                    return [dict(update) for update in history]
                except Exception as e:
                    logging.error(f"Error fetching update history for {lore_id}: {e}")
                    return []
    
    async def _fetch_elements_by_ids(self, element_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple elements by their IDs"""
        if not element_ids:
            return []
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Example query - adjust based on your schema
                    elements = await conn.fetch("""
                        SELECT lore_id, name, lore_type, description
                        FROM LoreElements
                        WHERE lore_id = ANY($1)
                    """, element_ids)
                    
                    return [dict(elem) for elem in elements]
                except Exception as e:
                    logging.error(f"Error fetching elements by IDs: {e}")
                    return []
    
    async def _update_world_power_balance(self, power_shifts: Dict[str, float]) -> None:
        """Update the world state to reflect power shifts"""
        if not power_shifts:
            return
            
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                try:
                    # Get current power hierarchy
                    world_state = await self._fetch_world_state()
                    power_hierarchy = world_state.get('power_hierarchy', {})
                    
                    # Update power values
                    for faction_id, shift in power_shifts.items():
                        current_power = power_hierarchy.get(faction_id, 5)
                        new_power = max(1, min(10, current_power + shift))
                        power_hierarchy[faction_id] = new_power
                    
                    # Update the database
                    await conn.execute("""
                        UPDATE WorldState
                        SET power_hierarchy = $1,
                            last_updated = $2
                        WHERE user_id = $3 AND conversation_id = $4
                    """, json.dumps(power_hierarchy), datetime.datetime.now(), 
                    self.user_id, self.conversation_id)
                except Exception as e:
                    logging.error(f"Error updating world power balance: {e}")
    
      
    # Register this system with governance
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_dynamics",
            directive_text="Evolve, expand, and develop world lore through emergent events and natural maturation.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        
        logging.info(f"LoreDynamicsSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")

#===========================================================================
    # GEOPOLITICAL INTERACTION METHODS
    #===========================================================================
    
    async def _handle_nation_relationships(self, nation_id: int, event_impact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update relationships between nations based on an event impact
        
        Args:
            nation_id: ID of the primary nation involved
            event_impact: Impact data from the event
            
        Returns:
            Dictionary of relationship changes
        """
        relationship_changes = {}
        
        # Get nation relationships
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get current relations
                relations = await conn.fetch("""
                    SELECT r.nation1_id, r.nation2_id, r.relationship_type, r.relationship_quality,
                           n1.name as nation1_name, n2.name as nation2_name
                    FROM InternationalRelations r
                    JOIN Nations n1 ON r.nation1_id = n1.id
                    JOIN Nations n2 ON r.nation2_id = n2.id
                    WHERE r.nation1_id = $1 OR r.nation2_id = $1
                """, nation_id)
                
                # Determine severity of impact
                severity = event_impact.get('stability_impact', 5)
                
                # Update each relationship based on the event impact
                for relation in relations:
                    # Determine which nation is the other one
                    other_nation_id = relation['nation1_id'] if relation['nation2_id'] == nation_id else relation['nation2_id']
                    other_nation_name = relation['nation1_name'] if relation['nation2_id'] == nation_id else relation['nation2_name']
                    
                    # Calculate change based on relationship type and event severity
                    rel_type = relation['relationship_type']
                    rel_quality = relation['relationship_quality']
                    
                    # Bigger impacts make bigger changes
                    change_magnitude = severity / 10
                    
                    if rel_type == 'ally':
                        # Allies become stronger in conflicts, weaker in scandals
                        if 'conflict' in event_impact.get('event_type', '').lower():
                            change = change_magnitude * 0.5  # Strengthen alliance
                        elif 'scandal' in event_impact.get('event_type', '').lower():
                            change = -change_magnitude * 0.3  # Weaken alliance
                        else:
                            change = change_magnitude * 0.1  # Slight strengthen
                    elif rel_type == 'rival':
                        # Rivals become more hostile in conflicts
                        if 'conflict' in event_impact.get('event_type', '').lower():
                            change = -change_magnitude * 0.7  # Intensify rivalry
                        else:
                            change = -change_magnitude * 0.2  # Slight intensify
                    else:  # neutral
                        # Neutral relationships could go either way
                        if random.random() > 0.5:
                            change = change_magnitude * 0.2  # Improve
                        else:
                            change = -change_magnitude * 0.2  # Worsen
                    
                    # Apply the change
                    new_quality = max(1, min(10, rel_quality + change))
                    
                    # Update the relationship
                    await conn.execute("""
                        UPDATE InternationalRelations
                        SET relationship_quality = $1
                        WHERE (nation1_id = $2 AND nation2_id = $3)
                           OR (nation1_id = $3 AND nation2_id = $2)
                    """, new_quality, nation_id, other_nation_id)
                    
                    # Record the change
                    relationship_changes[other_nation_name] = {
                        "old_quality": rel_quality,
                        "new_quality": new_quality,
                        "change": change
                    }
        
        return relationship_changes
    
    async def _create_nation_description(self, nation_data: Dict[str, Any]) -> str:
        """
        Create a detailed nation description based on its characteristics
        
        Args:
            nation_data: Core nation data
            
        Returns:
            Detailed nation description
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Create a prompt for the LLM
        prompt = f"""
        Create a detailed description for this nation in a matriarchal fantasy world:
        
        NATION DATA:
        {json.dumps(nation_data, indent=2)}
        
        Write a rich, detailed description that:
        1. Captures the nation's character and feel
        2. Emphasizes its matriarchal elements (level {nation_data.get('matriarchy_level', 5)}/10)
        3. Describes its governance, culture, and notable features
        4. Mentions its resources and relationship with neighbors
        5. Has a consistent and engaging tone
        6. Is 3-4 paragraphs in length
        
        Return only the description text with no preamble or explanation.
        """
        
        # Create an agent for nation description
        nation_agent = Agent(
            name="NationDescriptionAgent",
            instructions="You create rich descriptions for nations in fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(nation_agent, prompt, context=run_ctx.context)
        return result.final_output
    
    #===========================================================================
    # ADDITIONAL EVENT PROMPT METHODS
    #===========================================================================
    
    def _create_military_conflict_prompt(self, faction_data, nation_data, location_data):
        """Create a prompt for military conflict events"""
        return f"""
        Generate a military conflict event for this world:
        
        POTENTIAL NATIONS INVOLVED:
        {json.dumps(nation_data, indent=2)}
        
        POTENTIAL FACTIONS INVOLVED:
        {json.dumps(faction_data, indent=2)}
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        Create a military conflict that:
        1. Involves clear aggressor and defender sides
        2. Has a specific cause and objective
        3. Creates opportunities for heroism and character development
        4. Shows how warfare operates in a matriarchal society
        
        Return a JSON object with:
        - event_name: Name of the conflict
        - event_type: "military_conflict"
        - description: Detailed description
        - aggressor: Entity that initiated the conflict
        - defender: Entity defending against aggression
        - location: Where the conflict is primarily occurring
        - cause: What caused the conflict
        - scale: Scale of the conflict (skirmish, battle, war, etc.)
        - current_status: Current status of the conflict
        - casualties: Description of casualties
        - affected_lore_categories: Array of lore categories this would affect
        """
    
    def _create_natural_disaster_prompt(self, location_data):
        """Create a prompt for natural disaster events"""
        return f"""
        Generate a natural disaster event for this world:
        
        POTENTIAL LOCATIONS AFFECTED:
        {json.dumps(location_data, indent=2)}
        
        Create a natural disaster that:
        1. Has a significant impact on the affected area
        2. Could have supernatural or magical elements
        3. Creates opportunities for societal response that highlights values
        4. Changes the physical landscape in some way
        
        Return a JSON object with:
        - event_name: Name of the disaster
        - event_type: "natural_disaster"
        - description: Detailed description
        - disaster_type: Type of disaster (earthquake, flood, magical storm, etc.)
        - primary_location: Primary location affected
        - secondary_locations: Array of secondarily affected locations
        - severity: Severity level (1-10)
        - immediate_impact: Description of immediate impact
        - response: How communities and leadership are responding
        - supernatural_elements: Any supernatural aspects (if applicable)
        - affected_lore_categories: Array of lore categories this would affect
        """
    
    def _create_cultural_development_prompt(self, faction_data, location_data):
        """Create a prompt for cultural development events"""
        return f"""
        Generate a cultural development event for this world:
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        POTENTIAL FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        Create a cultural development that:
        1. Introduces a new tradition, art form, or cultural practice
        2. Comes from a specific community or demographic
        3. Has meaning within the matriarchal power structure
        4. Could spread to other communities
        
        Return a JSON object with:
        - event_name: Name of the cultural development
        - event_type: "cultural_development"
        - description: Detailed description
        - development_type: Type of development (tradition, art form, practice, etc.)
        - originating_group: Group where it originated
        - significance: Social significance
        - symbolism: Symbolic meaning
        - reception: How different groups have received it
        - spread_potential: Likelihood of spreading (1-10)
        - affected_lore_categories: Array of lore categories this would affect
        """
    
    def _create_technological_advancement_prompt(self, faction_data, location_data):
        """Create a prompt for technological advancement events"""
        return f"""
        Generate a technological or magical advancement event for this world:
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        POTENTIAL FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        Create a technological or magical advancement that:
        1. Changes how something is done in the world
        2. Was developed by specific individuals or groups
        3. Has both benefits and potential drawbacks
        4. Reinforces matriarchal control of knowledge or resources
        
        Return a JSON object with:
        - event_name: Name of the advancement
        - event_type: "technological_advancement"
        - description: Detailed description
        - advancement_type: Type (technological, magical, alchemical, etc.)
        - creator: Who created or discovered it
        - applications: Potential applications
        - limitations: Current limitations
        - control: Who controls access to this advancement
        - societal_impact: How it impacts society
        - affected_lore_categories: Array of lore categories this would affect
        """
    
    def _create_religious_event_prompt(self, location_data):
        """Create a prompt for religious events"""
        return f"""
        Generate a religious event for this world:
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        Create a religious event that:
        1. Has significance within an existing faith system
        2. Could be interpreted as divine intervention or revelation
        3. Affects religious practices or beliefs
        4. Reinforces the feminine divine nature of the world
        
        Return a JSON object with:
        - event_name: Name of the religious event
        - event_type: "religious_event"
        - description: Detailed description
        - event_category: Type of event (miracle, prophecy, divine manifestation, etc.)
        - location: Where it occurred
        - witnesses: Who witnessed it
        - religious_significance: Significance to believers
        - skeptic_explanation: How skeptics explain it
        - faith_impact: How it impacts faith practices
        - affected_lore_categories: Array of lore categories this would affect
        """
    
    def _create_economic_change_prompt(self, faction_data, nation_data):
        """Create a prompt for economic change events"""
        return f"""
        Generate an economic change event for this world:
        
        POTENTIAL NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        POTENTIAL FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        Create an economic change that:
        1. Shifts resource distribution or trade patterns
        2. Affects multiple groups or regions
        3. Creates winners and losers
        4. Shows how economic power relates to matriarchal control
        
        Return a JSON object with:
        - event_name: Name of the economic change
        - event_type: "economic_change"
        - description: Detailed description
        - change_type: Type of change (trade shift, resource discovery, currency change, etc.)
        - primary_causes: What caused the change
        - beneficiaries: Who benefits
        - disadvantaged: Who is disadvantaged
        - resource_involved: Primary resource or commodity involved
        - wealth_redistribution: How wealth is being redistributed
        - affected_lore_categories: Array of lore categories this would affect
        """
    
    def _create_diplomatic_incident_prompt(self, faction_data, nation_data):
        """Create a prompt for diplomatic incident events"""
        return f"""
        Generate a diplomatic incident for this world:
        
        POTENTIAL NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        POTENTIAL FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        Create a diplomatic incident that:
        1. Creates tension between two or more groups
        2. Stems from a specific action or misunderstanding
        3. Requires diplomatic resolution
        4. Highlights cultural differences or values
        
        Return a JSON object with:
        - event_name: Name of the diplomatic incident
        - event_type: "diplomatic_incident"
        - description: Detailed description
        - parties_involved: Array of involved parties
        - instigating_action: What triggered the incident
        - cultural_factors: Cultural factors at play
        - severity: Severity level (1-10)
        - potential_resolutions: Possible ways to resolve it
        - current_status: Current diplomatic status
        - affected_lore_categories: Array of lore categories this would affect
        """
    
    def _create_event_prompt(self, event_type, faction_data, nation_data, location_data):
        """
        Create a tailored prompt for generating a specific type of event
        
        Args:
            event_type: Type of event to generate
            faction_data: Faction context data
            nation_data: Nation context data
            location_data: Location context data
            
        Returns:
            Prompt string
        """
        if event_type == "political_shift":
            entities = faction_data if faction_data else nation_data
            return f"""
            Generate a political shift event for this world:
            
            POTENTIAL ENTITIES INVOLVED:
            {json.dumps(entities, indent=2)}
            
            Create a significant political shift event that:
            1. Changes power dynamics in a meaningful way
            2. Is specific and detailed enough to impact the world
            3. Reinforces or challenges matriarchal power structures
            4. Could lead to interesting narrative developments
            
            Return a JSON object with:
            - event_name: Name of the event
            - event_type: "political_shift"
            - description: Detailed description
            - entities_involved: Array of entity names involved
            - instigator: The primary instigating entity
            - immediate_consequences: Array of immediate consequences
            - potential_long_term_effects: Array of potential long-term effects
            - affected_lore_categories: Array of lore categories this would affect
            """
        elif event_type == "military_conflict":
            return self._create_military_conflict_prompt(faction_data, nation_data, location_data)
        elif event_type == "natural_disaster":
            return self._create_natural_disaster_prompt(location_data)
        elif event_type == "cultural_development":
            return self._create_cultural_development_prompt(faction_data, location_data)
        elif event_type == "technological_advancement":
            return self._create_technological_advancement_prompt(faction_data, location_data)
        elif event_type == "religious_event":
            return self._create_religious_event_prompt(location_data)
        elif event_type == "economic_change":
            return self._create_economic_change_prompt(faction_data, nation_data)
        else:  # diplomatic_incident
            return self._create_diplomatic_incident_prompt(faction_data, nation_data)
    
    #===========================================================================
    # TERRITORY AND REGION METHODS
    #===========================================================================
    
    async def _get_region_details(self, region_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a geographic region
        
        Args:
            region_id: ID of the region
            
        Returns:
            Region details
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get region details
                region = await conn.fetchrow("""
                    SELECT * FROM GeographicRegions
                    WHERE id = $1
                """, region_id)
                
                if not region:
                    return {}
                    
                # Get locations in this region
                locations = await conn.fetch("""
                    SELECT l.id, l.location_name, l.location_type, l.description
                    FROM Locations l
                    JOIN RegionLocations rl ON l.id = rl.location_id
                    WHERE rl.region_id = $1
                """, region_id)
                
                # Compile result
                result = dict(region)
                result["locations"] = [dict(loc) for loc in locations]
                
                return result
    
    async def _get_territory_conflicts(self) -> List[Dict[str, Any]]:
        """
        Get all current territorial conflicts
        
        Returns:
            List of territorial conflicts
        """
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if we have a dedicated conflicts table
                has_conflicts_table = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'territorialconflicts'
                    );
                """)
                
                if has_conflicts_table:
                    # Get from conflicts table
                    conflicts = await conn.fetch("""
                        SELECT * FROM TerritorialConflicts
                        WHERE status != 'resolved'
                    """)
                    
                    return [dict(conflict) for conflict in conflicts]
                else:
                    # Get from national conflicts with territory disputes
                    conflicts = await conn.fetch("""
                        SELECT * FROM NationalConflicts
                        WHERE conflict_type = 'territory_dispute'
                        AND status != 'resolved'
                    """)
                    
                    return [dict(conflict) for conflict in conflicts]
    
    async def _calculate_territory_changes(self, conflict_id: int, winner_id: int, loser_id: int) -> Dict[str, Any]:
        """
        Calculate territorial changes based on conflict outcome
        
        Args:
            conflict_id: ID of the conflict
            winner_id: ID of the winning nation
            loser_id: ID of the losing nation
            
        Returns:
            Dictionary of territory changes
        """
        # This would implement territory transfer logic
        # For this example, we'll use a simplified placeholder
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get conflict details
                conflict = await conn.fetchrow("""
                    SELECT * FROM NationalConflicts
                    WHERE id = $1
                """, conflict_id)
                
                if not conflict:
                    return {}
                    
                # Get winner and loser details
                winner = await conn.fetchrow("""
                    SELECT name, territory FROM Nations
                    WHERE id = $1
                """, winner_id)
                
                loser = await conn.fetchrow("""
                    SELECT name, territory FROM Nations
                    WHERE id = $1
                """, loser_id)
                
                if not winner or not loser:
                    return {}
                
                # Calculate territory changes
                winner_territory = winner['territory'] if winner['territory'] else []
                loser_territory = loser['territory'] if loser['territory'] else []
                
                # Transfer a territory (simplified)
                contested_territory = None
                for territory in loser_territory:
                    if "border" in territory.lower() or "disputed" in territory.lower():
                        contested_territory = territory
                        break
                
                if not contested_territory and loser_territory:
                    contested_territory = loser_territory[0]
                
                if contested_territory:
                    # Transfer territory
                    if contested_territory in loser_territory:
                        loser_territory.remove(contested_territory)
                    if contested_territory not in winner_territory:
                        winner_territory.append(contested_territory)
                    
                    # Update nations
                    await conn.execute("""
                        UPDATE Nations
                        SET territory = $1
                        WHERE id = $2
                    """, winner_territory, winner_id)
                    
                    await conn.execute("""
                        UPDATE Nations
                        SET territory = $1
                        WHERE id = $2
                    """, loser_territory, loser_id)
                    
                    return {
                        "transferred_territory": contested_territory,
                        "from_nation": loser['name'],
                        "to_nation": winner['name']
                    }
                else:
                    return {
                        "transferred_territory": None,
                        "from_nation": loser['name'],
                        "to_nation": winner['name'],
                        "result": "No suitable territory to transfer"
                    }
    
    #===========================================================================
    # INTEGRATION WITH OTHER LORE SYSTEMS
    #===========================================================================
    
    async def get_consolidated_world_state(self) -> Dict[str, Any]:
        """
        Get a consolidated view of the current world state for use by other systems
        
        Returns:
            Dictionary with comprehensive world state
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Collect data from various systems
        world_state = await self._fetch_world_state()
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        # Get active conflicts
        conflicts = []
        try:
            conflicts_system = NationalConflictSystem(self.user_id, self.conversation_id)
            conflicts = await conflicts_system.get_active_conflicts(run_ctx)
        except Exception as e:
            logging.error(f"Error getting conflicts: {e}")
            
        # Get major factions
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get major factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description 
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 5
                """)
                
                faction_data = [dict(faction) for faction in factions]
                
                # Get recent events
                recent_events = await conn.fetch("""
                    SELECT id, name, date_description, description
                    FROM HistoricalEvents
                    ORDER BY id DESC
                    LIMIT 3
                """)
                
                event_data = [dict(event) for event in recent_events]
        
        # Compile comprehensive state
        return {
            "world_state": world_state,
            "nations": nations,
            "active_conflicts": conflicts,
            "major_factions": faction_data,
            "recent_events": event_data,
            "stability_index": world_state.get('stability_index', 8),
            "power_dynamics": world_state.get('power_dynamics', 'strict_hierarchy'),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    async def integrate_character_lore(self, character_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate character data with world lore
        
        Args:
            character_data: Character information
            
        Returns:
            Dictionary with character lore integration results
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Generate potential connections to existing lore
        connections = []
        
        # Check for potential faction connections
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Find factions that might connect to this character
                character_description = character_data.get('description', '')
                character_embedding = await generate_embedding(character_description)
                
                faction_matches = await conn.fetch("""
                    SELECT id, name, type, description,
                           1 - (embedding <=> $1) as relevance
                    FROM Factions
                    WHERE 1 - (embedding <=> $1) > 0.6
                    ORDER BY relevance DESC
                    LIMIT 3
                """, character_embedding)
                
                for faction in faction_matches:
                    connections.append({
                        "type": "faction",
                        "entity_id": faction['id'],
                        "entity_name": faction['name'],
                        "entity_type": faction['type'],
                        "relevance": faction['relevance'],
                        "suggested_relationship": self._suggest_character_faction_relationship(
                            character_data, dict(faction)
                        )
                    })
                
                # Find locations that might connect to this character
                location_matches = await conn.fetch("""
                    SELECT id, location_name, location_type, description,
                           1 - (embedding <=> $1) as relevance
                    FROM Locations
                    WHERE 1 - (embedding <=> $1) > 0.6
                    ORDER BY relevance DESC
                    LIMIT 3
                """, character_embedding)
                
                for location in location_matches:
                    connections.append({
                        "type": "location",
                        "entity_id": location['id'],
                        "entity_name": location['location_name'],
                        "entity_type": location['location_type'],
                        "relevance": location['relevance'],
                        "suggested_relationship": self._suggest_character_location_relationship(
                            character_data, dict(location)
                        )
                    })
        
        # Generate appropriate cultural background
        cultural_background = await self._generate_character_cultural_background(
            run_ctx, character_data
        )
        
        # Determine character's position in the matriarchal hierarchy
        hierarchy_position = self._calculate_character_hierarchy_position(character_data)
        
        return {
            "character_id": character_data.get('id'),
            "name": character_data.get('name'),
            "suggested_connections": connections,
            "cultural_background": cultural_background,
            "hierarchy_position": hierarchy_position,
            "suggested_story_hooks": await self._generate_character_story_hooks(
                run_ctx, character_data, connections
            )
        }
    
    def _suggest_character_faction_relationship(
        self, 
        character_data: Dict[str, Any], 
        faction_data: Dict[str, Any]
    ) -> str:
        """Suggest relationship between character and faction"""
        gender = character_data.get('gender', 'unknown').lower()
        faction_type = faction_data.get('type', '').lower()
        
        if gender == 'female':
            if 'leadership' in faction_type or 'noble' in faction_type:
                return "potential member or leader"
            elif 'military' in faction_type:
                return "officer or strategist"
            elif 'religious' in faction_type:
                return "priestess or devotee"
            else:
                return "respected affiliate"
        elif gender == 'male':
            if 'service' in faction_type or 'support' in faction_type:
                return "dedicated servant or aide"
            elif 'military' in faction_type:
                return "soldier or support personnel"
            elif 'religious' in faction_type:
                return "acolyte or temple guardian"
            else:
                return "subordinate member"
        else:
            return "potential associate"
    
    def _suggest_character_location_relationship(
        self, 
        character_data: Dict[str, Any], 
        location_data: Dict[str, Any]
    ) -> str:
        """Suggest relationship between character and location"""
        gender = character_data.get('gender', 'unknown').lower()
        location_type = location_data.get('location_type', '').lower()
        
        if gender == 'female':
            if 'city' in location_type or 'settlement' in location_type:
                return "resident noble or authority figure"
            elif 'temple' in location_type or 'sacred' in location_type:
                return "high priestess or official"
            elif 'wilderness' in location_type or 'natural' in location_type:
                return "guardian or expert guide"
            else:
                return "notable visitor"
        elif gender == 'male':
            if 'city' in location_type or 'settlement' in location_type:
                return "resident servant or tradesperson"
            elif 'temple' in location_type or 'sacred' in location_type:
                return "devotee or caretaker"
            elif 'wilderness' in location_type or 'natural' in location_type:
                return "scout or harvester"
            else:
                return "frequent visitor"
        else:
            return "has connections to this place"
    
    async def _generate_character_cultural_background(
        self, 
        ctx, 
        character_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate appropriate cultural background for a character"""
        # Create an agent for cultural background generation
        culture_agent = Agent(
            name="CharacterCultureAgent",
            instructions="You create cultural backgrounds for characters in a matriarchal fantasy world.",
            model="o3-mini"
        )
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a cultural background for this character in a matriarchal fantasy world:
        
        CHARACTER:
        {json.dumps(character_data, indent=2)}
        
        Create a detailed cultural background that:
        1. Fits the character's background and traits
        2. Reflects their position in a matriarchal society based on their gender
        3. Includes customs, traditions, and beliefs they grew up with
        4. Explains how their upbringing shapes their worldview
        5. Includes at least one unique cultural tradition or practice
        
        Return a JSON object with:
        - home_culture: Name of their home culture/region
        - cultural_values: Array of core values from their culture
        - gender_expectations: How their gender is viewed in their culture
        - traditions: Array of traditions or practices they're familiar with
        - taboos: Array of taboos or restrictions from their culture
        - cultural_quirks: Unique mannerisms or behaviors from their culture
        - cultural_influences: How their culture affects their personality
        """
        
        # Get the response
        result = await Runner.run(culture_agent, prompt, context=ctx.context)
        
        try:
            # Parse the JSON response
            return json.loads(result.final_output)
        except json.JSONDecodeError:
            # Return a simplified fallback
            return {
                "home_culture": "Unknown",
                "cultural_values": ["respect", "tradition", "hierarchy"],
                "gender_expectations": self._get_default_gender_expectations(character_data),
                "traditions": ["seasonal ceremonies", "respect rituals"],
                "taboos": ["disrespect to authority", "violation of gender norms"],
                "cultural_quirks": ["particular greeting style", "formal speech patterns"],
                "cultural_influences": "Shaped by their cultural upbringing and experiences"
            }
    
    def _get_default_gender_expectations(self, character_data: Dict[str, Any]) -> str:
        """Get default gender expectations based on character gender"""
        gender = character_data.get('gender', 'unknown').lower()
        
        if gender == 'female':
            return "Expected to develop leadership skills, authority, and decision-making abilities."
        elif gender == 'male':
            return "Expected to be supportive, deferential to female authority, and excel in service roles."
        else:
            return "Expected to find their own path within the matriarchal structure."
    
    def _calculate_character_hierarchy_position(self, character_data: Dict[str, Any]) -> int:
        """Calculate character's position in the matriarchal hierarchy (1-10)"""
        gender = character_data.get('gender', 'unknown').lower()
        background = character_data.get('background', '').lower()
        traits = character_data.get('traits', [])
        description = character_data.get('description', '').lower()
        
        # Calculate base position
        if gender == 'female':
            base_position = 4  # Default female position
            
            # Adjust based on background
            if 'noble' in background or 'aristocrat' in background:
                base_position -= 2
            elif 'leader' in background or 'ruler' in background:
                base_position -= 3
            elif 'peasant' in background or 'commoner' in background:
                base_position += 1
                
            # Adjust based on traits
            if 'authoritative' in traits or 'commanding' in traits:
                base_position -= 1
            if 'submissive' in traits or 'meek' in traits:
                base_position += 2
                
            # Check for keywords in description
            if 'queen' in description or 'empress' in description:
                base_position = 1
            elif 'princess' in description or 'duchess' in description:
                base_position = 2
                
        elif gender == 'male':
            base_position = 7  # Default male position
            
            # Adjust based on background
            if 'noble' in background or 'aristocrat' in background:
                base_position -= 1
            elif 'servant' in background or 'slave' in background:
                base_position += 2
                
            # Adjust based on traits
            if 'obedient' in traits or 'submissive' in traits:
                base_position += 1
            if 'rebellious' in traits or 'defiant' in traits:
                base_position -= 1
                
            # Check for keywords in description
            if 'consort' in description or 'favored' in description:
                base_position = 5
            elif 'rebel' in description:
                base_position = 6
                
        else:
            base_position = 5  # Default other gender position
            
        # Ensure within bounds
        return max(1, min(10, base_position))
    
    async def _generate_character_story_hooks(
        self,
        ctx,
        character_data: Dict[str, Any],
        connections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate story hooks for a character based on their connections"""
        # Create an agent for story hook generation
        hook_agent = Agent(
            name="StoryHookAgent",
            instructions="You create story hooks for characters in a matriarchal fantasy world.",
            model="o3-mini"
        )
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 2-3 story hooks for this character in a matriarchal fantasy world:
        
        CHARACTER:
        {json.dumps(character_data, indent=2)}
        
        POTENTIAL CONNECTIONS:
        {json.dumps(connections, indent=2)}
        
        Create story hooks that:
        1. Relate to the character's background, traits, and connections
        2. Fit naturally within a matriarchal society
        3. Offer interesting narrative opportunities
        4. Include at least one hook that explores gender dynamics
        5. Can be played out in short or long-form scenarios
        
        Return a JSON array where each hook has:
        - title: A catchy title for the hook
        - hook_type: Type of hook (personal quest, relationship drama, duty, etc.)
        - description: Brief description of the hook
        - stakes: What's at stake for the character
        - potential_outcomes: Array of possible outcomes
        - difficulty: How challenging it would be (1-10)
        """
        
        # Get the response
        result = await Runner.run(hook_agent, prompt, context=ctx.context)
        
        try:
            # Parse the JSON response
            hooks = json.loads(result.final_output)
            
            # Ensure we got a list
            if not isinstance(hooks, list):
                if isinstance(hooks, dict):
                    hooks = [hooks]
                else:
                    hooks = []
                    
            return hooks
        except json.JSONDecodeError:
            # Return a simplified fallback
            return [{
                "title": "Navigating the Hierarchy",
                "hook_type": "personal challenge",
                "description": "The character must navigate the complex social hierarchy of their matriarchal society.",
                "stakes": "Their social standing and future opportunities",
                "potential_outcomes": ["improved status", "valuable connections", "social disgrace"],
                "difficulty": 6
            }]
    
#===========================================================================
    # LORE UPDATE HELPER METHODS
    #===========================================================================
    
    async def _build_enhanced_lore_prompt(
        self,
        element: Dict[str, Any],
        event_description: str,
        societal_impact: Dict[str, Any],
        related_elements: List[Dict[str, Any]],
        hierarchy_position: int,
        update_history: List[Dict[str, Any]],
        player_character: Dict[str, Any] = None,
        dominant_npcs: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build a sophisticated prompt for lore updates with rich context
        
        Args:
            element: The lore element to update
            event_description: Description of the event
            societal_impact: Impact assessment on society
            related_elements: Elements connected to this one
            hierarchy_position: Position in power hierarchy (lower is more powerful)
            update_history: Recent update history for this element
            player_character: Optional player character data
            dominant_npcs: Optional list of ruling NPCs
            
        Returns:
            A detailed prompt for the LLM
        """
        # Format update history as context
        history_context = ""
        if update_history:
            history_items = []
            for update in update_history:
                history_items.append(f"- {update['timestamp']}: {update['update_reason']}")
            history_context = "UPDATE HISTORY:\n" + "\n".join(history_items)
        
        # Format related elements as context
        relationships_context = ""
        if related_elements:
            rel_items = []
            for rel in related_elements:
                rel_items.append(f"- {rel['name']} ({rel['lore_type']}): {rel['relationship_type']} - {rel['relationship_strength']}/10")
            relationships_context = "RELATIONSHIPS:\n" + "\n".join(rel_items)
        
        # Format player character context if available
        player_context = ""
        if player_character:
            player_context = f"""
            PLAYER CHARACTER CONTEXT:
            Name: {player_character['name']}
            Status: {player_character['status']}
            Recent Actions: {player_character['recent_actions']}
            Position in Hierarchy: {player_character.get('hierarchy_position', 'subordinate')}
            """
        
        # Format dominant NPCs context if available
        dominant_context = ""
        if dominant_npcs:
            dom_items = []
            for npc in dominant_npcs:
                dom_items.append(f"- {npc['name']}: {npc['position']} - {npc['attitude']} toward situation")
            dominant_context = "RELEVANT AUTHORITY FIGURES:\n" + "\n".join(dom_items)
        
        # Hierarchy-appropriate directive
        hierarchy_directive = await self._get_hierarchy_directive(hierarchy_position)
        
        # Build the complete prompt
        prompt = f"""
        The following lore element in our matriarchal-themed RPG world requires updating based on recent events:
        
        LORE ELEMENT:
        Type: {element['lore_type']}
        Name: {element['name']}
        Current Description: {element['description']}
        Position in Hierarchy: {hierarchy_position}/10 (lower number = higher authority)
        
        {relationships_context}
        
        {history_context}
        
        EVENT THAT OCCURRED:
        {event_description}
        
        SOCIETAL IMPACT ASSESSMENT:
        Stability Impact: {societal_impact['stability_impact']}/10
        Power Structure Change: {societal_impact['power_structure_change']}
        Public Perception Shift: {societal_impact['public_perception']}
        
        {player_context}
        
        {dominant_context}
        
        {hierarchy_directive}
        
        Generate a sophisticated update for this lore element that incorporates the impact of this event.
        The update should maintain narrative consistency while allowing for meaningful development.
        
        Return your response as a JSON object with:
        {{
            "new_description": "The updated description that reflects event impact",
            "update_reason": "Detailed explanation of why this update makes sense",
            "impact_level": A number from 1-10 indicating how significantly this event affects this element,
            "narrative_themes": ["List", "of", "relevant", "themes"],
            "adaptation_quality": A number from 1-10 indicating how well the element adapts to change,
            "hierarchy_changes": {{"entity_id": change_value, ...}},
            "mood_shift": "emotional tone shift (e.g. 'neutral', 'tense', 'relieved')"
        }}
        
        ADDITIONAL TYPE-SPECIFIC FIELDS:
        """
        
        # Add type-specific fields to the prompt
        if element['lore_type'] == 'character':
            prompt += """
            "character_development": {
                "confidence": 1-10,
                "resolve": 1-10,
                "ambition": 1-10
            },
            "motivation_shift": "description of how character's motivations might change"
            """
        elif element['lore_type'] == 'location':
            prompt += """
            "atmosphere_change": "how the location's feel or atmosphere changes",
            "accessibility_change": "how access to this location may have changed"
            """
        elif element['lore_type'] == 'faction':
            prompt += """
            "internal_stability": "how stable the faction is after events",
            "external_influence": "how the faction's influence has changed"
            """
        
        return prompt
    
    async def _get_hierarchy_directive(self, hierarchy_position: int) -> str:
        """
        Get an appropriate directive based on the element's position in hierarchy
        
        Args:
            hierarchy_position: Position in the power hierarchy (1-10, lower is more powerful)
            
        Returns:
            A directive string appropriate to the hierarchy level
        """
        if hierarchy_position <= 2:
            return """
            DIRECTIVE: This element represents a highest-tier authority figure. 
            Their decisions significantly impact the world. 
            They rarely change their core principles but may adjust strategies.
            They maintain control and authority in all situations.
            """
        elif hierarchy_position <= 4:
            return """
            DIRECTIVE: This element represents high authority.
            They have significant influence but answer to the highest tier.
            They strongly maintain the established order while pursuing their ambitions.
            They assert dominance in their domain but show deference to higher authority.
            """
        elif hierarchy_position <= 7:
            return """
            DIRECTIVE: This element has mid-level authority.
            They implement the will of higher authorities while managing those below.
            They may have personal aspirations but function within established boundaries.
            They balance compliance with higher authority against control of subordinates.
            """
        else:
            return """
            DIRECTIVE: This element has low authority in the hierarchy.
            They follow directives from above and have limited autonomy.
            They may seek to improve their position but must navigate carefully.
            They show appropriate deference to those of higher status.
            """
    
    async def _parse_lore_update_response(self, response_text: str, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM response with advanced error handling
        
        Args:
            response_text: Raw response from the LLM
            element: Original lore element
            
        Returns:
            Parsed update data
        """
        try:
            # First try to parse as JSON
            update_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['new_description', 'update_reason', 'impact_level']
            for field in required_fields:
                if field not in update_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return update_data
            
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON response for {element['name']}")
            
            # Try regex extraction for common patterns
            patterns = {
                'new_description': r'"new_description"\s*:\s*"([^"]+)"',
                'update_reason': r'"update_reason"\s*:\s*"([^"]+)"',
                'impact_level': r'"impact_level"\s*:\s*(\d+)',
                'narrative_themes': r'"narrative_themes"\s*:\s*\[(.*?)\]',
                'mood_shift': r'"mood_shift"\s*:\s*"([^"]+)"'
            }
            
            extracted_data = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    if key == 'impact_level':
                        extracted_data[key] = int(match.group(1))
                    elif key == 'narrative_themes':
                        themes_str = match.group(1)
                        themes = [t.strip().strip('"\'') for t in themes_str.split(',')]
                        extracted_data[key] = themes
                    else:
                        extracted_data[key] = match.group(1)
            
            # Fill in missing required fields with defaults
            if 'new_description' not in extracted_data:
                # Find the longest paragraph as a fallback description
                paragraphs = re.split(r'\n\n+', response_text)
                paragraphs = [p for p in paragraphs if len(p) > 50]
                if paragraphs:
                    extracted_data['new_description'] = max(paragraphs, key=len)
                else:
                    extracted_data['new_description'] = element['description']
            
            if 'update_reason' not in extracted_data:
                extracted_data['update_reason'] = "Event impact (extracted from unstructured response)"
                
            if 'impact_level' not in extracted_data:
                # Look for numbers in text that might indicate impact level
                numbers = re.findall(r'\b([1-9]|10)\b', response_text)
                if numbers:
                    extracted_data['impact_level'] = int(numbers[0])
                else:
                    extracted_data['impact_level'] = 5
            
            return extracted_data
    
    async def _calculate_cascade_effects(
        self, 
        element: Dict[str, Any], 
        update_data: Dict[str, Any],
        related_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate how updates to one element affect related elements
        
        Args:
            element: The updated lore element
            update_data: The update data for this element
            related_elements: Elements related to this one
            
        Returns:
            Dictionary of cascade effects
        """
        cascade_effects = {
            'relationship_changes': {},
            'power_shifts': {}
        }
        
        impact_level = update_data.get('impact_level', 5)
        
        # Calculate relationship changes based on impact
        for related in related_elements:
            rel_id = related.get('lore_id', '')
            if not rel_id:
                continue
                
            rel_strength = related.get('relationship_strength', 5)
            rel_type = related.get('relationship_type', 'neutral')
            
            # Calculate relationship change based on event impact and current relationship
            # Higher impact events cause more relationship change
            if rel_type in ['subservient', 'loyal']:
                # Loyal/subservient relationships strengthen during impactful events
                change = (impact_level - 5) * 0.3
            elif rel_type in ['authority', 'dominant']:
                # Authority relationships may weaken slightly during high-impact events
                change = (5 - impact_level) * 0.2
            elif rel_type in ['rival', 'adversarial']:
                # Rivalries intensify during impactful events
                change = -abs(impact_level - 5) * 0.4
            else:
                # Neutral relationships shift based on impact direction
                change = (impact_level - 5) * 0.1
            
            # Adjust for hierarchy differences - larger gaps mean more significant changes
            hierarchy_diff = abs(
                element.get('hierarchy_position', 5) - 
                related.get('hierarchy_position', 5)
            )
            
            change *= (1 + (hierarchy_diff * 0.1))
            
            cascade_effects['relationship_changes'][rel_id] = round(change, 1)
        
        # Calculate power shifts for relevant factions
        if element.get('lore_type', '') == 'faction':
            # Direct power shift for the affected faction
            faction_id = element.get('lore_id', '')
            if faction_id:
                power_shift = (impact_level - 5) * 0.5
                cascade_effects['power_shifts'][faction_id] = power_shift
        
        # Calculate power shifts for factions related to the element
        faction_relations = [r for r in related_elements if r.get('lore_type', '') == 'faction']
        for faction in faction_relations:
            faction_id = faction.get('lore_id', '')
            if not faction_id:
                continue
                
            rel_type = faction.get('relationship_type', 'neutral')
            
            # Calculate power shift based on relationship type
            if rel_type in ['allied', 'supportive']:
                # Allied factions shift in the same direction
                shift = (impact_level - 5) * 0.3
            elif rel_type in ['rival', 'opposed']:
                # Rival factions shift in the opposite direction
                shift = (5 - impact_level) * 0.3
            else:
                # Neutral factions have minimal shifts
                shift = (impact_level - 5) * 0.1
                
            cascade_effects['power_shifts'][faction_id] = round(shift, 1)
        
        return cascade_effects
    
    async def _generate_cascade_updates(
        self,
        cascade_elements: List[Dict[str, Any]],
        event_description: str,
        relationship_changes: Dict[str, float],
        power_shifts: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate simplified updates for elements affected indirectly
        
        Args:
            cascade_elements: List of elements affected by cascade
            event_description: Original event description
            relationship_changes: Dictionary of relationship changes by ID
            power_shifts: Dictionary of power shifts by faction ID
            
        Returns:
            List of update records for cascade elements
        """
        cascade_updates = []
        
        for element in cascade_elements:
            element_id = element.get('lore_id', '')
            if not element_id:
                continue
                
            # Get the relationship change if any
            rel_change = relationship_changes.get(element_id, 0)
            
            # Get power shift if this is a faction
            power_shift = 0
            if element.get('lore_type', '') == 'faction':
                power_shift = power_shifts.get(element_id, 0)
            
            # Calculate impact level based on relationship change and power shift
            impact_level = min(10, max(1, round(5 + abs(rel_change) * 2 + abs(power_shift) * 2)))
            
            # Generate a simplified update
            if abs(rel_change) > 1 or abs(power_shift) > 1:
                # Significant enough to warrant an update
                
                # Determine the nature of the update based on changes
                if element.get('lore_type', '') == 'character':
                    if rel_change > 0:
                        update_reason = f"Strengthened position due to recent events"
                        description_modifier = "more confident and assured"
                    else:
                        update_reason = f"Position weakened by recent events"
                        description_modifier = "more cautious and reserved"
                
                elif element.get('lore_type', '') == 'faction':
                    if power_shift > 0:
                        update_reason = f"Gained influence following recent events"
                        description_modifier = "increasing their authority and reach"
                    else:
                        update_reason = f"Lost influence due to recent events"
                        description_modifier = "adapting to their diminished standing"
                
                elif element.get('lore_type', '') == 'location':
                    if rel_change > 0:
                        update_reason = f"Increased importance after recent events"
                        description_modifier = "now sees more activity and attention"
                    else:
                        update_reason = f"Decreased importance after recent events"
                        description_modifier = "now sees less activity and attention"
                
                else:
                    update_reason = f"Indirectly affected by recent events"
                    description_modifier = "subtly changed by recent developments"
                
                # Create a new description with the modifier
                new_description = element.get('description', '')
                if "." in new_description:
                    parts = new_description.split(".")
                    parts[-2] += f", {description_modifier}"
                    new_description = ".".join(parts)
                else:
                    new_description = f"{new_description} {description_modifier}."
                
                # Create update record
                cascade_updates.append({
                    'lore_type': element.get('lore_type', ''),
                    'lore_id': element_id,
                    'name': element.get('name', ''),
                    'old_description': element.get('description', ''),
                    'new_description': new_description,
                    'update_reason': update_reason,
                    'impact_level': impact_level,
                    'is_cascade_update': True,
                    'timestamp': datetime.datetime.now().isoformat()
                })
        
        return cascade_updates
    
    async def _calculate_societal_impact(
        self,
        event_description: str,
        stability_index: int,
        power_hierarchy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the societal impact of an event
        
        Args:
            event_description: Description of the event
            stability_index: Current stability of society (1-10)
            power_hierarchy: Current power structure data
            
        Returns:
            Dictionary of societal impact metrics
        """
        # Analyze event text for impact keywords
        impact_keywords = {
            'high_impact': [
                'overthrown', 'revolution', 'usurped', 'conquered', 'rebellion',
                'assassination', 'coup', 'catastrophe', 'disaster'
            ],
            'medium_impact': [
                'challenge', 'conflict', 'dispute', 'tension', 'unrest',
                'scandal', 'controversy', 'uprising', 'demonstration'
            ],
            'low_impact': [
                'minor', 'small', 'limited', 'isolated', 'contained',
                'private', 'personal', 'individual', 'trivial'
            ]
        }
        
        # Count keyword occurrences
        high_count = sum(1 for word in impact_keywords['high_impact'] if word.lower() in event_description.lower())
        medium_count = sum(1 for word in impact_keywords['medium_impact'] if word.lower() in event_description.lower())
        low_count = sum(1 for word in impact_keywords['low_impact'] if word.lower() in event_description.lower())
        
        # Calculate base stability impact
        if high_count > 0:
            base_stability_impact = 7 + min(high_count, 3)  # Max 10
        elif medium_count > 0:
            base_stability_impact = 4 + min(medium_count, 3)  # Max 7
        elif low_count > 0:
            base_stability_impact = 2 + min(low_count, 2)  # Max 4
        else:
            base_stability_impact = 3  # Default moderate impact
        
        # Adjust for current stability
        # Higher stability means events have less impact
        stability_modifier = (10 - stability_index) / 10
        adjusted_impact = base_stability_impact * (0.5 + stability_modifier)
        
        # Determine power structure change
        if adjusted_impact >= 8:
            power_change = "significant realignment of authority"
        elif adjusted_impact >= 6:
            power_change = "moderate shift in power dynamics"
        elif adjusted_impact >= 4:
            power_change = "subtle adjustments to authority structures"
        else:
            power_change = "minimal change to established order"
        
        # Determine public perception
        if adjusted_impact >= 7:
            if "rebellion" in event_description.lower() or "uprising" in event_description.lower():
                perception = "widespread questioning of authority"
            else:
                perception = "significant public concern"
        elif adjusted_impact >= 5:
            perception = "notable public interest and discussion"
        else:
            perception = "limited public awareness or interest"
        
        return {
            'stability_impact': round(adjusted_impact),
            'power_structure_change': power_change,
            'public_perception': perception
        }
    
    #===========================================================================
    # URBAN MYTH AND CULTURE EVOLUTION METHODS
    #===========================================================================
    
    async def _evolve_urban_myths(self) -> List[Dict[str, Any]]:
        """
        Evolve urban myths over time - they grow, change, or fade
        
        Returns:
            List of changes to urban myths
        """
        changes = []
        
        # Choose a random sample of urban myths to evolve
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if we have a dedicated urban myths table
                has_urban_myths_table = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'urbanmyths'
                    );
                """)
                
                if has_urban_myths_table:
                    # Get myths from dedicated table
                    rows = await conn.fetch("""
                        SELECT id, name, description, believability, spread_rate
                        FROM UrbanMyths
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                else:
                    # Fall back to world lore with urban_myth category
                    rows = await conn.fetch("""
                        SELECT id, name, description
                        FROM WorldLore
                        WHERE category = 'urban_myth'
                        ORDER BY RANDOM()
                        LIMIT 3
                    """)
                
                for row in rows:
                    myth_id = row['id']
                    myth_name = row['name']
                    old_description = row['description']
                    
                    # Determine change type
                    change_type = random.choice(["grow", "evolve", "fade"])
                    
                    # Customize change based on type
                    if change_type == "grow":
                        # Myth becomes more widespread or elaborate
                        new_believability = min(10, row.get('believability', 5) + random.randint(1, 2))
                        new_spread = min(10, row.get('spread_rate', 5) + random.randint(1, 2))
                        
                        # Use LLM to expand the myth
                        run_ctx = RunContextWrapper(context={
                            "user_id": self.user_id,
                            "conversation_id": self.conversation_id
                        })
                        
                        prompt = f"""
                        This urban myth is growing in popularity and becoming more elaborate:
                        
                        MYTH: {myth_name}
                        CURRENT DESCRIPTION: {old_description}
                        
                        Expand this myth to include new details, aspects or variations that have emerged.
                        The myth should become more widespread and elaborate, but maintain its core essence.
                        
                        Return only the new expanded description, written in the same style as the original.
                        """
                        
                        myth_agent = Agent(
                            name="MythEvolutionAgent",
                            instructions="You develop and evolve urban myths over time.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        change_description = "Myth has grown in popularity and become more elaborate."
                        
                    elif change_type == "evolve":
                        # Myth changes in some significant way
                        # Use LLM to transform the myth
                        run_ctx = RunContextWrapper(context={
                            "user_id": self.user_id,
                            "conversation_id": self.conversation_id
                        })
                        
                        prompt = f"""
                        This urban myth is evolving with new variations:
                        
                        MYTH: {myth_name}
                        CURRENT DESCRIPTION: {old_description}
                        
                        Evolve this myth by changing some details while keeping its core recognizable.
                        Perhaps add a twist, change the outcome, or incorporate a new element.
                        
                        Return only the new evolved description, written in the same style as the original.
                        """
                        
                        myth_agent = Agent(
                            name="MythEvolutionAgent",
                            instructions="You develop and evolve urban myths over time.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        change_description = "Myth has evolved with new variations or interpretations."
                        
                    else:  # fade
                        # Myth becomes less prominent or believable
                        new_believability = max(1, row.get('believability', 5) - random.randint(1, 2))
                        new_spread = max(1, row.get('spread_rate', 5) - random.randint(1, 2))
                        
                        # Use LLM to show the myth fading
                        run_ctx = RunContextWrapper(context={
                            "user_id": self.user_id,
                            "conversation_id": self.conversation_id
                        })
                        
                        prompt = f"""
                        This urban myth is fading from public consciousness:
                        
                        MYTH: {myth_name}
                        CURRENT DESCRIPTION: {old_description}
                        
                        Modify this description to indicate the myth is becoming less believed or remembered.
                        Add phrases like "once widely believed" or "few now remember" to show its decline.
                        
                        Return only the new faded description, written in the same style as the original.
                        """
                        
                        myth_agent = Agent(
                            name="MythEvolutionAgent",
                            instructions="You develop and evolve urban myths over time.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        change_description = "Myth is fading from public consciousness or becoming less believed."
                    
                    # Apply the update
                    try:
                        if has_urban_myths_table:
                            # Update in dedicated table
                            await conn.execute("""
                                UPDATE UrbanMyths
                                SET description = $1,
                                    believability = $2,
                                    spread_rate = $3
                                WHERE id = $4
                            """, new_description, 
                            new_believability if 'new_believability' in locals() else row.get('believability', 5),
                            new_spread if 'new_spread' in locals() else row.get('spread_rate', 5),
                            myth_id)
                        else:
                            # Update in WorldLore
                            # Generate new embedding
                            embedding_text = f"{myth_name} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            await conn.execute("""
                                UPDATE WorldLore
                                SET description = $1, embedding = $2
                                WHERE id = $3
                            """, new_description, new_embedding, myth_id)
                        
                        # Record the change
                        changes.append({
                            "myth_id": myth_id,
                            "name": myth_name,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description,
                            "change_description": change_description
                        })
                        
                        # Clear cache
                        if has_urban_myths_table:
                            LORE_CACHE.invalidate_pattern(f"UrbanMyths_{myth_id}")
                        else:
                            LORE_CACHE.invalidate_pattern(f"WorldLore_{myth_id}")
                    except Exception as e:
                        logging.error(f"Error updating myth {myth_id}: {e}")
        
        return changes
        
    async def _develop_cultural_elements(self) -> List[Dict[str, Any]]:
        """
        Develop cultural elements over time - they formalize, adapt, or merge
        
        Returns:
            List of changes to cultural elements
        """
        changes = []
        
        # Choose a random sample of cultural elements to develop
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, name, type, description, significance, practiced_by
                    FROM CulturalElements
                    ORDER BY RANDOM()
                    LIMIT 2
                """)
                
                for row in rows:
                    element_id = row['id']
                    element_name = row['name']
                    element_type = row['type']
                    old_description = row['description']
                    old_significance = row['significance']
                    practiced_by = row['practiced_by'] if row['practiced_by'] else []
                    
                    # Determine change type
                    change_type = random.choice(["formalize", "adapt", "spread", "codify"])
                    
                    # Create the run context
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    # Customize change based on type
                    if change_type == "formalize":
                        # Cultural element becomes more formalized
                        prompt = f"""
                        This cultural element is becoming more formalized and structured:
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this cultural element has become more formalized.
                        Add details about specific rules, structures, or ceremonies that have developed.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = min(10, old_significance + 1)
                        change_description = "Element has become more formal and structured."
                        
                    elif change_type == "adapt":
                        # Cultural element adapts to changing circumstances
                        prompt = f"""
                        This cultural element is adapting to changing social circumstances:
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this cultural element has adapted to changing times.
                        Show how it maintains its essence while evolving to remain relevant.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = old_significance  # Remains the same
                        change_description = "Element has adapted to changing social circumstances."
                        
                    elif change_type == "spread":
                        # Cultural element spreads to new groups
                        new_groups = ["neighboring regions", "younger generations", "urban centers", "rural communities"]
                        added_groups = random.sample(new_groups, 1)
                        
                        # Don't add groups that are already listed
                        if practiced_by:
                            added_groups = [g for g in added_groups if g not in practiced_by]
                        
                        prompt = f"""
                        This cultural element is spreading to new groups, specifically: {', '.join(added_groups)}
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        CURRENTLY PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this cultural element has spread to these new groups.
                        How might they adapt or interpret it slightly differently?
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = min(10, old_significance + 1)
                        change_description = f"Element has spread to new groups: {', '.join(added_groups)}"
                        
                        # Update the practiced_by list
                        if practiced_by:
                            practiced_by.extend(added_groups)
                        else:
                            practiced_by = added_groups
                            
                    else:  # codify
                        # Cultural element becomes codified in writing or law
                        prompt = f"""
                        This cultural element is becoming codified into formal rules, laws, or texts:
                        
                        ELEMENT: {element_name} ({element_type})
                        CURRENT DESCRIPTION: {old_description}
                        PRACTICED BY: {', '.join(practiced_by) if practiced_by else 'Various groups'}
                        
                        Update this description to show how this practice has become codified.
                        Add details about written texts, legal codes, or formal teachings that preserve it.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_significance = min(10, old_significance + 2)
                        change_description = "Element has been codified into formal rules, laws, or texts."
                    
                    # Get updated description from LLM
                    culture_agent = Agent(
                        name="CultureEvolutionAgent",
                        instructions="You develop and evolve cultural elements over time.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(culture_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply the update
                    try:
                        # Generate new embedding
                        embedding_text = f"{element_name} {element_type} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        await conn.execute("""
                            UPDATE CulturalElements
                            SET description = $1, 
                                significance = $2, 
                                practiced_by = $3,
                                embedding = $4
                            WHERE id = $5
                        """, new_description, new_significance, practiced_by, new_embedding, element_id)
                        
                        # Record the change
                        changes.append({
                            "element_id": element_id,
                            "name": element_name,
                            "type": element_type,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description,
                            "change_description": change_description
                        })
                        
                        # Clear cache
                        LORE_CACHE.invalidate_pattern(f"CulturalElements_{element_id}")
                    except Exception as e:
                        logging.error(f"Error updating cultural element {element_id}: {e}")
        
        return changes
    
    async def _shift_geopolitical_landscape(self) -> List[Dict[str, Any]]:
        """
        Apply subtle shifts to the geopolitical landscape
        
        Returns:
            List of geopolitical changes
        """
        changes = []
        
        # Load some faction data to work with
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description, territory, rivals, allies
                    FROM Factions
                    ORDER BY RANDOM()
                    LIMIT 3
                """)
                
                if not factions:
                    return changes
                    
                # Get a random region if we have them
                regions = await conn.fetch("""
                    SELECT id, name, description, governing_faction
                    FROM GeographicRegions
                    ORDER BY RANDOM()
                    LIMIT 1
                """)
                
                region = regions[0] if regions else None
                
                # Determine what kind of shift to generate
                shift_types = ["alliance_change", "territory_dispute", "influence_growth"]
                
                # Add "regional_governance" if we have a region
                if region:
                    shift_types.append("regional_governance")
                    
                # Pick a random shift type
                shift_type = random.choice(shift_types)
                
                # Create the run context
                run_ctx = RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                if shift_type == "alliance_change":
                    # Two factions change their relationship
                    if len(factions) >= 2:
                        faction1 = factions[0]
                        faction2 = factions[1]
                        
                        # Determine new relationship
                        current_allies1 = faction1['allies'] if faction1['allies'] else []
                        current_rivals1 = faction1['rivals'] if faction1['rivals'] else []
                        
                        if faction2['name'] in current_allies1:
                            # They were allies, now becoming rivals
                            new_relationship = "rivalry"
                            
                            # Update allies and rivals lists
                            if faction2['name'] in current_allies1:
                                current_allies1.remove(faction2['name'])
                            if faction2['name'] not in current_rivals1:
                                current_rivals1.append(faction2['name'])
                                
                            prompt = f"""
                            Two factions that were once allies are now becoming rivals:
                            
                            FACTION 1: {faction1['name']} ({faction1['type']})
                            Description: {faction1['description']}
                            
                            FACTION 2: {faction2['name']} ({faction2['type']})
                            Description: {faction2['description']}
                            
                            Update the description of Faction 1 to reflect this change in relationship.
                            Include information about why they've become rivals and what tensions exist now.
                            Maintain the matriarchal power dynamics in your description.
                            
                            Return only the new description, written in the same style as the original.
                            """
                            
                        elif faction2['name'] in current_rivals1:
                            # They were rivals, now becoming allies
                            new_relationship = "alliance"
                            
                            # Update allies and rivals lists
                            if faction2['name'] in current_rivals1:
                                current_rivals1.remove(faction2['name'])
                            if faction2['name'] not in current_allies1:
                                current_allies1.append(faction2['name'])
                                
                            prompt = f"""
                            Two factions that were once rivals are now becoming allies:
                            
                            FACTION 1: {faction1['name']} ({faction1['type']})
                            Description: {faction1['description']}
                            
                            FACTION 2: {faction2['name']} ({faction2['type']})
                            Description: {faction2['description']}
                            
                            Update the description of Faction 1 to reflect this change in relationship.
                            Include information about why they've become allies and what common interests they now share.
                            Maintain the matriarchal power dynamics in your description.
                            
                            Return only the new description, written in the same style as the original.
                            """
                            
                        else:
                            # They had no specific relationship, now becoming allies
                            new_relationship = "new alliance"
                            
                            # Update allies list
                            if faction2['name'] not in current_allies1:
                                current_allies1.append(faction2['name'])
                                
                            prompt = f"""
                            Two factions are forming a new alliance:
                            
                            FACTION 1: {faction1['name']} ({faction1['type']})
                            Description: {faction1['description']}
                            
                            FACTION 2: {faction2['name']} ({faction2['type']})
                            Description: {faction2['description']}
                            
                            Update the description of Faction 1 to reflect this new alliance.
                            Include information about why they've become allies and what mutual benefits exist.
                            Maintain the matriarchal power dynamics in your description.
                            
                            Return only the new description, written in the same style as the original.
                            """
                        
                        # Get updated description from LLM
                        geo_agent = Agent(
                            name="GeopoliticsAgent",
                            instructions="You evolve geopolitical relationships between factions.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        # Apply the update to faction 1
                        try:
                            # Generate new embedding
                            embedding_text = f"{faction1['name']} {faction1['type']} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            await conn.execute("""
                                UPDATE Factions
                                SET description = $1, 
                                    allies = $2,
                                    rivals = $3,
                                    embedding = $4
                                WHERE id = $5
                            """, new_description, current_allies1, current_rivals1, new_embedding, faction1['id'])
                            
                            # Need to also update faction 2's allies/rivals
                            current_allies2 = faction2['allies'] if faction2['allies'] else []
                            current_rivals2 = faction2['rivals'] if faction2['rivals'] else []
                            
                            if new_relationship == "alliance" or new_relationship == "new alliance":
                                if faction1['name'] in current_rivals2:
                                    current_rivals2.remove(faction1['name'])
                                if faction1['name'] not in current_allies2:
                                    current_allies2.append(faction1['name'])
                            else:  # rivalry
                                if faction1['name'] in current_allies2:
                                    current_allies2.remove(faction1['name'])
                                if faction1['name'] not in current_rivals2:
                                    current_rivals2.append(faction1['name'])
                            
                            await conn.execute("""
                                UPDATE Factions
                                SET allies = $1,
                                    rivals = $2
                                WHERE id = $3
                            """, current_allies2, current_rivals2, faction2['id'])
                            
                            # Record the change
                            changes.append({
                                "change_type": "alliance_change",
                                "faction1_id": faction1['id'],
                                "faction1_name": faction1['name'],
                                "faction2_id": faction2['id'],
                                "faction2_name": faction2['name'],
                                "new_relationship": new_relationship,
                                "old_description": faction1['description'],
                                "new_description": new_description
                            })
                            
                            # Clear cache
                            LORE_CACHE.invalidate_pattern(f"Factions_{faction1['id']}")
                            LORE_CACHE.invalidate_pattern(f"Factions_{faction2['id']}")
                        except Exception as e:
                            logging.error(f"Error updating faction relationship: {e}")
                
                elif shift_type == "territory_dispute":
                    # A faction is expanding its territory
                    faction = factions[0]
                    
                    prompt = f"""
                    This faction is expanding its territorial claims or influence:
                    
                    FACTION: {faction['name']} ({faction['type']})
                    Current Description: {faction['description']}
                    Current Territory: {faction['territory'] if faction['territory'] else "Unspecified areas"}
                    
                    Update the description to show how this faction is claiming new territory or expanding influence.
                    Include what areas they're moving into and any resistance or consequences this creates.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    # Get updated description from LLM
                    geo_agent = Agent(
                        name="GeopoliticsAgent",
                        instructions="You evolve geopolitical relationships between factions.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply the update
                    try:
                        # Generate new embedding
                        embedding_text = f"{faction['name']} {faction['type']} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        # Also update territory to include expansion
                        new_territory = faction['territory'] if faction['territory'] else []
                        new_areas = ["expanding into neighboring regions", "claiming new settlements"]
                        
                        if not isinstance(new_territory, list):
                            # Handle string case
                            new_territory = [new_territory, new_areas[0]]
                        else:
                            new_territory.append(random.choice(new_areas))
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET description = $1,
                                territory = $2,
                                embedding = $3
                            WHERE id = $4
                        """, new_description, new_territory, new_embedding, faction['id'])
                        
                        # Record the change
                        changes.append({
                            "change_type": "territory_dispute",
                            "faction_id": faction['id'],
                            "faction_name": faction['name'],
                            "old_description": faction['description'],
                            "new_description": new_description,
                            "new_territory": new_territory
                        })
                        
                        # Clear cache
                        LORE_CACHE.invalidate_pattern(f"Factions_{faction['id']}")
                    except Exception as e:
                        logging.error(f"Error updating faction territory: {e}")
                
                elif shift_type == "regional_governance" and region:
                    # A region changes its governing faction
                    faction = factions[0]
                    
                    # Only proceed if this is a different faction than current governance
                    if region['governing_faction'] != faction['name']:
                        prompt = f"""
                        A geographic region is changing its governing faction:
                        
                        REGION: {region['name']}
                        Current Description: {region['description']}
                        Previous Governing Faction: {region['governing_faction'] if region['governing_faction'] else "None"}
                        New Governing Faction: {faction['name']} ({faction['type']})
                        
                        Update the region's description to reflect this change in governance.
                        Include how this transition occurred and what changes the new faction is implementing.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        # Get updated description from LLM
                        geo_agent = Agent(
                            name="GeopoliticsAgent",
                            instructions="You evolve geopolitical relationships between factions.",
                            model="o3-mini"
                        )
                        
                        result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                        new_description = result.final_output
                        
                        # Apply the update
                        try:
                            # Generate new embedding
                            embedding_text = f"{region['name']} {new_description}"
                            new_embedding = await generate_embedding(embedding_text)
                            
                            await conn.execute("""
                                UPDATE GeographicRegions
                                SET description = $1,
                                    governing_faction = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, faction['name'], new_embedding, region['id'])
                            
                            # Record the change
                            changes.append({
                                "change_type": "regional_governance",
                                "region_id": region['id'],
                                "region_name": region['name'],
                                "old_governing_faction": region['governing_faction'] if region['governing_faction'] else "None",
                                "new_governing_faction": faction['name'],
                                "old_description": region['description'],
                                "new_description": new_description
                            })
                            
                            # Clear cache
                            LORE_CACHE.invalidate_pattern(f"GeographicRegions_{region['id']}")
                        except Exception as e:
                            logging.error(f"Error updating region governance: {e}")
                    
                else:  # influence_growth
                    # A faction is growing in influence
                    faction = factions[0]
                    
                    prompt = f"""
                    This faction is experiencing a significant growth in influence and power:
                    
                    FACTION: {faction['name']} ({faction['type']})
                    Current Description: {faction['description']}
                    
                    Update the description to show how this faction is becoming more influential.
                    Include new strategies they're employing, resources they're acquiring, or alliances they're leveraging.
                    Maintain the matriarchal power dynamics in your description.
                    
                    Return only the new description, written in the same style as the original.
                    """
                    
                    # Get updated description from LLM
                    geo_agent = Agent(
                        name="GeopoliticsAgent",
                        instructions="You evolve geopolitical relationships between factions.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(geo_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply the update
                    try:
                        # Generate new embedding
                        embedding_text = f"{faction['name']} {faction['type']} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        await conn.execute("""
                            UPDATE Factions
                            SET description = $1,
                                embedding = $2
                            WHERE id = $3
                        """, new_description, new_embedding, faction['id'])
                        
                        # Record the change
                        changes.append({
                            "change_type": "influence_growth",
                            "faction_id": faction['id'],
                            "faction_name": faction['name'],
                            "old_description": faction['description'],
                            "new_description": new_description
                        })
                        
                        # Clear cache
                        LORE_CACHE.invalidate_pattern(f"Factions_{faction['id']}")
                    except Exception as e:
                        logging.error(f"Error updating faction influence: {e}")
        
        return changes
    
    async def _evolve_notable_figures(self) -> List[Dict[str, Any]]:
        """
        Evolve reputations and stories of notable figures
        
        Returns:
            List of changes to notable figures
        """
        changes = []
        
        # Check if we have a dedicated notable figures table
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                has_notable_figures_table = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'notablefigures'
                    );
                """)
                
                if has_notable_figures_table:
                    # Get figures from dedicated table
                    rows = await conn.fetch("""
                        SELECT id, name, description, significance, reputation
                        FROM NotableFigures
                        ORDER BY RANDOM()
                        LIMIT 2
                    """)
                else:
                    # Fall back to world lore with notable_figure category
                    rows = await conn.fetch("""
                        SELECT id, name, description, significance
                        FROM WorldLore
                        WHERE category = 'notable_figure'
                        ORDER BY RANDOM()
                        LIMIT 2
                    """)
                
                for row in rows:
                    figure_id = row['id']
                    figure_name = row['name']
                    old_description = row['description']
                    old_significance = row['significance'] if 'significance' in row else 5
                    
                    # Determine change type
                    change_options = ["reputation_rise", "reputation_fall", "scandal", "achievement", "reform"]
                    
                    # If we have reputation field, use it to influence likely changes
                    reputation = row.get('reputation', 5)
                    if reputation and reputation >= 7:
                        # High reputation more likely to fall or have scandal
                        weights = [1, 3, 2, 1, 1]
                    elif reputation and reputation <= 3:
                        # Low reputation more likely to rise or reform
                        weights = [3, 1, 1, 2, 2]
                    else:
                        # Mid reputation equal chances
                        weights = [1, 1, 1, 1, 1]
                    
                    # Weighted choice
                    change_type = random.choices(
                        change_options, 
                        weights=weights,
                        k=1
                    )[0]
                    
                    # Create the run context
                    run_ctx = RunContextWrapper(context={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    # Setup for each change type
                    if change_type == "reputation_rise":
                        prompt = f"""
                        This notable figure is experiencing a rise in reputation and influence:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to show how this figure's reputation has improved.
                        Include their recent positive actions, public perception changes, or new supporters.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = min(10, (reputation if reputation else 5) + random.randint(1, 2))
                        reputation_change = "improved"
                        
                    elif change_type == "reputation_fall":
                        prompt = f"""
                        This notable figure is experiencing a decline in reputation and influence:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to show how this figure's reputation has declined.
                        Include their recent missteps, public perception changes, or lost support.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = max(1, (reputation if reputation else 5) - random.randint(1, 2))
                        reputation_change = "declined"
                        
                    elif change_type == "scandal":
                        prompt = f"""
                        This notable figure is involved in a recent scandal:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to incorporate a scandal this figure is facing.
                        Include the nature of the scandal and how they're responding to it.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = max(1, (reputation if reputation else 5) - random.randint(2, 3))
                        reputation_change = "significantly damaged"
                        
                    elif change_type == "achievement":
                        prompt = f"""
                        This notable figure has recently achieved something significant:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to incorporate a major achievement by this figure.
                        Include what they accomplished and how it has affected their standing.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        new_reputation = min(10, (reputation if reputation else 5) + random.randint(2, 3))
                        reputation_change = "significantly improved"
                        
                    else:  # reform
                        prompt = f"""
                        This notable figure has undergone a personal or political reform:
                        
                        FIGURE: {figure_name}
                        Current Description: {old_description}
                        
                        Update the description to show how this figure has changed their stance, approach, or values.
                        Include what prompted this change and how others have reacted to it.
                        Maintain the matriarchal power dynamics in your description.
                        
                        Return only the new description, written in the same style as the original.
                        """
                        
                        # Reform could go either way
                        direction = random.choice([-1, 1])
                        new_reputation = max(1, min(10, (reputation if reputation else 5) + direction * random.randint(1, 2)))
                        reputation_change = "transformed"
                    
                    # Get updated description from LLM
                    figure_agent = Agent(
                        name="NotableFigureAgent",
                        instructions="You evolve the stories and reputations of notable figures.",
                        model="o3-mini"
                    )
                    
                    result = await Runner.run(figure_agent, prompt, context=run_ctx.context)
                    new_description = result.final_output
                    
                    # Apply the update
                    try:
                        # Generate new embedding
                        embedding_text = f"{figure_name} {new_description}"
                        new_embedding = await generate_embedding(embedding_text)
                        
                        if has_notable_figures_table:
                            # Update in dedicated table
                            await conn.execute("""
                                UPDATE NotableFigures
                                SET description = $1,
                                    reputation = $2,
                                    embedding = $3
                                WHERE id = $4
                            """, new_description, new_reputation, new_embedding, figure_id)
                        else:
                            # Update in WorldLore
                            await conn.execute("""
                                UPDATE WorldLore
                                SET description = $1,
                                    embedding = $2
                                WHERE id = $3
                            """, new_description, new_embedding, figure_id)
                        
                        # Record the change
                        changes.append({
                            "figure_id": figure_id,
                            "name": figure_name,
                            "change_type": change_type,
                            "old_description": old_description,
                            "new_description": new_description,
                            "reputation_change": reputation_change
                        })
                        
                        # Clear cache
                        if has_notable_figures_table:
                            LORE_CACHE.invalidate_pattern(f"NotableFigures_{figure_id}")
                        else:
                            LORE_CACHE.invalidate_pattern(f"WorldLore_{figure_id}")
                    except Exception as e:
                        logging.error(f"Error updating notable figure {figure_id}: {e}")
        
        return changes
        
class LocalLoreManager(BaseLoreManager):
    """
    Consolidated manager for local lore elements including urban myths, local histories,
    landmarks, and other location-specific narratives.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_prefix = "locallore"
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
        
    async def initialize_tables(self):
        """Ensure all local lore tables exist"""
        table_definitions = {
            "UrbanMyths": """
                CREATE TABLE UrbanMyths (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    origin_location TEXT,
                    origin_event TEXT,
                    believability INTEGER CHECK (believability BETWEEN 1 AND 10),
                    spread_rate INTEGER CHECK (spread_rate BETWEEN 1 AND 10),
                    regions_known TEXT[],
                    variations TEXT[],
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_urbanmyths_embedding 
                ON UrbanMyths USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "LocalHistories": """
                CREATE TABLE LocalHistories (
                    id SERIAL PRIMARY KEY,
                    location_id INTEGER NOT NULL,
                    event_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    date_description TEXT,
                    significance INTEGER CHECK (significance BETWEEN 1 AND 10),
                    impact_type TEXT,
                    notable_figures TEXT[],
                    current_relevance TEXT,
                    commemoration TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_localhistories_embedding 
                ON LocalHistories USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_localhistories_location
                ON LocalHistories(location_id);
            """,
            
            "Landmarks": """
                CREATE TABLE Landmarks (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_id INTEGER NOT NULL,
                    landmark_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    historical_significance TEXT,
                    current_use TEXT,
                    controlled_by TEXT,
                    legends TEXT[],
                    embedding VECTOR(1536),
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_landmarks_embedding 
                ON Landmarks USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_landmarks_location
                ON Landmarks(location_id);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_urban_myth",
        action_description="Adding urban myth: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def add_urban_myth(
        self, 
        ctx,
        name: str, 
        description: str, 
        origin_location: Optional[str] = None,
        origin_event: Optional[str] = None,
        believability: int = 6,
        spread_rate: int = 5,
        regions_known: List[str] = None
    ) -> int:
        """
        Add a new urban myth to the database
        
        Args:
            name: Name of the urban myth
            description: Full description of the myth
            origin_location: Where the myth originated
            origin_event: What event spawned the myth
            believability: How believable the myth is (1-10)
            spread_rate: How quickly the myth is spreading (1-10)
            regions_known: List of regions where the myth is known
            
        Returns:
            ID of the created urban myth
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Generate embedding for the myth
        embedding_text = f"{name} {description}"
        embedding = await generate_embedding(embedding_text)
        
        # Set defaults
        regions_known = regions_known or ["local area"]
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                myth_id = await conn.fetchval("""
                    INSERT INTO UrbanMyths (
                        name, description, origin_location, origin_event,
                        believability, spread_rate, regions_known, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """, name, description, origin_location, origin_event,
                     believability, spread_rate, regions_known, embedding)
                
                return myth_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_myths_for_location",
        action_description="Getting myths for location: {location}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def get_myths_for_location(self, ctx, location: str) -> List[Dict[str, Any]]:
        """
        Get urban myths known in a specific location
        
        Args:
            location: The location name
            
        Returns:
            List of urban myths known in this location
        """
        # Check cache first
        cache_key = f"myths_for_{location}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        # Ensure tables exist
        await self.ensure_initialized()
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if we have a direct match where this location is the origin
                direct_myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate, variations
                    FROM UrbanMyths
                    WHERE origin_location = $1
                """, location)
                
                # Check for myths where this location is in the regions_known array
                known_myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate, variations
                    FROM UrbanMyths
                    WHERE $1 = ANY(regions_known)
                """, location)
                
                # Combine and deduplicate
                myth_ids = set()
                myths = []
                
                for myth in direct_myths:
                    myth_dict = dict(myth)
                    myth_dict["origin"] = True
                    myths.append(myth_dict)
                    myth_ids.add(myth["id"])
                
                for myth in known_myths:
                    if myth["id"] not in myth_ids:
                        myth_dict = dict(myth)
                        myth_dict["origin"] = False
                        myths.append(myth_dict)
                
                # Cache the result
                self.set_cache(cache_key, myths)
                
                return myths
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_local_history",
        action_description="Adding local history event: {event_name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def add_local_history(
        self, 
        ctx,
        location_id: int,
        event_name: str,
        description: str,
        date_description: str = "Some time ago",
        significance: int = 5,
        impact_type: str = "cultural",
        notable_figures: List[str] = None,
        current_relevance: str = None,
        commemoration: str = None
    ) -> int:
        """
        Add a local historical event to the database
        
        Args:
            location_id: ID of the associated location
            event_name: Name of the historical event
            description: Description of the event
            date_description: When it occurred
            significance: Importance from 1-10
            impact_type: Type of impact (political, cultural, etc.)
            notable_figures: People involved
            current_relevance: How it affects the present
            commemoration: How it's remembered/celebrated
            
        Returns:
            ID of the created local history event
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        notable_figures = notable_figures or []
        
        # Generate embedding
        embedding_text = f"{event_name} {description} {date_description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                event_id = await conn.fetchval("""
                    INSERT INTO LocalHistories (
                        location_id, event_name, description, date_description,
                        significance, impact_type, notable_figures,
                        current_relevance, commemoration, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """, location_id, event_name, description, date_description,
                     significance, impact_type, notable_figures,
                     current_relevance, commemoration, embedding)
                
                # Invalidate relevant cache
                self.invalidate_cache_pattern(f"local_history_{location_id}")
                
                return event_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_landmark",
        action_description="Adding landmark: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def add_landmark(
        self, 
        ctx,
        name: str,
        location_id: int,
        landmark_type: str,
        description: str,
        historical_significance: str = None,
        current_use: str = None,
        controlled_by: str = None,
        legends: List[str] = None
    ) -> int:
        """
        Add a landmark to the database
        
        Args:
            name: Name of the landmark
            location_id: ID of the associated location
            landmark_type: Type of landmark (monument, building, natural feature, etc.)
            description: Description of the landmark
            historical_significance: Historical importance
            current_use: How it's used today
            controlled_by: Who controls/owns it
            legends: Associated legends or stories
            
        Returns:
            ID of the created landmark
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        legends = legends or []
        
        # Generate embedding
        embedding_text = f"{name} {landmark_type} {description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                landmark_id = await conn.fetchval("""
                    INSERT INTO Landmarks (
                        name, location_id, landmark_type, description,
                        historical_significance, current_use, controlled_by,
                        legends, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, name, location_id, landmark_type, description,
                     historical_significance, current_use, controlled_by,
                     legends, embedding)
                
                # Invalidate relevant cache
                self.invalidate_cache_pattern(f"landmarks_{location_id}")
                
                return landmark_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_location_lore",
        action_description="Getting all lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def get_location_lore(self, ctx, location_id: int) -> Dict[str, Any]:
        """
        Get all lore associated with a location (myths, history, landmarks)
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with all lore for the location
        """
        # Check cache first
        cache_key = f"location_lore_{location_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        # Get location details
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get location name
                location = await conn.fetchrow("""
                    SELECT id, location_name
                    FROM Locations
                    WHERE id = $1
                """, location_id)
                
                if not location:
                    return {"error": "Location not found"}
                
                location_name = location["location_name"]
                
                # Get all local histories
                histories = await conn.fetch("""
                    SELECT id, event_name, description, date_description,
                           significance, impact_type, notable_figures,
                           current_relevance, commemoration
                    FROM LocalHistories
                    WHERE location_id = $1
                    ORDER BY significance DESC
                """, location_id)
                
                # Get all landmarks
                landmarks = await conn.fetch("""
                    SELECT id, name, landmark_type, description,
                           historical_significance, current_use,
                           controlled_by, legends
                    FROM Landmarks
                    WHERE location_id = $1
                """, location_id)
                
                # Get all myths
                myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate
                    FROM UrbanMyths
                    WHERE origin_location = $1 OR $1 = ANY(regions_known)
                """, location_name)
                
                # Compile result
                result = {
                    "location": dict(location),
                    "histories": [dict(hist) for hist in histories],
                    "landmarks": [dict(landmark) for landmark in landmarks],
                    "myths": [dict(myth) for myth in myths]
                }
                
                # Cache result
                self.set_cache(cache_key, result)
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_location_lore",
        action_description="Generating lore for location: {location_data['name']}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def generate_location_lore(self, ctx, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive lore for a location
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            Dictionary with generated lore
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Generate myths
        myths = await self.generate_myths_for_location(run_ctx, location_data)
        
        # Generate local histories
        histories = await self.generate_local_history(run_ctx, location_data)
        
        # Generate landmarks
        landmarks = await self.generate_landmarks(run_ctx, location_data)
        
        return {
            "location": location_data,
            "generated_myths": myths,
            "generated_histories": histories,
            "generated_landmarks": landmarks
        }
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_myths_for_location",
        action_description="Generating myths for location: {location_data['name']}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def generate_myths_for_location(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate urban myths for a specific location based on its characteristics
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            List of generated urban myths
        """
        # Create the run context
        run_ctx = self.create_run_context(ctx)
        
        # Extract relevant details
        location_name = location_data.get('name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        description = location_data.get('description', '')
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 2-3 urban myths or local legends associated with this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        
        Create urban myths that feel authentic to this location. Each myth should:
        1. Be somewhat believable but with fantastical elements
        2. Reflect local concerns, features, or history
        3. Have some connection to matriarchal power structures
        
        Format your response as a JSON array where each object has:
        - "name": The name/title of the myth
        - "description": A detailed description of the myth
        - "believability": Number from 1-10 indicating how believable it is
        - "spread_rate": Number from 1-10 indicating how widely it has spread
        - "origin": Brief statement of how the myth originated
        """
        
        # Create an agent for myth generation
        myth_agent = Agent(
            name="UrbanMythAgent",
            instructions="You create urban myths and local legends for locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(myth_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            myths = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(myths, list):
                if isinstance(myths, dict):
                    myths = [myths]
                else:
                    myths = []
            
            # Apply matriarchal theming to each myth
            for myth in myths:
                if 'description' in myth:
                    myth['description'] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        'myth', myth['description'], emphasis_level=1
                    )
            
            # Store each myth
            saved_myths = []
            for myth in myths:
                # Extract myth details
                name = myth.get('name')
                description = myth.get('description')
                believability = myth.get('believability', random.randint(4, 8))
                spread_rate = myth.get('spread_rate', random.randint(3, 7))
                
                if not name or not description:
                    continue
                
                # Save the myth
                try:
                    myth_id = await self.add_urban_myth(
                        run_ctx,
                        name=name,
                        description=description,
                        origin_location=location_name,
                        believability=believability,
                        spread_rate=spread_rate,
                        regions_known=[location_name]
                    )
                    
                    # Add to results
                    myth['id'] = myth_id
                    saved_myths.append(myth)
                except Exception as e:
                    logging.error(f"Error saving urban myth '{name}': {e}")
            
            return saved_myths
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for urban myths: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_local_history",
        action_description="Generating local history for: {location_data['name']}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def generate_local_history(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate local historical events for a location based on its characteristics
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            List of generated local historical events
        """
        # Create the run context
        run_ctx = self.create_run_context(ctx)
        
        # Extract relevant details
        location_id = location_data.get('id')
        location_name = location_data.get('name', 'Unknown Location')
        location_type = location_data.get('type', 'place')
        description = location_data.get('description', '')
        
        # Get world lore for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Try to get world history for context
                world_history = await conn.fetchval("""
                    SELECT description FROM WorldLore
                    WHERE category = 'world_history'
                    LIMIT 1
                """)
                
                if not world_history:
                    world_history = "The world has a rich history of matriarchal societies and feminine power."
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 3-4 local historical events specific to this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        
        WORLD HISTORY CONTEXT:
        {world_history[:500]}
        
        Create local historical events that feel authentic to this location. Each event should:
        1. Be specific to this location rather than world-changing
        2. Reflect local development, conflicts, or cultural shifts
        3. Include at least one event related to matriarchal power structures
        4. Include a range of timeframes (some recent, some older)
        
        Format your response as a JSON array where each object has:
        - "event_name": The name of the historical event
        - "description": A detailed description of what happened
        - "date_description": When it occurred (e.g., "50 years ago", "during the reign of...")
        - "significance": Number from 1-10 indicating historical importance
        - "impact_type": Type of impact (political, cultural, economic, religious, etc.)
        - "notable_figures": Array of names of people involved
        - "current_relevance": How it still affects the location today
        - "commemoration": How it's remembered (festival, monument, etc.)
        """
        
        # Create an agent for local history generation
        history_agent = Agent(
            name="LocalHistoryAgent",
            instructions="You create local historical events for specific locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(history_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            events = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(events, list):
                if isinstance(events, dict):
                    events = [events]
                else:
                    events = []
            
            # Apply matriarchal theming to each event
            for event in events:
                if 'description' in event:
                    event['description'] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        'history', event['description'], emphasis_level=1
                    )
            
            # Store each event
            saved_events = []
            for event in events:
                # Extract event details
                event_name = event.get('event_name')
                description = event.get('description')
                date_description = event.get('date_description', 'Some time ago')
                significance = event.get('significance', 5)
                impact_type = event.get('impact_type', 'cultural')
                notable_figures = event.get('notable_figures', [])
                current_relevance = event.get('current_relevance')
                commemoration = event.get('commemoration')
                
                if not event_name or not description:
                    continue
                
                # Save the event
                try:
                    event_id = await self.add_local_history(
                        run_ctx,
                        location_id=location_id,
                        event_name=event_name,
                        description=description,
                        date_description=date_description,
                        significance=significance,
                        impact_type=impact_type,
                        notable_figures=notable_figures,
                        current_relevance=current_relevance,
                        commemoration=commemoration
                    )
                    
                    # Add to results
                    event['id'] = event_id
                    saved_events.append(event)
                except Exception as e:
                    logging.error(f"Error saving local historical event '{event_name}': {e}")
            
            return saved_events
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for local history: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_landmarks",
        action_description="Generating landmarks for: {location_data['name']}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def generate_landmarks(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate landmarks for a location based on its characteristics
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            List of generated landmarks
        """
        # Create the run context
        run_ctx = self.create_run_context(ctx)
        
        # Extract relevant details
        location_id = location_data.get('id')
        location_name = location_data.get('name', 'Unknown Location')
        location_type = location_data.get('type', 'place')
        description = location_data.get('description', '')
        
        # Get local history for context
        local_history = await self.get_location_lore(ctx, location_id)
        history_context = "\n".join([f"- {e['event_name']}: {e['description'][:100]}..." 
                                    for e in local_history.get('histories', [])[:2]])
        
        # Get controlling factions
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Try to find factions that control this location
                controlling_factions = []
                try:
                    factions = await conn.fetch("""
                        SELECT f.id, f.name, f.type
                        FROM Factions f
                        JOIN LoreConnections lc ON f.id = lc.source_id
                        WHERE lc.target_type = 'LocationLore'
                        AND lc.target_id = $1
                        AND lc.source_type = 'Factions'
                        AND lc.connection_type = 'influences'
                    """, location_id)
                    
                    controlling_factions = [dict(faction) for faction in factions]
                except:
                    # Just use empty list if table doesn't exist or query fails
                    pass
        
        faction_context = ", ".join([f['name'] for f in controlling_factions]) if controlling_factions else "Various groups"
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 3-5 landmarks found in this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        CONTROLLING FACTIONS: {faction_context}
        
        LOCAL HISTORY CONTEXT:
        {history_context}
        
        Create landmarks that feel authentic to this location. Include:
        1. At least one natural landmark (if appropriate)
        2. At least one architectural/built landmark
        3. At least one landmark related to matriarchal power structures
        4. A mix of ancient/historical and more recent landmarks
        
        Format your response as a JSON array where each object has:
        - "name": The name of the landmark
        - "landmark_type": Type of landmark (monument, building, natural feature, temple, etc.)
        - "description": A detailed physical description
        - "historical_significance": Its importance to local history
        - "current_use": How it's used today (ceremonial, practical, tourist attraction, etc.)
        - "controlled_by": Which faction or group controls it
        - "legends": Array of brief legends or stories associated with it
        """
        
        # Create an agent for landmark generation
        landmark_agent = Agent(
            name="LandmarkAgent",
            instructions="You create landmarks for specific locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(landmark_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            landmarks = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(landmarks, list):
                if isinstance(landmarks, dict):
                    landmarks = [landmarks]
                else:
                    landmarks = []
            
            # Apply matriarchal theming to each landmark
            for landmark in landmarks:
                if 'description' in landmark:
                    landmark['description'] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        'landmark', landmark['description'], emphasis_level=1
                    )
            
            # Store each landmark
            saved_landmarks = []
            for landmark in landmarks:
                # Extract landmark details
                name = landmark.get('name')
                landmark_type = landmark.get('landmark_type', 'building')
                description = landmark.get('description')
                historical_significance = landmark.get('historical_significance')
                current_use = landmark.get('current_use')
                controlled_by = landmark.get('controlled_by')
                legends = landmark.get('legends', [])
                
                if not name or not description:
                    continue
                
                # Save the landmark
                try:
                    landmark_id = await self.add_landmark(
                        run_ctx,
                        name=name,
                        location_id=location_id,
                        landmark_type=landmark_type,
                        description=description,
                        historical_significance=historical_significance,
                        current_use=current_use,
                        controlled_by=controlled_by,
                        legends=legends
                    )
                    
                    # Add to results
                    landmark['id'] = landmark_id
                    saved_landmarks.append(landmark)
                except Exception as e:
                    logging.error(f"Error saving landmark '{name}': {e}")
            
            return saved_landmarks
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for landmarks: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_location_lore",
        action_description="Evolving lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def evolve_location_lore(self, ctx, location_id: int, event_description: str) -> Dict[str, Any]:
        """
        Evolve the lore of a location based on an event
        
        Args:
            location_id: ID of the location
            event_description: Description of the event affecting the location
            
        Returns:
            Dictionary with evolution results
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Get current location lore
        location_lore = await self.get_location_lore(ctx, location_id)
        
        if "error" in location_lore:
            return location_lore
        
        # Theming the event
        themed_event = MatriarchalThemingUtils.apply_matriarchal_theme(
            'event', event_description, emphasis_level=1
        )
        
        # Get location name
        location_name = location_lore.get('location', {}).get('location_name', 'Unknown Location')
        
        # Create an agent for lore evolution
        evolution_agent = Agent(
            name="LoreEvolutionAgent",
            instructions="You evolve location lore based on events that affect the location.",
            model="o3-mini"
        )
        
        # 1. Evolve myths first
        myths_prompt = f"""
        Based on this event that occurred at {location_name}, determine how it affects the local myths:
        
        EVENT:
        {themed_event}
        
        CURRENT MYTHS:
        {json.dumps(location_lore.get('myths', []), indent=2)}
        
        Create 1-2 reasonable evolutions for these myths based on the event.
        You can either modify an existing myth or create a new one.
        
        Format your response as a JSON object with:
        - "modified_myths": Array of objects with "id" (if updating existing) and "new_description"
        - "new_myths": Array of objects with "name", "description", "believability", "spread_rate"
        """
        
        # 2. Evolve history
        history_prompt = f"""
        Based on this event that occurred at {location_name}, create a new historical entry:
        
        EVENT:
        {themed_event}
        
        EXISTING HISTORY:
        {json.dumps(location_lore.get('histories', []), indent=2)}
        
        Create a new historical entry for this event.
        
        Format your response as a JSON object with:
        "new_history": {{
            "event_name": "Name for this historical event",
            "description": "Detailed description of what happened",
            "date_description": "Recently",
            "significance": Number from 1-10 indicating historical importance,
            "impact_type": Type of impact (political, cultural, etc.),
            "notable_figures": Array of people involved,
            "current_relevance": How it affects the location now
        }}
        """
        
        # 3. Evolve landmarks
        landmarks_prompt = f"""
        Based on this event that occurred at {location_name}, determine how it affects the landmarks:
        
        EVENT:
        {themed_event}
        
        CURRENT LANDMARKS:
        {json.dumps(location_lore.get('landmarks', []), indent=2)}
        
        Determine if this event would affect any existing landmarks or create new ones.
        
        Format your response as a JSON object with:
        - "modified_landmarks": Array of objects with "id" (of existing landmark) and "new_description"
        - "new_landmarks": Array of objects with landmark details for any new landmarks created
        """
        
        # Execute all three prompts
        myth_result = await Runner.run(evolution_agent, myths_prompt, context=run_ctx.context)
        history_result = await Runner.run(evolution_agent, history_prompt, context=run_ctx.context)
        landmark_result = await Runner.run(evolution_agent, landmarks_prompt, context=run_ctx.context)
        
        # Process the results
        try:
            myth_changes = json.loads(myth_result.final_output)
            history_changes = json.loads(history_result.final_output)
            landmark_changes = json.loads(landmark_result.final_output)
            
            # Apply myth changes
            myth_updates = []
            if "modified_myths" in myth_changes:
                for update in myth_changes["modified_myths"]:
                    if "id" in update and "new_description" in update:
                        # Update the myth
                        async with self.get_connection_pool() as pool:
                            async with pool.acquire() as conn:
                                await conn.execute("""
                                    UPDATE UrbanMyths
                                    SET description = $1
                                    WHERE id = $2
                                """, update["new_description"], update["id"])
                                
                                # Get updated myth
                                updated_myth = await conn.fetchrow("""
                                    SELECT id, name, description, believability, spread_rate
                                    FROM UrbanMyths
                                    WHERE id = $1
                                """, update["id"])
                                
                                if updated_myth:
                                    myth_updates.append(dict(updated_myth))
            
            # Add new myths
            new_myths = []
            if "new_myths" in myth_changes:
                for new_myth in myth_changes["new_myths"]:
                    # Apply theming
                    if 'description' in new_myth:
                        new_myth['description'] = MatriarchalThemingUtils.apply_matriarchal_theme(
                            'myth', new_myth['description'], emphasis_level=1
                        )
                        
                    # Add the myth
                    try:
                        myth_id = await self.add_urban_myth(
                            run_ctx,
                            name=new_myth.get("name", "Unnamed Myth"),
                            description=new_myth.get("description", ""),
                            origin_location=location_name,
                            origin_event=themed_event,
                            believability=new_myth.get("believability", 6),
                            spread_rate=new_myth.get("spread_rate", 5),
                            regions_known=[location_name]
                        )
                        
                        new_myth["id"] = myth_id
                        new_myths.append(new_myth)
                    except Exception as e:
                        logging.error(f"Error adding new myth: {e}")
            
            # Add new history
            new_history = None
            if "new_history" in history_changes:
                history_entry = history_changes["new_history"]
                
                # Apply theming
                if 'description' in history_entry:
                    history_entry['description'] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        'history', history_entry['description'], emphasis_level=1
                    )
                
                # Add the history entry
                try:
                    history_id = await self.add_local_history(
                        run_ctx,
                        location_id=location_id,
                        event_name=history_entry.get("event_name", "Recent Event"),
                        description=history_entry.get("description", ""),
                        date_description=history_entry.get("date_description", "Recently"),
                        significance=history_entry.get("significance", 5),
                        impact_type=history_entry.get("impact_type", "event"),
                        notable_figures=history_entry.get("notable_figures", []),
                        current_relevance=history_entry.get("current_relevance")
                    )
                    
                    history_entry["id"] = history_id
                    new_history = history_entry
                except Exception as e:
                    logging.error(f"Error adding new history entry: {e}")
            
            # Apply landmark changes
            landmark_updates = []
            if "modified_landmarks" in landmark_changes:
                for update in landmark_changes["modified_landmarks"]:
                    if "id" in update and "new_description" in update:
                        # Apply theming
                        update["new_description"] = MatriarchalThemingUtils.apply_matriarchal_theme(
                            'landmark', update["new_description"], emphasis_level=1
                        )
                        
                        # Update the landmark
                        async with self.get_connection_pool() as pool:
                            async with pool.acquire() as conn:
                                await conn.execute("""
                                    UPDATE Landmarks
                                    SET description = $1
                                    WHERE id = $2
                                """, update["new_description"], update["id"])
                                
                                # Get updated landmark
                                updated_landmark = await conn.fetchrow("""
                                    SELECT id, name, landmark_type, description
                                    FROM Landmarks
                                    WHERE id = $1
                                """, update["id"])
                                
                                if updated_landmark:
                                    landmark_updates.append(dict(updated_landmark))
            
            # Add new landmarks
            new_landmarks = []
            if "new_landmarks" in landmark_changes:
                for new_landmark in landmark_changes["new_landmarks"]:
                    # Apply theming
                    if 'description' in new_landmark:
                        new_landmark['description'] = MatriarchalThemingUtils.apply_matriarchal_theme(
                            'landmark', new_landmark['description'], emphasis_level=1
                        )
                        
                    # Add the landmark
                    try:
                        landmark_id = await self.add_landmark(
                            run_ctx,
                            name=new_landmark.get("name", "Unnamed Landmark"),
                            location_id=location_id,
                            landmark_type=new_landmark.get("landmark_type", "structure"),
                            description=new_landmark.get("description", ""),
                            historical_significance=new_landmark.get("historical_significance"),
                            current_use=new_landmark.get("current_use"),
                            controlled_by=new_landmark.get("controlled_by"),
                            legends=new_landmark.get("legends", [])
                        )
                        
                        new_landmark["id"] = landmark_id
                        new_landmarks.append(new_landmark)
                    except Exception as e:
                        logging.error(f"Error adding new landmark: {e}")
            
            # Invalidate cache for this location
            self.invalidate_cache_pattern(f"location_lore_{location_id}")
            
            # Return results
            return {
                "event": themed_event,
                "location_id": location_id,
                "location_name": location_name,
                "myth_updates": myth_updates,
                "new_myths": new_myths,
                "new_history": new_history,
                "landmark_updates": landmark_updates,
                "new_landmarks": new_landmarks
            }
            
        except Exception as e:
            logging.error(f"Error processing location lore evolution: {e}")
            return {"error": f"Failed to evolve location lore: {str(e)}"}

class EducationalSystemManager(BaseLoreManager):
    """
    Manages how knowledge is taught and passed down across generations
    within the matriarchal society, including formal and informal systems.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
        
    async def initialize_tables(self):
        """Ensure educational system tables exist"""
        table_definitions = {
            "EducationalSystems": """
                CREATE TABLE EducationalSystems (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    system_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_demographics TEXT[],
                    controlled_by TEXT,
                    core_teachings TEXT[],
                    teaching_methods TEXT[],
                    coming_of_age_rituals TEXT,
                    knowledge_restrictions TEXT,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_educationalsystems_embedding 
                ON EducationalSystems USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "KnowledgeTraditions": """
                CREATE TABLE KnowledgeTraditions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    tradition_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    knowledge_domain TEXT NOT NULL,
                    preservation_method TEXT,
                    access_requirements TEXT,
                    associated_group TEXT,
                    examples TEXT[],
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledgetraditions_embedding 
                ON KnowledgeTraditions USING ivfflat (embedding vector_cosine_ops);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_educational_system",
        action_description="Adding educational system: {name}",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def add_educational_system(
        self, 
        ctx,
        name: str,
        system_type: str,
        description: str,
        target_demographics: List[str],
        controlled_by: str,
        core_teachings: List[str],
        teaching_methods: List[str],
        coming_of_age_rituals: str = None,
        knowledge_restrictions: str = None
    ) -> int:
        """
        Add an educational system to the database
        
        Args:
            name: Name of the educational system
            system_type: Type of system (formal, apprenticeship, etc.)
            description: Detailed description
            target_demographics: Who is educated in this system
            controlled_by: Which faction/group controls this system
            core_teachings: What is taught
            teaching_methods: How it's taught
            coming_of_age_rituals: Rituals marking educational milestones
            knowledge_restrictions: Limitations on who can learn what
            
        Returns:
            ID of the created educational system
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Generate embedding
        embedding_text = f"{name} {system_type} {description} {' '.join(core_teachings)}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                system_id = await conn.fetchval("""
                    INSERT INTO EducationalSystems (
                        name, system_type, description, target_demographics,
                        controlled_by, core_teachings, teaching_methods, 
                        coming_of_age_rituals, knowledge_restrictions, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """, name, system_type, description, target_demographics,
                     controlled_by, core_teachings, teaching_methods,
                     coming_of_age_rituals, knowledge_restrictions, embedding)
                
                return system_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_knowledge_tradition",
        action_description="Adding knowledge tradition: {name}",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def add_knowledge_tradition(
        self, 
        ctx,
        name: str,
        tradition_type: str,
        description: str,
        knowledge_domain: str,
        preservation_method: str = None,
        access_requirements: str = None,
        associated_group: str = None,
        examples: List[str] = None
    ) -> int:
        """
        Add a knowledge tradition to the database
        
        Args:
            name: Name of the knowledge tradition
            tradition_type: Type of tradition (oral, written, experiential, etc.)
            description: Detailed description
            knowledge_domain: Domain of knowledge (medicine, crafts, etc.)
            preservation_method: How the knowledge is preserved
            access_requirements: Requirements to access this knowledge
            associated_group: Group that maintains this tradition
            examples: Examples of specific knowledge transmitted
            
        Returns:
            ID of the created knowledge tradition
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        examples = examples or []
        
        # Generate embedding
        embedding_text = f"{name} {tradition_type} {description} {knowledge_domain}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                tradition_id = await conn.fetchval("""
                    INSERT INTO KnowledgeTraditions (
                        name, tradition_type, description, knowledge_domain,
                        preservation_method, access_requirements,
                        associated_group, examples, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, name, tradition_type, description, knowledge_domain,
                     preservation_method, access_requirements,
                     associated_group, examples, embedding)
                
                return tradition_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_educational_systems",
        action_description="Generating educational systems for the setting",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def generate_educational_systems(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate a comprehensive set of educational systems for the setting
        
        Returns:
            List of generated educational systems
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get factions for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                factions = await conn.fetch("""
                    SELECT name, type FROM Factions
                    LIMIT 5
                """)
                
                faction_names = [f"{f['name']} ({f['type']})" for f in factions]
                faction_context = ", ".join(faction_names)
                
                # Get cultural elements for context
                elements = await conn.fetch("""
                    SELECT name, type FROM CulturalElements
                    LIMIT 5
                """)
                
                element_names = [f"{e['name']} ({e['type']})" for e in elements]
                culture_context = ", ".join(element_names)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 3-4 educational systems for a matriarchal society.
        
        SOCIETAL CONTEXT:
        Factions: {faction_context}
        Cultural Elements: {culture_context}
        
        Design educational systems that:
        1. Reflect matriarchal power structures and values
        2. Include different approaches for different demographics
        3. Show how gender roles are taught and reinforced
        4. Include both formal and informal education systems
        
        Format your response as a JSON array where each object has:
        - "name": The name of the educational system
        - "system_type": Type of system (formal academy, mentorship, etc.)
        - "description": A detailed description of how the system works
        - "target_demographics": Array of who is educated in this system
        - "controlled_by": Which faction/group controls this system
        - "core_teachings": Array of what is taught
        - "teaching_methods": Array of how it's taught
        - "coming_of_age_rituals": Rituals marking educational milestones
        - "knowledge_restrictions": Limitations on who can learn what
        """
        
        # Create an agent for educational system generation
        education_agent = Agent(
            name="EducationalSystemAgent",
            instructions="You create educational systems for fictional societies.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(education_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            systems = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(systems, list):
                if isinstance(systems, dict):
                    systems = [systems]
                else:
                    systems = []
            
            # Store each system
            saved_systems = []
            for system in systems:
                # Extract system details
                name = system.get('name')
                system_type = system.get('system_type', 'formal')
                description = system.get('description')
                target_demographics = system.get('target_demographics', ['general population'])
                controlled_by = system.get('controlled_by', 'various factions')
                core_teachings = system.get('core_teachings', ['general knowledge'])
                teaching_methods = system.get('teaching_methods', ['lecture', 'practice'])
                coming_of_age_rituals = system.get('coming_of_age_rituals')
                knowledge_restrictions = system.get('knowledge_restrictions')
                
                if not name or not description:
                    continue
                
                # Save the system
                try:
                    system_id = await self.add_educational_system(
                        run_ctx,
                        name=name,
                        system_type=system_type,
                        description=description,
                        target_demographics=target_demographics,
                        controlled_by=controlled_by,
                        core_teachings=core_teachings,
                        teaching_methods=teaching_methods,
                        coming_of_age_rituals=coming_of_age_rituals,
                        knowledge_restrictions=knowledge_restrictions
                    )
                    
                    # Add to results
                    system['id'] = system_id
                    saved_systems.append(system)
                except Exception as e:
                    logging.error(f"Error saving educational system '{name}': {e}")
            
            return saved_systems
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for educational systems: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_knowledge_traditions",
        action_description="Generating knowledge traditions for the setting",
        id_from_context=lambda ctx: "educational_system_manager"
    )
    async def generate_knowledge_traditions(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate knowledge traditions that represent how knowledge is
        passed down through generations in informal ways
        
        Returns:
            List of generated knowledge traditions
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get cultural elements for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                elements = await conn.fetch("""
                    SELECT name, type, description FROM CulturalElements
                    LIMIT 3
                """)
                
                culture_context = "\n".join([f"- {e['name']} ({e['type']}): {e['description'][:100]}..." for e in elements])
                
                # Get educational systems for context
                systems = await conn.fetch("""
                    SELECT name, system_type FROM EducationalSystems
                    LIMIT 2
                """)
                
                systems_context = ", ".join([f"{s['name']} ({s['system_type']})" for s in systems]) if systems else "No formal systems yet"
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 4-5 knowledge traditions for how specific types of knowledge are passed down
        through generations in a matriarchal society.
        
        CULTURAL CONTEXT:
        {culture_context}
        
        FORMAL EDUCATIONAL SYSTEMS:
        {systems_context}
        
        Design knowledge traditions that:
        1. Cover a range of knowledge domains (spiritual, practical, artistic, etc.)
        2. Include both female-exclusive and general knowledge traditions
        3. Show how generational knowledge transfer reinforces power structures
        4. Include different preservation methods (oral, written, ritual, etc.)
        
        Format your response as a JSON array where each object has:
        - "name": The name of the tradition
        - "tradition_type": Type of tradition (oral, written, experiential, etc.)
        - "description": A detailed description of the tradition
        - "knowledge_domain": Domain of knowledge (medicine, crafts, leadership, etc.)
        - "preservation_method": How the knowledge is preserved over generations
        - "access_requirements": Requirements to access this knowledge
        - "associated_group": Group that maintains this tradition
        - "examples": Array of specific examples of knowledge transmitted this way
        """
        
        # Create an agent for knowledge tradition generation
        tradition_agent = Agent(
            name="KnowledgeTraditionAgent",
            instructions="You create knowledge transmission traditions for fictional societies.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(tradition_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            traditions = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(traditions, list):
                if isinstance(traditions, dict):
                    traditions = [traditions]
                else:
                    traditions = []
            
            # Store each tradition
            saved_traditions = []
            for tradition in traditions:
                # Extract tradition details
                name = tradition.get('name')
                tradition_type = tradition.get('tradition_type', 'oral')
                description = tradition.get('description')
                knowledge_domain = tradition.get('knowledge_domain', 'general knowledge')
                preservation_method = tradition.get('preservation_method')
                access_requirements = tradition.get('access_requirements')
                associated_group = tradition.get('associated_group')
                examples = tradition.get('examples', [])
                
                if not name or not description:
                    continue
                
                # Save the tradition
                try:
                    tradition_id = await self.add_knowledge_tradition(
                        run_ctx,
                        name=name,
                        tradition_type=tradition_type,
                        description=description,
                        knowledge_domain=knowledge_domain,
                        preservation_method=preservation_method,
                        access_requirements=access_requirements,
                        associated_group=associated_group,
                        examples=examples
                    )
                    
                    # Add to results
                    tradition['id'] = tradition_id
                    saved_traditions.append(tradition)
                except Exception as e:
                    logging.error(f"Error saving knowledge tradition '{name}': {e}")
            
            return saved_traditions
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for knowledge traditions: {response_text}")
            return []

class GeopoliticalSystemManager(BaseLoreManager):
    """
    Manages the geopolitical landscape of the world, including countries,
    regions, foreign relations, and international power dynamics.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
        
    async def initialize_tables(self):
        """Ensure geopolitical system tables exist"""
        table_definitions = {
            "Nations": """
                CREATE TABLE Nations (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    government_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    relative_power INTEGER CHECK (relative_power BETWEEN 1 AND 10),
                    matriarchy_level INTEGER CHECK (matriarchy_level BETWEEN 1 AND 10),
                    population_scale TEXT,
                    major_resources TEXT[],
                    major_cities TEXT[],
                    cultural_traits TEXT[],
                    notable_features TEXT,
                    neighboring_nations TEXT[],
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_nations_embedding 
                ON Nations USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "InternationalRelations": """
                CREATE TABLE InternationalRelations (
                    id SERIAL PRIMARY KEY,
                    nation1_id INTEGER NOT NULL,
                    nation2_id INTEGER NOT NULL,
                    relationship_type TEXT NOT NULL,
                    relationship_quality INTEGER CHECK (relationship_quality BETWEEN 1 AND 10),
                    description TEXT NOT NULL,
                    notable_conflicts TEXT[],
                    notable_alliances TEXT[],
                    trade_relations TEXT,
                    cultural_exchanges TEXT,
                    FOREIGN KEY (nation1_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (nation2_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    UNIQUE (nation1_id, nation2_id)
                );
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_nation",
        action_description="Adding nation: {name}",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def add_nation(
        self, 
        ctx,
        name: str,
        government_type: str,
        description: str,
        relative_power: int,
        matriarchy_level: int,
        population_scale: str = None,
        major_resources: List[str] = None,
        major_cities: List[str] = None,
        cultural_traits: List[str] = None,
        notable_features: str = None,
        neighboring_nations: List[str] = None
    ) -> int:
        """
        Add a nation to the database
        
        Args:
            name: Name of the nation
            government_type: Type of government
            description: Detailed description
            relative_power: Power level (1-10)
            matriarchy_level: How matriarchal (1-10)
            population_scale: Scale of population
            major_resources: Key resources
            major_cities: Key cities/settlements
            cultural_traits: Defining cultural traits
            notable_features: Other notable features
            neighboring_nations: Nations that border this one
            
        Returns:
            ID of the created nation
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        major_resources = major_resources or []
        major_cities = major_cities or []
        cultural_traits = cultural_traits or []
        neighboring_nations = neighboring_nations or []
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation_id = await conn.fetchval("""
                    INSERT INTO Nations (
                        name, government_type, description, relative_power,
                        matriarchy_level, population_scale, major_resources,
                        major_cities, cultural_traits, notable_features,
                        neighboring_nations
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, government_type, description, relative_power,
                     matriarchy_level, population_scale, major_resources,
                     major_cities, cultural_traits, notable_features,
                     neighboring_nations)
                
                # Generate and store embedding
                embedding_text = f"{name} {government_type} {description}"
                await self.generate_and_store_embedding(embedding_text, conn, "Nations", "id", nation_id)
                
                return nation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_international_relation",
        action_description="Adding relation between nations",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def add_international_relation(
        self, 
        ctx,
        nation1_id: int,
        nation2_id: int,
        relationship_type: str,
        relationship_quality: int,
        description: str,
        notable_conflicts: List[str] = None,
        notable_alliances: List[str] = None,
        trade_relations: str = None,
        cultural_exchanges: str = None
    ) -> int:
        """
        Add a relation between two nations
        
        Args:
            nation1_id: ID of first nation
            nation2_id: ID of second nation
            relationship_type: Type of relationship (ally, rival, etc.)
            relationship_quality: Quality level (1-10)
            description: Description of relationship
            notable_conflicts: Notable conflicts
            notable_alliances: Notable alliances
            trade_relations: Description of trade
            cultural_exchanges: Description of cultural exchanges
            
        Returns:
            ID of the created relation
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults
        notable_conflicts = notable_conflicts or []
        notable_alliances = notable_alliances or []
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                relation_id = await conn.fetchval("""
                    INSERT INTO InternationalRelations (
                        nation1_id, nation2_id, relationship_type,
                        relationship_quality, description, notable_conflicts,
                        notable_alliances, trade_relations, cultural_exchanges
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (nation1_id, nation2_id) DO UPDATE
                    SET relationship_type = $3,
                        relationship_quality = $4,
                        description = $5,
                        notable_conflicts = $6,
                        notable_alliances = $7,
                        trade_relations = $8,
                        cultural_exchanges = $9
                    RETURNING id
                """, nation1_id, nation2_id, relationship_type,
                     relationship_quality, description, notable_conflicts,
                     notable_alliances, trade_relations, cultural_exchanges)
                
                return relation_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_all_nations",
        action_description="Getting all nations in the world",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def get_all_nations(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all nations in the world
        
        Returns:
            List of all nations
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nations = await conn.fetch("""
                    SELECT id, name, government_type, description, relative_power,
                           matriarchy_level, population_scale, major_resources,
                           major_cities, cultural_traits, notable_features, 
                           neighboring_nations
                    FROM Nations
                    ORDER BY relative_power DESC
                """)
                
                return [dict(nation) for nation in nations]
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_relations",
        action_description="Getting relations for nation: {nation_name}",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def get_nation_relations(self, ctx, nation_id: int, nation_name: str) -> List[Dict[str, Any]]:
        """
        Get all international relations for a specific nation
        
        Args:
            nation_id: ID of the nation
            nation_name: Name of the nation (for logs)
            
        Returns:
            List of international relations
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get relations where this nation is either nation1 or nation2
                relations = await conn.fetch("""
                    SELECT r.id, r.relationship_type, r.relationship_quality, 
                           r.description, r.notable_conflicts, r.notable_alliances,
                           r.trade_relations, r.cultural_exchanges,
                           CASE
                               WHEN r.nation1_id = $1 THEN r.nation2_id
                               ELSE r.nation1_id
                           END AS other_nation_id
                    FROM InternationalRelations r
                    WHERE r.nation1_id = $1 OR r.nation2_id = $1
                """, nation_id)
                
                # Get names for the related nations
                result = []
                for relation in relations:
                    other_id = relation['other_nation_id']
                    other_nation = await conn.fetchrow("""
                        SELECT name, government_type
                        FROM Nations
                        WHERE id = $1
                    """, other_id)
                    
                    if other_nation:
                        relation_dict = dict(relation)
                        relation_dict['other_nation_name'] = other_nation['name']
                        relation_dict['other_nation_government'] = other_nation['government_type']
                        del relation_dict['other_nation_id']
                        result.append(relation_dict)
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world_nations",
        action_description="Generating nations for the world",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def generate_world_nations(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate a set of nations for the world, ensuring a mix of matriarchal levels
        
        Returns:
            List of generated nations
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get world lore for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Try to get world history for context
                world_history = await conn.fetchval("""
                    SELECT description FROM WorldLore
                    WHERE category = 'world_history'
                    LIMIT 1
                """)
                
                social_structure = await conn.fetchval("""
                    SELECT description FROM WorldLore
                    WHERE category = 'social_structure'
                    LIMIT 1
                """)
                
                context = f"World History: {world_history[:300] if world_history else 'Not available'}\n\n"
                context += f"Social Structure: {social_structure[:300] if social_structure else 'Not available'}"
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 5-7 nations for a fantasy world with varying levels of matriarchal governance.
        
        WORLD CONTEXT:
        {context}
        
        Create a politically diverse world with nations that:
        1. Include at least one strongly matriarchal society (player's home nation)
        2. Include nations with varying degrees of matriarchal governance
        3. Show realistic geopolitical dynamics including alliances and rivalries
        4. Have distinct cultural identities and government structures
        5. Vary in size, power, and resources
        
        Format your response as a JSON array where each object has:
        - "name": The name of the nation
        - "government_type": Type of government (oligarchy, monarchy, etc.)
        - "description": A detailed description
        - "relative_power": Power level (1-10)
        - "matriarchy_level": How matriarchal (1-10, with 10 being totally female dominated)
        - "population_scale": Scale of population (small, medium, large, vast)
        - "major_resources": Array of key resources
        - "major_cities": Array of key cities/settlements
        - "cultural_traits": Array of defining cultural traits
        - "notable_features": Other notable features
        - "neighboring_nations": Array of nations that border this one (use the same names you create)
        
        NOTE: Make sure the player's home nation has a matriarchy_level of at least 8.
        """
        
        # Create an agent for nation generation
        nation_agent = Agent(
            name="GeopoliticalAgent",
            instructions="You create nations and geopolitical systems for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(nation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            nations = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(nations, list):
                if isinstance(nations, dict):
                    nations = [nations]
                else:
                    nations = []
            
            # Ensure player home nation is highly matriarchal
            home_nation_exists = False
            for nation in nations:
                if nation.get('matriarchy_level', 0) >= 8:
                    home_nation_exists = True
                    break
                    
            if not home_nation_exists and nations:
                # Make the first nation highly matriarchal
                nations[0]['matriarchy_level'] = 9
                nations[0]['description'] = f"The player's home nation. {nations[0].get('description', '')}"
            
            # Store each nation
            saved_nations = []
            nation_name_to_id = {}
            
            for nation in nations:
                # Extract nation details
                name = nation.get('name')
                government_type = nation.get('government_type', 'unknown')
                description = nation.get('description')
                relative_power = nation.get('relative_power', 5)
                matriarchy_level = nation.get('matriarchy_level', 5)
                population_scale = nation.get('population_scale')
                major_resources = nation.get('major_resources', [])
                major_cities = nation.get('major_cities', [])
                cultural_traits = nation.get('cultural_traits', [])
                notable_features = nation.get('notable_features')
                neighboring_nations = nation.get('neighboring_nations', [])
                
                if not name or not description:
                    continue
                
                # Save the nation
                try:
                    nation_id = await self.add_nation(
                        run_ctx,
                        name=name,
                        government_type=government_type,
                        description=description,
                        relative_power=relative_power,
                        matriarchy_level=matriarchy_level,
                        population_scale=population_scale,
                        major_resources=major_resources,
                        major_cities=major_cities,
                        cultural_traits=cultural_traits,
                        notable_features=notable_features,
                        neighboring_nations=neighboring_nations
                    )
                    
                    # Add to results
                    nation['id'] = nation_id
                    saved_nations.append(nation)
                    nation_name_to_id[name] = nation_id
                except Exception as e:
                    logging.error(f"Error saving nation '{name}': {e}")
            
            # Now generate international relations
            for nation in saved_nations:
                nation_id = nation['id']
                name = nation['name']
                
                for neighbor_name in nation.get('neighboring_nations', []):
                    if neighbor_name in nation_name_to_id:
                        neighbor_id = nation_name_to_id[neighbor_name]
                        
                        # Only generate relation if neighbor_id > nation_id to avoid duplicates
                        if neighbor_id > nation_id:
                            # Determine relationship based on matriarchy levels
                            matriarchy_diff = abs(nation.get('matriarchy_level', 5) - 
                                                 next((n.get('matriarchy_level', 5) for n in saved_nations if n['name'] == neighbor_name), 5))
                            
                            # Greater difference means more tension
                            relationship_quality = max(1, 10 - matriarchy_diff)
                            
                            if relationship_quality >= 7:
                                relationship_type = "ally"
                                description = f"{name} and {neighbor_name} maintain friendly diplomatic relations."
                            elif relationship_quality >= 4:
                                relationship_type = "neutral"
                                description = f"{name} and {neighbor_name} maintain cautious but functional diplomatic relations."
                            else:
                                relationship_type = "rival"
                                description = f"{name} and {neighbor_name} have tense and occasionally hostile relations."
                            
                            # Create the relation
                            try:
                                await self.add_international_relation(
                                    run_ctx,
                                    nation1_id=nation_id,
                                    nation2_id=neighbor_id,
                                    relationship_type=relationship_type,
                                    relationship_quality=relationship_quality,
                                    description=description
                                )
                            except Exception as e:
                                logging.error(f"Error saving relation between {name} and {neighbor_name}: {e}")
            
            return saved_nations
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for nations: {response_text}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_international_relations",
        action_description="Generating detailed international relations",
        id_from_context=lambda ctx: "geopolitical_system_manager"
    )
    async def generate_international_relations(self, ctx) -> List[Dict[str, Any]]:
        """
        Generate detailed international relations between existing nations
        
        Returns:
            List of generated international relations
        """
        # Create the run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Get existing nations
        nations = await self.get_all_nations(run_ctx)
        
        if len(nations) < 2:
            return []
        
        # Create nation pairs to generate relations for
        nation_pairs = []
        for i, nation1 in enumerate(nations):
            for nation2 in nations[i+1:]:
                # Check if they're neighbors
                is_neighbor = (nation2['name'] in nation1.get('neighboring_nations', []) or
                              nation1['name'] in nation2.get('neighboring_nations', []))
                
                # Add the pair
                nation_pairs.append({
                    "nation1_id": nation1['id'],
                    "nation1_name": nation1['name'],
                    "nation1_govt": nation1['government_type'],
                    "nation1_power": nation1['relative_power'],
                    "nation1_matriarchy": nation1['matriarchy_level'],
                    "nation2_id": nation2['id'],
                    "nation2_name": nation2['name'],
                    "nation2_govt": nation2['government_type'],
                    "nation2_power": nation2['relative_power'],
                    "nation2_matriarchy": nation2['matriarchy_level'],
                    "is_neighbor": is_neighbor
                })
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate detailed international relations between these {len(nation_pairs)} pairs of nations:
        
        {json.dumps(nation_pairs, indent=2)}
        
        For each pair, create a realistic international relationship that:
        1. Considers their relative matriarchy levels (more different = more tension)
        2. Considers their relative power levels (power imbalance = different dynamics)
        3. Creates specific historical events (conflicts/alliances) between them
        4. Describes trade and cultural exchange patterns
        
        Format your response as a JSON array where each object has:
        - "nation1_id": First nation's ID (use the provided values exactly)
        - "nation2_id": Second nation's ID (use the provided values exactly)
        - "relationship_type": Type of relationship (ally, rival, neutral, vassal, etc.)
        - "relationship_quality": Quality level (1-10, with 10 being extremely positive)
        - "description": Detailed description of relationship dynamics
        - "notable_conflicts": Array of 0-2 notable conflicts (if any)
        - "notable_alliances": Array of 0-2 notable alliances or treaties (if any)
        - "trade_relations": Description of trade patterns
        - "cultural_exchanges": Description of cultural influences
        """
        
        # Create an agent for relation generation
        relation_agent = Agent(
            name="GeopoliticalRelationsAgent",
            instructions="You create detailed international relations for fictional worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(relation_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            relations = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(relations, list):
                if isinstance(relations, dict):
                    relations = [relations]
                else:
                    relations = []
            
            # Store each relation
            saved_relations = []
            for relation in relations:
                # Extract relation details
                nation1_id = relation.get('nation1_id')
                nation2_id = relation.get('nation2_id')
                relationship_type = relation.get('relationship_type', 'neutral')
                relationship_quality = relation.get('relationship_quality', 5)
                description = relation.get('description', 'Relations between these nations.')
                notable_conflicts = relation.get('notable_conflicts', [])
                notable_alliances = relation.get('notable_alliances', [])
                trade_relations = relation.get('trade_relations')
                cultural_exchanges = relation.get('cultural_exchanges')
                
                if not nation1_id or not nation2_id or not description:
                    continue
                
                # Save the relation
                try:
                    relation_id = await self.add_international_relation(
                        run_ctx,
                        nation1_id=nation1_id,
                        nation2_id=nation2_id,
                        relationship_type=relationship_type,
                        relationship_quality=relationship_quality,
                        description=description,
                        notable_conflicts=notable_conflicts,
                        notable_alliances=notable_alliances,
                        trade_relations=trade_relations,
                        cultural_exchanges=cultural_exchanges
                    )
                    
                    # Add to results
                    relation['id'] = relation_id
                    saved_relations.append(relation)
                except Exception as e:
                    logging.error(f"Error saving international relation: {e}")
            
            return saved_relations
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for international relations: {response_text}")
            return []

# Initialize cache for faiths
FAITH_CACHE = LoreCache(max_size=200, ttl=7200)  # 2 hour TTL

class ReligionManager(BaseLoreManager):
    """
    Comprehensive system for managing religions, faiths, and belief systems
    within the matriarchal society, including both creation and distribution.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
        
    async def initialize_tables(self):
        """Ensure all religion-related tables exist"""
        table_definitions = {
            "Deities": """
                CREATE TABLE Deities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    gender TEXT NOT NULL, -- female, male, non-binary, etc.
                    domain TEXT[] NOT NULL, -- love, war, knowledge, etc.
                    description TEXT NOT NULL,
                    iconography TEXT,
                    holy_symbol TEXT,
                    sacred_animals TEXT[],
                    sacred_colors TEXT[],
                    relationships JSONB, -- relationships with other deities
                    rank INTEGER CHECK (rank BETWEEN 1 AND 10), -- importance in pantheon
                    worshippers TEXT[], -- types of people who worship
                    pantheon_id INTEGER,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_deities_embedding 
                ON Deities USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "Pantheons": """
                CREATE TABLE Pantheons (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    origin_story TEXT NOT NULL,
                    major_holy_days TEXT[],
                    cosmic_structure TEXT, -- how the cosmos is organized
                    afterlife_beliefs TEXT,
                    creation_myth TEXT,
                    geographical_spread TEXT[], -- regions where worshipped
                    dominant_nations TEXT[], -- nations where dominant
                    primary_worshippers TEXT[], -- demographics who worship
                    matriarchal_elements TEXT NOT NULL, -- how it reinforces matriarchy
                    taboos TEXT[],
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_pantheons_embedding 
                ON Pantheons USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "ReligiousPractices": """
                CREATE TABLE ReligiousPractices (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    practice_type TEXT NOT NULL, -- ritual, ceremony, prayer, etc.
                    description TEXT NOT NULL,
                    frequency TEXT, -- daily, weekly, yearly, etc.
                    required_elements TEXT[], -- components needed
                    performed_by TEXT[], -- priests, all worshippers, etc.
                    purpose TEXT NOT NULL, -- blessing, protection, etc.
                    restricted_to TEXT[], -- if limited to certain people
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_religiouspractices_embedding 
                ON ReligiousPractices USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "HolySites": """
                CREATE TABLE HolySites (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    site_type TEXT NOT NULL, -- temple, shrine, sacred grove, etc.
                    description TEXT NOT NULL,
                    location_id INTEGER, -- reference to Locations table
                    location_description TEXT, -- if not linked to location
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    clergy_type TEXT, -- priestesses, clerics, etc.
                    clergy_hierarchy TEXT[], -- ranks in order
                    pilgrimage_info TEXT,
                    miracles_reported TEXT[],
                    restrictions TEXT[], -- who can enter
                    architectural_features TEXT,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_holysites_embedding 
                ON HolySites USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "ReligiousTexts": """
                CREATE TABLE ReligiousTexts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    text_type TEXT NOT NULL, -- scripture, hymnal, prayer book, etc.
                    description TEXT NOT NULL,
                    authorship TEXT, -- divine, prophetic, etc.
                    key_teachings TEXT[] NOT NULL,
                    restricted_to TEXT[], -- if access is limited
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    notable_passages TEXT[],
                    age_description TEXT, -- how old it is
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_religioustexts_embedding 
                ON ReligiousTexts USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "ReligiousOrders": """
                CREATE TABLE ReligiousOrders (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    order_type TEXT NOT NULL, -- monastic, military, scholarly, etc.
                    description TEXT NOT NULL,
                    founding_story TEXT,
                    headquarters TEXT,
                    hierarchy_structure TEXT[],
                    vows TEXT[],
                    practices TEXT[],
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    gender_composition TEXT, -- female-only, primarily female, mixed, etc.
                    special_abilities TEXT[],
                    notable_members TEXT[],
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_religiousorders_embedding 
                ON ReligiousOrders USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "ReligiousConflicts": """
                CREATE TABLE ReligiousConflicts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    conflict_type TEXT NOT NULL, -- schism, holy war, theological debate, etc.
                    description TEXT NOT NULL,
                    beginning_date TEXT,
                    resolution_date TEXT,
                    status TEXT, -- ongoing, resolved, dormant, etc.
                    parties_involved TEXT[] NOT NULL,
                    core_disagreement TEXT NOT NULL,
                    casualties TEXT,
                    historical_impact TEXT,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_religiousconflicts_embedding 
                ON ReligiousConflicts USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "NationReligion": """
                CREATE TABLE NationReligion (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    state_religion BOOLEAN DEFAULT FALSE,
                    primary_pantheon_id INTEGER, -- Main pantheon if any
                    pantheon_distribution JSONB, -- Distribution of pantheons by percentage
                    religiosity_level INTEGER CHECK (religiosity_level BETWEEN 1 AND 10),
                    religious_tolerance INTEGER CHECK (religious_tolerance BETWEEN 1 AND 10),
                    religious_leadership TEXT, -- Who leads religion nationally
                    religious_laws JSONB, -- Religious laws in effect
                    religious_holidays TEXT[], -- Major religious holidays
                    religious_conflicts TEXT[], -- Current religious tensions
                    religious_minorities TEXT[], -- Description of minority faiths
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (primary_pantheon_id) REFERENCES Pantheons(id) ON DELETE SET NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_nationreligion_embedding 
                ON NationReligion USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_nationreligion_nation
                ON NationReligion(nation_id);
            """,
            
            "RegionalReligiousPractice": """
                CREATE TABLE RegionalReligiousPractice (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    practice_id INTEGER NOT NULL, -- Reference to ReligiousPractices
                    regional_variation TEXT, -- How practice differs in this region
                    importance INTEGER CHECK (importance BETWEEN 1 AND 10),
                    frequency TEXT, -- How often practiced locally
                    local_additions TEXT, -- Any local additions to the practice
                    gender_differences TEXT, -- Any local gender differences
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (practice_id) REFERENCES ReligiousPractices(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_regionalreligiouspractice_embedding 
                ON RegionalReligiousPractice USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_regionalreligiouspractice_nation
                ON RegionalReligiousPractice(nation_id);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
                    

                    
                    logging.info("RegionalReligiousPractice table created")
    
    # --- Core Faith System Methods ---
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_deity",
        action_description="Adding deity: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def add_deity(
        self, 
        ctx,
        name: str,
        gender: str,
        domain: List[str],
        description: str,
        pantheon_id: Optional[int] = None,
        iconography: Optional[str] = None,
        holy_symbol: Optional[str] = None,
        sacred_animals: Optional[List[str]] = None,
        sacred_colors: Optional[List[str]] = None,
        relationships: Optional[Dict[str, str]] = None,
        rank: int = 5,
        worshippers: Optional[List[str]] = None
    ) -> int:
        """
        Add a deity to the database with governance oversight.
        
        Args:
            name: Name of the deity
            gender: Gender of the deity (female, male, non-binary, etc.)
            domain: List of domains the deity controls
            description: Detailed description
            pantheon_id: Optional ID of the pantheon this deity belongs to
            iconography: Optional description of how the deity is depicted
            holy_symbol: Optional description of the deity's holy symbol
            sacred_animals: Optional list of sacred animals
            sacred_colors: Optional list of sacred colors
            relationships: Optional dict of relationships with other deities
            rank: Importance in pantheon (1-10)
            worshippers: Optional list of types of people who worship
            
        Returns:
            ID of the created deity
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        sacred_animals = sacred_animals or []
        sacred_colors = sacred_colors or []
        relationships = relationships or {}
        worshippers = worshippers or []
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                deity_id = await conn.fetchval("""
                    INSERT INTO Deities (
                        name, gender, domain, description, pantheon_id,
                        iconography, holy_symbol, sacred_animals, sacred_colors,
                        relationships, rank, worshippers
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    RETURNING id
                """, name, gender, domain, description, pantheon_id,
                     iconography, holy_symbol, sacred_animals, sacred_colors,
                     json.dumps(relationships), rank, worshippers)
                
                # Generate and store embedding
                embedding_text = f"{name} {gender} {' '.join(domain)} {description}"
                await self.generate_and_store_embedding(embedding_text, conn, "Deities", "id", deity_id)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("deity")
                
                return deity_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_pantheon",
        action_description="Adding pantheon: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def add_pantheon(
        self, 
        ctx,
        name: str,
        description: str,
        origin_story: str,
        matriarchal_elements: str,
        creation_myth: Optional[str] = None,
        afterlife_beliefs: Optional[str] = None,
        cosmic_structure: Optional[str] = None,
        major_holy_days: Optional[List[str]] = None,
        geographical_spread: Optional[List[str]] = None,
        dominant_nations: Optional[List[str]] = None,
        primary_worshippers: Optional[List[str]] = None,
        taboos: Optional[List[str]] = None
    ) -> int:
        """
        Add a pantheon to the database with governance oversight.
        
        Args:
            name: Name of the pantheon
            description: General description
            origin_story: How the pantheon came to be
            matriarchal_elements: How it reinforces matriarchy
            creation_myth: Optional creation myth
            afterlife_beliefs: Optional afterlife beliefs
            cosmic_structure: Optional cosmic structure
            major_holy_days: Optional list of major holy days
            geographical_spread: Optional list of regions where worshipped
            dominant_nations: Optional list of nations where dominant
            primary_worshippers: Optional list of demographics who worship
            taboos: Optional list of taboos
            
        Returns:
            ID of the created pantheon
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        major_holy_days = major_holy_days or []
        geographical_spread = geographical_spread or []
        dominant_nations = dominant_nations or []
        primary_worshippers = primary_worshippers or []
        taboos = taboos or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {description} {origin_story} {matriarchal_elements}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon_id = await conn.fetchval("""
                    INSERT INTO Pantheons (
                        name, description, origin_story, matriarchal_elements,
                        creation_myth, afterlife_beliefs, cosmic_structure,
                        major_holy_days, geographical_spread, dominant_nations,
                        primary_worshippers, taboos, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """, name, description, origin_story, matriarchal_elements,
                     creation_myth, afterlife_beliefs, cosmic_structure,
                     major_holy_days, geographical_spread, dominant_nations,
                     primary_worshippers, taboos, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("pantheon")
                
                return pantheon_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_practice",
        action_description="Adding religious practice: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def add_religious_practice(
        self, 
        ctx,
        name: str,
        practice_type: str,
        description: str,
        purpose: str,
        frequency: Optional[str] = None,
        required_elements: Optional[List[str]] = None,
        performed_by: Optional[List[str]] = None,
        restricted_to: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None
    ) -> int:
        """
        Add a religious practice to the database with governance oversight.
        
        Args:
            name: Name of the practice
            practice_type: Type of practice (ritual, ceremony, etc.)
            description: Detailed description
            purpose: Purpose of the practice
            frequency: Optional frequency (daily, yearly, etc.)
            required_elements: Optional list of components needed
            performed_by: Optional list of who performs it
            restricted_to: Optional list of who it's restricted to
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            
        Returns:
            ID of the created practice
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        required_elements = required_elements or []
        performed_by = performed_by or []
        restricted_to = restricted_to or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {practice_type} {description} {purpose}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practice_id = await conn.fetchval("""
                    INSERT INTO ReligiousPractices (
                        name, practice_type, description, purpose,
                        frequency, required_elements, performed_by,
                        restricted_to, deity_id, pantheon_id, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, practice_type, description, purpose,
                     frequency, required_elements, performed_by,
                     restricted_to, deity_id, pantheon_id, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("practice")
                
                return practice_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_holy_site",
        action_description="Adding holy site: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def add_holy_site(
        self, 
        ctx,
        name: str,
        site_type: str,
        description: str,
        clergy_type: str,
        location_id: Optional[int] = None,
        location_description: Optional[str] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        clergy_hierarchy: Optional[List[str]] = None,
        pilgrimage_info: Optional[str] = None,
        miracles_reported: Optional[List[str]] = None,
        restrictions: Optional[List[str]] = None,
        architectural_features: Optional[str] = None
    ) -> int:
        """
        Add a holy site to the database with governance oversight.
        
        Args:
            name: Name of the holy site
            site_type: Type of site (temple, shrine, etc.)
            description: Detailed description
            clergy_type: Type of clergy (priestesses, etc.)
            location_id: Optional ID in Locations table
            location_description: Optional description if no location_id
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            clergy_hierarchy: Optional list of ranks in hierarchy
            pilgrimage_info: Optional information on pilgrimages
            miracles_reported: Optional list of reported miracles
            restrictions: Optional list of restrictions on entry
            architectural_features: Optional description of features
            
        Returns:
            ID of the created holy site
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        clergy_hierarchy = clergy_hierarchy or []
        miracles_reported = miracles_reported or []
        restrictions = restrictions or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {site_type} {description} {clergy_type}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                site_id = await conn.fetchval("""
                    INSERT INTO HolySites (
                        name, site_type, description, clergy_type,
                        location_id, location_description, deity_id,
                        pantheon_id, clergy_hierarchy, pilgrimage_info,
                        miracles_reported, restrictions, architectural_features,
                        embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                """, name, site_type, description, clergy_type,
                     location_id, location_description, deity_id,
                     pantheon_id, clergy_hierarchy, pilgrimage_info,
                     miracles_reported, restrictions, architectural_features,
                     embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("site")
                
                return site_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_text",
        action_description="Adding religious text: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def add_religious_text(
        self, 
        ctx,
        name: str,
        text_type: str,
        description: str,
        key_teachings: List[str],
        authorship: Optional[str] = None,
        restricted_to: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        notable_passages: Optional[List[str]] = None,
        age_description: Optional[str] = None
    ) -> int:
        """
        Add a religious text to the database with governance oversight.
        
        Args:
            name: Name of the text
            text_type: Type of text (scripture, hymnal, etc.)
            description: Detailed description
            key_teachings: List of key teachings
            authorship: Optional description of authorship
            restricted_to: Optional list of who can access it
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            notable_passages: Optional list of notable passages
            age_description: Optional description of age
            
        Returns:
            ID of the created text
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        restricted_to = restricted_to or []
        notable_passages = notable_passages or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {text_type} {description} {' '.join(key_teachings)}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                text_id = await conn.fetchval("""
                    INSERT INTO ReligiousTexts (
                        name, text_type, description, key_teachings,
                        authorship, restricted_to, deity_id,
                        pantheon_id, notable_passages, age_description,
                        embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, text_type, description, key_teachings,
                     authorship, restricted_to, deity_id,
                     pantheon_id, notable_passages, age_description,
                     embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("text")
                
                return text_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_order",
        action_description="Adding religious order: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def add_religious_order(
        self, 
        ctx,
        name: str,
        order_type: str,
        description: str,
        gender_composition: str,
        founding_story: Optional[str] = None,
        headquarters: Optional[str] = None,
        hierarchy_structure: Optional[List[str]] = None,
        vows: Optional[List[str]] = None,
        practices: Optional[List[str]] = None,
        deity_id: Optional[int] = None,
        pantheon_id: Optional[int] = None,
        special_abilities: Optional[List[str]] = None,
        notable_members: Optional[List[str]] = None
    ) -> int:
        """
        Add a religious order to the database with governance oversight.
        
        Args:
            name: Name of the order
            order_type: Type of order (monastic, military, etc.)
            description: Detailed description
            gender_composition: Gender makeup (female-only, etc.)
            founding_story: Optional founding story
            headquarters: Optional headquarters location
            hierarchy_structure: Optional list of ranks
            vows: Optional list of vows taken
            practices: Optional list of practices
            deity_id: Optional ID of associated deity
            pantheon_id: Optional ID of associated pantheon
            special_abilities: Optional list of special abilities
            notable_members: Optional list of notable members
            
        Returns:
            ID of the created order
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Set defaults for optional parameters
        hierarchy_structure = hierarchy_structure or []
        vows = vows or []
        practices = practices or []
        special_abilities = special_abilities or []
        notable_members = notable_members or []
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {order_type} {description} {gender_composition}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                order_id = await conn.fetchval("""
                    INSERT INTO ReligiousOrders (
                        name, order_type, description, gender_composition,
                        founding_story, headquarters, hierarchy_structure,
                        vows, practices, deity_id, pantheon_id,
                        special_abilities, notable_members, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                """, name, order_type, description, gender_composition,
                     founding_story, headquarters, hierarchy_structure,
                     vows, practices, deity_id, pantheon_id,
                     special_abilities, notable_members, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("order")
                
                return order_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_conflict",
        action_description="Adding religious conflict: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def add_religious_conflict(
        self, 
        ctx,
        name: str,
        conflict_type: str,
        description: str,
        parties_involved: List[str],
        core_disagreement: str,
        beginning_date: Optional[str] = None,
        resolution_date: Optional[str] = None,
        status: str = "ongoing",
        casualties: Optional[str] = None,
        historical_impact: Optional[str] = None
    ) -> int:
        """
        Add a religious conflict to the database with governance oversight.
        
        Args:
            name: Name of the conflict
            conflict_type: Type of conflict (schism, holy war, etc.)
            description: Detailed description
            parties_involved: List of parties involved
            core_disagreement: Central point of disagreement
            beginning_date: Optional textual beginning date
            resolution_date: Optional textual resolution date
            status: Status (ongoing, resolved, dormant, etc.)
            casualties: Optional description of casualties
            historical_impact: Optional description of impact
            
        Returns:
            ID of the created conflict
        """
        # Ensure tables exist
        await self.initialize_tables()
        
        # Generate embedding for semantic search
        embedding_text = f"{name} {conflict_type} {description} {core_disagreement}"
        embedding = await generate_embedding(embedding_text)
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflict_id = await conn.fetchval("""
                    INSERT INTO ReligiousConflicts (
                        name, conflict_type, description, parties_involved,
                        core_disagreement, beginning_date, resolution_date,
                        status, casualties, historical_impact, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """, name, conflict_type, description, parties_involved,
                     core_disagreement, beginning_date, resolution_date,
                     status, casualties, historical_impact, embedding)
                
                # Clear relevant cache
                FAITH_CACHE.invalidate_pattern("conflict")
                
                return conflict_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_pantheon",
        action_description="Generating pantheon for the world",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_pantheon(self, ctx) -> Dict[str, Any]:
        """
        Generate a complete pantheon for the world with governance oversight.
        
        Returns:
            Dictionary with the pantheon and its deities
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get world info for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get foundation lore for context
                foundation_lore = await conn.fetch("""
                    SELECT category, description FROM WorldLore
                    WHERE category in ('cosmology', 'magic_system', 'social_structure')
                """)
                
                foundation_context = {}
                for row in foundation_lore:
                    foundation_context[row['category']] = row['description']
                
                # Get some geographical regions for context
                regions = await conn.fetch("""
                    SELECT name FROM GeographicRegions
                    LIMIT 5
                """)
                
                region_names = [r['name'] for r in regions]
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, matriarchy_level FROM Nations
                    ORDER BY matriarchy_level DESC
                    LIMIT 5
                """)
                
                nation_context = ""
                for row in nations:
                    nation_context += f"{row['name']} (matriarchy level: {row['matriarchy_level']}), "
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a complete feminine-dominated pantheon for a matriarchal fantasy world.
        
        WORLD CONTEXT:
        Cosmology: {foundation_context.get('cosmology', 'Not available')}
        Magic System: {foundation_context.get('magic_system', 'Not available')}
        Social Structure: {foundation_context.get('social_structure', 'Not available')}
        
        Geographic Regions: {', '.join(region_names)}
        Nations: {nation_context}
        
        Create a pantheon that:
        1. Is predominantly female with goddesses in all major positions of power
        2. Includes a few male deities in subservient or specialized roles
        3. Has a clear hierarchical structure reinforcing feminine dominance
        4. Includes domains that reflect gender power dynamics
        5. Has a cosmic structure that reinforces matriarchal principles
        
        Return a JSON object with:
        1. "pantheon" - details about the overall pantheon
        2. "deities" - array of deity objects
        
        For the pantheon include:
        - name, description, origin_story, matriarchal_elements, creation_myth,
          afterlife_beliefs, cosmic_structure, major_holy_days, geographical_spread,
          dominant_nations, primary_worshippers, taboos
        
        For each deity include:
        - name, gender, domain (array), description, iconography, holy_symbol,
          sacred_animals (array), sacred_colors (array), rank (1-10),
          worshippers (array), relationships (to other deities as an object)
        """
        
        # Create an agent for pantheon generation
        pantheon_agent = Agent(
            name="PantheonGenerationAgent",
            instructions="You create religious pantheons for matriarchal fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(pantheon_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            pantheon_data = json.loads(response_text)
            
            # Ensure we have both pantheon and deities
            if not all([
                isinstance(pantheon_data, dict),
                "pantheon" in pantheon_data,
                "deities" in pantheon_data,
                isinstance(pantheon_data["deities"], list)
            ]):
                raise ValueError("Invalid pantheon structure in response")
            
            # Save the pantheon
            pantheon_info = pantheon_data["pantheon"]
            deities_info = pantheon_data["deities"]
            
            # Create the pantheon
            pantheon_id = await self.add_pantheon(
                run_ctx,
                name=pantheon_info.get("name", "The Pantheon"),
                description=pantheon_info.get("description", ""),
                origin_story=pantheon_info.get("origin_story", ""),
                matriarchal_elements=pantheon_info.get("matriarchal_elements", ""),
                creation_myth=pantheon_info.get("creation_myth"),
                afterlife_beliefs=pantheon_info.get("afterlife_beliefs"),
                cosmic_structure=pantheon_info.get("cosmic_structure"),
                major_holy_days=pantheon_info.get("major_holy_days"),
                geographical_spread=pantheon_info.get("geographical_spread"),
                dominant_nations=pantheon_info.get("dominant_nations"),
                primary_worshippers=pantheon_info.get("primary_worshippers"),
                taboos=pantheon_info.get("taboos")
            )
            
            # Create each deity
            created_deities = []
            for deity in deities_info:
                try:
                    deity_id = await self.add_deity(
                        run_ctx,
                        name=deity.get("name", "Unnamed Deity"),
                        gender=deity.get("gender", "female"),
                        domain=deity.get("domain", []),
                        description=deity.get("description", ""),
                        pantheon_id=pantheon_id,
                        iconography=deity.get("iconography"),
                        holy_symbol=deity.get("holy_symbol"),
                        sacred_animals=deity.get("sacred_animals"),
                        sacred_colors=deity.get("sacred_colors"),
                        relationships=deity.get("relationships", {}),
                        rank=deity.get("rank", 5),
                        worshippers=deity.get("worshippers")
                    )
                    
                    deity["id"] = deity_id
                    created_deities.append(deity)
                except Exception as e:
                    logging.error(f"Error creating deity {deity.get('name')}: {e}")
            
            # Return the created pantheon and deities
            return {
                "pantheon": {**pantheon_info, "id": pantheon_id},
                "deities": created_deities
            }
            
        except Exception as e:
            logging.error(f"Error generating pantheon: {e}")
            return {"error": str(e)}
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_religious_practices",
        action_description="Generating religious practices for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_religious_practices(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate religious practices for a pantheon with governance oversight.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of generated religious practices
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious practices for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        DEITIES:
        {json.dumps(deities_data, indent=2)}
        
        Create 5-7 religious practices that:
        1. Reinforce matriarchal dominance themes
        2. Include varied practice types (daily rituals, seasonal ceremonies, rites of passage, etc.)
        3. Provide specific details on how they're performed
        4. Show which deities they are associated with
        
        Return a JSON array where each practice has:
        - name: Name of the practice
        - practice_type: Type of practice (ritual, ceremony, prayer, etc.)
        - description: Detailed description of the practice
        - purpose: Purpose of the practice
        - frequency: How often it's performed
        - required_elements: Array of required components
        - performed_by: Array of who performs it
        - restricted_to: Array of who it's restricted to (if applicable)
        - deity_id: ID of the associated deity (use exact IDs from the provided deity list)
        """
        
        # Create an agent for practice generation
        practice_agent = Agent(
            name="ReligiousPracticeAgent",
            instructions="You create religious practices for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(practice_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            practices = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(practices, list):
                if isinstance(practices, dict):
                    practices = [practices]
                else:
                    practices = []
            
            # Save each practice
            created_practices = []
            for practice in practices:
                try:
                    practice_id = await self.add_religious_practice(
                        run_ctx,
                        name=practice.get("name", "Unnamed Practice"),
                        practice_type=practice.get("practice_type", "ritual"),
                        description=practice.get("description", ""),
                        purpose=practice.get("purpose", "worship"),
                        frequency=practice.get("frequency"),
                        required_elements=practice.get("required_elements"),
                        performed_by=practice.get("performed_by"),
                        restricted_to=practice.get("restricted_to"),
                        deity_id=practice.get("deity_id"),
                        pantheon_id=pantheon_id
                    )
                    
                    practice["id"] = practice_id
                    created_practices.append(practice)
                except Exception as e:
                    logging.error(f"Error creating religious practice {practice.get('name')}: {e}")
            
            return created_practices
        except Exception as e:
            logging.error(f"Error generating religious practices: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_holy_sites",
        action_description="Generating holy sites for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_holy_sites(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate holy sites for a pantheon with governance oversight.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of generated holy sites
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon and location info
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, geographical_spread, dominant_nations
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                # Get the major deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain
                    FROM Deities
                    WHERE pantheon_id = $1 AND rank >= 6
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Get potential locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    LIMIT 10
                """)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
                location_data = [dict(location) for location in locations]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate holy sites for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        MAJOR DEITIES:
        {json.dumps(deities_data, indent=2)}
        
        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}
        
        Create 3-5 holy sites that:
        1. Reflect matriarchal dominance in architecture and function
        2. Include sites for major deities and some for the pantheon as a whole
        3. Have distinct clergy systems with feminine leadership
        4. Include varied site types (temples, shrines, sacred groves, etc.)
        
        Return a JSON array where each site has:
        - name: Name of the holy site
        - site_type: Type of site (temple, shrine, sacred grove, etc.)
        - description: Detailed description
        - clergy_type: Type of clergy (priestesses, etc.)
        - location_id: ID of the location (use exact IDs from the provided locations, or null)
        - location_description: Description of the location if no location_id provided
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - clergy_hierarchy: Array of ranks in the clergy hierarchy
        - pilgrimage_info: Information about pilgrimages (if applicable)
        - miracles_reported: Array of reported miracles (if applicable)
        - restrictions: Array of restrictions on entry
        - architectural_features: Architectural features of the site
        """
        
        # Create an agent for holy site generation
        site_agent = Agent(
            name="HolySiteAgent",
            instructions="You create holy sites for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(site_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            sites = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(sites, list):
                if isinstance(sites, dict):
                    sites = [sites]
                else:
                    sites = []
            
            # Save each holy site
            created_sites = []
            for site in sites:
                try:
                    site_id = await self.add_holy_site(
                        run_ctx,
                        name=site.get("name", "Unnamed Site"),
                        site_type=site.get("site_type", "temple"),
                        description=site.get("description", ""),
                        clergy_type=site.get("clergy_type", "priestesses"),
                        location_id=site.get("location_id"),
                        location_description=site.get("location_description"),
                        deity_id=site.get("deity_id"),
                        pantheon_id=pantheon_id,
                        clergy_hierarchy=site.get("clergy_hierarchy"),
                        pilgrimage_info=site.get("pilgrimage_info"),
                        miracles_reported=site.get("miracles_reported"),
                        restrictions=site.get("restrictions"),
                        architectural_features=site.get("architectural_features")
                    )
                    
                    site["id"] = site_id
                    created_sites.append(site)
                except Exception as e:
                    logging.error(f"Error creating holy site {site.get('name')}: {e}")
            
            return created_sites
        except Exception as e:
            logging.error(f"Error generating holy sites: {e}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_faith_system",
        action_description="Generating complete faith system for the world",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """
        Generate a complete faith system for the world with governance oversight.
        
        Returns:
            Dictionary with all faith system components
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # 1. Generate pantheon
        pantheon_data = await self.generate_pantheon(run_ctx)
        
        if "error" in pantheon_data:
            return pantheon_data
            
        pantheon_id = pantheon_data["pantheon"]["id"]
        
        # 2. Generate religious practices
        practices = await self.generate_religious_practices(run_ctx, pantheon_id)
        
        # 3. Generate holy sites
        holy_sites = await self.generate_holy_sites(run_ctx, pantheon_id)
        
        # 4. Generate religious texts
        religious_texts = await self._generate_religious_texts(run_ctx, pantheon_id)
        
        # 5. Generate religious orders
        religious_orders = await self._generate_religious_orders(run_ctx, pantheon_id)
        
        # 6. Generate religious conflicts
        religious_conflicts = await self._generate_religious_conflicts(run_ctx, pantheon_id)
        
        # Combine all results
        result = {
            "pantheon": pantheon_data["pantheon"],
            "deities": pantheon_data["deities"],
            "practices": practices,
            "holy_sites": holy_sites,
            "religious_texts": religious_texts,
            "religious_orders": religious_orders,
            "religious_conflicts": religious_conflicts
        }
        
        return result
        
    # -- Helper methods for faith generation --
    
    async def _generate_religious_texts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious texts for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, creation_myth
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious texts for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        CREATION MYTH: {pantheon_data.get('creation_myth')}
        
        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}
        
        Create 3-5 religious texts that:
        1. Reinforce matriarchal principles and feminine divine superiority
        2. Include varied text types (core scripture, commentaries, prayers, etc.)
        3. Describe who has access to each text
        4. Include specific key teachings
        
        Return a JSON array where each text has:
        - name: Name of the text
        - text_type: Type of text (scripture, hymnal, prayer book, etc.)
        - description: Detailed description
        - key_teachings: Array of key teachings
        - authorship: Description of authorship
        - restricted_to: Array of who can access it (if applicable)
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - notable_passages: Array of notable passages
        - age_description: Description of the text's age
        """
        
        # Create an agent for text generation
        text_agent = Agent(
            name="ReligiousTextAgent",
            instructions="You create religious texts for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(text_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            texts = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(texts, list):
                if isinstance(texts, dict):
                    texts = [texts]
                else:
                    texts = []
            
            # Save each religious text
            created_texts = []
            for text in texts:
                try:
                    text_id = await self.add_religious_text(
                        run_ctx,
                        name=text.get("name", "Unnamed Text"),
                        text_type=text.get("text_type", "scripture"),
                        description=text.get("description", ""),
                        key_teachings=text.get("key_teachings", []),
                        authorship=text.get("authorship"),
                        restricted_to=text.get("restricted_to"),
                        deity_id=text.get("deity_id"),
                        pantheon_id=pantheon_id,
                        notable_passages=text.get("notable_passages"),
                        age_description=text.get("age_description")
                    )
                    
                    text["id"] = text_id
                    created_texts.append(text)
                except Exception as e:
                    logging.error(f"Error creating religious text {text.get('name')}: {e}")
            
            return created_texts
        except Exception as e:
            logging.error(f"Error generating religious texts: {e}")
            return []
    
    async def _generate_religious_orders(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious orders for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get the deities in this pantheon
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                
                # Get holy sites for potential headquarters
                holy_sites = await conn.fetch("""
                    SELECT id, name, site_type
                    FROM HolySites
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                deities_data = [dict(deity) for deity in deities]
                site_data = [dict(site) for site in holy_sites]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious orders for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        MATRIARCHAL ELEMENTS: {pantheon_data.get('matriarchal_elements')}
        
        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}
        
        HOLY SITES (potential headquarters):
        {json.dumps(site_data, indent=2)}
        
        Create 3-4 religious orders that:
        1. Heavily emphasize female leadership and matriarchal structure
        2. Include varied order types (monastic, military, scholarly, etc.)
        3. Have clear gender compositions (most should be female-dominated)
        4. Include details on hierarchies and practices
        
        Return a JSON array where each order has:
        - name: Name of the order
        - order_type: Type of order (monastic, military, scholarly, etc.)
        - description: Detailed description
        - gender_composition: Gender makeup (female-only, primarily female, mixed, etc.)
        - founding_story: Founding story
        - headquarters: Headquarters location (can reference holy sites)
        - hierarchy_structure: Array of ranks in hierarchy (from highest to lowest)
        - vows: Array of vows taken by members
        - practices: Array of practices
        - deity_id: ID of the associated deity (use exact IDs from the provided deities, or null)
        - special_abilities: Array of special abilities (if applicable)
        - notable_members: Array of notable members (if applicable)
        """
        
        # Create an agent for order generation
        order_agent = Agent(
            name="ReligiousOrderAgent",
            instructions="You create religious orders for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(order_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            orders = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(orders, list):
                if isinstance(orders, dict):
                    orders = [orders]
                else:
                    orders = []
            
            # Save each religious order
            created_orders = []
            for order in orders:
                try:
                    order_id = await self.add_religious_order(
                        run_ctx,
                        name=order.get("name", "Unnamed Order"),
                        order_type=order.get("order_type", "monastic"),
                        description=order.get("description", ""),
                        gender_composition=order.get("gender_composition", "female-only"),
                        founding_story=order.get("founding_story"),
                        headquarters=order.get("headquarters"),
                        hierarchy_structure=order.get("hierarchy_structure"),
                        vows=order.get("vows"),
                        practices=order.get("practices"),
                        deity_id=order.get("deity_id"),
                        pantheon_id=pantheon_id,
                        special_abilities=order.get("special_abilities"),
                        notable_members=order.get("notable_members")
                    )
                    
                    order["id"] = order_id
                    created_orders.append(order)
                except Exception as e:
                    logging.error(f"Error creating religious order {order.get('name')}: {e}")
            
            return created_orders
        except Exception as e:
            logging.error(f"Error generating religious orders: {e}")
            return []
    
    async def _generate_religious_conflicts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Helper method to generate religious conflicts for a pantheon"""
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon info
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon details
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return []
                
                # Get religious orders for potential conflict parties
                orders = await conn.fetch("""
                    SELECT id, name, order_type, gender_composition
                    FROM ReligiousOrders
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
                
                # Get nations for potential conflicts
                nations = await conn.fetch("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    LIMIT 5
                """)
                
                # Convert to dictionaries
                pantheon_data = dict(pantheon)
                order_data = [dict(order) for order in orders]
                nation_data = [dict(nation) for nation in nations]
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate religious conflicts for this pantheon:
        
        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}
        
        RELIGIOUS ORDERS (potential conflict parties):
        {json.dumps(order_data, indent=2)}
        
        NATIONS (potential conflict locations):
        {json.dumps(nation_data, indent=2)}
        
        Create 2-3 religious conflicts that:
        1. Show theological or power struggles within the faith
        2. Include conflicts that highlight gender dynamics (not just female vs male)
        3. Include different conflict types (schisms, theological debates, holy wars)
        4. Have realistic core disagreements
        
        Return a JSON array where each conflict has:
        - name: Name of the conflict
        - conflict_type: Type of conflict (schism, holy war, theological debate, etc.)
        - description: Detailed description
        - parties_involved: Array of parties involved
        - core_disagreement: Central point of disagreement
        - beginning_date: Textual beginning date
        - resolution_date: Textual resolution date (if resolved)
        - status: Status (ongoing, resolved, dormant)
        - casualties: Description of casualties (if applicable)
        - historical_impact: Description of historical impact
        """
        
        # Create an agent for conflict generation
        conflict_agent = Agent(
            name="ReligiousConflictAgent",
            instructions="You create religious conflicts for fantasy pantheons.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            conflicts = json.loads(response_text)
            
            # Ensure we got a list
            if not isinstance(conflicts, list):
                if isinstance(conflicts, dict):
                    conflicts = [conflicts]
                else:
                    conflicts = []
            
            # Save each religious conflict
            created_conflicts = []
            for conflict in conflicts:
                try:
                    conflict_id = await self.add_religious_conflict(
                        run_ctx,
                        name=conflict.get("name", "Unnamed Conflict"),
                        conflict_type=conflict.get("conflict_type", "schism"),
                        description=conflict.get("description", ""),
                        parties_involved=conflict.get("parties_involved", []),
                        core_disagreement=conflict.get("core_disagreement", ""),
                        beginning_date=conflict.get("beginning_date"),
                        resolution_date=conflict.get("resolution_date"),
                        status=conflict.get("status", "ongoing"),
                        casualties=conflict.get("casualties"),
                        historical_impact=conflict.get("historical_impact")
                    )
                    
                    conflict["id"] = conflict_id
                    created_conflicts.append(conflict)
                except Exception as e:
                    logging.error(f"Error creating religious conflict {conflict.get('name')}: {e}")
            
            return created_conflicts
        except Exception as e:
            logging.error(f"Error generating religious conflicts: {e}")
            return []
    
    # --- Distribution Methods ---
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="distribute_religions",
        action_description="Distributing religions across nations",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        """
        Distribute religions across nations with governance oversight.
        
        Returns:
            List of national religion distributions
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nations and pantheons
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        # Get pantheons through the faith system
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheons = await conn.fetch("""
                    SELECT id, name, description, matriarchal_elements
                    FROM Pantheons
                """)
                
                # Convert to list of dicts
                pantheon_data = [dict(pantheon) for pantheon in pantheons]
        
        if not nations or not pantheon_data:
            return []
        
        # Create agent for religious distribution
        distribution_agent = Agent(
            name="ReligiousDistributionAgent",
            instructions="You distribute religious pantheons across fantasy nations in a matriarchal world.",
            model="o3-mini"
        )
        
        distributions = []
        for nation in nations:
            # Create prompt for distribution
            prompt = f"""
            Determine religious distribution for this nation:
            
            NATION:
            {json.dumps(nation, indent=2)}
            
            AVAILABLE PANTHEONS:
            {json.dumps(pantheon_data, indent=2)}
            
            Create a realistic religious distribution that:
            1. Considers the nation's matriarchy level ({nation.get("matriarchy_level", 5)}/10)
            2. Determines whether it has a state religion
            3. Distributes pantheons in percentages
            4. Establishes religious laws and practices
            5. Emphasizes matriarchal and feminine aspects of religion
            
            Return a JSON object with:
            - nation_id: The nation ID
            - state_religion: Boolean indicating if there's a state religion
            - primary_pantheon_id: ID of main pantheon (or null if none)
            - pantheon_distribution: Object mapping pantheon IDs to percentage of population
            - religiosity_level: Overall religiosity (1-10)
            - religious_tolerance: Tolerance level (1-10)
            - religious_leadership: Who leads religion nationally (favor matriarchal leadership)
            - religious_laws: Object describing religious laws in effect
            - religious_holidays: Array of major religious holidays
            - religious_conflicts: Array of current religious tensions
            - religious_minorities: Array of minority faith descriptions
            """
            
            # Get response from agent
            result = await Runner.run(distribution_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                distribution_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in distribution_data for k in ["nation_id", "religiosity_level"]):
                    continue
                
                # Generate embedding
                embedding_text = f"religion {nation['name']} {distribution_data.get('religious_leadership', '')} {distribution_data.get('religious_tolerance', 5)}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        distribution_id = await conn.fetchval("""
                            INSERT INTO NationReligion (
                                nation_id, state_religion, primary_pantheon_id, pantheon_distribution,
                                religiosity_level, religious_tolerance, religious_leadership,
                                religious_laws, religious_holidays, religious_conflicts,
                                religious_minorities, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            RETURNING id
                        """,
                        distribution_data.get("nation_id"),
                        distribution_data.get("state_religion", False),
                        distribution_data.get("primary_pantheon_id"),
                        json.dumps(distribution_data.get("pantheon_distribution", {})),
                        distribution_data.get("religiosity_level", 5),
                        distribution_data.get("religious_tolerance", 5),
                        distribution_data.get("religious_leadership"),
                        json.dumps(distribution_data.get("religious_laws", {})),
                        distribution_data.get("religious_holidays", []),
                        distribution_data.get("religious_conflicts", []),
                        distribution_data.get("religious_minorities", []),
                        embedding)
                        
                        distribution_data["id"] = distribution_id
                        distributions.append(distribution_data)
                        
                        # Now generate regional religious practices
                        await self._generate_regional_practices(run_ctx, distribution_data)
            
            except Exception as e:
                logging.error(f"Error distributing religion for nation {nation['id']}: {e}")
        
        return distributions

    async def _generate_regional_practices(self, ctx, distribution_data: Dict[str, Any]) -> None:
        """Generate regional variations of religious practices"""
        # Get pantheons and practices
        nation_id = distribution_data.get("nation_id")
        primary_pantheon_id = distribution_data.get("primary_pantheon_id")
        
        if not primary_pantheon_id:
            return
        
        # Get religious practices for this pantheon
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practices = await conn.fetch("""
                    SELECT id, name, practice_type, description, purpose
                    FROM ReligiousPractices
                    WHERE pantheon_id = $1
                """, primary_pantheon_id)
                
                # Convert to list of dicts
                practice_data = [dict(practice) for practice in practices]
                
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                nation_data = dict(nation) if nation else {}
        
        if not practice_data or not nation_data:
            return
        
        # Create agent for regional practice generation
        practice_agent = Agent(
            name="RegionalPracticeAgent",
            instructions="You create regional variations of religious practices for a matriarchal society.",
            model="o3-mini"
        )
        
        for practice in practice_data:
            # Create prompt for practice variation
            prompt = f"""
            Create a regional variation of this religious practice for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            RELIGIOUS PRACTICE:
            {json.dumps(practice, indent=2)}
            
            RELIGIOUS CONTEXT:
            Religiosity level: {distribution_data.get("religiosity_level", 5)}
            Religious tolerance: {distribution_data.get("religious_tolerance", 5)}
            
            Create a regional variation that:
            1. Adapts the practice to local culture
            2. Considers the nation's matriarchy level ({nation_data.get("matriarchy_level", 5)}/10)
            3. Emphasizes feminine power and authority
            4. Feels authentic to both the practice and the nation
            
            Return a JSON object with:
            - practice_id: ID of the original practice
            - regional_variation: How the practice is modified regionally
            - importance: Importance in this region (1-10)
            - frequency: How often practiced locally
            - local_additions: Any local additions to the practice
            - gender_differences: Any local gender differences
            """
            
            # Get response from agent
            result = await Runner.run(practice_agent, prompt, context=ctx.context)
            
            try:
                # Parse JSON response
                variation_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in variation_data for k in ["practice_id", "regional_variation"]):
                    continue
                
                # Generate embedding
                embedding_text = f"practice {practice['name']} {variation_data['regional_variation']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO RegionalReligiousPractice (
                                nation_id, practice_id, regional_variation,
                                importance, frequency, local_additions,
                                gender_differences, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        nation_id,
                        variation_data.get("practice_id"),
                        variation_data.get("regional_variation"),
                        variation_data.get("importance", 5),
                        variation_data.get("frequency"),
                        variation_data.get("local_additions"),
                        variation_data.get("gender_differences"),
                        embedding)
            
            except Exception as e:
                logging.error(f"Error generating regional practice variation: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_religion",
        action_description="Getting religious information for nation {nation_id}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def get_nation_religion(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive religious information about a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with nation's religious information
        """
        # Check cache first
        cache_key = f"nation_religion_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = FAITH_CACHE.get(cache_key)
        if cached:
            return cached
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return {"error": "Nation not found"}
                
                # Get religious distribution
                religion = await conn.fetchrow("""
                    SELECT * FROM NationReligion
                    WHERE nation_id = $1
                """, nation_id)
                
                if not religion:
                    return {"error": "No religious data for this nation"}
                
                # Get primary pantheon
                primary_pantheon_id = religion["primary_pantheon_id"]
                primary_pantheon = None
                if primary_pantheon_id:
                    pantheon = await conn.fetchrow("""
                        SELECT id, name, description, matriarchal_elements
                        FROM Pantheons
                        WHERE id = $1
                    """, primary_pantheon_id)
                    
                    if pantheon:
                        primary_pantheon = dict(pantheon)
                
                # Get regional practices
                practices = await conn.fetch("""
                    SELECT r.*, p.name as practice_name, p.practice_type, p.purpose
                    FROM RegionalReligiousPractice r
                    JOIN ReligiousPractices p ON r.practice_id = p.id
                    WHERE r.nation_id = $1
                """, nation_id)
                
                # Get holy sites in this nation
                holy_sites = await conn.fetch("""
                    SELECT h.* 
                    FROM HolySites h
                    JOIN Locations l ON h.location_id = l.id
                    JOIN LoreConnections lc ON l.id = lc.target_id
                    JOIN Nations n ON lc.source_id = n.id
                    WHERE n.id = $1 AND lc.source_type = 'Nations' AND lc.target_type = 'Locations'
                """, nation_id)
                
                # Compile result
                result = {
                    "nation": dict(nation),
                    "religion": dict(religion),
                    "primary_pantheon": primary_pantheon,
                    "regional_practices": [dict(practice) for practice in practices],
                    "holy_sites": [dict(site) for site in holy_sites]
                }
                
                # Parse JSON fields
                if "pantheon_distribution" in result["religion"] and result["religion"]["pantheon_distribution"]:
                    try:
                        result["religion"]["pantheon_distribution"] = json.loads(result["religion"]["pantheon_distribution"])
                    except:
                        pass
                
                if "religious_laws" in result["religion"] and result["religion"]["religious_laws"]:
                    try:
                        result["religion"]["religious_laws"] = json.loads(result["religion"]["religious_laws"])
                    except:
                        pass
                
                # Cache the result
                FAITH_CACHE.set(cache_key, result)
                
                return result

    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="religion_manager",
            directive_text="Create and manage faith systems that emphasize feminine divine superiority.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        
        logging.info(f"ReligionManager registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")

# -------------------------------------------------
# INTEGRATED MATRIARCHAL LORE SYSTEM
# -------------------------------------------------

class MatriarchalLoreSystem(LoreDynamicsSystem):
    """
    Master class that integrates all lore systems with a matriarchal theme focus.
    Acts as the primary interface for all lore generation and updates.
    Inherits from LoreDynamicsSystem for lore evolution capabilities.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        
        # Initialize specialized sub-systems
        self.culture_system = RegionalCultureSystem(user_id, conversation_id)
        self.conflict_system = NationalConflictSystem(user_id, conversation_id)
        self.religion_system = ReligionManager(user_id, conversation_id)
        self.local_lore_system = LocalLoreManager(user_id, conversation_id)
        # Note: geopolitical_manager is already initialized in the parent class
        self.cache_prefix = "matriarchal_lore"
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            # Initialize all sub-systems
            await self.initialize_all_systems()
    
    async def initialize_all_systems(self):
        """Initialize all subsystems and their tables"""
        await self.culture_system.ensure_initialized()
        await self.conflict_system.ensure_initialized()
        await self.religion_system.ensure_initialized()
        await self.local_lore_system.ensure_initialized()
        
        # Register systems with governance
        await self.culture_system.register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="culture_system",
            directive_text="Create and manage cultural norms, languages, and customs with matriarchal themes."
        )
        
        await self.conflict_system.register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="conflict_system",
            directive_text="Create and manage national conflicts that reflect matriarchal power dynamics."
        )
        
        await self.religion_system.register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="religion_system",
            directive_text="Create and manage faith systems that emphasize feminine divine superiority."
        )
        
        await self.local_lore_system.register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="local_lore_system",
            directive_text="Create and manage local lore, myths, and histories with matriarchal influences."
        )
        
        logging.info(f"All matriarchal lore systems initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_world",
        action_description="Generating complete matriarchal world lore",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def generate_complete_world(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete world with matriarchal theming
        
        Args:
            environment_desc: Description of the environment/setting
            
        Returns:
            Dictionary containing the complete world lore
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # 1. First generate base lore through DynamicLoreGenerator
        dynamic_lore = DynamicLoreGenerator(self.user_id, self.conversation_id)
        
        # Check for directives before proceeding
        await dynamic_lore._check_and_process_directives()
        
        # Generate foundation lore
        foundation_data = await dynamic_lore.initialize_world_lore(environment_desc)
        
        # Apply matriarchal theming to foundation lore
        for key, content in foundation_data.items():
            foundation_data[key] = MatriarchalThemingUtils.apply_matriarchal_theme(key, content)
        
        # 2. Generate factions with matriarchal power structures
        factions_data = await dynamic_lore.generate_factions(environment_desc, foundation_data)
        
        # 3. Generate cultural elements
        cultural_data = await dynamic_lore.generate_cultural_elements(environment_desc, factions_data)
        
        # 4. Generate historical events emphasizing matriarchal history
        historical_data = await dynamic_lore.generate_historical_events(
            environment_desc, foundation_data, factions_data
        )
        
        # 5. Generate locations
        locations_data = await dynamic_lore.generate_locations(environment_desc, factions_data)
        
        # 6. Generate quest hooks
        quests_data = await dynamic_lore.generate_quest_hooks(factions_data, locations_data)
        
        # 7. Generate world nations
        nations = await self.geopolitical_manager.generate_world_nations(run_ctx)
        
        # 8. Generate languages through our regional culture system
        languages = await self.culture_system.generate_languages(run_ctx, count=5)
        
        # 9. Generate and distribute religions
        religious_data = await self.religion_system.generate_complete_faith_system(run_ctx)
        
        # 10. Generate conflicts
        conflicts = await self.conflict_system.generate_initial_conflicts(run_ctx, count=3)
        
        # 11. For each nation, generate cultural norms and domestic issues
        nation_details = []
        for nation in nations:
            # Generate cultural norms
            norms = await self.culture_system.generate_cultural_norms(run_ctx, nation["id"])
            
            # Generate etiquette
            etiquette = await self.culture_system.generate_etiquette(run_ctx, nation["id"])
            
            # Generate domestic issues
            issues = await self.conflict_system.generate_domestic_issues(run_ctx, nation["id"])
            
            nation_details.append({
                "nation_id": nation["id"],
                "name": nation["name"],
                "cultural_norms": norms,
                "etiquette": etiquette,
                "domestic_issues": issues
            })
        
        # 12. Generate local lore for each location
        local_lore = []
        for location in locations_data.get("locations", [])[:5]:  # Limit to first 5 to avoid too much generation
            if "id" in location:
                lore = await self.local_lore_system.generate_location_lore(run_ctx, location)
                local_lore.append(lore)
        
        # Combine all results
        complete_lore = {
            "world_lore": foundation_data,
            "factions": factions_data,
            "cultural_elements": cultural_data,
            "historical_events": historical_data,
            "locations": locations_data,
            "quests": quests_data,
            "nations": nations,
            "languages": languages,
            "religions": religious_data,
            "conflicts": conflicts,
            "nation_details": nation_details,
            "local_lore": local_lore
        }
        
        return complete_lore
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="handle_narrative_event",
        action_description="Handling narrative event impacts",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def handle_narrative_event(
        self, 
        ctx,
        event_description: str,
        affected_lore_ids: List[str] = None,
        player_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle impacts of a narrative event on the world
        
        Args:
            event_description: Description of the event that occurred
            affected_lore_ids: Optional list of specifically affected lore IDs
            player_data: Optional player character data
            
        Returns:
            Dictionary with all updates applied
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # First apply matriarchal theming to the event description
        themed_event = MatriarchalThemingUtils.apply_matriarchal_theme("event", event_description, emphasis_level=1)
        
        # Use the LoreDynamicsSystem's evolve_lore_with_event method
        lore_updates = await self.evolve_lore_with_event(themed_event)
        
        # Process other impacts
        
        # Check if the event should affect conflicts or domestic issues
        event_impact = await self._calculate_event_impact(themed_event)
        
        # If significant impact, evolve conflicts and issues
        conflict_results = None
        if event_impact > 6:
            # Evolve conflicts and domestic issues
            conflict_results = await self.conflict_system.evolve_all_conflicts(run_ctx, days_passed=7)
        
        # Update local lore if the event is location-specific
        local_updates = []
        if "location" in themed_event.lower() or "place" in themed_event.lower():
            try:
                # Try to extract location references
                locations = await self._extract_location_references(themed_event)
                
                # Update each referenced location
                for location_id in locations:
                    local_update = await self.local_lore_system.evolve_location_lore(
                        run_ctx, location_id, themed_event
                    )
                    local_updates.append(local_update)
            except Exception as e:
                logging.error(f"Error updating local lore: {e}")
        
        return {
            "event": themed_event,
            "original_event": event_description,
            "lore_updates": lore_updates,
            "event_impact": event_impact,
            "conflict_results": conflict_results,
            "local_lore_updates": local_updates
        }
    
    async def _extract_location_references(self, event_text: str) -> List[int]:
        """
        Extract location references from event text
        
        Args:
            event_text: Event description
            
        Returns:
            List of location IDs
        """
        # Get all location names
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                locations = await conn.fetch("""
                    SELECT id, location_name
                    FROM Locations
                """)
                
                # Check for location names in event text
                referenced_ids = []
                for location in locations:
                    if location["location_name"].lower() in event_text.lower():
                        referenced_ids.append(location["id"])
                
                return referenced_ids
    
    async def get_comprehensive_world_state(self, ctx) -> Dict[str, Any]:
        """
        Get a comprehensive view of the current world state
        
        Returns:
            Dictionary with current world state
        """
        # Check cache first
        cache_key = "comprehensive_world_state"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        run_ctx = self.create_run_context(ctx)
        
        # Get basic world state
        world_state = await self._fetch_world_state()
        
        # Get nations
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        # Get active conflicts
        conflicts = await self.conflict_system.get_active_conflicts(run_ctx)
        
        # Get religions
        religions = await self.religion_system.get_pantheons(run_ctx)
        
        # Get a sample of locations
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                locations = await conn.fetch("""
                    SELECT id, location_name, location_type, description
                    FROM Locations
                    LIMIT 10
                """)
                
                location_data = [dict(loc) for loc in locations]
        
        # Compile result
        result = {
            "world_state": world_state,
            "nations": nations,
            "conflicts": conflicts,
            "religions": religions,
            "locations": location_data,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Cache result
        self.set_cache(cache_key, result)
        
        return result
    
    async def generate_additional_faction(self, ctx, faction_type: str = None) -> Dict[str, Any]:
        """
        Generate an additional faction with governance oversight.
        
        Args:
            faction_type: Optional type of faction to generate
            
        Returns:
            New faction details
        """
        # Create the run context
        run_ctx = self.create_run_context(ctx)
        
        # Get existing factions for context
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get existing factions
                existing_factions = await conn.fetch("""
                    SELECT name, type, description, values, goals
                    FROM Factions
                    LIMIT 10
                """)
                
                # Get nations for context
                nations = await conn.fetch("""
                    SELECT name, government_type, matriarchy_level
                    FROM Nations
                    LIMIT 5
                """)
                
                # Get some cultural elements for context
                cultural_elements = await conn.fetch("""
                    SELECT name, type, description
                    FROM CulturalElements
                    ORDER BY RANDOM()
                    LIMIT 5
                """)
                
                # Convert to lists
                faction_data = [dict(faction) for faction in existing_factions]
                nation_data = [dict(nation) for nation in nations]
                cultural_data = [dict(element) for element in cultural_elements]
        
        # If faction_type not specified, select one that's underrepresented
        if not faction_type:
            faction_types = ["political", "religious", "criminal", "mercantile", "academic", "military", "social"]
            existing_types = [f.get("type", "").lower() for f in faction_data]
            
            # Count each type
            type_counts = {}
            for t in faction_types:
                type_counts[t] = sum(1 for et in existing_types if t in et)
            
            # Find underrepresented types
            min_count = min(type_counts.values()) if type_counts else 0
            underrepresented = [t for t, c in type_counts.items() if c == min_count]
            
            # Choose one randomly from underrepresented
            faction_type = random.choice(underrepresented if underrepresented else faction_types)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a new {faction_type} faction for a matriarchal fantasy world:
        
        EXISTING FACTIONS:
        {json.dumps(faction_data, indent=2)}
        
        NATIONS:
        {json.dumps(nation_data, indent=2)}
        
        CULTURAL CONTEXT:
        {json.dumps(cultural_data, indent=2)}
        
        Create a {faction_type} faction that:
        1. Is distinct from existing factions
        2. Has a clear role and purpose in the world
        3. Has interesting values and goals
        4. Reflects matriarchal power dynamics
        5. Has potential for interesting stories and conflicts
        
        Return a JSON object with:
        - name: Name of the faction
        - type: Type of faction (should include "{faction_type}")
        - description: Detailed description
        - values: Array of core values
        - goals: Array of primary goals
        - headquarters: Main location or headquarters
        - founding_story: How the faction was founded
        - rivals: Array of rival factions or groups
        - allies: Array of allied factions or groups
        - territory: Areas where they operate
        - resources: Key resources they control
        - hierarchy_type: Structure of leadership
        - secret_knowledge: Any secret knowledge they possess
        - public_reputation: How they're viewed publicly
        - color_scheme: Colors associated with them
        - symbol_description: Their emblem or symbol
        """
        
        # Create an agent for faction generation
        faction_agent = Agent(
            name="FactionGenerationAgent",
            instructions="You create factions for fantasy worlds.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(faction_agent, prompt, context=run_ctx.context)
        response_text = result.final_output
        
        try:
            # Parse the JSON response
            faction_data = json.loads(response_text)
            
            # Apply matriarchal theming
            if "description" in faction_data:
                faction_data["description"] = MatriarchalThemingUtils.apply_matriarchal_theme(
                    "faction", faction_data["description"], emphasis_level=2
                )
            
            # Add faction to database
            faction_id = await self.lore_manager.add_faction(
                name=faction_data.get("name", f"Unnamed {faction_type.capitalize()} Faction"),
                faction_type=faction_data.get("type", faction_type),
                description=faction_data.get("description", ""),
                values=faction_data.get("values", []),
                goals=faction_data.get("goals", []),
                headquarters=faction_data.get("headquarters"),
                founding_story=faction_data.get("founding_story"),
                rivals=faction_data.get("rivals", []),
                allies=faction_data.get("allies", []),
                territory=faction_data.get("territory"),
                resources=faction_data.get("resources", []),
                hierarchy_type=faction_data.get("hierarchy_type"),
                secret_knowledge=faction_data.get("secret_knowledge"),
                public_reputation=faction_data.get("public_reputation"),
                color_scheme=faction_data.get("color_scheme"),
                symbol_description=faction_data.get("symbol_description")
            )
            
            # Update with the ID
            faction_data["id"] = faction_id
            
            return faction_data
            
        except Exception as e:
            logging.error(f"Error generating additional faction: {e}")
            return {"error": str(e)}
            
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="matriarchal_lore_system",
            directive_text="Create and maintain a cohesive world with matriarchal themes and power dynamics.",
            scope="world_building",
            priority=DirectivePriority.HIGH
        )
        
        logging.info(f"MatriarchalLoreSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")


# -------------------------------------------------
# MATRIARCHAL THEMING UTILITIES
# -------------------------------------------------

# A dictionary of regex patterns to replacement strings for basic feminization
# \b ensures "god" won't become "Goddess" inside words like "good"
_FEMDOM_WORD_MAP = {
    r"\bgod\b": "Goddess",
    r"\bgods\b": "Goddesses",
    r"\bgodhood\b": "Goddesshood",
    r"\bking\b": "Queen",
    r"\bkings\b": "Queens",
    r"\bkingdom\b": "Queendom",
    r"\bprince\b": "princess",
    r"\bprinces\b": "princesses",
    r"\bfather\b": "mother",
    r"\bfathers\b": "mothers",
    r"\bhim\b": "her",
    r"\bhis\b": "her",
    r"\bhe\b": "she",
    r"\blord\b": "lady",
    r"\blords\b": "ladies",
    r"\bman\b": "woman",
    r"\bmen\b": "women",
}

# Random synonyms for a supreme feminine figure
_GODDESS_SYNONYMS = [
    "Supreme Goddess",
    "High Empress",
    "Great Matriarch",
    "Divine Mother",
    "Infinite Mistress of Creation",
]

# Default emphasis level for matriarchal theming (1=low, 3=very high)
_DEFAULT_EMPHASIS_LEVEL = 2


def _apply_basic_replacements(text: str) -> str:
    """
    Runs a set of regex-based replacements to feminize words/phrases.
    Respects case; if the original word is capitalized, keep it capitalized.
    """
    result = text

    for pattern_str, replacement_str in _FEMDOM_WORD_MAP.items():
        pattern = re.compile(pattern_str, re.IGNORECASE)

        def _replacement_func(match):
            original = match.group(0)
            # If the original word starts with uppercase, we uppercase the replacement's first letter
            if original and original[0].isupper():
                return replacement_str.capitalize()
            return replacement_str

        result = pattern.sub(_replacement_func, result)

    return result


def _ensure_goddess_reference(text: str) -> str:
    """
    If there's no mention of 'Goddess' or a similar figure, insert a default reference
    to a supreme feminine force at the end of the text.
    """
    if not re.search(r"(goddess|divine mother|matriarch|empress of creation)", text, re.IGNORECASE):
        chosen_title = random.choice(_GODDESS_SYNONYMS)
        insertion = (
            f"\n\nAt the cosmic center stands {chosen_title}, "
            "the eternal wellspring of existence. Her dominion weaves reality itself."
        )
        text += insertion

    return text


def _inject_contextual_lore(text: str, insertion: str, label: str = "", append_if_not_found: bool = True) -> str:
    """
    Insert 'insertion' text after a specific label or heading (case-insensitive).
    If label not found and append_if_not_found is True, just append at the end.
    """
    if not label:
        # No label provided, always append at the end
        return text.strip() + "\n\n" + insertion

    pattern = re.compile(rf"({re.escape(label)})", re.IGNORECASE)
    match = pattern.search(text)

    if match:
        # Insert after the matched label
        idx = match.end()
        # Insert a blank line before the snippet
        return text[:idx] + "\n\n" + insertion + text[idx:]
    else:
        if append_if_not_found:
            return text.strip() + "\n\n" + insertion
        else:
            # Return unchanged if not found and we don't want to append
            return text


def _emphasize_feminine_power(text: str, emphasis_level: int = _DEFAULT_EMPHASIS_LEVEL) -> str:
    """
    Add extra lines emphasizing matriarchal/feminine power, depending on emphasis_level.
    1 = minimal emphasis
    2 = moderate emphasis
    3 = strong emphasis
    """
    if emphasis_level >= 2:
        text += (
            "\n\nThroughout every realm, the ascendancy of womanhood underpins all aspects of life, "
            "unquestioned and unassailable."
        )
    if emphasis_level >= 3:
        text += (
            "\n\nNo force—mortal or divine—can transcend the supremacy of the feminine essence, "
            "the primal wellspring of creation itself."
        )
    return text

class MatriarchalThemingUtils:
    """
    Streamlined utility class for applying matriarchal theming to different types of lore content.
    """
    
    # A dictionary of regex patterns to replacement strings for basic feminization
    _FEMDOM_WORD_MAP = {
        r"\bgod\b": "Goddess",
        r"\bgods\b": "Goddesses",
        r"\bgodhood\b": "Goddesshood",
        r"\bking\b": "Queen",
        r"\bkings\b": "Queens",
        r"\bkingdom\b": "Queendom",
        r"\bprince\b": "princess",
        r"\bprinces\b": "princesses",
        r"\bfather\b": "mother",
        r"\bfathers\b": "mothers",
        r"\bhim\b": "her",
        r"\bhis\b": "her",
        r"\bhe\b": "she",
        r"\blord\b": "lady",
        r"\blords\b": "ladies",
        r"\bman\b": "woman",
        r"\bmen\b": "women",
    }
    
    # Random synonyms for a supreme feminine figure
    _GODDESS_SYNONYMS = [
        "Supreme Goddess",
        "High Empress",
        "Great Matriarch",
        "Divine Mother",
        "Infinite Mistress of Creation",
    ]
    
    # Theme-specific content insertions
    _THEME_CONTENT = {
        "cosmology": "At the heart of all creation is the Feminine Principle, the source of all life and power. "
                    "The cosmos itself is understood as fundamentally feminine in nature, "
                    "with any masculine elements serving and supporting the greater feminine whole.",
                    
        "magic_system": "The flow and expression of magical energies reflect the natural order of feminine dominance. "
                        "Women typically possess greater innate magical potential and exclusive rights to the highest mysteries. "
                        "Men specializing in arcane arts often excel in supportive, protective, or enhancing magics, "
                        "operating in service to more powerful feminine traditions. "
                        "Only when guided by a woman's touch do the most potent rituals fully manifest.",
                        
        "world_history": "Throughout recorded chronicles, women have held the reins of power. "
                        "Great Empresses, Matriarchs, and female rulers have guided civilizations toward prosperity. "
                        "Though conflicts and rebellions against this natural order have arisen, "
                        "the unshakable principle of feminine dominance remains the bedrock of history.",
                        
        "calendar_system": "The calendar marks vital dates in feminine history, aligning festivals and holy days "
                          "with lunar cycles and the reigns of legendary Empresses. Major celebrations honor "
                          "the cyclical power of womanhood, reflecting its role in birth, renewal, and creation."
    }
    
    @staticmethod
    def _apply_basic_replacements(text: str) -> str:
        """
        Runs a set of regex-based replacements to feminize words/phrases.
        Respects case; if the original word is capitalized, keep it capitalized.
        """
        result = text

        for pattern_str, replacement_str in MatriarchalThemingUtils._FEMDOM_WORD_MAP.items():
            pattern = re.compile(pattern_str, re.IGNORECASE)

            def _replacement_func(match):
                original = match.group(0)
                # If the original word starts with uppercase, we uppercase the replacement's first letter
                if original and original[0].isupper():
                    return replacement_str.capitalize()
                return replacement_str

            result = pattern.sub(_replacement_func, result)

        return result
    
    @staticmethod
    def _ensure_goddess_reference(text: str) -> str:
        """
        If there's no mention of 'Goddess' or a similar figure, insert a default reference
        to a supreme feminine force at the end of the text.
        """
        if not re.search(r"(goddess|divine mother|matriarch|empress of creation)", text, re.IGNORECASE):
            chosen_title = random.choice(MatriarchalThemingUtils._GODDESS_SYNONYMS)
            insertion = (
                f"\n\nAt the cosmic center stands {chosen_title}, "
                "the eternal wellspring of existence. Her dominion weaves reality itself."
            )
            text += insertion

        return text
    
    @staticmethod
    def _inject_themed_content(text: str, lore_type: str) -> str:
        """
        Insert themed content based on lore type
        
        Args:
            text: Original text
            lore_type: Type of lore
            
        Returns:
            Text with inserted themed content
        """
        if lore_type in MatriarchalThemingUtils._THEME_CONTENT:
            themed_content = MatriarchalThemingUtils._THEME_CONTENT[lore_type]
            
            # Try to find an appropriate section heading
            pattern = re.compile(rf"({re.escape(lore_type)})", re.IGNORECASE)
            match = pattern.search(text)
            
            if match:
                # Insert after the matched label
                idx = match.end()
                return text[:idx] + "\n\n" + themed_content + text[idx:]
            else:
                # Append at the end if no appropriate place found
                return text.strip() + "\n\n" + themed_content
        
        return text
    
    @staticmethod
    def _emphasize_feminine_power(text: str, emphasis_level: int = 2) -> str:
        """
        Add extra lines emphasizing matriarchal/feminine power, depending on emphasis_level.
        1 = minimal emphasis
        2 = moderate emphasis
        3 = strong emphasis
        """
        if emphasis_level >= 2:
            text += (
                "\n\nThroughout every realm, the ascendancy of womanhood underpins all aspects of life, "
                "unquestioned and unassailable."
            )
        if emphasis_level >= 3:
            text += (
                "\n\nNo force—mortal or divine—can transcend the supremacy of the feminine essence, "
                "the primal wellspring of creation itself."
            )
        return text
    
    @staticmethod
    def apply_matriarchal_theme(lore_type: str, content: str, emphasis_level: int = 2) -> str:
        """
        Apply appropriate matriarchal theming based on lore type.
        
        Args:
            lore_type: Type of lore content ('cosmology', 'magic_system', 'history', etc.)
            content: Original content to modify
            emphasis_level: Level of emphasis on matriarchal themes (1-3)
            
        Returns:
            Modified content with matriarchal theming
        """
        # 1. Apply basic word replacements
        result = MatriarchalThemingUtils._apply_basic_replacements(content)
        
        # 2. Ensure goddess reference
        result = MatriarchalThemingUtils._ensure_goddess_reference(result)
        
        # 3. Inject theme-specific content
        result = MatriarchalThemingUtils._inject_themed_content(result, lore_type)
        
        # 4. Add emphasis
        result = MatriarchalThemingUtils._emphasize_feminine_power(result, emphasis_level)
        
        return result


# -------------------------------------------------
# REGIONAL CULTURE SYSTEM
# -------------------------------------------------

class RegionalCultureSystem(BaseLoreManager):
    """
    Manages culturally specific norms, customs, manners, and languages
    across different regions and nations.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
        
    async def initialize_tables(self):
        """Ensure regional culture tables exist"""
        table_definitions = {
            "Languages": """
                CREATE TABLE Languages (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    language_family TEXT,
                    description TEXT NOT NULL,
                    writing_system TEXT,
                    primary_regions INTEGER[], -- Nation IDs where primarily spoken
                    minority_regions INTEGER[], -- Nation IDs where spoken by minorities
                    formality_levels TEXT[], -- Different levels of formality
                    common_phrases JSONB, -- Basic phrases in this language
                    difficulty INTEGER CHECK (difficulty BETWEEN 1 AND 10),
                    relation_to_power TEXT, -- How language relates to power structures
                    dialects JSONB, -- Regional variations
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_languages_embedding 
                ON Languages USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "CulturalNorms": """
                CREATE TABLE CulturalNorms (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL, -- Nation this applies to
                    category TEXT NOT NULL, -- greeting, dining, authority, gift, etc.
                    description TEXT NOT NULL, -- Detailed description
                    formality_level TEXT, -- casual, formal, ceremonial
                    gender_specific BOOLEAN DEFAULT FALSE, -- If norm differs by gender
                    female_variation TEXT, -- Female-specific version if applicable
                    male_variation TEXT, -- Male-specific version if applicable
                    taboo_level INTEGER CHECK (taboo_level BETWEEN 0 AND 10), -- How taboo breaking this is
                    consequence TEXT, -- Consequence of breaking norm
                    regional_variations JSONB, -- Variations within the nation
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_culturalnorms_embedding 
                ON CulturalNorms USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_culturalnorms_nation
                ON CulturalNorms(nation_id);
            """,
            
            "Etiquette": """
                CREATE TABLE Etiquette (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL, -- Nation this applies to
                    context TEXT NOT NULL, -- Context (court, public, private, etc.)
                    title_system TEXT, -- How titles work
                    greeting_ritual TEXT, -- How people greet each other
                    body_language TEXT, -- Expected body language
                    eye_contact TEXT, -- Eye contact norms
                    distance_norms TEXT, -- Personal space norms
                    gift_giving TEXT, -- Gift-giving norms
                    dining_etiquette TEXT, -- Table manners
                    power_display TEXT, -- How power is displayed
                    respect_indicators TEXT, -- How respect is shown
                    gender_distinctions TEXT, -- How gender impacts etiquette
                    taboos TEXT[], -- Things never to do
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_etiquette_embedding 
                ON Etiquette USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_etiquette_nation
                ON Etiquette(nation_id);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_languages",
        action_description="Generating languages for the world",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_languages(self, ctx, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate languages for the world with governance oversight.
        
        Args:
            count: Number of languages to generate
            
        Returns:
            List of generated languages
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nations for context
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        if not nations:
            return []
        
        # Create agent for language generation
        language_agent = Agent(
            name="LanguageGenerationAgent",
            instructions="You create realistic languages for fantasy worlds with matriarchal power structures.",
            model="o3-mini"
        )
        
        languages = []
        for i in range(count):
            # Determine language distribution
            # For simplicity: create some major languages and some minor ones
            is_major = i < count // 2
            
            if is_major:
                # Major language used by multiple nations
                primary_nations = random.sample(nations, min(3, len(nations)))
                minority_nations = random.sample([n for n in nations if n not in primary_nations], 
                                               min(2, len(nations) - len(primary_nations)))
            else:
                # More localized language
                primary_nations = random.sample(nations, 1)
                minority_nations = random.sample([n for n in nations if n not in primary_nations], 
                                               min(2, len(nations) - 1))
            
            # Create prompt for language generation
            prompt = f"""
            Generate a detailed language for a fantasy world with matriarchal power structures.
            
            PRIMARY NATIONS:
            {json.dumps([{
                "name": n.get("name", "Unknown"),
                "government_type": n.get("government_type", "Unknown"),
                "matriarchy_level": n.get("matriarchy_level", 5)
            } for n in primary_nations], indent=2)}
            
            MINORITY NATIONS:
            {json.dumps([{
                "name": n.get("name", "Unknown"),
                "government_type": n.get("government_type", "Unknown"),
                "matriarchy_level": n.get("matriarchy_level", 5)
            } for n in minority_nations], indent=2)}
            
            Create a {'major regional' if is_major else 'localized'} language that:
            1. Reflects the matriarchal power structures of the world
            2. Has realistic features and complexity
            3. Includes information about how formality and power are expressed
            4. Includes feminine-dominant linguistic features
            5. Has some common phrases or expressions
            
            Return a JSON object with:
            - name: Name of the language
            - language_family: Linguistic family it belongs to
            - description: Detailed description of the language
            - writing_system: How it's written (if at all)
            - formality_levels: Array of formality levels (from casual to formal)
            - common_phrases: Object with key phrases (greeting, farewell, etc.)
            - difficulty: How hard it is to learn (1-10)
            - relation_to_power: How the language reflects power dynamics
            - dialects: Object with different regional dialects
            """
            
            # Get response from agent
            result = await Runner.run(language_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                language_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in language_data for k in ["name", "description"]):
                    continue
                
                # Add nation IDs
                language_data["primary_regions"] = [n["id"] for n in primary_nations]
                language_data["minority_regions"] = [n["id"] for n in minority_nations]
                
                # Generate embedding
                embedding_text = f"{language_data['name']} {language_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        language_id = await conn.fetchval("""
                            INSERT INTO Languages (
                                name, language_family, description, writing_system,
                                primary_regions, minority_regions, formality_levels,
                                common_phrases, difficulty, relation_to_power, dialects
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            RETURNING id
                        """,
                        language_data.get("name"),
                        language_data.get("language_family"),
                        language_data.get("description"),
                        language_data.get("writing_system"),
                        language_data.get("primary_regions"),
                        language_data.get("minority_regions"),
                        language_data.get("formality_levels"),
                        json.dumps(language_data.get("common_phrases", {})),
                        language_data.get("difficulty", 5),
                        language_data.get("relation_to_power"),
                        json.dumps(language_data.get("dialects", {})))
                        
                        # Generate and store embedding
                        embedding_text = f"{language_data['name']} {language_data['description']}"
                        await self.generate_and_store_embedding(embedding_text, conn, "Languages", "id", language_id)
                        
                        language_data["id"] = language_id
                        languages.append(language_data)
            
            except Exception as e:
                logging.error(f"Error generating language: {e}")
        
        return languages
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_cultural_norms",
        action_description="Generating cultural norms for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_cultural_norms(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate cultural norms for a specific nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of generated cultural norms
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nation details
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return []
                
                nation_data = dict(nation)
        
        # Create agent for cultural norm generation
        norm_agent = Agent(
            name="CulturalNormAgent",
            instructions="You create cultural norms for fantasy nations with matriarchal power structures.",
            model="o3-mini"
        )
        
        # Categories of norms to generate
        categories = [
            "greeting", "dining", "authority", "gift_giving", "personal_space", 
            "gender_relations", "age_relations", "public_behavior", "private_behavior", 
            "business_conduct", "religious_practice"
        ]
        
        norms = []
        for category in categories:
            # Create prompt for norm generation
            prompt = f"""
            Generate cultural norms about {category} for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            Create detailed cultural norms that:
            1. Reflect the nation's matriarchy level ({nation_data.get("matriarchy_level", 5)}/10)
            2. Feel authentic and consistent with the nation's traits
            3. Include both dos and don'ts
            4. Specify if norms differ by gender
            
            Return a JSON object with:
            - category: "{category}"
            - description: Detailed description of the norm
            - formality_level: Level of formality (casual, formal, ceremonial)
            - gender_specific: Boolean - whether norm differs by gender
            - female_variation: Female-specific version if applicable
            - male_variation: Male-specific version if applicable
            - taboo_level: How taboo breaking this is (0-10)
            - consequence: Consequence of breaking norm
            - regional_variations: Object with any variations within the nation
            """
            
            # Get response from agent
            result = await Runner.run(norm_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                norm_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in norm_data for k in ["category", "description"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{category} {norm_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        norm_id = await conn.fetchval("""
                            INSERT INTO CulturalNorms (
                                nation_id, category, description, formality_level,
                                gender_specific, female_variation, male_variation,
                                taboo_level, consequence, regional_variations, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            RETURNING id
                        """,
                        nation_id,
                        norm_data.get("category"),
                        norm_data.get("description"),
                        norm_data.get("formality_level"),
                        norm_data.get("gender_specific", False),
                        norm_data.get("female_variation"),
                        norm_data.get("male_variation"),
                        norm_data.get("taboo_level", 5),
                        norm_data.get("consequence"),
                        json.dumps(norm_data.get("regional_variations", {})),
                        embedding)
                        
                        norm_data["id"] = norm_id
                        norm_data["nation_id"] = nation_id
                        norms.append(norm_data)
            
            except Exception as e:
                logging.error(f"Error generating cultural norm for category {category}: {e}")
        
        return norms

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_etiquette",
        action_description="Generating etiquette for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_etiquette(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate etiquette systems for a specific nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of generated etiquette systems
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nation details
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return []
                
                nation_data = dict(nation)
        
        # Create agent for etiquette generation
        etiquette_agent = Agent(
            name="EtiquetteAgent",
            instructions="You create etiquette systems for fantasy nations with matriarchal power structures.",
            model="o3-mini"
        )
        
        # Contexts for etiquette
        contexts = ["court", "noble", "public", "private", "religious", "business"]
        
        etiquette_systems = []
        for context in contexts:
            # Create prompt for etiquette generation
            prompt = f"""
            Generate an etiquette system for {context} contexts in this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            Create a detailed etiquette system that:
            1. Reflects the nation's matriarchy level ({nation_data.get("matriarchy_level", 5)}/10)
            2. Provides clear rules for behavior in {context} settings
            3. Specifies how power and respect are shown
            4. Includes gender-specific elements that reflect feminine authority
            
            Return a JSON object with:
            - context: "{context}"
            - title_system: How titles and forms of address work
            - greeting_ritual: How people greet each other
            - body_language: Expected body language
            - eye_contact: Eye contact norms
            - distance_norms: Personal space norms
            - gift_giving: Gift-giving norms
            - dining_etiquette: Table manners
            - power_display: How power is displayed
            - respect_indicators: How respect is shown
            - gender_distinctions: How gender impacts etiquette
            - taboos: Array of things never to do
            """
            
            # Get response from agent
            result = await Runner.run(etiquette_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                etiquette_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in etiquette_data for k in ["context", "greeting_ritual"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{context} etiquette {etiquette_data['greeting_ritual']} {etiquette_data.get('respect_indicators', '')}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        etiquette_id = await conn.fetchval("""
                            INSERT INTO Etiquette (
                                nation_id, context, title_system, greeting_ritual,
                                body_language, eye_contact, distance_norms, gift_giving,
                                dining_etiquette, power_display, respect_indicators,
                                gender_distinctions, taboos, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                            RETURNING id
                        """,
                        nation_id,
                        etiquette_data.get("context"),
                        etiquette_data.get("title_system"),
                        etiquette_data.get("greeting_ritual"),
                        etiquette_data.get("body_language"),
                        etiquette_data.get("eye_contact"),
                        etiquette_data.get("distance_norms"),
                        etiquette_data.get("gift_giving"),
                        etiquette_data.get("dining_etiquette"),
                        etiquette_data.get("power_display"),
                        etiquette_data.get("respect_indicators"),
                        etiquette_data.get("gender_distinctions"),
                        etiquette_data.get("taboos", []),
                        embedding)
                        
                        etiquette_data["id"] = etiquette_id
                        etiquette_data["nation_id"] = nation_id
                        etiquette_systems.append(etiquette_data)
            
            except Exception as e:
                logging.error(f"Error generating etiquette for context {context}: {e}")
        
        return etiquette_systems

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_culture",
        action_description="Getting cultural information for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def get_nation_culture(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive cultural information about a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with nation's cultural information
        """
        # Check cache first
        cache_key = f"nation_culture_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = CULTURE_CACHE.get(cache_key)
        if cached:
            return cached
        
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return {"error": "Nation not found"}
                
                # Get languages
                languages = await conn.fetch("""
                    SELECT id, name, description, writing_system, formality_levels
                    FROM Languages
                    WHERE $1 = ANY(primary_regions) OR $1 = ANY(minority_regions)
                """, nation_id)
                
                # Get cultural norms
                norms = await conn.fetch("""
                    SELECT id, category, description, formality_level, gender_specific,
                           female_variation, male_variation, taboo_level, consequence
                    FROM CulturalNorms
                    WHERE nation_id = $1
                """, nation_id)
                
                # Get etiquette
                etiquette = await conn.fetch("""
                    SELECT id, context, title_system, greeting_ritual, power_display,
                           respect_indicators, gender_distinctions, taboos
                    FROM Etiquette
                    WHERE nation_id = $1
                """, nation_id)
                
                # Compile result
                result = {
                    "nation": dict(nation),
                    "languages": {
                        "primary": [dict(lang) for lang in languages if nation_id in lang["primary_regions"]],
                        "minority": [dict(lang) for lang in languages if nation_id in lang["minority_regions"]]
                    },
                    "cultural_norms": [dict(norm) for norm in norms],
                    "etiquette": [dict(etiq) for etiq in etiquette]
                }
                
                # Cache the result
                CULTURE_CACHE.set(cache_key, result)
                
                return result


# -------------------------------------------------
# NATIONAL CONFLICT SYSTEM
# -------------------------------------------------

class NationalConflictSystem(BaseLoreManager):
    """
    System for managing, generating, and evolving national and international
    conflicts that serve as background elements in the world.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
        
    async def initialize_tables(self):
        """Ensure conflict system tables exist"""
        table_definitions = {
            "NationalConflicts": """
                CREATE TABLE NationalConflicts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    conflict_type TEXT NOT NULL, -- war, trade_dispute, diplomatic_tension, etc.
                    description TEXT NOT NULL,
                    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                    status TEXT NOT NULL, -- active, resolved, escalating, de-escalating
                    start_date TEXT NOT NULL,
                    end_date TEXT, -- NULL if ongoing
                    involved_nations INTEGER[], -- IDs of nations involved
                    primary_aggressor INTEGER, -- Nation ID of aggressor
                    primary_defender INTEGER, -- Nation ID of defender
                    current_casualties TEXT, -- Description of casualties so far
                    economic_impact TEXT, -- Description of economic impact
                    diplomatic_consequences TEXT, -- Description of diplomatic fallout
                    public_opinion JSONB, -- Public opinion in different nations
                    recent_developments TEXT[], -- Recent events in the conflict
                    potential_resolution TEXT, -- Potential ways it might end
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_nationalconflicts_embedding 
                ON NationalConflicts USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "ConflictNews": """
                CREATE TABLE ConflictNews (
                    id SERIAL PRIMARY KEY,
                    conflict_id INTEGER NOT NULL,
                    headline TEXT NOT NULL,
                    content TEXT NOT NULL,
                    publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_nation INTEGER, -- Nation ID where this news originated
                    bias TEXT, -- pro_aggressor, pro_defender, neutral
                    embedding VECTOR(1536),
                    FOREIGN KEY (conflict_id) REFERENCES NationalConflicts(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_conflictnews_embedding 
                ON ConflictNews USING ivfflat (embedding vector_cosine_ops);
            """,
            
            "DomesticIssues": """
                CREATE TABLE DomesticIssues (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    issue_type TEXT NOT NULL, -- civil_rights, political_controversy, economic_crisis, etc.
                    description TEXT NOT NULL,
                    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                    status TEXT NOT NULL, -- emerging, active, waning, resolved
                    start_date TEXT NOT NULL,
                    end_date TEXT, -- NULL if ongoing
                    supporting_factions TEXT[], -- Groups supporting one side
                    opposing_factions TEXT[], -- Groups opposing
                    neutral_factions TEXT[], -- Groups remaining neutral
                    affected_demographics TEXT[], -- Demographics most affected
                    public_opinion JSONB, -- Opinion distribution
                    government_response TEXT, -- How the government is responding
                    recent_developments TEXT[], -- Recent events in this issue
                    political_impact TEXT, -- Impact on political landscape
                    social_impact TEXT, -- Impact on society
                    economic_impact TEXT, -- Economic consequences
                    potential_resolution TEXT, -- Potential ways it might resolve
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_domesticissues_embedding 
                ON DomesticIssues USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_domesticissues_nation
                ON DomesticIssues(nation_id);
            """,
            
            "DomesticNews": """
                CREATE TABLE DomesticNews (
                    id SERIAL PRIMARY KEY,
                    issue_id INTEGER NOT NULL,
                    headline TEXT NOT NULL,
                    content TEXT NOT NULL,
                    publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_faction TEXT, -- Faction perspective
                    bias TEXT, -- supporting, opposing, neutral
                    embedding VECTOR(1536),
                    FOREIGN KEY (issue_id) REFERENCES DomesticIssues(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_domesticnews_embedding 
                ON DomesticNews USING ivfflat (embedding vector_cosine_ops);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_domestic_issues",
        action_description="Generating domestic issues for nation {nation_id}",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def generate_domestic_issues(self, ctx, nation_id: int, count: int = 2) -> List[Dict[str, Any]]:
        """
        Generate domestic issues for a specific nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            count: Number of issues to generate
            
        Returns:
            List of generated domestic issues
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nation details
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return []
                
                nation_data = dict(nation)
                
                # Get factions in this nation
                factions = await conn.fetch("""
                    SELECT id, name, type, description, values
                    FROM Factions
                    WHERE $1 = ANY(territory)
                """, nation_data.get("name"))
                
                faction_data = [dict(faction) for faction in factions]
        
        # Create agent for domestic issue generation
        issue_agent = Agent(
            name="DomesticIssueAgent",
            instructions="You create realistic domestic political and social issues for fantasy nations with matriarchal power structures.",
            model="o3-mini"
        )
        
        # Determine issue types based on nation characteristics
        issue_types = []
        
        # Higher matriarchy has different issues than lower
        matriarchy_level = nation_data.get("matriarchy_level", 5)
        
        if matriarchy_level >= 8:
            # High matriarchy issues
            issue_types.extend([
                "male_rights_movement", "traditionalist_opposition", "matriarchy_reform", 
                "male_separatism", "gender_hierarchy_legislation"
            ])
        elif matriarchy_level <= 3:
            # Low matriarchy issues
            issue_types.extend([
                "feminist_movement", "equality_legislation", "patriarchal_opposition",
                "female_leadership_controversy", "gender_role_debates"
            ])
        else:
            # Balanced matriarchy issues
            issue_types.extend([
                "gender_balance_debate", "power_sharing_reform", "traditionalist_vs_progressive"
            ])
        
        # Universal issue types
        universal_issues = [
            "economic_crisis", "environmental_disaster", "disease_outbreak",
            "succession_dispute", "religious_controversy", "tax_reform",
            "military_service_debate", "trade_regulation", "education_policy",
            "infrastructure_development", "foreign_policy_shift", "corruption_scandal",
            "resource_scarcity", "technological_change", "constitutional_crisis",
            "land_rights_dispute", "criminal_justice_reform", "public_safety_concerns",
            "media_censorship", "social_services_funding"
        ]
        
        issue_types.extend(universal_issues)
        
        # Generate issues
        issues = []
        selected_types = random.sample(issue_types, min(count, len(issue_types)))
        
        for issue_type in selected_types:
            # Create prompt for the agent
            prompt = f"""
            Generate a domestic political or social issue for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a {issue_type} issue that:
            1. Makes sense given the nation's characteristics
            2. Creates realistic societal tension and debate
            3. Involves multiple factions or groups
            4. Considers the matriarchal level of the society ({matriarchy_level}/10)
            
            Return a JSON object with:
            - name: Name of the issue/controversy
            - issue_type: "{issue_type}"
            - description: Detailed description
            - severity: Severity level (1-10)
            - status: Current status (emerging, active, waning, resolved)
            - start_date: When it started (narrative date)
            - supporting_factions: Groups supporting one side
            - opposing_factions: Groups opposing
            - neutral_factions: Groups remaining neutral
            - affected_demographics: Demographics most affected
            - public_opinion: Object describing opinion distribution
            - government_response: How the government is responding
            - recent_developments: Array of recent events in this issue
            - political_impact: Impact on political landscape
            - social_impact: Impact on society
            - economic_impact: Economic consequences
            - potential_resolution: Potential ways it might resolve
            """
            
            # Get response from agent
            result = await Runner.run(issue_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse response
                issue_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in issue_data for k in ["name", "description", "issue_type"]):
                    continue
                
                # Add nation_id
                issue_data["nation_id"] = nation_id
                
                # Generate embedding
                embedding_text = f"{issue_data['name']} {issue_data['description']} {issue_data['issue_type']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        issue_id = await conn.fetchval("""
                            INSERT INTO DomesticIssues (
                                nation_id, name, issue_type, description, severity,
                                status, start_date, supporting_factions, opposing_factions,
                                neutral_factions, affected_demographics, public_opinion,
                                government_response, recent_developments, political_impact,
                                social_impact, economic_impact, potential_resolution, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                            RETURNING id
                        """, 
                        nation_id,
                        issue_data.get("name"), 
                        issue_data.get("issue_type"),
                        issue_data.get("description"),
                        issue_data.get("severity", 5),
                        issue_data.get("status", "active"),
                        issue_data.get("start_date", "Recently"),
                        issue_data.get("supporting_factions", []),
                        issue_data.get("opposing_factions", []),
                        issue_data.get("neutral_factions", []),
                        issue_data.get("affected_demographics", []),
                        json.dumps(issue_data.get("public_opinion", {})),
                        issue_data.get("government_response", ""),
                        issue_data.get("recent_developments", []),
                        issue_data.get("political_impact", ""),
                        issue_data.get("social_impact", ""),
                        issue_data.get("economic_impact", ""),
                        issue_data.get("potential_resolution", ""),
                        embedding)
                        
                        # Generate initial news about this issue
                        await self._generate_domestic_news(run_ctx, issue_id, issue_data, nation_data)
                        
                        # Add to result
                        issue_data["id"] = issue_id
                        issues.append(issue_data)
                        
            except Exception as e:
                logging.error(f"Error generating domestic issue: {e}")
        
        return issues
    
    async def _generate_domestic_news(
        self, 
        ctx, 
        issue_id: int, 
        issue_data: Dict[str, Any],
        nation_data: Dict[str, Any]
    ) -> None:
        """Generate initial news articles about a domestic issue"""
        # Create agent for news generation
        news_agent = Agent(
            name="DomesticNewsAgent",
            instructions="You create realistic news articles about domestic political issues in a matriarchal society.",
            model="o3-mini"
        )
        
        # Generate news articles from different perspectives
        biases = ["supporting", "opposing", "neutral"]
        
        for bias in biases:
            # Create prompt for the agent
            prompt = f"""
            Generate a news article about this domestic issue from a {bias} perspective:
            
            ISSUE:
            {json.dumps(issue_data, indent=2)}
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            Create a news article that:
            1. Has a clear {bias} bias toward the issue
            2. Includes quotes from relevant figures
            3. Covers the key facts but with the appropriate spin
            4. Has a catchy headline
            5. Reflects the matriarchal power structures of society
            
            Return a JSON object with:
            - headline: The article headline
            - content: The full article content (300-500 words)
            - source_faction: The faction or institution publishing this
            """
            
            # Get response from agent
            result = await Runner.run(news_agent, prompt, context=ctx.context)
            
            try:
                # Parse response
                news_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in news_data for k in ["headline", "content"]):
                    continue
                
                # Generate embedding
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        news_id = await conn.fetchval("""
                            INSERT INTO DomesticNews (
                                issue_id, headline, content, source_faction, bias
                            )
                            VALUES ($1, $2, $3, $4, $5)
                            RETURNING id
                        """, 
                        issue_id,
                        news_data.get("headline"), 
                        news_data.get("content"),
                        news_data.get("source_faction", "Unknown Source"),
                        bias)
                        
                        # Generate and store embedding
                        embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                        await self.generate_and_store_embedding(embedding_text, conn, "DomesticNews", "id", news_id)
                        
            except Exception as e:
                logging.error(f"Error generating domestic news: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_initial_conflicts",
        action_description="Generating initial national conflicts",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def generate_initial_conflicts(self, ctx, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate initial conflicts between nations with governance oversight.
        
        Args:
            count: Number of conflicts to generate
            
        Returns:
            List of generated conflicts
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nations for context
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        if len(nations) < 2:
            return []
        
        conflicts = []
        
        # Create agent for conflict generation
        conflict_agent = Agent(
            name="NationalConflictAgent",
            instructions="You create realistic international conflicts for a fantasy world with matriarchal power structures.",
            model="o3-mini"
        )
        
        for i in range(count):
            # Select random nations that aren't already in major conflicts
            available_nations = [n for n in nations if not any(
                n["id"] in c.get("involved_nations", []) for c in conflicts
            )]
            
            if len(available_nations) < 2:
                available_nations = nations  # Fallback if needed
            
            # Choose two random nations
            nation_pair = random.sample(available_nations, 2)
            
            # Determine conflict type based on nations' characteristics
            matriarchy_diff = abs(
                nation_pair[0].get("matriarchy_level", 5) - 
                nation_pair[1].get("matriarchy_level", 5)
            )
            
            # Higher difference makes ideological conflicts more likely
            if matriarchy_diff > 4:
                conflict_types = ["ideological_dispute", "cultural_tension", "religious_conflict", "proxy_war"]
            elif matriarchy_diff > 2:
                conflict_types = ["diplomatic_tension", "border_dispute", "trade_dispute", "resource_conflict"]
            else:
                conflict_types = ["territorial_dispute", "trade_war", "succession_crisis", "alliance_dispute"]
                
            # Randomly select conflict type
            conflict_type = random.choice(conflict_types)
            
            # Create prompt for the agent
            prompt = f"""
            Generate a detailed international conflict between these two nations:
            
            NATION 1:
            {json.dumps(nation_pair[0], indent=2)}
            
            NATION 2:
            {json.dumps(nation_pair[1], indent=2)}
            
            Create a {conflict_type} that:
            1. Makes sense given the nations' characteristics
            2. Has appropriate severity and clear causes
            3. Includes realistic consequences and casualties
            4. Considers the matriarchal nature of the world
            5. Reflects how the differing matriarchy levels ({matriarchy_diff} point difference) might cause tension
            
            Return a JSON object with:
            - name: Name of the conflict
            - conflict_type: "{conflict_type}"
            - description: Detailed description
            - severity: Severity level (1-10)
            - status: Current status (active, escalating, etc.)
            - start_date: When it started (narrative date)
            - involved_nations: IDs of involved nations
            - primary_aggressor: ID of the primary aggressor
            - primary_defender: ID of the primary defender
            - current_casualties: Description of casualties so far
            - economic_impact: Description of economic impact
            - diplomatic_consequences: Description of diplomatic fallout
            - public_opinion: Object with nation IDs as keys and opinion descriptions as values
            - recent_developments: Array of recent events in the conflict
            - potential_resolution: Potential ways it might end
            """
            
            # Get response from agent
            result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse response
                conflict_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in conflict_data for k in ["name", "description", "conflict_type", "severity", "status"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{conflict_data['name']} {conflict_data['description']} {conflict_data['conflict_type']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        conflict_id = await conn.fetchval("""
                            INSERT INTO NationalConflicts (
                                name, conflict_type, description, severity, status,
                                start_date, involved_nations, primary_aggressor, primary_defender,
                                current_casualties, economic_impact, diplomatic_consequences,
                                public_opinion, recent_developments, potential_resolution, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                            RETURNING id
                        """, 
                        conflict_data.get("name"), 
                        conflict_data.get("conflict_type"),
                        conflict_data.get("description"),
                        conflict_data.get("severity", 5),
                        conflict_data.get("status", "active"),
                        conflict_data.get("start_date", "Recently"),
                        conflict_data.get("involved_nations", [nation_pair[0]["id"], nation_pair[1]["id"]]),
                        conflict_data.get("primary_aggressor", nation_pair[0]["id"]),
                        conflict_data.get("primary_defender", nation_pair[1]["id"]),
                        conflict_data.get("current_casualties", "Unknown"),
                        conflict_data.get("economic_impact", "Unknown"),
                        conflict_data.get("diplomatic_consequences", "Unknown"),
                        json.dumps(conflict_data.get("public_opinion", {})),
                        conflict_data.get("recent_developments", []),
                        conflict_data.get("potential_resolution", "Unknown"),
                        embedding)
                        
                        # Generate initial news about this conflict
                        await self._generate_conflict_news(run_ctx, conflict_id, conflict_data, nation_pair)
                        
                        # Add to result
                        conflict_data["id"] = conflict_id
                        conflicts.append(conflict_data)
                        
            except Exception as e:
                logging.error(f"Error generating conflict: {e}")
        
        return conflicts
    
    async def _generate_conflict_news(
        self, 
        ctx, 
        conflict_id: int, 
        conflict_data: Dict[str, Any],
        nations: List[Dict[str, Any]]
    ) -> None:
        """Generate initial news articles about a conflict"""
        # Create agent for news generation
        news_agent = Agent(
            name="ConflictNewsAgent",
            instructions="You create realistic news articles about international conflicts in a matriarchal world.",
            model="o3-mini"
        )
        
        # Generate one news article from each nation's perspective
        for i, nation in enumerate(nations[:2]):
            bias = "pro_defender" if nation["id"] == conflict_data.get("primary_defender") else "pro_aggressor"
            
            # Create prompt for the agent
            prompt = f"""
            Generate a news article about this conflict from the perspective of {nation["name"]}:
            
            CONFLICT:
            {json.dumps(conflict_data, indent=2)}
            
            REPORTING NATION:
            {json.dumps(nation, indent=2)}
            
            Create a news article that:
            1. Has a clear {bias} bias
            2. Includes quotes from officials (primarily women in positions of power)
            3. Covers the key facts but with the nation's spin
            4. Has a catchy headline
            5. Reflects matriarchal power structures in its language and reporting style
            
            Return a JSON object with:
            - headline: The article headline
            - content: The full article content (300-500 words)
            """
            
            # Get response from agent
            result = await Runner.run(news_agent, prompt, context=ctx.context)
            
            try:
                # Parse response
                news_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in news_data for k in ["headline", "content"]):
                    continue
                
                # Apply matriarchal theming to content
                news_data["content"] = _apply_basic_replacements(news_data["content"])
                
                # Generate embedding
                embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO ConflictNews (
                                conflict_id, headline, content, source_nation, bias, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        conflict_id,
                        news_data.get("headline"), 
                        news_data.get("content"),
                        nation["id"],
                        bias,
                        embedding)
                        
            except Exception as e:
                logging.error(f"Error generating conflict news: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_active_conflicts",
        action_description="Getting active national conflicts",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_active_conflicts(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all active conflicts with governance oversight.
        
        Returns:
            List of active conflicts
        """
        # Check cache first
        cache_key = f"active_conflicts_{self.user_id}_{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        # Query database for active conflicts
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflicts = await conn.fetch("""
                    SELECT * FROM NationalConflicts
                    WHERE status != 'resolved'
                    ORDER BY severity DESC
                """)
                
                # Convert to list of dicts
                result = [dict(conflict) for conflict in conflicts]
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, result)
                
                return result

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_issues",
        action_description="Getting domestic issues for nation {nation_id}",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_nation_issues(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Get all domestic issues for a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of domestic issues
        """
        # Check cache first
        cache_key = f"nation_domestic_issues_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        # Query database for domestic issues
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                issues = await conn.fetch("""
                    SELECT * FROM DomesticIssues
                    WHERE nation_id = $1
                    ORDER BY severity DESC
                """, nation_id)
                
                # Convert to list of dicts
                result = [dict(issue) for issue in issues]
                
                # Parse JSON fields
                for issue in result:
                    if "public_opinion" in issue and issue["public_opinion"]:
                        try:
                            issue["public_opinion"] = json.loads(issue["public_opinion"])
                        except:
                            pass
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, result)
                
                return result

