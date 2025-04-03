# lore/managers/religion.py

import logging
import json
import random
from typing import Dict, List, Any, Optional

from agents import Agent, Runner
from agents.run_context import RunContextWrapper

from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from embedding.vector_store import generate_embedding

from lore.core.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils

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
                GLOBAL_LORE_CACHE.invalidate_pattern("deity")
                
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
                GLOBAL_LORE_CACHE.invalidate_pattern("pantheon")
                
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
                GLOBAL_LORE_CACHE.invalidate_pattern("practice")
                
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
                GLOBAL_LORE_CACHE.invalidate_pattern("site")
                
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
                GLOBAL_LORE_CACHE.invalidate_pattern("text")
                
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
                GLOBAL_LORE_CACHE.invalidate_pattern("order")
                
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
                GLOBAL_LORE_CACHE.invalidate_pattern("conflict")
                
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
        cached = GLOBAL_LORE_CACHE.get(cache_key)
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
                GLOBAL_LORE_CACHE.set(cache_key, result)
                
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
