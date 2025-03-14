# lore/regional_culture_system.py

import logging
import random
import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner
from agents.run_context import RunContextWrapper

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission

# Database functionality
from db.connection import get_db_connection
from embedding.vector_store import generate_embedding
from utils.caching import LoreCache

# Import existing modules
from lore.lore_manager import LoreManager
from lore.enhanced_lore import GeopoliticalSystemManager

# Initialize cache for cultural data
CULTURE_CACHE = LoreCache(max_size=100, ttl=7200)  # 2 hour TTL

class RegionalCultureSystem:
    """
    Manages culturally specific norms, customs, manners, and languages
    across different regions and nations.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
        
    async def initialize_tables(self):
        """Ensure regional culture tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if Languages table exists
                languages_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'languages'
                    );
                """)
                
                if not languages_exist:
                    # Create the table
                    await conn.execute("""
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
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_languages_embedding 
                        ON Languages USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("Languages table created")
                
                # Check if CulturalNorms table exists
                norms_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'culturalnorms'
                    );
                """)
                
                if not norms_exist:
                    # Create the table
                    await conn.execute("""
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
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_culturalnorms_embedding 
                        ON CulturalNorms USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_culturalnorms_nation
                        ON CulturalNorms(nation_id);
                    """)
                    
                    logging.info("CulturalNorms table created")
                
                # Check if Etiquette table exists
                etiquette_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'etiquette'
                    );
                """)
                
                if not etiquette_exist:
                    # Create the table
                    await conn.execute("""
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
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_etiquette_embedding 
                        ON Etiquette USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_etiquette_nation
                        ON Etiquette(nation_id);
                    """)
                    
                    logging.info("Etiquette table created")
    
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
            instructions="You create realistic languages for fantasy worlds.",
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
            4. Includes some common phrases or expressions
            
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
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        language_id = await conn.fetchval("""
                            INSERT INTO Languages (
                                name, language_family, description, writing_system,
                                primary_regions, minority_regions, formality_levels,
                                common_phrases, difficulty, relation_to_power, dialects, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
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
                        json.dumps(language_data.get("dialects", {})),
                        embedding)
                        
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
        async with self.lore_manager.get_connection_pool() as pool:
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
            instructions="You create cultural norms for fantasy nations.",
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
                async with self.lore_manager.get_connection_pool() as pool:
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
        async with self.lore_manager.get_connection_pool() as pool:
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
            instructions="You create etiquette systems for fantasy nations.",
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
            4. Includes gender-specific elements
            
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
                async with self.lore_manager.get_connection_pool() as pool:
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
        
        async with self.lore_manager.get_connection_pool() as pool:
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_relevant_cultural_info",
        action_description="Getting cultural info relevant to context",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def get_relevant_cultural_info(self, ctx, location_name: str, context: str = "public") -> Dict[str, Any]:
        """
        Get relevant cultural information for a location and context with governance oversight.
        
        Args:
            location_name: Name of the location
            context: Context (public, private, court, etc.)
            
        Returns:
            Dictionary with relevant cultural information
        """
        # First determine which nation this location is in
        from lore.lore_integration import LoreIntegrationSystem
        integration = LoreIntegrationSystem(self.user_id, self.conversation_id)
        
        location_details = await integration._get_location_details(location_name)
        if not location_details or "location_id" not in location_details:
            return {"error": "Location not found"}
        
        # Get the nation that controls this location
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Try to find the nation through LocationLore
                location_id = location_details["location_id"]
                location_lore = await integration._get_location_lore(location_name)
                
                controlling_factions = await integration._get_location_factions(location_name)
                
                # Find the nation these factions belong to
                nation_id = None
                for faction in controlling_factions:
                    nation = await conn.fetchval("""
                        SELECT id FROM Nations
                        WHERE name = $1
                    """, faction.get("name"))
                    
                    if nation:
                        nation_id = nation
                        break
                
                # If not found, try to get from GeographicRegions
                if not nation_id:
                    region = await conn.fetchrow("""
                        SELECT id, governing_faction FROM GeographicRegions
                        WHERE $1 = ANY(main_settlements)
                    """, location_name)
                    
                    if region and region["governing_faction"]:
                        nation = await conn.fetchval("""
                            SELECT id FROM Nations
                            WHERE name = $1
                        """, region["governing_faction"])
                        
                        if nation:
                            nation_id = nation
                
                # If all else fails, get the first nation
                if not nation_id:
                    nation_id = await conn.fetchval("""
                        SELECT id FROM Nations
                        LIMIT 1
                    """)
        
        if not nation_id:
            return {"error": "Could not determine nation for this location"}
        
        # Now get cultural information for this nation
        culture_data = await self.get_nation_culture(ctx, nation_id)
        
        # Filter for context-relevant information
        relevant_etiquette = [e for e in culture_data.get("etiquette", []) 
                             if e.get("context", "").lower() == context.lower()]
        
        # Add greeting information
        greeting_norms = [n for n in culture_data.get("cultural_norms", []) 
                         if n.get("category", "").lower() == "greeting"]
        
        # Add context-specific norms
        context_map = {
            "public": ["public_behavior", "authority", "personal_space"],
            "private": ["private_behavior", "gender_relations"],
            "court": ["authority", "power_display"],
            "business": ["business_conduct", "gift_giving"],
            "religious": ["religious_practice"]
        }
        
        relevant_categories = context_map.get(context.lower(), ["public_behavior"])
        relevant_norms = [n for n in culture_data.get("cultural_norms", []) 
                         if n.get("category", "").lower() in relevant_categories]
        
        # Compile relevant information
        return {
            "location": location_details,
            "nation": culture_data.get("nation"),
            "languages": culture_data.get("languages", {}).get("primary", []),
            "etiquette": relevant_etiquette,
            "greeting_norms": greeting_norms,
            "relevant_norms": relevant_norms
        }
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="regional_culture_system",
            agent_instance=self
        )
        
        # Issue a directive for regional culture system
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="regional_culture_system",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Generate and manage culturally-specific norms, customs, and languages.",
                "scope": "world_building"
            },
            priority=5,  # Medium priority
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"RegionalCultureSystem registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
