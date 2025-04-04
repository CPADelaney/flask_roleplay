# lore/systems/regional_culture.py

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
from lore.core.cache import GLOBAL_LORE_CACHE

class RegionalCultureSystem(BaseLoreManager):
    """
    Manages culturally specific norms, customs, manners, and languages
    across different regions and nations.
    """
    
    # Pydantic models for structured outputs and guardrails
    class NationValidation(BaseModel):
        """Validation model for nation IDs"""
        is_valid: bool
        reasoning: str
    
    class LanguageOutput(BaseModel):
        """Structured output for language generation"""
        name: str
        language_family: str
        description: str
        writing_system: str
        formality_levels: List[str]
        common_phrases: Dict[str, str]
        difficulty: int = Field(..., ge=1, le=10)
        relation_to_power: str
        dialects: Dict[str, str]
        primary_regions: List[int] = []
        minority_regions: List[int] = []
    
    class CulturalNormOutput(BaseModel):
        """Structured output for cultural norm generation"""
        category: str
        description: str
        formality_level: str
        gender_specific: bool
        female_variation: Optional[str] = None
        male_variation: Optional[str] = None
    
def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.cache_namespace = "regional_culture"
        
        # Initialize specialized agents for different cultural tasks
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized agents for different cultural tasks"""
        # Create base instructions that all cultural agents will use
        base_instructions = """
        You're creating content for a fantasy world featuring matriarchal power structures.
        Ensure all cultural elements reflect feminine authority and appropriate social hierarchies.
        Male elements should be presented in supportive or subordinate roles.
        Your content should be detailed, consistent, and reflect complex cultural elements.
        """
        
        # Language generation agent
        self.language_agent = Agent(
            name="LanguageGenerationAgent",
            instructions=f"""
            {base_instructions}
            
            You create realistic languages for fantasy worlds with matriarchal power structures.
            Languages should have realistic features including grammar, writing systems, and power expressions.
            They should reflect the social hierarchy with linguistic features showing feminine dominance.
            Consider how language encodes status, formality, and power relationships between genders.
            """,
            model="o3-mini"
        )
        
        # Cultural norm agent
        self.norm_agent = Agent(
            name="CulturalNormAgent",
            instructions=f"""
            {base_instructions}
            
            You create cultural norms for fantasy nations with matriarchal power structures.
            Norms should reflect social hierarchies, feminine authority, and provide clear behavioral guidelines.
            Consider how cultural norms differ by gender, status, and context.
            Specify taboos, consequences for breaking norms, and regional variations.
            """,
            model="o3-mini"
        )
        
        # Etiquette agent
        self.etiquette_agent = Agent(
            name="EtiquetteAgent",
            instructions=f"""
            {base_instructions}
            
            You create etiquette systems for fantasy nations with matriarchal power structures.
            Detail specific protocols for greetings, body language, titles, and other formal behaviors.
            Ensure etiquette reinforces feminine authority and appropriate male deference.
            Specify how etiquette varies by context (court, public, private, religious, etc).
            """,
            model="o3-mini"
        )
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
            await self.register_with_governance()
            self.initialized = True
    
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
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="regional_culture_system",
            directive_text="Create and manage cultural systems that reflect matriarchal power structures.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        
        logging.info(f"RegionalCultureSystem registered with governance for user {self.user_id}, conversation {self.conversation_id}")
    
    # Guardrail function for validating nation IDs
    async def _validate_nation_id(self, ctx, agent, input_data: int) -> GuardrailFunctionOutput:
        """Validate that the nation ID exists"""
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Query database to check if nation exists
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id FROM Nations WHERE id = $1
                """, input_data)
                
                is_valid = nation is not None
                reasoning = "Nation ID exists in database" if is_valid else "Nation ID does not exist"
                
                output = {
                    "is_valid": is_valid,
                    "reasoning": reasoning
                }
                
                return GuardrailFunctionOutput(
                    output_info=output,
                    tripwire_triggered=not is_valid
                )
    
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
        # Create a trace for the language generation process
        with trace(
            "LanguageGeneration", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "language_count": count
            }
        ):
            # Create the run context
            run_ctx = self.create_run_context(ctx)
            
            # Get nations for context
            nations = await self.geopolitical_manager.get_all_nations(run_ctx)
            
            if not nations:
                return []
            
            # Configure agent with proper output structure
            language_agent = self.language_agent.clone(
                output_type=self.LanguageOutput
            )
            
            # Run configuration for tracing
            run_config = RunConfig(
                workflow_name="LanguageGeneration",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "generation_type": "language"
                }
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
                
                Return a structured LanguageOutput object with all required fields:
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
                
                # Generate language using the language agent with structured output
                result = await Runner.run(
                    language_agent,
                    prompt,
                    context=run_ctx.context,
                    run_config=run_config
                )
                
                # Get the structured output
                language_data = result.final_output
                
                # Add nation IDs
                language_data.primary_regions = [n["id"] for n in primary_nations]
                language_data.minority_regions = [n["id"] for n in minority_nations]
                
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
                        language_data.name,
                        language_data.language_family,
                        language_data.description,
                        language_data.writing_system,
                        language_data.primary_regions,
                        language_data.minority_regions,
                        language_data.formality_levels,
                        json.dumps(language_data.common_phrases),
                        language_data.difficulty,
                        language_data.relation_to_power,
                        json.dumps(language_data.dialects))
                        
                        # Generate and store embedding
                        embedding_text = f"{language_data.name} {language_data.description}"
                        await self.generate_and_store_embedding(embedding_text, conn, "Languages", "id", language_id)
                        
                        # Add ID to data and convert to dict for return
                        language_dict = language_data.dict()
                        language_dict["id"] = language_id
                        languages.append(language_dict)
            
            return languages
    
    class CulturalNormCompleteOutput(BaseModel):
        """Complete structure for cultural norm output"""
        category: str
        description: str
        formality_level: str
        gender_specific: bool
        female_variation: Optional[str] = None
        male_variation: Optional[str] = None
        taboo_level: int = Field(..., ge=0, le=10)
        consequence: str
        regional_variations: Dict[str, str] = {}
    
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
        # Create a trace for the cultural norm generation process
        with trace(
            "CulturalNormGeneration", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "nation_id": nation_id
            }
        ):
            # Create the run context
            run_ctx = self.create_run_context(ctx)
            
            # Add input guardrail to validate nation ID
            input_guardrail = InputGuardrail(guardrail_function=self._validate_nation_id)
            
            # Get nation details first
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
            
            # Configure norm agent with structured output
            norm_agent = self.norm_agent.clone(
                name="CulturalNormAgent",
                output_type=self.CulturalNormCompleteOutput,
                input_guardrails=[input_guardrail]
            )
            
            # Categories of norms to generate
            categories = [
                "greeting", "dining", "authority", "gift_giving", "personal_space", 
                "gender_relations", "age_relations", "public_behavior", "private_behavior", 
                "business_conduct", "religious_practice"
            ]
            
            # Run configuration for tracing
            run_config = RunConfig(
                workflow_name="CulturalNormGeneration",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "nation_id": nation_id,
                    "generation_type": "cultural_norm"
                }
            )
            
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
                
                Return a structured CulturalNormCompleteOutput object with:
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
                
                # Get response from agent with structured output
                result = await Runner.run(
                    norm_agent, 
                    prompt, 
                    context=run_ctx.context,
                    run_config=run_config
                )
                
                # Get the structured norm data
                norm_data = result.final_output
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        norm_id = await conn.fetchval("""
                            INSERT INTO CulturalNorms (
                                nation_id, category, description, formality_level,
                                gender_specific, female_variation, male_variation,
                                taboo_level, consequence, regional_variations
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            RETURNING id
                        """,
                        nation_id,
                        norm_data.category,
                        norm_data.description,
                        norm_data.formality_level,
                        norm_data.gender_specific,
                        norm_data.female_variation,
                        norm_data.male_variation,
                        norm_data.taboo_level,
                        norm_data.consequence,
                        json.dumps(norm_data.regional_variations))
                        
                        # Generate and store embedding
                        embedding_text = f"{category} {norm_data.description}"
                        await self.generate_and_store_embedding(embedding_text, conn, "CulturalNorms", "id", norm_id)
                        
                        # Add ID and nation ID to the norm data for return
                        norm_dict = norm_data.dict()
                        norm_dict["id"] = norm_id
                        norm_dict["nation_id"] = nation_id
                        norms.append(norm_dict)
            
            return norms
    
    class EtiquetteOutput(BaseModel):
        """Model for etiquette system output"""
        context: str
        title_system: str
        greeting_ritual: str
        body_language: str
        eye_contact: str
        distance_norms: str
        gift_giving: str
        dining_etiquette: str
        power_display: str
        respect_indicators: str
        gender_distinctions: str
        taboos: List[str]

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
        # Create a trace for the etiquette generation process
        with trace(
            "EtiquetteGeneration", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "nation_id": nation_id
            }
        ):
            # Create the run context
            run_ctx = self.create_run_context(ctx)
            
            # Add input guardrail to validate nation ID
            input_guardrail = InputGuardrail(guardrail_function=self._validate_nation_id)
            
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
            
            # Configure etiquette agent with structured output
            etiquette_agent = self.etiquette_agent.clone(
                output_type=self.EtiquetteOutput,
                input_guardrails=[input_guardrail]
            )
            
            # Contexts for etiquette
            contexts = ["court", "noble", "public", "private", "religious", "business"]
            
            # Run configuration for tracing
            run_config = RunConfig(
                workflow_name="EtiquetteGeneration",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "nation_id": nation_id,
                    "generation_type": "etiquette"
                }
            )
            
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
                
                Return a structured EtiquetteOutput object with:
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
                
                # Get response from agent with structured output
                result = await Runner.run(
                    etiquette_agent, 
                    prompt, 
                    context=run_ctx.context,
                    run_config=run_config
                )
                
                # Get structured etiquette data
                etiquette_data = result.final_output
                
                # Store in database
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        etiquette_id = await conn.fetchval("""
                            INSERT INTO Etiquette (
                                nation_id, context, title_system, greeting_ritual,
                                body_language, eye_contact, distance_norms, gift_giving,
                                dining_etiquette, power_display, respect_indicators,
                                gender_distinctions, taboos
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                            RETURNING id
                        """,
                        nation_id,
                        etiquette_data.context,
                        etiquette_data.title_system,
                        etiquette_data.greeting_ritual,
                        etiquette_data.body_language,
                        etiquette_data.eye_contact,
                        etiquette_data.distance_norms,
                        etiquette_data.gift_giving,
                        etiquette_data.dining_etiquette,
                        etiquette_data.power_display,
                        etiquette_data.respect_indicators,
                        etiquette_data.gender_distinctions,
                        etiquette_data.taboos)
                        
                        # Generate and store embedding
                        embedding_text = f"{context} etiquette {etiquette_data.greeting_ritual} {etiquette_data.respect_indicators}"
                        await self.generate_and_store_embedding(embedding_text, conn, "Etiquette", "id", etiquette_id)
                        
                        # Add ID and nation ID to the data for return
                        etiquette_dict = etiquette_data.dict()
                        etiquette_dict["id"] = etiquette_id
                        etiquette_dict["nation_id"] = nation_id
                        etiquette_systems.append(etiquette_dict)
            
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
        # Create a trace for fetching cultural information
        with trace(
            "FetchNationCulture", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "nation_id": nation_id
            }
        ):
            # Add input guardrail to validate nation ID
            input_guardrail = InputGuardrail(guardrail_function=self._validate_nation_id)
            
            # Check cache first
            cache_key = f"nation_culture_{nation_id}"
            cached = self.get_cache(cache_key)
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
                    self.set_cache(cache_key, result)
                    
                    return result
    
    # Add utility function for summarizing cultural elements
    @function_tool
    async def summarize_culture(self, nation_id: int, format_type: str = "brief") -> str:
        """
        Generate a summary of a nation's culture.
        
        Args:
            nation_id: ID of the nation
            format_type: Type of summary (brief, detailed, academic)
            
        Returns:
            Formatted cultural summary
        """
        # Create a trace for summarizing culture
        with trace(
            "SummarizeCulture", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "nation_id": nation_id,
                "format": format_type
            }
        ):
            # Get comprehensive cultural data
            cultural_data = await self.get_nation_culture(
                RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                }), 
                nation_id
            )
            
            if "error" in cultural_data:
                return f"Error: {cultural_data['error']}"
            
            # Create an agent to summarize the culture
            culture_summary_agent = Agent(
                name="CultureSummaryAgent",
                instructions="""
                You create concise, informative summaries of fantasy cultures.
                Focus on highlighting the most distinctive elements while maintaining a coherent picture.
                Ensure the matriarchal power dynamics are properly emphasized.
                """,
                model="o3-mini"
            )
            
            # Create run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Create summary prompt
            prompt = f"""
            Create a {format_type} summary of this nation's culture:
            
            NATION: {cultural_data['nation']['name']}
            
            CULTURAL DATA:
            {json.dumps(cultural_data, indent=2)}
            
            Focus on:
            1. The most distinctive norms and customs
            2. How matriarchal power is expressed in daily life
            3. Key cultural elements visitors would need to understand
            4. Notable linguistic features
            
            Format as a coherent, readable summary that captures the essence of this culture.
            """
            
            # Get the summary
            result = await Runner.run(culture_summary_agent, prompt, context=run_ctx.context)
            return result.final_output
    
    # Add method to detect cultural conflicts
    @function_tool
    async def detect_cultural_conflicts(self, nation_id1: int, nation_id2: int) -> Dict[str, Any]:
        """
        Analyze potential cultural conflicts between two nations.
        
        Args:
            nation_id1: First nation ID
            nation_id2: Second nation ID
            
        Returns:
            Dictionary with analysis of potential conflicts
        """
        # Create a trace for conflict detection
        with trace(
            "CulturalConflictDetection", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "nation_id1": nation_id1,
                "nation_id2": nation_id2
            }
        ):
            # Get cultural data for both nations
            cultural_data1 = await self.get_nation_culture(
                RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                }), 
                nation_id1
            )
            
            cultural_data2 = await self.get_nation_culture(
                RunContextWrapper(context={
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                }), 
                nation_id2
            )
            
            if "error" in cultural_data1 or "error" in cultural_data2:
                return {"error": "One or both nations not found"}
            
            # Create an agent to analyze conflicts
            conflict_analysis_agent = Agent(
                name="CulturalConflictAnalyst",
                instructions="""
                You analyze potential cultural conflicts between different cultures.
                Identify areas where different cultural expectations might lead to misunderstandings.
                Focus on specific, concrete examples rather than generalities.
                Consider how matriarchal power structures might interact differently.
                """,
                model="o3-mini"
            )
            
            # Create run context
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Create analysis prompt
            prompt = f"""
            Analyze potential cultural conflicts between these two nations:
            
            NATION 1: {cultural_data1['nation']['name']}
            CULTURAL DATA 1:
            {json.dumps(cultural_data1, indent=2)}
            
            NATION 2: {cultural_data2['nation']['name']}
            CULTURAL DATA 2:
            {json.dumps(cultural_data2, indent=2)}
            
            For your analysis, consider:
            1. Different greeting expectations
            2. Conflicting etiquette norms
            3. Taboos that might be violated
            4. Different expectations around gender roles
            5. Conflict between power structures
            
            Return a JSON object with:
            - potential_conflicts: Array of specific conflicts
            - severity_level: Overall severity of potential conflicts (1-10)
            - recommendations: Suggestions for diplomatic protocols
            """
            
            # Get the analysis
            result = await Runner.run(conflict_analysis_agent, prompt, context=run_ctx.context)
            
            try:
                # Try to parse as JSON
                conflict_data = json.loads(result.final_output)
                return conflict_data
            except:
                # If parsing fails, return the raw text
                return {
                    "analysis": result.final_output,
                    "parsing_error": "Could not parse response as JSON"
                }
