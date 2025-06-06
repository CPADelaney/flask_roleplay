# lore/systems/regional_culture.py

import logging
import json
import random
from typing import Dict, List, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field

# ------------------ AGENTS SDK IMPORTS ------------------
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    trace,
    InputGuardrail,
    GuardrailFunctionOutput,
    RunContextWrapper
)
from agents.run import RunConfig

# ------------------ NYX/GOVERNANCE IMPORTS ------------------
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# ------------------ PROJECT IMPORTS ------------------
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils
from lore.core.cache import GLOBAL_LORE_CACHE

# ===========================================================================
# PYDANTIC MODELS FOR STRUCTURED DATA
# ===========================================================================

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

class LanguageDistribution(BaseModel):
    """Model for language distribution among nations"""
    primary_region_ids: List[int]
    minority_region_ids: List[int]

class NormCategories(BaseModel):
    """Model for cultural norm categories"""
    categories: List[str]
    count: int

class CulturalConflictAnalysis(BaseModel):
    """Model for cultural conflict analysis"""
    potential_conflicts: List[str]
    severity_level: int = Field(..., ge=1, le=10)
    recommendations: str

class DialectEvolutionModel(BaseModel):
    """Model for dialect evolution information"""
    dialect_name: str
    parent_language: str
    vocabulary_changes: Dict[str, str]
    grammatical_changes: List[str]
    pronunciation_shifts: List[str]
    social_context: str
    prestige_level: int = Field(..., ge=1, le=10)
    example_phrases: Dict[str, str]
    regional_distribution: List[str]

class CulturalDiffusionResult(BaseModel):
    """Model for cultural diffusion simulation results"""
    language_influence: Dict[str, Any]
    artistic_exchanges: Dict[str, Any]
    religious_practices: Dict[str, Any]
    fashion_changes: Dict[str, Any]
    cuisine_evolution: Dict[str, Any]
    social_customs: Dict[str, Any]
    timeline: List[Dict[str, Any]]

class DiffusionEffect(BaseModel):
    """Model for specific diffusion effects"""
    category: str
    from_nation: int
    to_nation: int
    elements_transferred: List[str]
    modifications: List[str]
    adoption_groups: List[str]
    resistance_groups: List[str]
    timeline_years: List[int]

class LanguageInfluenceEffect(BaseModel):
    """Model for language diffusion effects"""
    vocabulary_borrowed: Dict[str, str]
    idioms_adopted: List[str]
    accent_influences: str
    formality_changes: List[str]

class ArtisticDiffusionEffect(BaseModel):
    """Model for artistic diffusion"""
    art_forms: List[str]
    literary_influences: List[str]
    musical_elements: List[str]
    architectural_styles: List[str]

class CulturalSummary(BaseModel):
    """Model for cultural summary generation"""
    format_type: str
    content: str

# ===========================================================================
# MAIN REGIONAL CULTURE SYSTEM CLASS
# ===========================================================================

class RegionalCultureSystem(BaseLoreManager):
    """
    Manages culturally specific norms, customs, manners, and languages
    across different regions and nations, fully agent-ified for dynamic,
    thematically matriarchal outputs.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.cache_namespace = "regional_culture"
        self._agents_initialized = False
        self._agents = {}
    
    def _get_agent(self, agent_name: str) -> Agent:
        """Lazy-load agents on demand to avoid initialization overhead"""
        if not self._agents_initialized:
            self._initialize_agents()
            self._agents_initialized = True
        return self._agents.get(agent_name)
    
    def _initialize_agents(self):
        """Initialize specialized agents for different cultural tasks"""
        base_instructions = (
            "You're creating content for a fantasy world featuring matriarchal power structures. "
            "Ensure all cultural elements reflect feminine authority and appropriate social hierarchies. "
            "Male elements are typically supportive or subordinate. Provide detail and realism."
        )
        
        model_settings = ModelSettings(temperature=0.9)
        
        agent_configs = {
            "language": {
                "name": "LanguageGenerationAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create realistic languages for a fantasy world with matriarchal power. "
                    "Reflect how language encodes status, formality, and gender hierarchy."
                ),
                "model": "gpt-4o-mini",
                "settings": model_settings
            },
            "norm": {
                "name": "CulturalNormAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create cultural norms for fantasy nations. Norms must reflect matriarchal structures. "
                    "Consider differences by gender, status, and context. Provide taboos, consequences, variations."
                ),
                "model": "gpt-4o-mini",
                "settings": model_settings
            },
            "etiquette": {
                "name": "EtiquetteAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create detailed etiquette systems for matriarchal fantasy nations. "
                    "Include greetings, body language, titles, gift-giving, and display of power. "
                    "Be explicit about how men must defer to female authority."
                ),
                "model": "gpt-4o-mini",
                "settings": model_settings
            },
            "distribution": {
                "name": "LanguageDistributionAgent",
                "instructions": (
                    "You receive a list of nations and a desired number of languages to create. "
                    "Propose how to distribute each new language among these nations. "
                    "Consider geography, politics, and cultural influence. "
                    "Some languages might be major (used by multiple large nations), others minor."
                ),
                "model": "gpt-4o-mini",
                "settings": ModelSettings(temperature=0.8)
            },
            "category": {
                "name": "NormCategoryAgent",
                "instructions": (
                    "Given a single nation's data, decide which categories of cultural norms to generate "
                    "(e.g. greeting, dining, authority, gift_giving, personal_space, gender_relations, etc.), "
                    "and how many norms total. Consider the nation's specific characteristics."
                ),
                "model": "gpt-4o-mini",
                "settings": ModelSettings(temperature=0.8)
            },
            "context": {
                "name": "EtiquetteContextAgent",
                "instructions": (
                    "We have a matriarchal nation. Decide which contexts we need specific etiquette for. "
                    "Example contexts might be 'court', 'noble', 'public', 'private', 'religious', 'business'. "
                    "Consider the nation's government type and cultural traits."
                ),
                "model": "gpt-4o-mini",
                "settings": ModelSettings(temperature=0.8)
            },
            "summary": {
                "name": "CultureSummaryAgent",
                "instructions": (
                    "You create coherent, matriarchal-themed cultural summaries. "
                    "Highlight the most distinctive norms, how matriarchal authority is expressed, "
                    "and any notable linguistic or etiquette features."
                ),
                "model": "gpt-4o-mini",
                "settings": ModelSettings(temperature=0.8)
            },
            "conflict": {
                "name": "CulturalConflictAnalyst",
                "instructions": (
                    "You analyze how two different matriarchal cultures might conflict. "
                    "Focus on greeting norms, etiquette, taboos, religious differences, etc. "
                    "Provide specific conflicts, severity assessment, and diplomatic recommendations."
                ),
                "model": "gpt-4o-mini",
                "settings": ModelSettings(temperature=0.8)
            },
            "diffusion": {
                "name": "CulturalDiffusionAgent",
                "instructions": (
                    "You simulate cultural diffusion between two nations over time. "
                    "Model how language, customs, fashion, cuisine, arts, and other cultural elements "
                    "flow between societies based on proximity, relations, and power dynamics. "
                    "Maintain matriarchal power structures as the dominant framework."
                ),
                "model": "gpt-4o-mini",
                "settings": model_settings
            },
            "dialect": {
                "name": "DialectEvolutionAgent",
                "instructions": (
                    "You simulate linguistic evolution of dialects in fantasy languages. "
                    "Model vocabulary changes, grammatical shifts, pronunciation differences, "
                    "and social contexts. Pay special attention to how language reflects "
                    "matriarchal power structures and feminine-dominated society."
                ),
                "model": "gpt-4o-mini",
                "settings": model_settings
            }
        }
        
        # Create agents
        for key, config in agent_configs.items():
            self._agents[key] = Agent(
                name=config["name"],
                instructions=config["instructions"],
                model=config["model"],
                model_settings=config["settings"]
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
                CREATE TABLE IF NOT EXISTS Languages (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    language_family TEXT,
                    description TEXT NOT NULL,
                    writing_system TEXT,
                    primary_regions INTEGER[],
                    minority_regions INTEGER[],
                    formality_levels TEXT[],
                    common_phrases JSONB,
                    difficulty INTEGER CHECK (difficulty BETWEEN 1 AND 10),
                    relation_to_power TEXT,
                    dialects JSONB,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_languages_embedding 
                ON Languages USING ivfflat (embedding vector_cosine_ops);
            """,
            "CulturalNorms": """
                CREATE TABLE IF NOT EXISTS CulturalNorms (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    formality_level TEXT,
                    gender_specific BOOLEAN DEFAULT FALSE,
                    female_variation TEXT,
                    male_variation TEXT,
                    taboo_level INTEGER CHECK (taboo_level BETWEEN 0 AND 10),
                    consequence TEXT,
                    regional_variations JSONB,
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_culturalnorms_embedding 
                ON CulturalNorms USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_culturalnorms_nation
                ON CulturalNorms(nation_id);
            """,
            "Etiquette": """
                CREATE TABLE IF NOT EXISTS Etiquette (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    context TEXT NOT NULL,
                    title_system TEXT,
                    greeting_ritual TEXT,
                    body_language TEXT,
                    eye_contact TEXT,
                    distance_norms TEXT,
                    gift_giving TEXT,
                    dining_etiquette TEXT,
                    power_display TEXT,
                    respect_indicators TEXT,
                    gender_distinctions TEXT,
                    taboos TEXT[],
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_etiquette_embedding 
                ON Etiquette USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_etiquette_nation
                ON Etiquette(nation_id);
            """,
            "LanguageDialects": """
                CREATE TABLE IF NOT EXISTS LanguageDialects (
                    id SERIAL PRIMARY KEY,
                    language_id INTEGER NOT NULL,
                    region_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    parent_language TEXT,
                    vocabulary_changes JSONB,
                    grammatical_changes TEXT[],
                    pronunciation_shifts TEXT[],
                    social_context TEXT,
                    prestige_level INTEGER CHECK (prestige_level BETWEEN 1 AND 10),
                    example_phrases JSONB,
                    regional_distribution TEXT[],
                    embedding VECTOR(1536),
                    UNIQUE(language_id, region_id),
                    FOREIGN KEY (language_id) REFERENCES Languages(id) ON DELETE CASCADE,
                    FOREIGN KEY (region_id) REFERENCES Nations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_languagedialects_embedding
                ON LanguageDialects USING ivfflat (embedding vector_cosine_ops);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="regional_culture_system",
            directive_text=(
                "Create and manage cultural systems that reflect matriarchal power structures."
            ),
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        
        logging.info(f"RegionalCultureSystem registered with governance for user {self.user_id}, conversation {self.conversation_id}")
    
    # ---------------------------------------------------------------------------
    # Guardrail function for validating nation IDs
    # ---------------------------------------------------------------------------
    async def _validate_nation_id(self, ctx, agent, input_data: int) -> GuardrailFunctionOutput:
        """Validate that the given nation ID actually exists in the DB."""
        run_ctx = self.create_run_context(ctx)
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id FROM Nations WHERE id = $1
                """, input_data)
                
                is_valid = nation is not None
                reasoning = "Nation ID exists in database" if is_valid else "Nation ID does not exist"
                
                return GuardrailFunctionOutput(
                    output_info={"is_valid": is_valid, "reasoning": reasoning},
                    tripwire_triggered=not is_valid
                )
    
    # ---------------------------------------------------------------------------
    # (1) Generate Languages
    # ---------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_languages",
        action_description="Generating languages for the world",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_languages(self, ctx, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate languages for the world with governance oversight.
        Now uses an LLM to decide distribution among nations, 
        rather than code-based random.
        """
        with trace(
            "LanguageGeneration", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "language_count": count}
        ):
            run_ctx = self.create_run_context(ctx)
            nations = await self.geopolitical_manager.get_all_nations(run_ctx)
            if not nations:
                return []
            
            # 1) Ask the LLM how to distribute primary vs minority usage for each planned language
            #    instead of random code-based logic.
            distribution_agent = self._get_agent("distribution").clone(
                output_type=List[LanguageDistribution]
            )
            
            # Build distribution prompt
            nation_list = [{"id": n["id"], "name": n["name"]} for n in nations]
            dist_prompt = f"""
            We have the following nations:
            {json.dumps(nation_list, indent=2)}

            We want to create {count} new languages. 
            Decide which nations are primary speakers vs. minority speakers for each new language.
            Return exactly {count} LanguageDistribution objects with:
              - "primary_region_ids": array of nation IDs
              - "minority_region_ids": array of nation IDs
            """
            
            dist_config = RunConfig(
                workflow_name="LanguageDistribution",
                trace_metadata=self.trace_metadata
            )
            dist_result = await Runner.run(distribution_agent, dist_prompt, context=run_ctx.context, run_config=dist_config)
            
            try:
                distribution_data = dist_result.final_output
            except Exception:
                # Fallback if the LLM didn't produce valid output
                distribution_data = []
            
            # If distribution_data is malformed or doesn't match count, fallback:
            if not isinstance(distribution_data, list) or len(distribution_data) != count:
                # fallback - code-based approach
                distribution_data = []
                for i in range(count):
                    # Simplistic fallback: pick random primary & minority
                    if len(nations) <= 1:
                        distribution_data.append(LanguageDistribution(
                            primary_region_ids=[nations[0]["id"]], 
                            minority_region_ids=[]
                        ))
                    else:
                        primary_choice = random.sample(nations, min(2, len(nations)))
                        minority_candidates = [n for n in nations if n not in primary_choice]
                        minority_choice = random.sample(minority_candidates, min(1, len(minority_candidates))) if minority_candidates else []
                        distribution_data.append(LanguageDistribution(
                            primary_region_ids=[n["id"] for n in primary_choice],
                            minority_region_ids=[n["id"] for n in minority_choice]
                        ))
            
            # 2) Now for each language distribution, we call the language generation agent
            language_agent = self._get_agent("language").clone(output_type=LanguageOutput)
            run_config = RunConfig(
                workflow_name="LanguageGeneration",
                trace_metadata={"user_id": str(self.user_id), "conversation_id": str(self.conversation_id)}
            )
            
            languages = []
            for i, dist in enumerate(distribution_data):
                # Build the prompt
                # We'll provide details about the nations in dist["primary_region_ids"] & dist["minority_region_ids"]
                def find_nation_data(nid):
                    for nn in nations:
                        if nn["id"] == nid:
                            return nn
                    return {}
                
                primary_nations = [find_nation_data(nid) for nid in dist.primary_region_ids]
                minority_nations = [find_nation_data(nid) for nid in dist.minority_region_ids]
                
                gen_prompt = (
                    "Create a new language for a matriarchal fantasy setting. Some nations speak it primarily; "
                    "others use it as a minority language. Output a LanguageOutput object."
                    f"\nPRIMARY NATIONS:\n{json.dumps(primary_nations, indent=2)}"
                    f"\nMINORITY NATIONS:\n{json.dumps(minority_nations, indent=2)}"
                )
                
                result = await Runner.run(language_agent, gen_prompt, context=run_ctx.context, run_config=run_config)
                language_data = result.final_output
                # Attach the distribution
                language_data.primary_regions = dist.primary_region_ids
                language_data.minority_regions = dist.minority_region_ids
                
                # 3) Insert into DB
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
                        
                        # Generate embedding
                        embed_text = f"{language_data.name} {language_data.description}"
                        await self.generate_and_store_embedding(embed_text, conn, "Languages", "id", language_id)
                        
                        lang_dict = language_data.dict()
                        lang_dict["id"] = language_id
                        languages.append(lang_dict)
            
            return languages
    
    # ---------------------------------------------------------------------------
    # (2) Generate Cultural Norms
    # ---------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_cultural_norms",
        action_description="Generating cultural norms for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_cultural_norms(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate cultural norms for a specific nation with governance oversight.
        We let the LLM pick how many norms and which categories to generate 
        (instead of a fixed local list).
        """
        with trace(
            "CulturalNormGeneration", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_id": nation_id}
        ):
            run_ctx = self.create_run_context(ctx)
            # Input guardrail to validate the nation
            input_guardrail = InputGuardrail(guardrail_function=self._validate_nation_id)
            
            # Fetch the nation data
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
            
            # Let the LLM decide how many norms to generate, and for which categories
            category_agent = self._get_agent("category").clone(
                output_type=NormCategories
            )
            
            cat_prompt = f"""
            We have the following nation data:
            {json.dumps(nation_data, indent=2)}

            Decide which categories of cultural norms we should generate (like greeting, authority, dining, etc.), 
            and how many norms total. Consider categories like:
            - greeting, dining, authority, gift_giving, personal_space
            - gender_relations, religious_practice, public_behavior
            - business_conduct, family_dynamics
            
            Return a NormCategories object with:
            - categories: array of category strings
            - count: total number of norms to generate
            """
            
            cat_config = RunConfig(
                workflow_name="DetermineNormCategories",
                trace_metadata=self.trace_metadata
            )
            cat_result = await Runner.run(category_agent, cat_prompt, context=run_ctx.context, run_config=cat_config)
            
            try:
                category_data = cat_result.final_output
                categories = category_data.categories
                norms_count = category_data.count
            except Exception:
                # fallback
                categories = ["greeting","dining","authority","gift_giving","gender_relations"]
                norms_count = len(categories)
            
            # Now we generate that many norms with the norm agent
            norm_agent = self._get_agent("norm").clone(
                output_type=CulturalNormOutput,
                input_guardrails=[input_guardrail]
            )
            
            run_config = RunConfig(
                workflow_name="CulturalNormGeneration",
                trace_metadata={"user_id": str(self.user_id), "conversation_id": str(self.conversation_id), "nation_id": nation_id}
            )
            
            norms = []
            
            # If the user wants more norms_count than categories, we can repeat categories or some sub-lists
            # For simplicity, we'll ensure we produce exactly norms_count norms total. We'll cycle categories
            cat_cycle = categories if categories else ["greeting","dining"]
            
            for i in range(norms_count):
                cat = cat_cycle[i % len(cat_cycle)]
                
                prompt = f"""
                Generate cultural norms about {cat} for this nation:
                NATION DATA:
                {json.dumps(nation_data, indent=2)}
                
                Return a CulturalNormOutput object with:
                - category: {cat}
                - description, formality_level, gender_specific
                - female_variation, male_variation (if gender_specific)
                - taboo_level (0-10), consequence
                - regional_variations dictionary
                
                Ensure strong matriarchal themes.
                """
                
                result = await Runner.run(norm_agent, prompt, context=run_ctx.context, run_config=run_config)
                norm_data = result.final_output
                
                # Insert into DB
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
                        
                        emb_text = f"{cat} {norm_data.description}"
                        await self.generate_and_store_embedding(emb_text, conn, "CulturalNorms", "id", norm_id)
                        
                        ndict = norm_data.dict()
                        ndict["id"] = norm_id
                        ndict["nation_id"] = nation_id
                        norms.append(ndict)
            
            return norms
    
    # ---------------------------------------------------------------------------
    # (3) Generate Etiquette
    # ---------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_etiquette",
        action_description="Generating etiquette for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_etiquette(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate etiquette systems for a specific nation with governance oversight.
        Now we use an LLM to pick which contexts to produce (instead of a fixed list).
        """
        with trace(
            "EtiquetteGeneration", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_id": nation_id}
        ):
            run_ctx = self.create_run_context(ctx)
            input_guardrail = InputGuardrail(guardrail_function=self._validate_nation_id)
            
            # Fetch nation details
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
            
            # Let an agent decide which contexts to produce
            context_agent = self._get_agent("context").clone(
                output_type=List[str]
            )
            
            ctx_prompt = f"""
            We have the following nation:
            {json.dumps(nation_data, indent=2)}

            Suggest an array of relevant 'contexts' for which we should generate distinct etiquette systems.
            Consider contexts like: court, noble, public, private, religious, business, military, academic, artistic, intimate.
            Return an array of context strings.
            """
            
            ctx_config = RunConfig(
                workflow_name="EtiquetteContexts",
                trace_metadata=self.trace_metadata
            )
            ctx_result = await Runner.run(context_agent, ctx_prompt, context=run_ctx.context, run_config=ctx_config)
            
            try:
                contexts = ctx_result.final_output
                if not isinstance(contexts, list):
                    contexts = ["court","public","private"]
            except Exception:
                contexts = ["court","public","private"]
            
            # Now produce etiquette for each context
            etiquette_agent = self._get_agent("etiquette").clone(
                output_type=EtiquetteOutput, 
                input_guardrails=[input_guardrail]
            )
            run_config = RunConfig(workflow_name="EtiquetteGeneration", trace_metadata=self.trace_metadata)
            
            etiquette_systems = []
            for context_item in contexts:
                prompt = f"""
                Generate a detailed etiquette system for {context_item} contexts in this matriarchal nation:
                {json.dumps(nation_data, indent=2)}
                
                Return an EtiquetteOutput with:
                - context: "{context_item}"
                - title_system: how titles and honorifics work
                - greeting_ritual: detailed greeting procedures
                - body_language: expected postures and gestures
                - eye_contact: rules about looking at others
                - distance_norms: personal space expectations
                - gift_giving: protocols for gifts
                - dining_etiquette: meal behavior rules
                - power_display: how authority is shown
                - respect_indicators: signs of deference
                - gender_distinctions: different rules by gender
                - taboos: list of forbidden behaviors
                
                Be explicit about male deference to female authority.
                """
                result = await Runner.run(etiquette_agent, prompt, context=run_ctx.context, run_config=run_config)
                etiquette_data = result.final_output
                
                # Insert into DB
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        etiq_id = await conn.fetchval("""
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
                        
                        embed_text = f"{context_item} etiquette {etiquette_data.greeting_ritual} {etiquette_data.respect_indicators}"
                        await self.generate_and_store_embedding(embed_text, conn, "Etiquette", "id", etiq_id)
                        
                        et_dict = etiquette_data.dict()
                        et_dict["id"] = etiq_id
                        et_dict["nation_id"] = nation_id
                        etiquette_systems.append(et_dict)
            
            return etiquette_systems
    
    # ---------------------------------------------------------------------------
    # Fetching & Summarizing Cultural Data
    # ---------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_culture",
        action_description="Getting cultural information for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def get_nation_culture(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive cultural information about a nation with governance oversight.
        We do basic DB lookups, not agent calls, since this is purely retrieval.
        """
        with trace(
            "FetchNationCulture", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_id": nation_id}
        ):
            input_guardrail = InputGuardrail(guardrail_function=self._validate_nation_id)
            
            cache_key = f"nation_culture_{nation_id}"
            cached = self.get_cache(cache_key)
            if cached:
                return cached
            
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    nation = await conn.fetchrow("""
                        SELECT id, name, government_type, matriarchy_level, cultural_traits
                        FROM Nations
                        WHERE id = $1
                    """, nation_id)
                    
                    if not nation:
                        return {"error": "Nation not found"}
                    
                    languages = await conn.fetch("""
                        SELECT id, name, description, writing_system, formality_levels,
                               primary_regions, minority_regions
                        FROM Languages
                        WHERE $1 = ANY(primary_regions) OR $1 = ANY(minority_regions)
                    """, nation_id)
                    
                    norms = await conn.fetch("""
                        SELECT id, category, description, formality_level, gender_specific,
                               female_variation, male_variation, taboo_level, consequence
                        FROM CulturalNorms
                        WHERE nation_id = $1
                    """, nation_id)
                    
                    etiquette = await conn.fetch("""
                        SELECT id, context, title_system, greeting_ritual, power_display,
                               respect_indicators, gender_distinctions, taboos
                        FROM Etiquette
                        WHERE nation_id = $1
                    """, nation_id)
                    
                    result = {
                        "nation": dict(nation),
                        "languages": {
                            "primary": [
                                dict(lang) for lang in languages
                                if nation_id in lang["primary_regions"]
                            ],
                            "minority": [
                                dict(lang) for lang in languages
                                if nation_id in lang["minority_regions"]
                            ]
                        },
                        "cultural_norms": [dict(norm) for norm in norms],
                        "etiquette": [dict(e) for e in etiquette]
                    }
                    
                    self.set_cache(cache_key, result)
                    return result
    
    @function_tool(strict_mode=False)
    async def summarize_culture(self, nation_id: int, format_type: str = "brief") -> str:
        """
        Generate a textual summary of a nation's culture, using an LLM.
        """
        with trace(
            "SummarizeCulture", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_id": nation_id, "format": format_type}
        ):
            cultural_data = await self.get_nation_culture(
                RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id}),
                nation_id
            )
            if "error" in cultural_data:
                return f"Error: {cultural_data['error']}"
            
            # Summarize with an agent
            summary_agent = self._get_agent("summary")
            
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            format_instructions = {
                "brief": "Create a concise 2-3 paragraph summary",
                "detailed": "Create a comprehensive multi-section summary with headers",
                "narrative": "Write an immersive narrative description as if from a traveler's journal"
            }.get(format_type, "Create an appropriate summary")
            
            prompt = f"""
            {format_instructions} of this nation's culture:
            
            NATION: {cultural_data['nation']['name']}
            
            CULTURAL DATA:
            {json.dumps(cultural_data, indent=2)}
            
            Emphasize matriarchal power in daily life, key traditions, and important linguistic notes.
            """
            
            result = await Runner.run(summary_agent, prompt, context=run_ctx.context)
            return result.final_output
    
    @function_tool(strict_mode=False)
    async def detect_cultural_conflicts(self, nation_id1: int, nation_id2: int) -> Dict[str, Any]:
        """
        Analyze potential cultural conflicts between two nations,
        leveraging an LLM to identify points of friction or misunderstanding.
        """
        with trace(
            "CulturalConflictDetection", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "nation_id1": nation_id1, "nation_id2": nation_id2}
        ):
            data1 = await self.get_nation_culture(
                RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id}),
                nation_id1
            )
            data2 = await self.get_nation_culture(
                RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id}),
                nation_id2
            )
            if "error" in data1 or "error" in data2:
                return {"error": "One or both nations not found"}
            
            conflict_agent = self._get_agent("conflict").clone(
                output_type=CulturalConflictAnalysis
            )
            
            run_ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
            prompt = f"""
            Analyze potential cultural conflicts between these two nations:
            
            NATION 1 DATA:
            {json.dumps(data1, indent=2)}
            
            NATION 2 DATA:
            {json.dumps(data2, indent=2)}
            
            Consider:
            - Greeting and etiquette mismatches
            - Taboo violations
            - Language barriers and miscommunications
            - Conflicting power structure expectations
            - Religious or value differences
            - Gender role expectations
            - Gift-giving faux pas
            
            Return a CulturalConflictAnalysis with:
            - potential_conflicts: list of specific conflict scenarios
            - severity_level: 1-10 overall severity
            - recommendations: diplomatic advice
            """
            
            result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
            return result.final_output.dict()
                
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_cultural_diffusion",
        action_description="Simulating cultural diffusion between nations",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def simulate_cultural_diffusion(self, ctx, nation1_id: int, nation2_id: int, years: int = 50) -> Dict[str, Any]:
        """
        Simulate how culture diffuses between two nations over time, including
        language influence, customs, fashion, food, and other cultural elements.
        """
        with trace(
            "CulturalDiffusionSimulation", 
            group_id=self.trace_group_id,
            metadata={"nation1_id": nation1_id, "nation2_id": nation2_id, "years": years}
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Validate both nations
            for nation_id in [nation1_id, nation2_id]:
                validation = await self._validate_nation_id(ctx, None, nation_id)
                if not validation.output_info["is_valid"]:
                    return {"error": f"Nation with ID {nation_id} not found"}
            
            # Get both nations' cultural data
            nation1_culture = await self.get_nation_culture(run_ctx, nation1_id)
            nation2_culture = await self.get_nation_culture(run_ctx, nation2_id)
            
            # Create a diffusion simulation agent
            diffusion_agent = self._get_agent("diffusion").clone(
                output_type=CulturalDiffusionResult
            )
            
            # Get geopolitical data about the two nations' relationship
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    relation = await conn.fetchrow("""
                        SELECT * FROM InternationalRelations
                        WHERE (nation1_id = $1 AND nation2_id = $2)
                        OR (nation1_id = $2 AND nation2_id = $1)
                    """, nation1_id, nation2_id)
                    
                    # If no relation exists, create a minimal one
                    relation_data = dict(relation) if relation else {
                        "relationship_type": "neutral",
                        "relationship_quality": 5,
                        "description": "No formal relations"
                    }
            
            # Build the prompt
            prompt = f"""
            Simulate cultural diffusion between these two nations over {years} years:
            
            NATION 1 CULTURE:
            {json.dumps(nation1_culture, indent=2)}
            
            NATION 2 CULTURE:
            {json.dumps(nation2_culture, indent=2)}
            
            RELATIONSHIP:
            {json.dumps(relation_data, indent=2)}
            
            Simulate the following cultural exchanges:
            1. Language influence (vocabulary, idioms, accent)
            2. Artistic and literary exchanges
            3. Religious practices and beliefs
            4. Fashion and clothing
            5. Cuisine and food
            6. Social customs and etiquette
            
            For each category, specify:
            - What elements transfer from each nation to the other
            - How they are modified in the process
            - Social groups that adopt or resist the changes
            - Timeline of adoption over {years} years
            
            Return a CulturalDiffusionResult with detailed diffusion patterns, maintaining matriarchal frameworks.
            """
            
            # Run the simulation
            result = await Runner.run(diffusion_agent, prompt, context=run_ctx.context)
            diffusion_data = result.final_output
            
            # Apply the diffusion effects to the database
            await self._apply_diffusion_effects(nation1_id, nation2_id, diffusion_data)
            
            return {
                "nations": [nation1_id, nation2_id],
                "years_simulated": years,
                "diffusion_results": diffusion_data.dict()
            }
    
    async def _apply_diffusion_effects(self, nation1_id: int, nation2_id: int, diffusion_data: CulturalDiffusionResult) -> None:
        """Apply cultural diffusion effects to both nations."""
        # This implementation depends on your database structure
        # Here's a simplified approach
        
        # For each diffusion category
        for category, effects in diffusion_data.dict().items():
            if category in ["language_influence", "vocabulary", "idioms"]:
                await self._apply_language_diffusion(nation1_id, nation2_id, effects)
            
            elif category in ["artistic_exchanges", "literary", "art"]:
                await self._apply_artistic_diffusion(nation1_id, nation2_id, effects)
            
            elif category in ["religious_practices", "beliefs", "practices"]:
                await self._apply_religious_diffusion(nation1_id, nation2_id, effects)
            
            elif category in ["fashion_changes", "clothing", "appearance"]:
                await self._apply_fashion_diffusion(nation1_id, nation2_id, effects)
            
            elif category in ["cuisine_evolution", "food", "culinary"]:
                await self._apply_cuisine_diffusion(nation1_id, nation2_id, effects)
            
            elif category in ["social_customs", "customs", "etiquette"]:
                await self._apply_customs_diffusion(nation1_id, nation2_id, effects)
        
        # Invalidate caches for both nations
        self.invalidate_cache_pattern(f"nation_culture_{nation1_id}")
        self.invalidate_cache_pattern(f"nation_culture_{nation2_id}")
    
    async def _apply_language_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply language diffusion effects."""
        # Create an agent to determine specific language changes
        language_diffusion_agent = Agent(
            name="LanguageDiffusionAgent",
            instructions="Generate specific vocabulary borrowings and linguistic influences between nations.",
            model="gpt-4o-mini",
            output_type=LanguageInfluenceEffect
        )
        
        prompt = f"""
        Based on these diffusion effects:
        {json.dumps(effects, indent=2)}
        
        Generate specific language influences including:
        - vocabulary_borrowed: dictionary of borrowed words with meanings
        - idioms_adopted: list of adopted idiomatic expressions
        - accent_influences: description of pronunciation changes
        - formality_changes: list of changes in formal/informal speech
        """
        
        result = await Runner.run(language_diffusion_agent, prompt, context={})
        influence_data = result.final_output
        
        # Store the influence data (implementation depends on your DB structure)
        # This is a placeholder - you might want to create a LanguageInfluences table
        logging.info(f"Applied language diffusion between nations {nation1_id} and {nation2_id}")
    
    async def _apply_artistic_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply artistic and literary diffusion effects."""
        # Similar pattern to language diffusion
        artistic_agent = Agent(
            name="ArtisticDiffusionAgent",
            instructions="Generate specific artistic and literary influences between nations.",
            model="gpt-4o-mini",
            output_type=ArtisticDiffusionEffect
        )
        
        prompt = f"""
        Based on these diffusion effects:
        {json.dumps(effects, indent=2)}
        
        Generate specific artistic influences including:
        - art_forms: list of shared or influenced art forms
        - literary_influences: list of literary styles or themes
        - musical_elements: list of musical influences
        - architectural_styles: list of architectural borrowings
        """
        
        result = await Runner.run(artistic_agent, prompt, context={})
        artistic_data = result.final_output
        
        logging.info(f"Applied artistic diffusion between nations {nation1_id} and {nation2_id}")
    
    async def _apply_religious_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply religious practice diffusion effects."""
        # Placeholder implementation
        logging.info(f"Applied religious diffusion between nations {nation1_id} and {nation2_id}")
    
    async def _apply_fashion_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply fashion and clothing diffusion effects."""
        # Placeholder implementation
        logging.info(f"Applied fashion diffusion between nations {nation1_id} and {nation2_id}")
    
    async def _apply_cuisine_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply culinary diffusion effects."""
        # Placeholder implementation
        logging.info(f"Applied cuisine diffusion between nations {nation1_id} and {nation2_id}")
    
    async def _apply_customs_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply social customs diffusion effects."""
        # Placeholder implementation
        logging.info(f"Applied customs diffusion between nations {nation1_id} and {nation2_id}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_dialect",
        action_description="Evolving regional dialect through language agents",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def evolve_dialect(self, ctx, language_id: int, region_id: int, years: int = 100) -> Dict[str, Any]:
        """
        Evolve a regional dialect using language agents that simulate linguistic evolution.
        """
        with trace(
            "DialectEvolution", 
            group_id=self.trace_group_id,
            metadata={"language_id": language_id, "region_id": region_id, "years": years}
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Get language and region data
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    language = await conn.fetchrow("""
                        SELECT * FROM Languages WHERE id = $1
                    """, language_id)
                    
                    region = await conn.fetchrow("""
                        SELECT * FROM Nations WHERE id = $1
                    """, region_id)
                    
                    if not language or not region:
                        return {"error": "Language or region not found"}
                    
                    language_data = dict(language)
                    region_data = dict(region)
                    
                    # Parse JSON fields
                    if language_data.get('common_phrases'):
                        language_data['common_phrases'] = json.loads(language_data['common_phrases'])
                    if language_data.get('dialects'):
                        language_data['dialects'] = json.loads(language_data['dialects'])
                    
                    # Get existing dialects in this language
                    existing_dialects = language_data.get('dialects', {})
                    
                    # Get cultural elements for context
                    cultural_elements = await conn.fetch("""
                        SELECT * FROM CulturalElements 
                        WHERE $1 = ANY(practiced_by)
                        LIMIT 10
                    """, region_data["name"])
                    
                    cultural_data = [dict(c) for c in cultural_elements]
            
            # Create language evolution agent
            dialect_agent = self._get_agent("dialect").clone(
                output_type=DialectEvolutionModel
            )
            
            # Build the prompt
            prompt = f"""
            Evolve a regional dialect for this language and region over {years} years:
            
            LANGUAGE:
            {json.dumps(language_data, indent=2)}
            
            REGION:
            {json.dumps(region_data, indent=2)}
            
            EXISTING DIALECTS:
            {json.dumps(existing_dialects, indent=2)}
            
            CULTURAL CONTEXT:
            {json.dumps(cultural_data, indent=2)}
            
            Create a DialectEvolutionModel for a new or evolved dialect that:
            1. Reflects the region's culture and social structure
            2. Shows matriarchal power in feminine-dominant language forms
            3. Includes specific vocabulary and grammatical changes
            4. Has example phrases showing the dialect in use
            5. Explains its social context and prestige level
            
            Include:
            - dialect_name: unique name for this dialect
            - parent_language: the base language name
            - vocabulary_changes: dict of original->dialectal words
            - grammatical_changes: list of grammar shifts
            - pronunciation_shifts: list of sound changes
            - social_context: who speaks it and when
            - prestige_level: 1-10 social status
            - example_phrases: dict showing usage
            - regional_distribution: list of areas where spoken
            """
            
            # Run the simulation
            result = await Runner.run(dialect_agent, prompt, context=run_ctx.context)
            dialect_model = result.final_output
            
            # Store the dialect in the database
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Update the language's dialects
                    current_dialects = language_data.get("dialects", {})
                    if not current_dialects:
                        current_dialects = {}
                    
                    # Add or update the dialect for this region
                    region_name = region_data["name"]
                    current_dialects[region_name] = dialect_model.dialect_name
                    
                    # Update the dialect details in the Languages table
                    await conn.execute("""
                        UPDATE Languages
                        SET dialects = $1
                        WHERE id = $2
                    """, json.dumps(current_dialects), language_id)
                    
                    # Store detailed dialect information in LanguageDialects table
                    try:
                        await conn.execute("""
                            INSERT INTO LanguageDialects
                            (language_id, region_id, name, parent_language, vocabulary_changes,
                             grammatical_changes, pronunciation_shifts, social_context,
                             prestige_level, example_phrases, regional_distribution)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            ON CONFLICT (language_id, region_id) DO UPDATE
                            SET name = EXCLUDED.name,
                                vocabulary_changes = EXCLUDED.vocabulary_changes,
                                grammatical_changes = EXCLUDED.grammatical_changes,
                                pronunciation_shifts = EXCLUDED.pronunciation_shifts,
                                social_context = EXCLUDED.social_context,
                                prestige_level = EXCLUDED.prestige_level,
                                example_phrases = EXCLUDED.example_phrases,
                                regional_distribution = EXCLUDED.regional_distribution
                        """,
                        language_id, 
                        region_id,
                        dialect_model.dialect_name,
                        dialect_model.parent_language,
                        json.dumps(dialect_model.vocabulary_changes),
                        dialect_model.grammatical_changes,
                        dialect_model.pronunciation_shifts,
                        dialect_model.social_context,
                        dialect_model.prestige_level,
                        json.dumps(dialect_model.example_phrases),
                        dialect_model.regional_distribution)
                        
                        # Generate embedding for the dialect
                        embed_text = f"{dialect_model.dialect_name} {dialect_model.social_context}"
                        
                        # Get the ID of the inserted/updated dialect
                        dialect_id = await conn.fetchval("""
                            SELECT id FROM LanguageDialects 
                            WHERE language_id = $1 AND region_id = $2
                        """, language_id, region_id)
                        
                        if dialect_id:
                            await self.generate_and_store_embedding(
                                embed_text, conn, "LanguageDialects", "id", dialect_id
                            )
                    except Exception as e:
                        logging.error(f"Error storing dialect data: {e}")
                        # Continue anyway since we've already updated the main language record
            
            # Return the dialect evolution results
            return {
                "language_id": language_id,
                "region_id": region_id,
                "years_simulated": years,
                "dialect": dialect_model.dict()
            }
    
    # ---------------------------------------------------------------------------
    # Additional Helper Methods
    # ---------------------------------------------------------------------------
    
    @function_tool(strict_mode=False)
    async def get_all_languages(self) -> List[Dict[str, Any]]:
        """Get all languages in the world."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                languages = await conn.fetch("""
                    SELECT id, name, language_family, description, 
                           writing_system, difficulty, relation_to_power
                    FROM Languages
                    ORDER BY name
                """)
                return [dict(lang) for lang in languages]
    
    @function_tool(strict_mode=False)
    async def get_language_details(self, language_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific language."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                language = await conn.fetchrow("""
                    SELECT * FROM Languages WHERE id = $1
                """, language_id)
                
                if not language:
                    return {"error": "Language not found"}
                
                lang_data = dict(language)
                
                # Parse JSON fields
                if lang_data.get('common_phrases'):
                    lang_data['common_phrases'] = json.loads(lang_data['common_phrases'])
                if lang_data.get('dialects'):
                    lang_data['dialects'] = json.loads(lang_data['dialects'])
                
                # Get dialects
                dialects = await conn.fetch("""
                    SELECT * FROM LanguageDialects
                    WHERE language_id = $1
                """, language_id)
                
                lang_data['dialect_details'] = []
                for dialect in dialects:
                    d = dict(dialect)
                    if d.get('vocabulary_changes'):
                        d['vocabulary_changes'] = json.loads(d['vocabulary_changes'])
                    if d.get('example_phrases'):
                        d['example_phrases'] = json.loads(d['example_phrases'])
                    lang_data['dialect_details'].append(d)
                
                # Get nations that speak this language
                nations = await conn.fetch("""
                    SELECT id, name FROM Nations
                    WHERE id = ANY($1) OR id = ANY($2)
                """, lang_data.get('primary_regions', []), 
                    lang_data.get('minority_regions', []))
                
                lang_data['speaking_nations'] = {
                    'primary': [
                        dict(n) for n in nations 
                        if n['id'] in lang_data.get('primary_regions', [])
                    ],
                    'minority': [
                        dict(n) for n in nations 
                        if n['id'] in lang_data.get('minority_regions', [])
                    ]
                }
                
                return lang_data
    
    @function_tool(strict_mode=False)
    async def compare_etiquette(self, nation_id1: int, nation_id2: int, context: str) -> Dict[str, Any]:
        """Compare etiquette between two nations for a specific context."""
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                etiq1 = await conn.fetchrow("""
                    SELECT * FROM Etiquette
                    WHERE nation_id = $1 AND context = $2
                """, nation_id1, context)
                
                etiq2 = await conn.fetchrow("""
                    SELECT * FROM Etiquette
                    WHERE nation_id = $1 AND context = $2
                """, nation_id2, context)
                
                if not etiq1 or not etiq2:
                    return {"error": "Etiquette not found for one or both nations in this context"}
                
                # Create comparison agent
                comparison_agent = Agent(
                    name="EtiquetteComparisonAgent",
                    instructions="Compare etiquette systems and highlight key differences and potential misunderstandings.",
                    model="gpt-4o-mini"
                )
                
                prompt = f"""
                Compare these two etiquette systems for {context} contexts:
                
                NATION 1 ETIQUETTE:
                {json.dumps(dict(etiq1), indent=2)}
                
                NATION 2 ETIQUETTE:
                {json.dumps(dict(etiq2), indent=2)}
                
                Highlight:
                - Major differences that could cause offense
                - Conflicting expectations
                - Areas of compatibility
                - Advice for visitors from each nation
                """
                
                result = await Runner.run(comparison_agent, prompt, context={})
                
                return {
                    "nation1_id": nation_id1,
                    "nation2_id": nation_id2,
                    "context": context,
                    "comparison": result.final_output
                }
    
    @function_tool(strict_mode=False)
    async def generate_diplomatic_protocol(self, nation_id1: int, nation_id2: int) -> Dict[str, Any]:
        """Generate a diplomatic protocol guide for interactions between two nations."""
        with trace(
            "DiplomaticProtocolGeneration", 
            group_id=self.trace_group_id,
            metadata={"nation1_id": nation_id1, "nation2_id": nation_id2}
        ):
            # Get cultural data for both nations
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            culture1 = await self.get_nation_culture(run_ctx, nation_id1)
            culture2 = await self.get_nation_culture(run_ctx, nation_id2)
            
            if "error" in culture1 or "error" in culture2:
                return {"error": "Failed to retrieve cultural data"}
            
            # Get conflict analysis
            conflicts = await self.detect_cultural_conflicts(nation_id1, nation_id2)
            
            # Create protocol generation agent
            protocol_agent = Agent(
                name="DiplomaticProtocolAgent",
                instructions=(
                    "Create comprehensive diplomatic protocols for interactions between nations. "
                    "Consider cultural sensitivities, power dynamics, and matriarchal structures. "
                    "Provide specific guidance for successful diplomatic engagement."
                ),
                model="gpt-4o-mini"
            )
            
            prompt = f"""
            Create a diplomatic protocol guide for interactions between these nations:
            
            NATION 1:
            {json.dumps(culture1, indent=2)}
            
            NATION 2:
            {json.dumps(culture2, indent=2)}
            
            KNOWN CONFLICTS:
            {json.dumps(conflicts, indent=2)}
            
            Create a comprehensive guide including:
            1. Pre-meeting preparations
            2. Arrival and initial greetings
            3. Gift exchange protocols
            4. Meeting room arrangements
            5. Speaking order and titles
            6. Topics to avoid
            7. Meal protocols
            8. Closing ceremonies
            9. Follow-up expectations
            10. Emergency conflict resolution
            
            Ensure the protocol respects both nations' matriarchal structures.
            """
            
            result = await Runner.run(protocol_agent, prompt, context=run_ctx.context)
            
            return {
                "nation1_id": nation_id1,
                "nation2_id": nation_id2,
                "protocol_guide": result.final_output,
                "based_on_conflicts": conflicts
            }
