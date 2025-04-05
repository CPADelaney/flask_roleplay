# lore/systems/regional_culture.py

import logging
import json
import random
from typing import Dict, List, Any, Optional
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
from lore.core.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils
from lore.core.cache import GLOBAL_LORE_CACHE

# ---------------------------------------------------------------------------
# Pydantic Models for structured outputs & guardrails
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------
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
        
        # Initialize specialized agents for different cultural tasks
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized agents for different cultural tasks"""
        base_instructions = (
            "You're creating content for a fantasy world featuring matriarchal power structures. "
            "Ensure all cultural elements reflect feminine authority and appropriate social hierarchies. "
            "Male elements are typically supportive or subordinate. Provide detail and realism."
        )
        
        # Language generation agent
        self.language_agent = Agent(
            name="LanguageGenerationAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create realistic languages for a fantasy world with matriarchal power. "
                "Reflect how language encodes status, formality, and gender hierarchy."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Cultural norm agent
        self.norm_agent = Agent(
            name="CulturalNormAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create cultural norms for fantasy nations. Norms must reflect matriarchal structures. "
                "Consider differences by gender, status, and context. Provide taboos, consequences, variations."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Etiquette agent
        self.etiquette_agent = Agent(
            name="EtiquetteAgent",
            instructions=(
                f"{base_instructions}\n\n"
                "You create detailed etiquette systems for matriarchal fantasy nations. "
                "Include greetings, body language, titles, gift-giving, and display of power. "
                "Be explicit about how men must defer to female authority."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
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
                CREATE TABLE CulturalNorms (
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
                CREATE TABLE Etiquette (
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
            distribution_agent = Agent(
                name="LanguageDistributionAgent",
                instructions=(
                    "You receive a list of nations and a desired number of languages to create. "
                    "Propose how to distribute each new language among these nations. "
                    "You must return JSON containing an array of languages, each with a list "
                    "of primary_region_ids and minority_region_ids. Example:\n\n"
                    "[\n"
                    "  {\n"
                    '    "primary_region_ids": [1,2],\n'
                    '    "minority_region_ids": [3]\n'
                    "  },\n"
                    "  ...\n"
                    "]\n\n"
                    "Remember we have matriarchal nations of varying size and importance; "
                    "some languages might be major (used by multiple large nations), others minor."
                ),
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.8)
            )
            
            # Build distribution prompt
            nation_list = [{"id": n["id"], "name": n["name"]} for n in nations]
            dist_prompt = f"""
            We have the following nations:
            {json.dumps(nation_list, indent=2)}

            We want to create {count} new languages. 
            Decide which nations are primary speakers vs. minority speakers for each new language.
            Return a JSON array with exactly {count} items, each item having:
              - "primary_region_ids": array of nation IDs
              - "minority_region_ids": array of nation IDs
            """
            
            dist_config = RunConfig(
                workflow_name="LanguageDistribution",
                trace_metadata=self.trace_metadata
            )
            dist_result = await Runner.run(distribution_agent, dist_prompt, context=run_ctx.context, run_config=dist_config)
            
            try:
                distribution_data = json.loads(dist_result.final_output)
            except json.JSONDecodeError:
                # Fallback if the LLM didn't produce valid JSON
                distribution_data = []
            
            # If distribution_data is malformed or doesn't match count, fallback:
            if not isinstance(distribution_data, list) or len(distribution_data) != count:
                # fallback - code-based approach
                distribution_data = []
                for i in range(count):
                    # Simplistic fallback: pick random primary & minority
                    if len(nations) <= 1:
                        distribution_data.append({"primary_region_ids": [nations[0]["id"]], "minority_region_ids": []})
                    else:
                        primary_choice = random.sample(nations, min(2, len(nations)))
                        minority_candidates = [n for n in nations if n not in primary_choice]
                        minority_choice = random.sample(minority_candidates, min(1, len(minority_candidates)))
                        distribution_data.append({
                            "primary_region_ids": [n["id"] for n in primary_choice],
                            "minority_region_ids": [n["id"] for n in minority_choice]
                        })
            
            # 2) Now for each language distribution, we call the language generation agent
            language_agent = self.language_agent.clone(output_type=LanguageOutput)
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
                
                primary_nations = [find_nation_data(nid) for nid in dist.get("primary_region_ids", [])]
                minority_nations = [find_nation_data(nid) for nid in dist.get("minority_region_ids", [])]
                
                prompt_data = {
                    "primary_nations": primary_nations,
                    "minority_nations": minority_nations
                }
                gen_prompt = (
                    "Create a new language for a matriarchal fantasy setting. Some nations speak it primarily; "
                    "others use it as a minority language. Output a LanguageOutput object."
                    f"\nPRIMARY NATIONS:\n{json.dumps(primary_nations, indent=2)}"
                    f"\nMINORITY NATIONS:\n{json.dumps(minority_nations, indent=2)}"
                )
                
                result = await Runner.run(language_agent, gen_prompt, context=run_ctx.context, run_config=run_config)
                language_data = result.final_output
                # Attach the distribution
                language_data.primary_regions = dist.get("primary_region_ids", [])
                language_data.minority_regions = dist.get("minority_region_ids", [])
                
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
            category_agent = Agent(
                name="NormCategoryAgent",
                instructions=(
                    "Given a single nation's data, decide which categories of cultural norms to generate "
                    "(e.g. greeting, dining, authority, gift_giving, personal_space, gender_relations, etc.), "
                    "and how many norms total. Return a JSON array of categories or a JSON object with categories + number. "
                    "Example:\n\n"
                    "[\"greeting\",\"authority\",\"dining\"]\n\n"
                    "or\n"
                    "{ \"categories\": [\"greeting\",\"public_behavior\",\"religious_practice\"], \"count\": 5 }"
                ),
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.8)
            )
            
            cat_prompt = f"""
            We have the following nation data:
            {json.dumps(nation_data, indent=2)}

            Decide which categories of cultural norms we should generate (like greeting, authority, dining, etc.), 
            and how many norms total. Return valid JSON with either an array or an object that includes categories 
            and an overall count.
            """
            
            cat_config = RunConfig(
                workflow_name="DetermineNormCategories",
                trace_metadata=self.trace_metadata
            )
            cat_result = await Runner.run(category_agent, cat_prompt, context=run_ctx.context, run_config=cat_config)
            
            # Attempt to parse
            try:
                category_data = json.loads(cat_result.final_output)
            except json.JSONDecodeError:
                # fallback
                category_data = ["greeting","dining","authority","gift_giving","gender_relations"]
            
            # Extract categories from category_data
            # e.g. it might be a list or an object with keys "categories" & "count"
            if isinstance(category_data, list):
                categories = category_data
                norms_count = len(category_data)
            elif isinstance(category_data, dict):
                categories = category_data.get("categories", [])
                norms_count = category_data.get("count", len(categories) or 5)
            else:
                categories = ["greeting","dining","authority"]
                norms_count = len(categories)
            
            # Now we generate that many norms with the norm agent
            norm_agent = self.norm_agent.clone(
                name="CulturalNormAgent",
                output_type=CulturalNormCompleteOutput,
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
                
                Return a CulturalNormCompleteOutput object with:
                - category: {cat}
                - description, formality_level, gender_specific, taboo_level, consequence, etc.
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
            context_agent = Agent(
                name="EtiquetteContextAgent",
                instructions=(
                    "We have a matriarchal nation. Decide which contexts we need specific etiquette for. "
                    "Example contexts might be 'court', 'noble', 'public', 'private', 'religious', 'business'. "
                    "Return a JSON array of strings for contexts. Example:\n[\"court\",\"public\",\"business\"]"
                ),
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.8)
            )
            
            ctx_prompt = f"""
            We have the following nation:
            {json.dumps(nation_data, indent=2)}

            Suggest an array of relevant 'contexts' for which we should generate distinct etiquette systems.
            Only valid JSON, e.g. ["court","religious","military"].
            """
            
            ctx_config = RunConfig(
                workflow_name="EtiquetteContexts",
                trace_metadata=self.trace_metadata
            )
            ctx_result = await Runner.run(context_agent, ctx_prompt, context=run_ctx.context, run_config=ctx_config)
            
            try:
                contexts = json.loads(ctx_result.final_output)
                if not isinstance(contexts, list):
                    contexts = ["court","public","private"]
            except json.JSONDecodeError:
                contexts = ["court","public","private"]
            
            # Now produce etiquette for each context
            etiquette_agent = self.etiquette_agent.clone(output_type=EtiquetteOutput, input_guardrails=[input_guardrail])
            run_config = RunConfig(workflow_name="EtiquetteGeneration", trace_metadata=self.trace_metadata)
            
            etiquette_systems = []
            for context_item in contexts:
                prompt = f"""
                Generate a detailed etiquette system for {context_item} contexts in this matriarchal nation:
                {json.dumps(nation_data, indent=2)}
                
                Return an EtiquetteOutput with fields describing how titles, greetings, body language, etc. vary.
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
    
    @function_tool
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
            summary_agent = Agent(
                name="CultureSummaryAgent",
                instructions=(
                    "You create coherent, matriarchal-themed cultural summaries. "
                    "Highlight the most distinctive norms, how matriarchal authority is expressed, "
                    "and any notable linguistic or etiquette features."
                ),
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.8)
            )
            
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            prompt = f"""
            Create a {format_type} summary of this nation's culture:
            
            NATION: {cultural_data['nation']['name']}
            
            CULTURAL DATA:
            {json.dumps(cultural_data, indent=2)}
            
            Emphasize matriarchal power in daily life, key traditions, and important linguistic notes.
            """
            
            result = await Runner.run(summary_agent, prompt, context=run_ctx.context)
            return result.final_output
    
    @function_tool
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
            
            conflict_agent = Agent(
                name="CulturalConflictAnalyst",
                instructions=(
                    "You analyze how two different matriarchal cultures might conflict. "
                    "Focus on greeting norms, etiquette, taboos, religious differences, etc. "
                    "Return a JSON object with fields 'potential_conflicts', 'severity_level', and 'recommendations'."
                ),
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.8)
            )
            
            run_ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
            prompt = f"""
            Analyze potential cultural conflicts between these two nations:
            
            NATION 1 DATA:
            {json.dumps(data1, indent=2)}
            
            NATION 2 DATA:
            {json.dumps(data2, indent=2)}
            
            Return JSON with:
            {{
              "potential_conflicts": [...],
              "severity_level": <1-10>,
              "recommendations": "..."
            }}
            """
            
            result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
            try:
                return json.loads(result.final_output)
            except json.JSONDecodeError:
                return {
                    "analysis": result.final_output,
                    "parsing_error": "Could not parse response as JSON"
                }
