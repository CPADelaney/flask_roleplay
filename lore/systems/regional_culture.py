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
                "model": "gpt-4.1-nano",
                "settings": model_settings
            },
            "norm": {
                "name": "CulturalNormAgent",
                "instructions": (
                    f"{base_instructions}\n\n"
                    "You create cultural norms for fantasy nations. Norms must reflect matriarchal structures. "
                    "Consider differences by gender, status, and context. Provide taboos, consequences, variations."
                ),
                "model": "gpt-4.1-nano",
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
                "model": "gpt-4.1-nano",
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
                "model": "gpt-4.1-nano",
                "settings": ModelSettings(temperature=0.8)
            },
            "category": {
                "name": "NormCategoryAgent",
                "instructions": (
                    "Given a single nation's data, decide which categories of cultural norms to generate "
                    "(e.g. greeting, dining, authority, gift_giving, personal_space, gender_relations, etc.), "
                    "and how many norms total. Consider the nation's specific characteristics."
                ),
                "model": "gpt-4.1-nano",
                "settings": ModelSettings(temperature=0.8)
            },
            "context": {
                "name": "EtiquetteContextAgent",
                "instructions": (
                    "We have a matriarchal nation. Decide which contexts we need specific etiquette for. "
                    "Example contexts might be 'court', 'noble', 'public', 'private', 'religious', 'business'. "
                    "Consider the nation's government type and cultural traits."
                ),
                "model": "gpt-4.1-nano",
                "settings": ModelSettings(temperature=0.8)
            },
            "summary": {
                "name": "CultureSummaryAgent",
                "instructions": (
                    "You create coherent, matriarchal-themed cultural summaries. "
                    "Highlight the most distinctive norms, how matriarchal authority is expressed, "
                    "and any notable linguistic or etiquette features."
                ),
                "model": "gpt-4.1-nano",
                "settings": ModelSettings(temperature=0.8)
            },
            "conflict": {
                "name": "CulturalConflictAnalyst",
                "instructions": (
                    "You analyze how two different matriarchal cultures might conflict. "
                    "Focus on greeting norms, etiquette, taboos, religious differences, etc. "
                    "Provide specific conflicts, severity assessment, and diplomatic recommendations."
                ),
                "model": "gpt-4.1-nano",
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
                "model": "gpt-4.1-nano",
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
                "model": "gpt-4.1-nano",
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
        async with await self.get_connection_pool() as pool:
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
        Generate languages with full canon establishment and relationship tracking.
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
            
            # Get existing languages and analyze patterns
            async with await self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    existing_languages = await conn.fetch("""
                        SELECT l.*, 
                               array_agg(DISTINCT n1.name) FILTER (WHERE n1.id = ANY(l.primary_regions)) as primary_nation_names,
                               array_agg(DISTINCT n2.name) FILTER (WHERE n2.id = ANY(l.minority_regions)) as minority_nation_names
                        FROM Languages l
                        LEFT JOIN Nations n1 ON n1.id = ANY(l.primary_regions)
                        LEFT JOIN Nations n2 ON n2.id = ANY(l.minority_regions)
                        GROUP BY l.id
                    """)
                    
                    # Analyze language families
                    language_families = {}
                    for lang in existing_languages:
                        family = lang['language_family']
                        if family:
                            if family not in language_families:
                                language_families[family] = []
                            language_families[family].append({
                                'id': lang['id'],
                                'name': lang['name'],
                                'regions': lang['primary_regions'] + lang['minority_regions']
                            })
                    
                    # Get geopolitical relationships
                    relations = await conn.fetch("""
                        SELECT r.*, n1.name as nation1_name, n2.name as nation2_name
                        FROM InternationalRelations r
                        JOIN Nations n1 ON r.nation1_id = n1.id
                        JOIN Nations n2 ON r.nation2_id = n2.id
                        WHERE r.relationship_quality >= 6
                    """)
            
            # Create distribution strategy agent
            distribution_agent = self._get_agent("distribution")
            
            # Build comprehensive context
            nation_data = []
            for n in nations:
                nation_info = {
                    "id": n["id"],
                    "name": n["name"],
                    "matriarchy_level": n.get("matriarchy_level", 5),
                    "cultural_traits": n.get("cultural_traits", []),
                    "neighboring_nations": n.get("neighboring_nations", []),
                    "existing_languages": []
                }
                
                # Add existing languages
                for lang in existing_languages:
                    if n["id"] in lang.get("primary_regions", []):
                        nation_info["existing_languages"].append({
                            "name": lang["name"],
                            "role": "primary"
                        })
                    elif n["id"] in lang.get("minority_regions", []):
                        nation_info["existing_languages"].append({
                            "name": lang["name"],
                            "role": "minority"
                        })
                
                nation_data.append(nation_info)
            
            # Generate distribution plan
            dist_prompt = f"""
            Plan the distribution of {count} new languages across these nations:
            
            NATIONS:
            {json.dumps(nation_data, indent=2)}
            
            EXISTING LANGUAGE FAMILIES:
            {json.dumps(list(language_families.keys()), indent=2)}
            
            DIPLOMATIC RELATIONS:
            {json.dumps([{"nations": [r["nation1_name"], r["nation2_name"]], "quality": r["relationship_quality"]} for r in relations], indent=2)}
            
            Consider:
            1. Geographic proximity (neighboring nations might share languages)
            2. Political alliances (allied nations might share trade languages)
            3. Cultural similarities (nations with similar traits might have related languages)
            4. Existing language gaps (nations without languages need them)
            5. Create both major languages (many speakers) and minor ones
            6. Some languages should form new families, others join existing ones
            
            For each of the {count} languages, return:
            - primary_region_ids: array of nation IDs where it's a primary language
            - minority_region_ids: array of nation IDs where it's a minority language
            - suggested_family: either an existing family name or "new:[family_name]"
            - distribution_reasoning: brief explanation
            """
            
            dist_config = RunConfig(
                workflow_name="LanguageDistribution",
                trace_metadata=self.trace_metadata
            )
            
            dist_result = await Runner.run(distribution_agent, dist_prompt, context=run_ctx.context, run_config=dist_config)
            
            try:
                distribution_plan = json.loads(dist_result.final_output)
                if not isinstance(distribution_plan, list):
                    distribution_plan = distribution_plan.get("languages", [])
            except:
                # Fallback distribution
                distribution_plan = self._create_fallback_distribution(nations, count, existing_languages)
            
            # Generate languages based on plan
            language_agent = self._get_agent("language").clone(output_type=LanguageOutput)
            languages = []
            
            async with await self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    for i, dist in enumerate(distribution_plan[:count]):
                        # Get detailed nation info for context
                        primary_nations = [n for n in nations if n["id"] in dist.get("primary_region_ids", [])]
                        minority_nations = [n for n in nations if n["id"] in dist.get("minority_region_ids", [])]
                        
                        if not primary_nations:
                            continue
                        
                        # Determine language family
                        suggested_family = dist.get("suggested_family", "")
                        if suggested_family.startswith("new:"):
                            family_name = suggested_family[4:]
                        elif suggested_family in language_families:
                            family_name = suggested_family
                        else:
                            family_name = f"Family_{i+1}"
                        
                        # Generate language details
                        gen_prompt = f"""
                        Create a new language for a matriarchal fantasy world.
                        
                        PRIMARY SPEAKERS (nations where this is the main language):
                        {json.dumps(primary_nations, indent=2)}
                        
                        MINORITY SPEAKERS (nations where this is a secondary language):
                        {json.dumps(minority_nations, indent=2)}
                        
                        LANGUAGE FAMILY: {family_name}
                        {"Related languages in family: " + json.dumps([l['name'] for l in language_families.get(family_name, [])]) if family_name in language_families else "This starts a new language family"}
                        
                        Create a language that:
                        1. Reflects the matriarchal power structures (pronouns, titles, formal speech)
                        2. Has vocabulary related to the nations' cultural traits
                        3. Shows influence from neighboring languages if applicable
                        4. Has appropriate complexity (difficulty 1-10)
                        5. Includes common phrases that reflect the culture
                        
                        Return a LanguageOutput object with all required fields.
                        """
                        
                        run_config = RunConfig(
                            workflow_name="LanguageGeneration",
                            trace_metadata={"language_index": i}
                        )
                        
                        result = await Runner.run(language_agent, gen_prompt, context=run_ctx.context, run_config=run_config)
                        language_data = result.final_output
                        
                        # Override with planned distribution
                        language_data.primary_regions = dist.get("primary_region_ids", [])
                        language_data.minority_regions = dist.get("minority_region_ids", [])
                        language_data.language_family = family_name
                        
                        # Check for duplicates using canon system
                        from lore.core import canon
                        embed_text = f"{language_data.name} {language_data.description} {language_data.language_family}"
                        
                        # Custom duplicate check for languages
                        similar_language = await self._check_similar_language(
                            conn, language_data.name, embed_text, language_data.language_family
                        )
                        
                        if similar_language:
                            # Ask validation agent
                            validation_agent = canon.CanonValidationAgent()
                            prompt = f"""
                            I'm creating a new language but found a similar one. Are these the same?
                            
                            Proposed Language:
                            - Name: {language_data.name}
                            - Family: {language_data.language_family}
                            - Description: {language_data.description}
                            - Primary regions: {[n['name'] for n in primary_nations]}
                            
                            Existing Language:
                            - Name: {similar_language['name']}
                            - Family: {similar_language['language_family']}
                            - Description: {similar_language['description']}
                            
                            Consider that languages can have similar names but be different.
                            Answer only 'true' or 'false'.
                            """
                            
                            validation_result = await Runner.run(validation_agent.agent, prompt)
                            if validation_result.final_output.strip().lower() == 'true':
                                # Use existing language, maybe expand its reach
                                language_id = similar_language['id']
                                
                                # Add new regions if needed
                                updated_primary = list(set(similar_language['primary_regions'] + language_data.primary_regions))
                                updated_minority = list(set(similar_language['minority_regions'] + language_data.minority_regions))
                                
                                await conn.execute("""
                                    UPDATE Languages
                                    SET primary_regions = $1,
                                        minority_regions = $2
                                    WHERE id = $3
                                """, updated_primary, updated_minority, language_id)
                                
                                lang_dict = dict(similar_language)
                                lang_dict['primary_regions'] = updated_primary
                                lang_dict['minority_regions'] = updated_minority
                                languages.append(lang_dict)
                                continue
                        
                        # Create new language
                        language_id = await conn.fetchval("""
                            INSERT INTO Languages (
                                name, language_family, description, writing_system,
                                primary_regions, minority_regions, formality_levels,
                                common_phrases, difficulty, relation_to_power, dialects,
                                embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
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
                            json.dumps(language_data.dialects),
                            await generate_embedding(embed_text)
                        )
                        
                        # Log canonical event
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"New language '{language_data.name}' emerged in the {language_data.language_family} family",
                            tags=['language', 'culture', 'creation'],
                            significance=7
                        )
                        
                        # Create initial dialects for major regions
                        for nation_id in language_data.primary_regions[:3]:  # First 3 primary regions
                            nation = next((n for n in nations if n['id'] == nation_id), None)
                            if nation:
                                await self._create_initial_dialect(
                                    conn, language_id, nation_id, 
                                    language_data.name, nation['name']
                                )
                        
                        lang_dict = language_data.dict()
                        lang_dict["id"] = language_id
                        languages.append(lang_dict)
            
            # Update language relationships
            await self._establish_language_relationships(languages, existing_languages)
            
            return languages

    async def _check_similar_language(self, conn, name: str, embed_text: str, family: str) -> Optional[Dict[str, Any]]:
        """Check for similar languages with sophisticated matching."""
        # Check exact name
        exact = await conn.fetchrow("""
            SELECT * FROM Languages WHERE LOWER(name) = LOWER($1)
        """, name)
        if exact:
            return dict(exact)
        
        # Check fuzzy name match
        fuzzy = await conn.fetch("""
            SELECT *, similarity(name, $1) as sim
            FROM Languages
            WHERE similarity(name, $1) > 0.7
            ORDER BY sim DESC
            LIMIT 1
        """, name)
        if fuzzy:
            return dict(fuzzy[0])
        
        # Check same family with similar embedding
        if family:
            family_similar = await conn.fetch("""
                SELECT *, 1 - (embedding <=> $1) as similarity
                FROM Languages
                WHERE language_family = $2
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> $1) > 0.85
                ORDER BY embedding <=> $1
                LIMIT 1
            """, await generate_embedding(embed_text), family)
            if family_similar:
                return dict(family_similar[0])
        
        return None
    
    async def _create_initial_dialect(self, conn, language_id: int, nation_id: int, 
                                     language_name: str, nation_name: str) -> None:
        """Create an initial dialect for a language in a specific nation."""
        dialect_name = f"{language_name} ({nation_name} dialect)"
        
        try:
            await conn.execute("""
                INSERT INTO LanguageDialects (
                    language_id, region_id, name, parent_language,
                    vocabulary_changes, grammatical_changes, pronunciation_shifts,
                    social_context, prestige_level, example_phrases,
                    regional_distribution, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (language_id, region_id) DO NOTHING
            """,
                language_id, nation_id, dialect_name, language_name,
                json.dumps({}),  # No initial vocabulary changes
                [],  # No initial grammatical changes
                [],  # No initial pronunciation shifts
                f"Standard dialect spoken in {nation_name}",
                7,  # Default prestige
                json.dumps({"greeting": f"Standard greeting in {nation_name}"}),
                [nation_name],
                await generate_embedding(dialect_name)
            )
        except Exception as e:
            logger.error(f"Error creating initial dialect: {e}")
    
    async def _establish_language_relationships(self, new_languages: List[Dict[str, Any]], 
                                              existing_languages: List[Any]) -> None:
        """Establish relationships between languages (borrowings, influences, etc)."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Group languages by family
                families = {}
                for lang in new_languages + [dict(l) for l in existing_languages]:
                    family = lang.get('language_family', 'Unknown')
                    if family not in families:
                        families[family] = []
                    families[family].append(lang)
                
                # Create relationships within families
                for family, langs in families.items():
                    if len(langs) > 1:
                        # Create family relationship entries
                        for i, lang1 in enumerate(langs):
                            for lang2 in langs[i+1:]:
                                await conn.execute("""
                                    INSERT INTO LanguageRelationships (
                                        language1_id, language2_id, relationship_type,
                                        description, influence_level
                                    )
                                    VALUES ($1, $2, $3, $4, $5)
                                    ON CONFLICT (language1_id, language2_id) DO NOTHING
                                """,
                                    lang1['id'], lang2['id'], 'family',
                                    f"Related languages in the {family} family",
                                    5  # Medium influence
                                )
    
    def _create_fallback_distribution(self, nations: List[Dict[str, Any]], count: int, 
                                     existing: List[Any]) -> List[Dict[str, Any]]:
        """Create a fallback distribution plan if LLM fails."""
        distribution = []
        
        # Find nations without primary languages
        nations_with_primary = set()
        for lang in existing:
            nations_with_primary.update(lang.get('primary_regions', []))
        
        nations_without_primary = [n for n in nations if n['id'] not in nations_with_primary]
        
        # Distribute languages
        for i in range(count):
            if i < len(nations_without_primary):
                # Give languages to nations without any
                primary = [nations_without_primary[i]['id']]
                # Add neighbors as minority speakers
                neighbors = nations_without_primary[i].get('neighboring_nations', [])
                minority = [n for n in neighbors if isinstance(n, int)][:2]
            else:
                # Create regional languages
                available = [n for n in nations if n['id'] not in nations_with_primary]
                if not available:
                    available = nations
                
                primary_nation = random.choice(available)
                primary = [primary_nation['id']]
                
                # Add neighbors
                neighbors = primary_nation.get('neighboring_nations', [])
                minority = [n for n in neighbors if isinstance(n, int)][:2]
            
            distribution.append({
                'primary_region_ids': primary,
                'minority_region_ids': minority,
                'suggested_family': f'Family_{(i % 3) + 1}',
                'distribution_reasoning': 'Fallback distribution'
            })
        
        return distribution
    
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
            async with await self.get_connection_pool() as pool:
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
                async with await self.get_connection_pool() as pool:
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
            async with await self.get_connection_pool() as pool:
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
                async with await self.get_connection_pool() as pool:
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
            
            async with await self.get_connection_pool() as pool:
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
            async with await self.get_connection_pool() as pool:
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
        """Apply language diffusion effects between nations."""
        if not effects or not isinstance(effects, dict):
            return
            
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get languages for both nations
                nation1_langs = await conn.fetch("""
                    SELECT l.*, n.name as nation_name, n.matriarchy_level
                    FROM Languages l
                    JOIN Nations n ON n.id = ANY(l.primary_regions) OR n.id = ANY(l.minority_regions)
                    WHERE n.id = $1
                """, nation1_id)
                
                nation2_langs = await conn.fetch("""
                    SELECT l.*, n.name as nation_name, n.matriarchy_level
                    FROM Languages l
                    JOIN Nations n ON n.id = ANY(l.primary_regions) OR n.id = ANY(l.minority_regions)
                    WHERE n.id = $1
                """, nation2_id)
                
                if not nation1_langs or not nation2_langs:
                    return
                
                # Process vocabulary diffusion
                if "vocabulary" in effects or "vocabulary_borrowed" in effects:
                    vocab_changes = effects.get("vocabulary", effects.get("vocabulary_borrowed", {}))
                    
                    for lang in nation2_langs:
                        lang_id = lang["id"]
                        
                        # Get current common phrases
                        common_phrases = json.loads(lang.get("common_phrases", "{}"))
                        
                        # Add borrowed vocabulary with cultural adaptation
                        for original_word, meaning in vocab_changes.items():
                            # Adapt the word to the receiving language's phonology
                            adapted_word = await self._adapt_word_phonology(original_word, lang)
                            common_phrases[adapted_word] = f"{meaning} (borrowed from {nation1_langs[0]['name']})"
                        
                        # Update the language
                        await conn.execute("""
                            UPDATE Languages
                            SET common_phrases = $1,
                                dialects = dialects || jsonb_build_object($2, 'influenced')
                            WHERE id = $3
                        """, json.dumps(common_phrases), nation1_langs[0]['name'], lang_id)
                        
                        # Create dialect entry if significant changes
                        if len(vocab_changes) > 5:
                            try:
                                await conn.execute("""
                                    INSERT INTO LanguageDialects (
                                        language_id, region_id, name, parent_language,
                                        vocabulary_changes, social_context, prestige_level
                                    )
                                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                                    ON CONFLICT (language_id, region_id) 
                                    DO UPDATE SET vocabulary_changes = 
                                        LanguageDialects.vocabulary_changes || EXCLUDED.vocabulary_changes
                                """,
                                    lang_id, nation2_id,
                                    f"{lang['name']} ({nation2_langs[0]['nation_name']} variant)",
                                    lang['name'],
                                    json.dumps(vocab_changes),
                                    f"Influenced by contact with {nation1_langs[0]['name']} speakers",
                                    7  # Prestige level
                                )
                            except Exception as e:
                                logger.error(f"Error creating dialect: {e}")
                
                # Process idiomatic expressions
                if "idioms" in effects or "idioms_adopted" in effects:
                    idioms = effects.get("idioms", effects.get("idioms_adopted", []))
                    
                    for lang in nation2_langs:
                        phrases = json.loads(lang.get("common_phrases", "{}"))
                        
                        for idiom in idioms:
                            # Translate idiom concept to local cultural context
                            localized_idiom = await self._localize_idiom(idiom, lang, nation2_id)
                            phrases[localized_idiom["phrase"]] = localized_idiom["meaning"]
                        
                        await conn.execute("""
                            UPDATE Languages SET common_phrases = $1 WHERE id = $2
                        """, json.dumps(phrases), lang["id"])
                
                # Record the cultural exchange
                exchange_id = await conn.fetchval("""
                    INSERT INTO CulturalExchanges (
                        nation1_id, nation2_id, exchange_type, exchange_details, 
                        impact_level, timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, 
                    nation1_id, nation2_id, "language_diffusion", 
                    json.dumps(effects), 
                    self._calculate_impact_level(effects),
                    datetime.now()
                )
                
                # Create canonical event
                from lore.core import canon
                await canon.log_canonical_event(
                    self.create_run_context({}), conn,
                    f"Linguistic exchange occurred between nations {nation1_id} and {nation2_id}",
                    tags=['cultural_exchange', 'language', 'diffusion'],
                    significance=6
                )

    
    async def _apply_artistic_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply artistic and creative diffusion between nations."""
        if not effects:
            return
            
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get cultural context for both nations
                nation1_data = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT ce.name) as cultural_elements
                    FROM Nations n
                    LEFT JOIN CulturalElements ce ON n.name = ANY(ce.practiced_by)
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation1_id)
                
                nation2_data = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT ce.name) as cultural_elements
                    FROM Nations n
                    LEFT JOIN CulturalElements ce ON n.name = ANY(ce.practiced_by)
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation2_id)
                
                if not nation1_data or not nation2_data:
                    return
                
                # Process artistic elements
                artistic_elements = effects.get("artistic_elements", effects.get("art_forms", []))
                
                for element in artistic_elements:
                    # Check if element already exists
                    existing = await conn.fetchrow("""
                        SELECT id, name, practiced_by FROM CulturalElements
                        WHERE name = $1 AND element_type = 'artistic'
                    """, element.get("name", element))
                    
                    if existing:
                        # Update to include new nation
                        practiced_by = existing['practiced_by'] or []
                        if nation2_data['name'] not in practiced_by:
                            practiced_by.append(nation2_data['name'])
                            
                        await conn.execute("""
                            UPDATE CulturalElements
                            SET practiced_by = $1,
                                description = description || E'\n\n' || $2,
                                significance = GREATEST(significance, $3)
                            WHERE id = $4
                        """, 
                            practiced_by,
                            f"Adopted by {nation2_data['name']} through cultural exchange with {nation1_data['name']}",
                            element.get("significance", 5),
                            existing['id']
                        )
                    else:
                        # Create new cultural element
                        element_data = element if isinstance(element, dict) else {"name": element}
                        
                        ce_id = await conn.fetchval("""
                            INSERT INTO CulturalElements (
                                name, element_type, description, practiced_by,
                                significance, historical_origin, matriarchal_elements,
                                associated_practices, required_materials, 
                                gender_associations, prestige_level, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            RETURNING id
                        """,
                            element_data.get("name", f"Art form {len(artistic_elements)}"),
                            "artistic",
                            element_data.get("description", f"Artistic tradition shared between {nation1_data['name']} and {nation2_data['name']}"),
                            [nation1_data['name'], nation2_data['name']],
                            element_data.get("significance", 6),
                            f"Originated in {nation1_data['name']}, spread to {nation2_data['name']}",
                            element_data.get("matriarchal_elements", "Celebrates feminine creative power"),
                            element_data.get("associated_practices", []),
                            element_data.get("required_materials", []),
                            element_data.get("gender_associations", {"primary": "female", "secondary": "all"}),
                            element_data.get("prestige_level", 7),
                            await generate_embedding(element_data.get("name", "") + " artistic tradition")
                        )
                
                # Process literary influences
                if "literary_influences" in effects:
                    for lit_influence in effects["literary_influences"]:
                        # Create literary tradition entries
                        await conn.execute("""
                            INSERT INTO CulturalElements (
                                name, element_type, description, practiced_by,
                                significance, historical_origin, matriarchal_elements
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (name, element_type) DO UPDATE
                            SET practiced_by = array_append(
                                COALESCE(CulturalElements.practiced_by, ARRAY[]::text[]), 
                                $8
                            )
                        """,
                            lit_influence.get("name", f"Literary tradition"),
                            "literary",
                            lit_influence.get("description", "Shared literary form"),
                            [nation1_data['name'], nation2_data['name']],
                            lit_influence.get("significance", 5),
                            f"Literary exchange between {nation1_data['name']} and {nation2_data['name']}",
                            "Female authors and perspectives emphasized",
                            nation2_data['name']
                        )
    
    async def _apply_religious_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply religious practice and belief diffusion between nations."""
        if not effects:
            return
            
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get religious data for both nations
                nation1_religion = await conn.fetchrow("""
                    SELECT nr.*, p.name as pantheon_name
                    FROM NationReligion nr
                    LEFT JOIN Pantheons p ON nr.primary_pantheon_id = p.id
                    WHERE nr.nation_id = $1
                """, nation1_id)
                
                nation2_religion = await conn.fetchrow("""
                    SELECT nr.*, p.name as pantheon_name
                    FROM NationReligion nr
                    LEFT JOIN Pantheons p ON nr.primary_pantheon_id = p.id
                    WHERE nr.nation_id = $1
                """, nation2_id)
                
                if not nation1_religion:
                    return
                    
                religious_practices = effects.get("religious_practices", [])
                
                for practice in religious_practices:
                    # Check if practice already exists
                    existing = await conn.fetchrow("""
                        SELECT id FROM ReligiousPractices
                        WHERE name = $1
                    """, practice.get("name", practice))
                    
                    if existing:
                        # Add to regional variations
                        await conn.execute("""
                            INSERT INTO RegionalReligiousPractice (
                                nation_id, practice_id, regional_variation,
                                importance, frequency, local_additions,
                                gender_differences
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (nation_id, practice_id) DO UPDATE
                            SET regional_variation = EXCLUDED.regional_variation,
                                importance = EXCLUDED.importance
                        """,
                            nation2_id,
                            existing['id'],
                            practice.get("variation", f"Adapted from {nation1_religion['pantheon_name']} tradition"),
                            practice.get("importance", 5),
                            practice.get("frequency", "seasonal"),
                            practice.get("local_additions", f"Incorporated local customs of nation {nation2_id}"),
                            practice.get("gender_differences", "Maintains matriarchal hierarchy")
                        )
                    else:
                        # Create new syncretic practice
                        if nation1_religion.get('primary_pantheon_id'):
                            await conn.execute("""
                                INSERT INTO ReligiousPractices (
                                    name, practice_type, description, purpose,
                                    frequency, pantheon_id, embedding
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """,
                                practice.get("name", "Syncretic practice"),
                                "syncretic",
                                practice.get("description", f"Practice combining traditions"),
                                practice.get("purpose", "cultural_harmony"),
                                practice.get("frequency", "seasonal"),
                                nation1_religion['primary_pantheon_id'],
                                await generate_embedding(practice.get("name", "") + " religious practice")
                            )
                
                # Update religious minorities
                if nation2_religion:
                    minorities = nation2_religion.get('religious_minorities', [])
                    if nation1_religion.get('pantheon_name') and nation1_religion['pantheon_name'] not in minorities:
                        minorities.append(f"{nation1_religion['pantheon_name']} practitioners")
                        
                    await conn.execute("""
                        UPDATE NationReligion
                        SET religious_minorities = $1
                        WHERE nation_id = $2
                    """, minorities, nation2_id)
    
    async def _adapt_word_phonology(self, word: str, target_language: Dict[str, Any]) -> str:
        """Adapt a word to fit the phonology of the target language."""
        # Use agent to adapt the word
        adaptation_agent = Agent(
            name="PhonologyAdaptationAgent",
            instructions="""You adapt foreign words to fit a target language's sound system.
            Consider the language's phonological constraints and typical sound patterns.""",
            model="gpt-4.1-nano"
        )
        
        prompt = f"""
        Adapt this word to fit the target language's phonology:
        
        Word: {word}
        Target Language: {target_language.get('name')}
        Writing System: {target_language.get('writing_system')}
        
        Make minimal changes to preserve recognizability while fitting the language's sounds.
        Return only the adapted word.
        """
        
        result = await Runner.run(adaptation_agent, prompt)
        return result.final_output.strip()
    
    async def _localize_idiom(self, idiom: str, target_language: Dict[str, Any], nation_id: int) -> Dict[str, str]:
        """Localize an idiom to fit the target culture."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT name, cultural_traits, matriarchy_level
                    FROM Nations WHERE id = $1
                """, nation_id)
        
        localization_agent = Agent(
            name="IdiomLocalizationAgent",
            instructions="""You adapt idioms to fit different cultural contexts while preserving meaning.
            Consider local culture, values, and matriarchal themes.""",
            model="gpt-4.1-nano"
        )
        
        prompt = f"""
        Localize this idiom for the target culture:
        
        Original Idiom: {idiom}
        Target Nation: {nation['name']}
        Cultural Traits: {nation.get('cultural_traits', [])}
        Matriarchy Level: {nation.get('matriarchy_level', 5)}/10
        
        Create a culturally appropriate version that:
        1. Preserves the core meaning
        2. Uses local cultural references
        3. Reflects matriarchal values where appropriate
        
        Return JSON with:
        - phrase: the localized idiom
        - meaning: explanation of what it means
        """
        
        result = await Runner.run(localization_agent, prompt)
        try:
            return json.loads(result.final_output)
        except:
            return {"phrase": idiom, "meaning": "Borrowed saying"}
    
    def _calculate_impact_level(self, effects: Dict[str, Any]) -> int:
        """Calculate the impact level of cultural diffusion (1-10)."""
        impact = 0
        
        # Count elements
        for key, value in effects.items():
            if isinstance(value, list):
                impact += min(len(value), 3)
            elif isinstance(value, dict):
                impact += min(len(value), 3)
            elif value:
                impact += 1
        
        return min(impact, 10)
    
    async def _apply_fashion_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply fashion and clothing diffusion effects between nations."""
        if not effects:
            return
            
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get nation data for context
                nation1 = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT ce.name) FILTER (WHERE ce.element_type = 'fashion') as fashion_elements
                    FROM Nations n
                    LEFT JOIN CulturalElements ce ON n.name = ANY(ce.practiced_by)
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation1_id)
                
                nation2 = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT ce.name) FILTER (WHERE ce.element_type = 'fashion') as fashion_elements
                    FROM Nations n
                    LEFT JOIN CulturalElements ce ON n.name = ANY(ce.practiced_by)
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation2_id)
                
                if not nation1 or not nation2:
                    return
                
                # Process fashion elements
                fashion_elements = effects.get("fashion_elements", effects.get("clothing_styles", []))
                
                # Create fashion adaptation agent
                fashion_agent = Agent(
                    name="FashionAdaptationAgent",
                    instructions="""You adapt fashion and clothing styles between cultures.
                    Consider climate, materials, cultural values, and matriarchal power display.""",
                    model="gpt-4.1-nano"
                )
                
                for element in fashion_elements:
                    element_data = element if isinstance(element, dict) else {"name": element}
                    
                    # Check if similar fashion already exists
                    existing = await conn.fetchrow("""
                        SELECT id, name, practiced_by, description
                        FROM CulturalElements
                        WHERE similarity(name, $1) > 0.7
                          AND element_type = 'fashion'
                    """, element_data.get("name", ""))
                    
                    if existing:
                        # Update existing fashion element
                        practiced_by = existing['practiced_by'] or []
                        if nation2['name'] not in practiced_by:
                            practiced_by.append(nation2['name'])
                            
                            # Get adaptation details
                            prompt = f"""
                            Fashion style "{existing['name']}" from {nation1['name']} is being adopted by {nation2['name']}.
                            
                            Nation 1 (origin): Matriarchy level {nation1['matriarchy_level']}/10
                            Nation 2 (adopting): Matriarchy level {nation2['matriarchy_level']}/10
                            
                            How would this fashion be adapted? Consider:
                            - Local materials and climate
                            - Cultural sensibilities
                            - Power display differences
                            - Gender-specific adaptations
                            
                            Return JSON with:
                            - adapted_name: localized name
                            - modifications: list of changes
                            - gender_variations: how it differs by gender
                            - status_indicators: how it shows social rank
                            """
                            
                            result = await Runner.run(fashion_agent, prompt)
                            try:
                                adaptation = json.loads(result.final_output)
                            except:
                                adaptation = {
                                    "adapted_name": existing['name'],
                                    "modifications": ["Adapted to local styles"],
                                    "gender_variations": {"female": "Elaborate", "male": "Simple"},
                                    "status_indicators": "Quality of materials"
                                }
                            
                            await conn.execute("""
                                UPDATE CulturalElements
                                SET practiced_by = $1,
                                    description = description || E'\n\n' || $2,
                                    regional_variations = COALESCE(regional_variations, '{}'::jsonb) || $3::jsonb
                                WHERE id = $4
                            """, 
                                practiced_by,
                                f"In {nation2['name']}: {', '.join(adaptation.get('modifications', []))}",
                                json.dumps({nation2['name']: adaptation}),
                                existing['id']
                            )
                    else:
                        # Create new fashion element
                        fashion_name = element_data.get("name", f"Fashion style {len(fashion_elements)}")
                        
                        # Generate detailed fashion description
                        prompt = f"""
                        Create a fashion element spreading from {nation1['name']} to {nation2['name']}.
                        
                        Fashion: {fashion_name}
                        Origin culture: Matriarchy level {nation1['matriarchy_level']}/10
                        Adopting culture: Matriarchy level {nation2['matriarchy_level']}/10
                        
                        Design fashion that:
                        1. Shows clear matriarchal power structures
                        2. Has distinct versions for different genders
                        3. Indicates social status
                        4. Uses available materials
                        
                        Return JSON with:
                        - description: detailed description
                        - materials: list of materials used
                        - gender_specific: dict of gender variations
                        - occasions: when it's worn
                        - status_levels: how elites vs commoners wear it
                        - colors: significant colors and meanings
                        - accessories: related items
                        """
                        
                        result = await Runner.run(fashion_agent, prompt)
                        try:
                            fashion_details = json.loads(result.final_output)
                        except:
                            fashion_details = {
                                "description": element_data.get("description", "Imported fashion style"),
                                "materials": ["Local fabrics"],
                                "gender_specific": {"female": "Elaborate", "male": "Simple"},
                                "occasions": ["Formal events"],
                                "status_levels": {"elite": "Luxurious", "common": "Basic"},
                                "colors": {"purple": "nobility", "white": "purity"},
                                "accessories": []
                            }
                        
                        ce_id = await conn.fetchval("""
                            INSERT INTO CulturalElements (
                                name, element_type, description, practiced_by,
                                significance, historical_origin, matriarchal_elements,
                                associated_practices, required_materials,
                                gender_associations, prestige_level, 
                                seasonal_variations, age_variations,
                                regional_variations, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                            RETURNING id
                        """,
                            fashion_name,
                            "fashion",
                            fashion_details.get("description"),
                            [nation1['name'], nation2['name']],
                            element_data.get("significance", 6),
                            f"Originated in {nation1['name']}, adopted by {nation2['name']}",
                            "Emphasizes feminine authority through elaborate designs for women",
                            fashion_details.get("occasions", []),
                            fashion_details.get("materials", []),
                            fashion_details.get("gender_specific", {}),
                            7,  # Prestige level
                            element_data.get("seasonal_variations", {}),
                            element_data.get("age_variations", {}),
                            json.dumps({
                                nation1['name']: "Original style",
                                nation2['name']: fashion_details
                            }),
                            await generate_embedding(fashion_name + " fashion clothing")
                        )
                        
                        # Create specific garment entries
                        if fashion_details.get("gender_specific"):
                            for gender, style in fashion_details["gender_specific"].items():
                                await conn.execute("""
                                    INSERT INTO FashionGarments (
                                        cultural_element_id, garment_name, gender,
                                        description, occasions, status_requirement
                                    )
                                    VALUES ($1, $2, $3, $4, $5, $6)
                                    ON CONFLICT (cultural_element_id, garment_name, gender) DO NOTHING
                                """,
                                    ce_id,
                                    f"{fashion_name} ({gender})",
                                    gender,
                                    style,
                                    fashion_details.get("occasions", []),
                                    "varies"
                                )
                
                # Process accessories and jewelry
                if "accessories" in effects or "jewelry" in effects:
                    accessories = effects.get("accessories", effects.get("jewelry", []))
                    
                    for accessory in accessories:
                        accessory_data = accessory if isinstance(accessory, dict) else {"name": accessory}
                        
                        await conn.execute("""
                            INSERT INTO CulturalElements (
                                name, element_type, description, practiced_by,
                                significance, matriarchal_elements, gender_associations
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (name, element_type) DO UPDATE
                            SET practiced_by = array_append(
                                COALESCE(CulturalElements.practiced_by, ARRAY[]::text[]), $8
                            )
                        """,
                            accessory_data.get("name", "Accessory"),
                            "accessory",
                            accessory_data.get("description", f"Fashion accessory adopted from {nation1['name']}"),
                            [nation2['name']],
                            accessory_data.get("significance", 5),
                            "Status symbols emphasizing feminine power",
                            accessory_data.get("gender_associations", {"primary": "female"}),
                            nation2['name']
                        )
                
                # Record the cultural exchange
                exchange_id = await conn.fetchval("""
                    INSERT INTO CulturalExchanges (
                        nation1_id, nation2_id, exchange_type, exchange_details,
                        impact_level, cultural_resistance, adoption_timeline,
                        timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """,
                    nation1_id, nation2_id, "fashion_diffusion",
                    json.dumps(effects),
                    self._calculate_impact_level(effects),
                    effects.get("resistance_level", 20),  # Some resistance to foreign fashion
                    effects.get("adoption_timeline", "gradual"),
                    datetime.now()
                )
                
                # Log canonical event
                from lore.core import canon
                await canon.log_canonical_event(
                    self.create_run_context({}), conn,
                    f"Fashion trends from {nation1['name']} influenced styles in {nation2['name']}",
                    tags=['cultural_exchange', 'fashion', 'diffusion'],
                    significance=5
                )
    
    async def _apply_cuisine_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply culinary and food diffusion effects between nations."""
        if not effects:
            return
            
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get nation data including existing cuisine
                nation1 = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT ct.name) FILTER (WHERE ct.id IS NOT NULL) as cuisine_traditions
                    FROM Nations n
                    LEFT JOIN CulinaryTraditions ct ON ct.nation_origin = n.id
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation1_id)
                
                nation2 = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT ct.name) FILTER (WHERE ct.id IS NOT NULL) as cuisine_traditions
                    FROM Nations n
                    LEFT JOIN CulinaryTraditions ct ON ct.nation_origin = n.id
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation2_id)
                
                if not nation1 or not nation2:
                    return
                
                # Create culinary adaptation agent
                cuisine_agent = Agent(
                    name="CuisineAdaptationAgent",
                    instructions="""You adapt culinary traditions between cultures.
                    Consider local ingredients, dietary restrictions, cultural significance,
                    and how food reflects matriarchal social structures.""",
                    model="gpt-4.1-nano"
                )
                
                # Process cuisine elements
                cuisine_elements = effects.get("cuisine_elements", effects.get("dishes", []))
                
                for dish in cuisine_elements:
                    dish_data = dish if isinstance(dish, dict) else {"name": dish}
                    
                    # Check if dish already exists
                    existing = await conn.fetchrow("""
                        SELECT * FROM CulinaryTraditions
                        WHERE similarity(name, $1) > 0.7
                    """, dish_data.get("name", ""))
                    
                    if existing:
                        # Update adopted_by list
                        adopted_by = existing['adopted_by'] or []
                        if nation2_id not in adopted_by:
                            adopted_by.append(nation2_id)
                            
                            # Get localization details
                            prompt = f"""
                            The dish "{existing['name']}" from {nation1['name']} is being adopted by {nation2['name']}.
                            
                            Original ingredients: {existing.get('ingredients', [])}
                            Nation 2 available ingredients: Consider local availability
                            
                            How would this dish be adapted? Consider:
                            - Ingredient substitutions
                            - Cooking method changes
                            - Cultural dietary restrictions
                            - Presentation changes
                            - Gender-specific serving customs
                            
                            Return JSON with:
                            - local_name: what it's called in the new culture
                            - ingredient_substitutions: dict of original->local ingredients
                            - preparation_changes: list of method adaptations
                            - serving_customs: how it's served (who serves, who eats first)
                            - occasions: when it's eaten
                            - status_association: is it elite or common food
                            """
                            
                            result = await Runner.run(cuisine_agent, prompt)
                            try:
                                adaptation = json.loads(result.final_output)
                            except:
                                adaptation = {
                                    "local_name": existing['name'],
                                    "ingredient_substitutions": {},
                                    "preparation_changes": ["Adapted to local methods"],
                                    "serving_customs": "Women serve first",
                                    "occasions": ["Daily meals"],
                                    "status_association": "common"
                                }
                            
                            await conn.execute("""
                                UPDATE CulinaryTraditions
                                SET adopted_by = $1,
                                    regional_variations = COALESCE(regional_variations, '{}'::jsonb) || $2::jsonb
                                WHERE id = $3
                            """,
                                adopted_by,
                                json.dumps({nation2['name']: adaptation}),
                                existing['id']
                            )
                    else:
                        # Create new culinary tradition
                        dish_name = dish_data.get("name", f"Dish {len(cuisine_elements)}")
                        
                        # Generate detailed culinary information
                        prompt = f"""
                        Create a culinary tradition spreading from {nation1['name']} to {nation2['name']}.
                        
                        Dish: {dish_name}
                        Origin: {nation1['name']} (Matriarchy level {nation1['matriarchy_level']}/10)
                        Adopting: {nation2['name']} (Matriarchy level {nation2['matriarchy_level']}/10)
                        
                        Create a dish that:
                        1. Has cultural significance beyond nutrition
                        2. Reflects matriarchal customs (who cooks, who serves, eating order)
                        3. Uses available ingredients
                        4. Has ceremonial or social importance
                        
                        Return JSON with:
                        - description: detailed description
                        - ingredients: list of main ingredients
                        - preparation: step-by-step method
                        - cooking_time: how long it takes
                        - difficulty: easy/medium/hard
                        - cultural_significance: why it matters
                        - gender_roles: who prepares and how it's served
                        - occasions: when it's made
                        - taboos: any restrictions
                        - nutritional_focus: health benefits emphasized
                        - presentation: how it's presented
                        """
                        
                        result = await Runner.run(cuisine_agent, prompt)
                        try:
                            culinary_details = json.loads(result.final_output)
                        except:
                            culinary_details = {
                                "description": dish_data.get("description", "Traditional dish"),
                                "ingredients": ["Local ingredients"],
                                "preparation": "Traditional method",
                                "cooking_time": "Variable",
                                "difficulty": "medium",
                                "cultural_significance": "Social bonding",
                                "gender_roles": {"preparer": "women", "server": "young women"},
                                "occasions": ["Festivals"],
                                "taboos": [],
                                "nutritional_focus": "Balanced",
                                "presentation": "Decorative"
                            }
                        
                        ct_id = await conn.fetchval("""
                            INSERT INTO CulinaryTraditions (
                                name, nation_origin, description, ingredients,
                                preparation, cultural_significance, adopted_by,
                                occasions, difficulty, cooking_time,
                                gender_roles, taboos, nutritional_focus,
                                presentation_style, meal_type,
                                seasonal_availability, preservation_methods,
                                regional_variations, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                            RETURNING id
                        """,
                            dish_name,
                            nation1_id,
                            culinary_details.get("description"),
                            culinary_details.get("ingredients", []),
                            culinary_details.get("preparation"),
                            culinary_details.get("cultural_significance"),
                            [nation2_id],
                            culinary_details.get("occasions", []),
                            culinary_details.get("difficulty", "medium"),
                            culinary_details.get("cooking_time"),
                            json.dumps(culinary_details.get("gender_roles", {})),
                            culinary_details.get("taboos", []),
                            culinary_details.get("nutritional_focus"),
                            culinary_details.get("presentation"),
                            dish_data.get("meal_type", "main"),
                            dish_data.get("seasonal_availability", "year-round"),
                            dish_data.get("preservation_methods", []),
                            json.dumps({
                                nation1['name']: "Original version",
                                nation2['name']: culinary_details
                            }),
                            await generate_embedding(dish_name + " cuisine food")
                        )
                        
                        # Create recipe variations
                        if culinary_details.get("occasions"):
                            for occasion in culinary_details["occasions"][:3]:
                                await conn.execute("""
                                    INSERT INTO RecipeVariations (
                                        tradition_id, variation_name, occasion,
                                        modifications, special_ingredients
                                    )
                                    VALUES ($1, $2, $3, $4, $5)
                                    ON CONFLICT (tradition_id, variation_name) DO NOTHING
                                """,
                                    ct_id,
                                    f"{dish_name} ({occasion})",
                                    occasion,
                                    f"Prepared specially for {occasion}",
                                    []
                                )
                
                # Process food customs and etiquette
                if "dining_customs" in effects:
                    for custom in effects["dining_customs"]:
                        custom_data = custom if isinstance(custom, dict) else {"description": custom}
                        
                        await conn.execute("""
                            INSERT INTO CulturalElements (
                                name, element_type, description, practiced_by,
                                significance, matriarchal_elements,
                                associated_practices, gender_associations
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (name, element_type) DO UPDATE
                            SET practiced_by = array_append(
                                COALESCE(CulturalElements.practiced_by, ARRAY[]::text[]), $9
                            )
                        """,
                            custom_data.get("name", "Dining custom"),
                            "dining_etiquette",
                            custom_data.get("description", "Adopted dining practice"),
                            [nation2['name']],
                            custom_data.get("significance", 5),
                            "Women eat first or are served first",
                            custom_data.get("associated_practices", []),
                            {"primary": "all", "specifics": custom_data.get("gender_rules", {})},
                            nation2['name']
                        )
                
                # Record the exchange
                exchange_id = await conn.fetchval("""
                    INSERT INTO CulturalExchanges (
                        nation1_id, nation2_id, exchange_type, exchange_details,
                        impact_level, cultural_resistance, health_impacts,
                        economic_impacts, timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """,
                    nation1_id, nation2_id, "cuisine_diffusion",
                    json.dumps(effects),
                    self._calculate_impact_level(effects),
                    effects.get("resistance_level", 15),  # Some dietary conservatism
                    effects.get("health_impacts", "Dietary diversity increased"),
                    effects.get("economic_impacts", "New trade in ingredients"),
                    datetime.now()
                )
                
                # Create trade routes for ingredients if needed
                if effects.get("creates_trade", True):
                    await conn.execute("""
                        INSERT INTO TradeRoutes (
                            nation1_id, nation2_id, trade_goods,
                            route_type, establishment_reason
                        )
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (nation1_id, nation2_id, route_type) 
                        DO UPDATE SET trade_goods = array_cat(
                            TradeRoutes.trade_goods, 
                            EXCLUDED.trade_goods
                        )
                    """,
                        nation1_id, nation2_id,
                        ["spices", "ingredients", "preserved foods"],
                        "culinary",
                        "Cultural cuisine exchange"
                    )
                
                # Log canonical event
                from lore.core import canon
                await canon.log_canonical_event(
                    self.create_run_context({}), conn,
                    f"Culinary traditions from {nation1['name']} enriched the cuisine of {nation2['name']}",
                    tags=['cultural_exchange', 'cuisine', 'diffusion', 'trade'],
                    significance=4
                )
    
    async def _apply_customs_diffusion(self, nation1_id: int, nation2_id: int, effects: Dict[str, Any]) -> None:
        """Apply social customs and etiquette diffusion effects between nations."""
        if not effects:
            return
            
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get detailed nation data including existing customs
                nation1 = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT sc.name) FILTER (WHERE sc.id IS NOT NULL) as social_customs,
                           array_agg(DISTINCT e.context) FILTER (WHERE e.id IS NOT NULL) as etiquette_contexts
                    FROM Nations n
                    LEFT JOIN SocialCustoms sc ON sc.nation_origin = n.id
                    LEFT JOIN Etiquette e ON e.nation_id = n.id
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation1_id)
                
                nation2 = await conn.fetchrow("""
                    SELECT n.*, 
                           array_agg(DISTINCT sc.name) FILTER (WHERE sc.id IS NOT NULL) as social_customs,
                           array_agg(DISTINCT e.context) FILTER (WHERE e.id IS NOT NULL) as etiquette_contexts
                    FROM Nations n
                    LEFT JOIN SocialCustoms sc ON sc.nation_origin = n.id
                    LEFT JOIN Etiquette e ON e.nation_id = n.id
                    WHERE n.id = $1
                    GROUP BY n.id
                """, nation2_id)
                
                if not nation1 or not nation2:
                    return
                
                # Create customs adaptation agent
                customs_agent = Agent(
                    name="CustomsAdaptationAgent",
                    instructions="""You adapt social customs and etiquette between cultures.
                    Consider power dynamics, gender roles, formality levels, and how customs
                    reinforce matriarchal social structures.""",
                    model="gpt-4.1-nano"
                )
                
                # Process social customs
                social_customs = effects.get("social_customs", effects.get("customs", []))
                
                for custom in social_customs:
                    custom_data = custom if isinstance(custom, dict) else {"name": custom}
                    
                    # Check for similar existing customs
                    existing = await conn.fetchrow("""
                        SELECT * FROM SocialCustoms
                        WHERE similarity(name, $1) > 0.7
                           OR similarity(description, $2) > 0.6
                    """, custom_data.get("name", ""), custom_data.get("description", ""))
                    
                    if existing:
                        # Update adoption list
                        adopted_by = existing['adopted_by'] or []
                        if nation2_id not in adopted_by:
                            adopted_by.append(nation2_id)
                            
                            # Get adaptation details
                            prompt = f"""
                            The custom "{existing['name']}" from {nation1['name']} is being adopted by {nation2['name']}.
                            
                            Original custom: {existing['description']}
                            Context: {existing.get('context', 'social')}
                            Formality: {existing.get('formality_level', 'medium')}
                            
                            Nation 1: Matriarchy level {nation1['matriarchy_level']}/10
                            Nation 2: Matriarchy level {nation2['matriarchy_level']}/10
                            
                            How would this custom be adapted? Consider:
                            - Power structure differences
                            - Local cultural sensitivities
                            - Gender role variations
                            - Formality adjustments
                            - Enforcement mechanisms
                            
                            Return JSON with:
                            - local_name: what it's called locally
                            - modifications: list of key changes
                            - gender_variations: how it differs by gender
                            - enforcement: who ensures compliance
                            - exceptions: when it doesn't apply
                            - integration_method: how it merges with existing customs
                            """
                            
                            result = await Runner.run(customs_agent, prompt)
                            try:
                                adaptation = json.loads(result.final_output)
                            except:
                                adaptation = {
                                    "local_name": existing['name'],
                                    "modifications": ["Adapted to local norms"],
                                    "gender_variations": {"female": "Leading role", "male": "Supporting role"},
                                    "enforcement": "Social pressure",
                                    "exceptions": ["Private settings"],
                                    "integration_method": "Gradual adoption"
                                }
                            
                            await conn.execute("""
                                UPDATE SocialCustoms
                                SET adopted_by = $1,
                                    regional_variations = COALESCE(regional_variations, '{}'::jsonb) || $2::jsonb,
                                    adoption_date = array_append(
                                        COALESCE(adoption_date, ARRAY[]::timestamp[]), 
                                        $3
                                    )
                                WHERE id = $4
                            """,
                                adopted_by,
                                json.dumps({nation2['name']: adaptation}),
                                datetime.now(),
                                existing['id']
                            )
                    else:
                        # Create new social custom
                        custom_name = custom_data.get("name", f"Custom {len(social_customs)}")
                        
                        # Generate detailed custom information
                        prompt = f"""
                        Create a social custom spreading from {nation1['name']} to {nation2['name']}.
                        
                        Custom: {custom_name}
                        Origin: {nation1['name']} (Matriarchy level {nation1['matriarchy_level']}/10)
                        Adopting: {nation2['name']} (Matriarchy level {nation2['matriarchy_level']}/10)
                        
                        Design a custom that:
                        1. Reinforces matriarchal power structures
                        2. Has clear gender-differentiated behaviors
                        3. Shows respect and hierarchy
                        4. Has social consequences for violation
                        5. Can be adapted between cultures
                        
                        Return JSON with:
                        - description: detailed description
                        - context: where/when it applies (public, private, formal, etc)
                        - formality_level: low/medium/high/sacred
                        - gender_specific_rules: dict of behaviors by gender
                        - age_variations: how it changes with age
                        - class_variations: elite vs common practice
                        - violation_consequences: what happens if broken
                        - teaching_method: how it's passed down
                        - symbolic_meaning: deeper significance
                        - related_customs: other customs it connects to
                        """
                        
                        result = await Runner.run(customs_agent, prompt)
                        try:
                            custom_details = json.loads(result.final_output)
                        except:
                            custom_details = {
                                "description": custom_data.get("description", "Social custom"),
                                "context": "public",
                                "formality_level": "medium",
                                "gender_specific_rules": {"female": "Initiate", "male": "Respond"},
                                "age_variations": {"youth": "Learn", "adult": "Practice", "elder": "Enforce"},
                                "class_variations": {"elite": "Elaborate", "common": "Simple"},
                                "violation_consequences": "Social disapproval",
                                "teaching_method": "Observation and practice",
                                "symbolic_meaning": "Respect for hierarchy",
                                "related_customs": []
                            }
                        
                        sc_id = await conn.fetchval("""
                            INSERT INTO SocialCustoms (
                                name, nation_origin, description, context,
                                formality_level, gender_rules, age_variations,
                                class_variations, violation_consequences,
                                adopted_by, adoption_date, teaching_method,
                                symbolic_meaning, related_customs,
                                required_participants, prohibited_participants,
                                seasonal_practice, regional_variations,
                                embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                            RETURNING id
                        """,
                            custom_name,
                            nation1_id,
                            custom_details.get("description"),
                            custom_details.get("context", "social"),
                            custom_details.get("formality_level", "medium"),
                            json.dumps(custom_details.get("gender_specific_rules", {})),
                            json.dumps(custom_details.get("age_variations", {})),
                            json.dumps(custom_details.get("class_variations", {})),
                            custom_details.get("violation_consequences"),
                            [nation2_id],
                            [datetime.now()],
                            custom_details.get("teaching_method"),
                            custom_details.get("symbolic_meaning"),
                            custom_details.get("related_customs", []),
                            custom_data.get("required_participants", []),
                            custom_data.get("prohibited_participants", []),
                            custom_data.get("seasonal_practice", False),
                            json.dumps({
                                nation1['name']: "Original form",
                                nation2['name']: custom_details
                            }),
                            await generate_embedding(custom_name + " social custom etiquette")
                        )
                        
                        # Create specific etiquette rules derived from this custom
                        if custom_details.get("context"):
                            await self._create_derived_etiquette(
                                conn, nation2_id, sc_id, custom_name, custom_details
                            )
                
                # Process greeting customs specifically
                if "greeting_customs" in effects:
                    for greeting in effects["greeting_customs"]:
                        greeting_data = greeting if isinstance(greeting, dict) else {"description": greeting}
                        
                        # Update or create greeting etiquette
                        existing_greeting = await conn.fetchrow("""
                            SELECT id FROM Etiquette
                            WHERE nation_id = $1 AND context = 'greeting'
                        """, nation2_id)
                        
                        if existing_greeting:
                            # Merge with existing
                            await conn.execute("""
                                UPDATE Etiquette
                                SET greeting_ritual = greeting_ritual || E'\n\n' || $1,
                                    gender_distinctions = gender_distinctions || E'\n\n' || $2
                                WHERE id = $3
                            """,
                                f"Influenced by {nation1['name']}: {greeting_data.get('description', '')}",
                                greeting_data.get('gender_distinctions', ''),
                                existing_greeting['id']
                            )
                        else:
                            # Create new greeting etiquette
                            await conn.execute("""
                                INSERT INTO Etiquette (
                                    nation_id, context, greeting_ritual,
                                    body_language, eye_contact, distance_norms,
                                    gender_distinctions, power_display
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                                nation2_id,
                                'greeting',
                                greeting_data.get('description', 'Adopted greeting custom'),
                                greeting_data.get('body_language', 'Respectful posture'),
                                greeting_data.get('eye_contact', 'Based on status'),
                                greeting_data.get('distance_norms', 'Arm\'s length'),
                                greeting_data.get('gender_distinctions', 'Women greeted first'),
                                greeting_data.get('power_display', 'Lower status initiates')
                            )
                
                # Process gift-giving customs
                if "gift_customs" in effects:
                    for gift_custom in effects["gift_customs"]:
                        gift_data = gift_custom if isinstance(gift_custom, dict) else {"description": gift_custom}
                        
                        await conn.execute("""
                            INSERT INTO CulturalElements (
                                name, element_type, description, practiced_by,
                                significance, matriarchal_elements,
                                associated_practices, required_materials,
                                gender_associations, occasions
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT (name, element_type) DO UPDATE
                            SET practiced_by = array_append(
                                COALESCE(CulturalElements.practiced_by, ARRAY[]::text[]), $11
                            ),
                            description = CulturalElements.description || E'\n\n' || $12
                        """,
                            gift_data.get("name", "Gift-giving custom"),
                            "gift_custom",
                            gift_data.get("description", "Adopted gift custom"),
                            [nation2['name']],
                            gift_data.get("significance", 6),
                            "Women receive gifts first, men present through female relatives",
                            gift_data.get("associated_practices", []),
                            gift_data.get("required_materials", []),
                            gift_data.get("gender_associations", {"giver": "any", "receiver": "female_first"}),
                            gift_data.get("occasions", ["festivals", "ceremonies"]),
                            nation2['name'],
                            f"Adapted from {nation1['name']}"
                        )
                
                # Record the cultural exchange
                exchange_id = await conn.fetchval("""
                    INSERT INTO CulturalExchanges (
                        nation1_id, nation2_id, exchange_type, exchange_details,
                        impact_level, cultural_resistance, integration_period,
                        social_impacts, political_impacts, timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """,
                    nation1_id, nation2_id, "customs_diffusion",
                    json.dumps(effects),
                    self._calculate_impact_level(effects),
                    effects.get("resistance_level", 25),  # Higher resistance to social customs
                    effects.get("integration_period", "1-2 generations"),
                    effects.get("social_impacts", "Increased cultural complexity"),
                    effects.get("political_impacts", "Enhanced diplomatic relations"),
                    datetime.now()
                )
                
                # Create custom adoption tracking
                for custom_type in ["social_customs", "greeting_customs", "gift_customs"]:
                    if custom_type in effects:
                        await conn.execute("""
                            INSERT INTO CustomAdoptionTracking (
                                exchange_id, custom_type, adoption_rate,
                                demographic_breakdown, resistance_factors,
                                success_factors, timeline
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                            exchange_id,
                            custom_type,
                            effects.get(f"{custom_type}_adoption_rate", 60),
                            json.dumps(effects.get(f"{custom_type}_demographics", {
                                "elite": 80,
                                "merchant": 70,
                                "common": 40
                            })),
                            effects.get(f"{custom_type}_resistance", ["traditionalists", "rural areas"]),
                            effects.get(f"{custom_type}_success", ["youth adoption", "practical benefits"]),
                            effects.get(f"{custom_type}_timeline", "gradual")
                        )
                
                # Log canonical event
                from lore.core import canon
                await canon.log_canonical_event(
                    self.create_run_context({}), conn,
                    f"Social customs from {nation1['name']} began influencing the culture of {nation2['name']}",
                    tags=['cultural_exchange', 'customs', 'diffusion', 'society'],
                    significance=6
                )
    
    async def _create_derived_etiquette(self, conn, nation_id: int, custom_id: int, 
                                       custom_name: str, custom_details: Dict[str, Any]) -> None:
        """Create specific etiquette rules derived from a social custom."""
        context = custom_details.get("context", "social")
        
        # Check if etiquette for this context already exists
        existing = await conn.fetchrow("""
            SELECT id FROM Etiquette
            WHERE nation_id = $1 AND context = $2
        """, nation_id, context)
        
        if not existing:
            await conn.execute("""
                INSERT INTO Etiquette (
                    nation_id, context, title_system, greeting_ritual,
                    body_language, eye_contact, distance_norms,
                    power_display, respect_indicators,
                    gender_distinctions, taboos, derived_from_custom
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                nation_id,
                context,
                f"Titles reflect {custom_name} hierarchy",
                custom_details.get("greeting_component", "Standard greeting with custom elements"),
                custom_details.get("body_language", "Posture shows deference"),
                custom_details.get("eye_contact", "Status-dependent"),
                custom_details.get("distance", "Respectful distance maintained"),
                "Female authority prominently displayed",
                custom_details.get("respect_indicators", "Multiple verbal and physical cues"),
                json.dumps(custom_details.get("gender_specific_rules", {})),
                custom_details.get("related_taboos", []),
                custom_id
            )
    
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
            async with await self.get_connection_pool() as pool:
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
            async with await self.get_connection_pool() as pool:
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
        async with await self.get_connection_pool() as pool:
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
        async with await self.get_connection_pool() as pool:
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
        async with await self.get_connection_pool() as pool:
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
                    model="gpt-4.1-nano"
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
                model="gpt-4.1-nano"
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
