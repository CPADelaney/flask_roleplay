# lore/managers/religion.py

import logging
import json
import random
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum

# Agents SDK imports
from agents import (
    Agent, 
    function_tool, 
    Runner, 
    ModelSettings,
    RunConfig,
    handoff,
    InputGuardrail,
    GuardrailFunctionOutput
)
from agents.run_context import RunContextWrapper

# Pydantic for structured data
from pydantic import BaseModel, Field, validator

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils
from lore.core.cache import GLOBAL_LORE_CACHE

logger = logging.getLogger(__name__)

# ===========================
# Pydantic Models for Type Safety
# ===========================

class DeityRelationship(BaseModel):
    """Structured deity relationships to avoid additionalProperties issues."""
    deity_name: str
    relationship_type: str
    description: Optional[str] = None

class DeityParams(BaseModel):
    """Parameters for creating a deity."""
    name: str = Field(..., description="Name of the deity")
    gender: str = Field(..., description="Gender of the deity")
    domain: List[str] = Field(..., description="Domains the deity presides over")
    description: str = Field(..., description="Description of the deity")
    pantheon_id: Optional[int] = Field(None, description="ID of the pantheon")
    iconography: Optional[str] = Field(None, description="Visual representation")
    holy_symbol: Optional[str] = Field(None, description="Sacred symbol")
    sacred_animals: List[str] = Field(default_factory=list, description="Sacred animals")
    sacred_colors: List[str] = Field(default_factory=list, description="Sacred colors")
    relationships: List[DeityRelationship] = Field(default_factory=list, description="Relationships with other deities")
    rank: int = Field(5, ge=1, le=10, description="Divine rank (1-10)")
    worshippers: List[str] = Field(default_factory=list, description="Groups that worship this deity")

class PantheonParams(BaseModel):
    """Parameters for creating a pantheon."""
    name: str
    description: str
    origin_story: str
    matriarchal_elements: str
    creation_myth: Optional[str] = None
    afterlife_beliefs: Optional[str] = None
    cosmic_structure: Optional[str] = None
    major_holy_days: List[str] = Field(default_factory=list)
    geographical_spread: List[str] = Field(default_factory=list)
    dominant_nations: List[str] = Field(default_factory=list)
    primary_worshippers: List[str] = Field(default_factory=list)
    taboos: List[str] = Field(default_factory=list)

class ReligiousPracticeParams(BaseModel):
    """Parameters for religious practices."""
    name: str
    practice_type: str
    description: str
    purpose: str
    frequency: Optional[str] = None
    required_elements: List[str] = Field(default_factory=list)
    performed_by: List[str] = Field(default_factory=list)
    restricted_to: List[str] = Field(default_factory=list)
    deity_id: Optional[int] = None
    pantheon_id: Optional[int] = None

class HolySiteParams(BaseModel):
    """Parameters for holy sites."""
    name: str
    site_type: str
    description: str
    clergy_type: str
    location_id: Optional[int] = None
    location_description: Optional[str] = None
    deity_id: Optional[int] = None
    pantheon_id: Optional[int] = None
    clergy_hierarchy: List[str] = Field(default_factory=list)
    pilgrimage_info: Optional[str] = None
    miracles_reported: List[str] = Field(default_factory=list)
    restrictions: List[str] = Field(default_factory=list)
    architectural_features: Optional[str] = None

class ReligiousTextParams(BaseModel):
    """Parameters for religious texts."""
    name: str
    text_type: str
    description: str
    key_teachings: List[str]
    authorship: Optional[str] = None
    restricted_to: List[str] = Field(default_factory=list)
    deity_id: Optional[int] = None
    pantheon_id: Optional[int] = None
    notable_passages: List[str] = Field(default_factory=list)
    age_description: Optional[str] = None

class ReligiousOrderParams(BaseModel):
    """Parameters for religious orders."""
    name: str
    order_type: str
    description: str
    gender_composition: str
    founding_story: Optional[str] = None
    headquarters: Optional[str] = None
    hierarchy_structure: List[str] = Field(default_factory=list)
    vows: List[str] = Field(default_factory=list)
    practices: List[str] = Field(default_factory=list)
    deity_id: Optional[int] = None
    pantheon_id: Optional[int] = None
    special_abilities: List[str] = Field(default_factory=list)
    notable_members: List[str] = Field(default_factory=list)

class ReligiousConflictParams(BaseModel):
    """Parameters for religious conflicts."""
    name: str
    conflict_type: str
    description: str
    parties_involved: List[str]
    core_disagreement: str
    beginning_date: Optional[str] = None
    resolution_date: Optional[str] = None
    status: str = "ongoing"
    casualties: Optional[str] = None
    historical_impact: Optional[str] = None

# Structured outputs for LLM responses
class MatriarchalThemeValidation(BaseModel):
    """Output schema for matriarchal theme validation."""
    is_matriarchal: bool
    reasoning: str
    suggestions: List[str] = Field(default_factory=list)

class GeneratedPantheon(BaseModel):
    """Structured output for pantheon generation."""
    pantheon: PantheonParams
    deities: List[DeityParams]

class TheologicalPosition(BaseModel):
    """A theological position in a dispute."""
    name: str
    core_belief: str
    scriptural_basis: str
    key_arguments: List[str]

class DisputeConclusion(BaseModel):
    """Conclusion of a theological dispute."""
    conclusion: str
    implications: str
    winning_position: Optional[str] = None
    synthesis_achieved: bool = False

class RitualComponent(BaseModel):
    """Component of a religious ritual."""
    name: str
    description: str
    purpose: str
    participants: List[str]
    required_items: List[str]
    symbolic_meaning: str

class CompleteRitual(BaseModel):
    """Complete religious ritual."""
    name: str
    purpose: str
    occasion: str
    duration: str
    preparation: str
    components: List[RitualComponent]
    variations: Dict[str, str]
    restrictions: List[str]
    theological_significance: str

class NationReligionDistribution(BaseModel):
    """Religious distribution for a nation."""
    nation_id: int
    state_religion: bool = False
    primary_pantheon_id: Optional[int] = None
    pantheon_distribution: Dict[int, float] = Field(default_factory=dict)
    religiosity_level: int = Field(5, ge=1, le=10)
    religious_tolerance: int = Field(5, ge=1, le=10)
    religious_leadership: str
    religious_laws: Dict[str, Any] = Field(default_factory=dict)
    religious_holidays: List[str] = Field(default_factory=list)
    religious_conflicts: List[str] = Field(default_factory=list)
    religious_minorities: List[str] = Field(default_factory=list)

class RegionalPracticeVariation(BaseModel):
    """Regional variation of a religious practice."""
    practice_id: int
    regional_variation: str
    importance: int = Field(5, ge=1, le=10)
    frequency: str
    local_additions: str
    gender_differences: str

class ReligiousEvolution(BaseModel):
    """Religious evolution data."""
    new_practices: List[Dict[str, Any]] = Field(default_factory=list)
    modified_rituals: List[Dict[str, Any]] = Field(default_factory=list)
    deity_reinterpretations: List[Dict[str, Any]] = Field(default_factory=list)
    syncretic_elements: List[Dict[str, Any]] = Field(default_factory=list)
    new_roles: List[Dict[str, Any]] = Field(default_factory=list)
    religious_changes: Dict[str, Any] = Field(default_factory=dict)

class SectarianPosition(BaseModel):
    """A sectarian position."""
    sect_name: str
    interpretation: str
    doctrinal_position: str
    prescribed_practices: List[str]
    view_of_other_sects: str
    gender_composition: str = "mixed"

# ===========================
# Main Religion Manager Class
# ===========================

class ReligionManager(BaseLoreManager):
    """
    Comprehensive system for managing religions, faiths, and belief systems
    within the matriarchal society, using OpenAI Agents SDK best practices.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.initialized = False
        
        # Initialize all agents upfront
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents used by the religion system."""
        
        # Theme validation agent
        self.theme_validator = Agent(
            name="MatriarchalThemeValidator",
            instructions="""You verify that all religious content maintains matriarchal themes. 
            You identify elements that might contradict a female-dominant religious structure.
            Focus on power dynamics, gender roles, and divine hierarchy.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            output_type=MatriarchalThemeValidation
        )
        
        # Pantheon generation agent
        self.pantheon_generator = Agent(
            name="PantheonGenerator",
            instructions="""You create religious pantheons for matriarchal fantasy worlds.
            Focus on feminine divine power structures with goddesses in positions of supreme authority.
            Male deities should exist but in supporting or specialized roles.
            Ensure the cosmic structure reinforces matriarchal principles.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=GeneratedPantheon,
            input_guardrails=[
                InputGuardrail(guardrail_function=self._matriarchal_theme_guardrail)
            ]
        )
        
        # Religious practice generator
        self.practice_generator = Agent(
            name="PracticeGenerator",
            instructions="""You create religious practices that reinforce matriarchal dominance.
            Design rituals, ceremonies, and observances that emphasize feminine divine authority.
            Ensure practices reflect the power dynamics of the society.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=List[ReligiousPracticeParams]
        )
        
        # Holy site generator
        self.site_generator = Agent(
            name="HolySiteGenerator",
            instructions="""You design holy sites and temples for matriarchal religions.
            Focus on architecture and features that emphasize feminine divine power.
            Consider how the sites reinforce social hierarchies.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=List[HolySiteParams]
        )
        
        # Religious text generator
        self.text_generator = Agent(
            name="TextGenerator",
            instructions="""You create religious texts and scriptures for matriarchal faiths.
            Ensure teachings emphasize feminine divine supremacy and proper gender roles.
            Create varied text types including scripture, commentary, and hymns.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=List[ReligiousTextParams]
        )
        
        # Religious order generator
        self.order_generator = Agent(
            name="OrderGenerator",
            instructions="""You establish religious orders for matriarchal faiths.
            Emphasize female leadership and gender-appropriate hierarchies.
            Design distinct orders with unique purposes and practices.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=List[ReligiousOrderParams]
        )
        
        # Conflict generator
        self.conflict_generator = Agent(
            name="ConflictGenerator",
            instructions="""You create religious conflicts and schisms.
            Focus on theological disputes that don't threaten the matriarchal order.
            Design conflicts that add depth while maintaining core power structures.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=List[ReligiousConflictParams]
        )
        
        # Religious distribution agent
        self.distribution_agent = Agent(
            name="DistributionAgent",
            instructions="""You distribute religions across nations considering their matriarchy levels.
            Create realistic religious demographics and state religion policies.
            Ensure distribution reflects cultural and political realities.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            output_type=NationReligionDistribution
        )
        
        # Theological dispute agents
        self.theological_arbiter = Agent(
            name="TheologicalArbiter",
            instructions="""You are a senior religious authority who evaluates theological arguments.
            You maintain orthodoxy while allowing for legitimate theological diversity.
            Your judgments uphold the matriarchal religious order.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            output_type=DisputeConclusion
        )
        
        # Evolution agent
        self.evolution_agent = Agent(
            name="ReligiousEvolutionAgent",
            instructions="""You simulate how religions evolve through cultural interaction over time.
            Maintain core matriarchal themes while allowing for organic development.
            Consider how local customs influence religious practice.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=ReligiousEvolution
        )
        
        # Ritual generator
        self.ritual_generator = Agent(
            name="RitualGenerator",
            instructions="""You create detailed religious rituals for matriarchal faiths.
            Focus on symbolism, required components, and theological significance.
            Design rituals that reinforce feminine divine authority.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=CompleteRitual
        )
        
        # Regional practice agent
        self.regional_practice_agent = Agent(
            name="RegionalPracticeAgent",
            instructions="""You adapt religious practices to regional contexts.
            Consider how local culture influences religious expression.
            Maintain core theological principles while allowing variation.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=RegionalPracticeVariation
        )
        
        # Sectarian development agents
        self.sect_generator = Agent(
            name="SectGenerator",
            instructions="""You create religious sects and schisms.
            Design theological divisions that don't threaten matriarchal order.
            Focus on interpretive differences rather than fundamental challenges.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Setup handoffs between agents
        self._setup_agent_handoffs()

    def _setup_agent_handoffs(self):
        """Configure handoffs between specialized agents."""
        # Pantheon generator can hand off to other generators
        self.pantheon_generator.handoffs = [
            self.practice_generator,
            self.site_generator,
            self.text_generator,
            self.order_generator
        ]
        
        # Order generator can trigger conflict generator
        self.order_generator.handoffs = [self.conflict_generator]
        
        # Distribution agent can hand off to regional practice agent
        self.distribution_agent.handoffs = [self.regional_practice_agent]

    async def ensure_initialized(self):
        """Ensure system is initialized."""
        if not self.initialized:
            await super().ensure_initialized()
            await self.initialize_tables()
            self.initialized = True

    async def initialize_tables(self):
        """Ensure all religion-related tables exist."""
        table_definitions = {
            "Deities": """
                CREATE TABLE IF NOT EXISTS Deities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    domain TEXT[] NOT NULL,
                    description TEXT NOT NULL,
                    iconography TEXT,
                    holy_symbol TEXT,
                    sacred_animals TEXT[],
                    sacred_colors TEXT[],
                    relationships JSONB,
                    rank INTEGER CHECK (rank BETWEEN 1 AND 10),
                    worshippers TEXT[],
                    pantheon_id INTEGER,
                    embedding VECTOR(1536)
                );
                CREATE INDEX IF NOT EXISTS idx_deities_embedding
                ON Deities USING ivfflat (embedding vector_cosine_ops);
            """,
            "Pantheons": """
                CREATE TABLE IF NOT EXISTS Pantheons (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    origin_story TEXT NOT NULL,
                    major_holy_days TEXT[],
                    cosmic_structure TEXT,
                    afterlife_beliefs TEXT,
                    creation_myth TEXT,
                    geographical_spread TEXT[],
                    dominant_nations TEXT[],
                    primary_worshippers TEXT[],
                    matriarchal_elements TEXT NOT NULL,
                    taboos TEXT[],
                    embedding VECTOR(1536)
                );
                CREATE INDEX IF NOT EXISTS idx_pantheons_embedding
                ON Pantheons USING ivfflat (embedding vector_cosine_ops);
            """,
            "ReligiousPractices": """
                CREATE TABLE IF NOT EXISTS ReligiousPractices (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    practice_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    frequency TEXT,
                    required_elements TEXT[],
                    performed_by TEXT[],
                    purpose TEXT NOT NULL,
                    restricted_to TEXT[],
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    embedding VECTOR(1536)
                );
                CREATE INDEX IF NOT EXISTS idx_religiouspractices_embedding
                ON ReligiousPractices USING ivfflat (embedding vector_cosine_ops);
            """,
            "HolySites": """
                CREATE TABLE IF NOT EXISTS HolySites (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    site_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    location_id INTEGER,
                    location_description TEXT,
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    clergy_type TEXT,
                    clergy_hierarchy TEXT[],
                    pilgrimage_info TEXT,
                    miracles_reported TEXT[],
                    restrictions TEXT[],
                    architectural_features TEXT,
                    embedding VECTOR(1536)
                );
                CREATE INDEX IF NOT EXISTS idx_holysites_embedding
                ON HolySites USING ivfflat (embedding vector_cosine_ops);
            """,
            "ReligiousTexts": """
                CREATE TABLE IF NOT EXISTS ReligiousTexts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    text_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    authorship TEXT,
                    key_teachings TEXT[] NOT NULL,
                    restricted_to TEXT[],
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    notable_passages TEXT[],
                    age_description TEXT,
                    embedding VECTOR(1536)
                );
                CREATE INDEX IF NOT EXISTS idx_religioustexts_embedding
                ON ReligiousTexts USING ivfflat (embedding vector_cosine_ops);
            """,
            "ReligiousOrders": """
                CREATE TABLE IF NOT EXISTS ReligiousOrders (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    founding_story TEXT,
                    headquarters TEXT,
                    hierarchy_structure TEXT[],
                    vows TEXT[],
                    practices TEXT[],
                    deity_id INTEGER,
                    pantheon_id INTEGER,
                    gender_composition TEXT,
                    special_abilities TEXT[],
                    notable_members TEXT[],
                    embedding VECTOR(1536)
                );
                CREATE INDEX IF NOT EXISTS idx_religiousorders_embedding
                ON ReligiousOrders USING ivfflat (embedding vector_cosine_ops);
            """,
            "ReligiousConflicts": """
                CREATE TABLE IF NOT EXISTS ReligiousConflicts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    beginning_date TEXT,
                    resolution_date TEXT,
                    status TEXT,
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
                CREATE TABLE IF NOT EXISTS NationReligion (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    state_religion BOOLEAN DEFAULT FALSE,
                    primary_pantheon_id INTEGER,
                    pantheon_distribution JSONB,
                    religiosity_level INTEGER CHECK (religiosity_level BETWEEN 1 AND 10),
                    religious_tolerance INTEGER CHECK (religious_tolerance BETWEEN 1 AND 10),
                    religious_leadership TEXT,
                    religious_laws JSONB,
                    religious_holidays TEXT[],
                    religious_conflicts TEXT[],
                    religious_minorities TEXT[],
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (primary_pantheon_id) REFERENCES Pantheons(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nationreligion_embedding
                ON NationReligion USING ivfflat (embedding vector_cosine_ops);
            """,
            "RegionalReligiousPractice": """
                CREATE TABLE IF NOT EXISTS RegionalReligiousPractice (
                    id SERIAL PRIMARY KEY,
                    nation_id INTEGER NOT NULL,
                    practice_id INTEGER NOT NULL,
                    regional_variation TEXT,
                    importance INTEGER CHECK (importance BETWEEN 1 AND 10),
                    frequency TEXT,
                    local_additions TEXT,
                    gender_differences TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                    FOREIGN KEY (practice_id) REFERENCES ReligiousPractices(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_regionalreligiouspractice_embedding
                ON RegionalReligiousPractice USING ivfflat (embedding vector_cosine_ops);
            """
        }

        await self.initialize_tables_for_class(table_definitions)

    # Guardrail function
    async def _matriarchal_theme_guardrail(self, ctx: RunContextWrapper, agent, input_data: str) -> GuardrailFunctionOutput:
        """Guardrail to ensure content maintains matriarchal themes."""
        result = await Runner.run(self.theme_validator, input_data, context=ctx.context)
        validation_result = result.final_output_as(MatriarchalThemeValidation)
        
        return GuardrailFunctionOutput(
            output_info=validation_result.dict(),
            tripwire_triggered=not validation_result.is_matriarchal
        )

    # ===========================
    # Core CRUD Operations with Structured Types
    # ===========================

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_deity",
        action_description="Adding deity to the pantheon",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def add_deity(self, ctx, params: DeityParams) -> int:
        """Prepares deity data and uses the Canon to create the entity."""
        await self.ensure_initialized()
        
        # This manager's job is to PREPARE the data.
        embed_text = f"{params.name} {params.gender} {' '.join(params.domain)} {params.description}"
        embedding = await generate_embedding(embed_text)

        relationships_json = json.dumps({
            rel.deity_name: {"type": rel.relationship_type, "description": rel.description}
            for rel in params.relationships
        })

        # Assemble all data into a dictionary for the Canon.
        deity_data_package = {
            "name": params.name,
            "gender": params.gender,
            "domain": params.domain,
            "description": params.description,
            "pantheon_id": params.pantheon_id,
            "iconography": params.iconography,
            "holy_symbol": params.holy_symbol,
            "sacred_animals": params.sacred_animals,
            "sacred_colors": params.sacred_colors,
            "relationships": relationships_json,
            "rank": params.rank,
            "worshippers": params.worshippers,
            "embedding": embedding
        }
        
        # The LoreSystem will manage the transaction, so we just need the connection.
        async with self.get_connection_pool() as conn:
            # Call the SINGLE, CENTRALIZED function from the canon.
            deity_id = await canon.find_or_create_deity(ctx, conn, **deity_data_package)
        
        self.invalidate_cache_pattern("deity")
        return deity_id
            
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_pantheon",
        action_description="Adding pantheon to the world",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def add_pantheon(self, ctx, params: PantheonParams) -> int:
        """Prepares pantheon data and uses the Canon to create it."""
        await self.ensure_initialized()
        
        pantheon_data_package = params.dict()
        pantheon_data_package['embedding'] = await generate_embedding(
            f"{params.name} {params.description}"
        )

        # Call the Canon...
        async with self.get_connection_pool() as conn:
            pantheon_id = await canon.find_or_create_pantheon(ctx, conn, **pantheon_data_package)

        self.invalidate_cache_pattern("pantheon")
        return pantheon_id


    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_practice",
        action_description="Adding religious practice",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def add_religious_practice(self, ctx, params: ReligiousPracticeParams) -> int:
        """Add a religious practice using canon establishment."""
        await self.ensure_initialized()
        
        embed_text = f"{params.name} {params.practice_type} {params.description} {params.purpose}"
        
        # Prepare data package for canon
        practice_data_package = {
            "name": params.name,
            "practice_type": params.practice_type,
            "description": params.description,
            "purpose": params.purpose,
            "frequency": params.frequency,
            "required_elements": params.required_elements,
            "performed_by": params.performed_by,
            "restricted_to": params.restricted_to,
            "deity_id": params.deity_id,
            "pantheon_id": params.pantheon_id
        }
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                from lore.core import canon
                practice_id = await canon.find_or_create_religious_practice(
                    ctx, conn, **practice_data_package, embedding_text=embed_text
                )
        
        GLOBAL_LORE_CACHE.invalidate_pattern("practice")
        return practice_id

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_holy_site",
        action_description="Adding holy site",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def add_holy_site(self, ctx, params: HolySiteParams) -> int:
        """Add a holy site using canon establishment."""
        await self.ensure_initialized()
        
        embed_text = f"{params.name} {params.site_type} {params.description} {params.clergy_type}"
        
        # Prepare data package for canon
        site_data_package = {
            "name": params.name,
            "site_type": params.site_type,
            "description": params.description,
            "clergy_type": params.clergy_type,
            "location_id": params.location_id,
            "location_description": params.location_description,
            "deity_id": params.deity_id,
            "pantheon_id": params.pantheon_id,
            "clergy_hierarchy": params.clergy_hierarchy,
            "pilgrimage_info": params.pilgrimage_info,
            "miracles_reported": params.miracles_reported,
            "restrictions": params.restrictions,
            "architectural_features": params.architectural_features
        }
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                from lore.core import canon
                site_id = await canon.find_or_create_holy_site(
                    ctx, conn, **site_data_package, embedding_text=embed_text
                )
        
        GLOBAL_LORE_CACHE.invalidate_pattern("site")
        return site_id

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_text",
        action_description="Adding religious text",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def add_religious_text(self, ctx, params: ReligiousTextParams) -> int:
        """Add a religious text using canon establishment."""
        await self.ensure_initialized()
        
        embed_text = f"{params.name} {params.text_type} {params.description} {' '.join(params.key_teachings)}"
        
        # Prepare data package for canon
        text_data_package = {
            "name": params.name,
            "text_type": params.text_type,
            "description": params.description,
            "key_teachings": params.key_teachings,
            "authorship": params.authorship,
            "restricted_to": params.restricted_to,
            "deity_id": params.deity_id,
            "pantheon_id": params.pantheon_id,
            "notable_passages": params.notable_passages,
            "age_description": params.age_description
        }
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                from lore.core import canon
                text_id = await canon.create_religious_text(
                    ctx, conn, **text_data_package, embedding_text=embed_text
                )
        
        GLOBAL_LORE_CACHE.invalidate_pattern("text")
        return text_id

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_order",
        action_description="Adding religious order",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def add_religious_order(self, ctx, params: ReligiousOrderParams) -> int:
        """Add a religious order using canon establishment."""
        await self.ensure_initialized()
        
        embed_text = f"{params.name} {params.order_type} {params.description} {params.gender_composition}"
        
        # Prepare data package for canon
        order_data_package = {
            "name": params.name,
            "order_type": params.order_type,
            "description": params.description,
            "gender_composition": params.gender_composition,
            "founding_story": params.founding_story,
            "headquarters": params.headquarters,
            "hierarchy_structure": params.hierarchy_structure,
            "vows": params.vows,
            "practices": params.practices,
            "deity_id": params.deity_id,
            "pantheon_id": params.pantheon_id,
            "special_abilities": params.special_abilities,
            "notable_members": params.notable_members
        }
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                from lore.core import canon
                order_id = await canon.find_or_create_religious_order(
                    ctx, conn, **order_data_package, embedding_text=embed_text
                )
        
        GLOBAL_LORE_CACHE.invalidate_pattern("order")
        return order_id

    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_conflict",
        action_description="Adding religious conflict",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def add_religious_conflict(self, ctx, params: ReligiousConflictParams) -> int:
        """Add a religious conflict using canon establishment."""
        await self.ensure_initialized()
        
        embed_text = f"{params.name} {params.conflict_type} {params.description} {params.core_disagreement}"
        
        # Prepare data package for canon
        conflict_data_package = {
            "name": params.name,
            "conflict_type": params.conflict_type,
            "description": params.description,
            "parties_involved": params.parties_involved,
            "core_disagreement": params.core_disagreement,
            "beginning_date": params.beginning_date,
            "resolution_date": params.resolution_date,
            "status": params.status,
            "casualties": params.casualties,
            "historical_impact": params.historical_impact
        }
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                from lore.core import canon
                conflict_id = await canon.log_religious_conflict(
                    ctx, conn, **conflict_data_package, embedding_text=embed_text
                )
        
        GLOBAL_LORE_CACHE.invalidate_pattern("conflict")
        return conflict_id

    # ===========================
    # Generation Methods Using Agents
    # ===========================

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_pantheon",
        action_description="Generating complete pantheon",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def generate_pantheon(self, ctx) -> Dict[str, Any]:
        """Generate a complete pantheon using specialized agents."""
        await self.ensure_initialized()
        
        # Gather context
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                foundation_lore = await conn.fetch("""
                    SELECT category, description FROM WorldLore
                    WHERE category in ('cosmology', 'magic_system', 'social_structure')
                """)
                foundation_context = {
                    row['category']: row['description'] 
                    for row in foundation_lore
                }

                regions = await conn.fetch("""
                    SELECT name FROM GeographicRegions LIMIT 5
                """)
                region_names = [r['name'] for r in regions]

                nations = await conn.fetch("""
                    SELECT name, matriarchy_level FROM Nations
                    ORDER BY matriarchy_level DESC LIMIT 5
                """)
                nation_info = [
                    f"{n['name']} (matriarchy level: {n['matriarchy_level']})"
                    for n in nations
                ]

        prompt = f"""
        Generate a complete feminine-dominated pantheon for a matriarchal fantasy world.

        WORLD CONTEXT:
        Cosmology: {foundation_context.get('cosmology', 'Not specified')}
        Magic System: {foundation_context.get('magic_system', 'Not specified')}
        Social Structure: {foundation_context.get('social_structure', 'Not specified')}

        Geographic Regions: {', '.join(region_names)}
        Nations: {', '.join(nation_info)}

        Requirements:
        1. Goddesses hold all supreme positions of power
        2. Male deities exist but in supporting/specialized roles
        3. Clear divine hierarchy reinforcing feminine dominance
        4. Domains that reflect matriarchal power dynamics
        5. Cosmic structure supporting matriarchal principles
        """

        run_config = RunConfig(
            workflow_name="PantheonGeneration",
            trace_metadata={"user_id": self.user_id, "conversation_id": self.conversation_id}
        )
        
        result = await Runner.run(
            self.pantheon_generator, 
            prompt, 
            context=ctx.context,
            run_config=run_config
        )
        
        generated_data = result.final_output_as(GeneratedPantheon)
        
        # Store the pantheon
        pantheon_id = await self.add_pantheon(ctx, generated_data.pantheon)
        
        # Store all deities
        created_deities = []
        for deity_params in generated_data.deities:
            deity_params.pantheon_id = pantheon_id
            try:
                deity_id = await self.add_deity(ctx, deity_params)
                created_deities.append({
                    "id": deity_id,
                    **deity_params.dict()
                })
            except Exception as e:
                logger.error(f"Error creating deity {deity_params.name}: {e}")
        
        return {
            "pantheon": {
                "id": pantheon_id,
                **generated_data.pantheon.dict()
            },
            "deities": created_deities
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_religious_practices",
        action_description="Generating religious practices",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def generate_religious_practices(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Generate religious practices for a pantheon."""
        await self.ensure_initialized()
        
        # Get pantheon context
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return [{"error": "Pantheon not found"}]
                
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)

        pantheon_data = dict(pantheon)
        deities_data = [dict(d) for d in deities]

        prompt = f"""
        Generate 5-7 religious practices for the pantheon: {pantheon_data['name']}
        
        PANTHEON DESCRIPTION: {pantheon_data['description']}
        
        DEITIES:
        {json.dumps(deities_data[:10], indent=2)}
        
        Create practices that:
        1. Reinforce matriarchal dominance
        2. Include varied practice types (rituals, ceremonies, observances)
        3. Show clear gender role expectations
        4. Associate with specific deities where appropriate
        """

        run_config = RunConfig(workflow_name="PracticeGeneration")
        result = await Runner.run(
            self.practice_generator,
            prompt,
            context=ctx.context,
            run_config=run_config
        )
        
        practices = result.final_output_as(List[ReligiousPracticeParams])
        
        created_practices = []
        for practice in practices:
            practice.pantheon_id = pantheon_id
            try:
                practice_id = await self.add_religious_practice(ctx, practice)
                created_practices.append({
                    "id": practice_id,
                    **practice.dict()
                })
            except Exception as e:
                logger.error(f"Error creating practice {practice.name}: {e}")
        
        return created_practices

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_faith_system",
        action_description="Generating complete faith system",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """Generate a complete faith system with all components."""
        # Generate pantheon first
        pantheon_result = await self.generate_pantheon(ctx)
        
        if "error" in pantheon_result:
            return pantheon_result
        
        pantheon_id = pantheon_result["pantheon"]["id"]
        
        # Generate all other components
        practices = await self.generate_religious_practices(ctx, pantheon_id)
        holy_sites = await self._generate_holy_sites(ctx, pantheon_id)
        texts = await self._generate_religious_texts(ctx, pantheon_id)
        orders = await self._generate_religious_orders(ctx, pantheon_id)
        conflicts = await self._generate_religious_conflicts(ctx, pantheon_id)
        
        return {
            "pantheon": pantheon_result["pantheon"],
            "deities": pantheon_result["deities"],
            "practices": practices,
            "holy_sites": holy_sites,
            "texts": texts,
            "orders": orders,
            "conflicts": conflicts
        }

    # Helper generation methods
    async def _generate_holy_sites(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Generate holy sites for a pantheon."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT name, description FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain
                    FROM Deities
                    WHERE pantheon_id = $1 AND rank >= 6
                    ORDER BY rank DESC
                """, pantheon_id)
                
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations LIMIT 10
                """)

        if not pantheon:
            return []

        prompt = f"""
        Generate 3-5 holy sites for pantheon: {dict(pantheon)['name']}
        
        MAJOR DEITIES:
        {json.dumps([dict(d) for d in deities], indent=2)}
        
        AVAILABLE LOCATIONS:
        {json.dumps([dict(l) for l in locations], indent=2)}
        
        Create sites with proper clergy hierarchies and architectural features
        that emphasize feminine divine authority.
        """

        run_config = RunConfig(workflow_name="HolySiteGeneration")
        result = await Runner.run(
            self.site_generator,
            prompt,
            context=ctx.context,
            run_config=run_config
        )
        
        sites = result.final_output_as(List[HolySiteParams])
        
        created_sites = []
        for site in sites:
            site.pantheon_id = pantheon_id
            try:
                site_id = await self.add_holy_site(ctx, site)
                created_sites.append({
                    "id": site_id,
                    **site.dict()
                })
            except Exception as e:
                logger.error(f"Error creating holy site {site.name}: {e}")
        
        return created_sites

    async def _generate_religious_texts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Generate religious texts for a pantheon."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT name, description, creation_myth
                    FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                    LIMIT 5
                """, pantheon_id)

        if not pantheon:
            return []

        prompt = f"""
        Generate 3-5 religious texts for pantheon: {dict(pantheon)['name']}
        
        DESCRIPTION: {dict(pantheon)['description']}
        CREATION MYTH: {dict(pantheon).get('creation_myth', 'Not specified')}
        
        TOP DEITIES:
        {json.dumps([dict(d) for d in deities], indent=2)}
        
        Create texts that emphasize matriarchal religious authority and
        proper gender roles according to divine law.
        """

        run_config = RunConfig(workflow_name="TextGeneration")
        result = await Runner.run(
            self.text_generator,
            prompt,
            context=ctx.context,
            run_config=run_config
        )
        
        texts = result.final_output_as(List[ReligiousTextParams])
        
        created_texts = []
        for text in texts:
            text.pantheon_id = pantheon_id
            try:
                text_id = await self.add_religious_text(ctx, text)
                created_texts.append({
                    "id": text_id,
                    **text.dict()
                })
            except Exception as e:
                logger.error(f"Error creating text {text.name}: {e}")
        
        return created_texts

    async def _generate_religious_orders(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Generate religious orders with full canon integration."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                    LIMIT 7
                """, pantheon_id)
                
                holy_sites = await conn.fetch("""
                    SELECT id, name, site_type, clergy_type
                    FROM HolySites
                    WHERE pantheon_id = $1
                """, pantheon_id)
                
                # Check existing orders to avoid duplicates
                existing_orders = await conn.fetch("""
                    SELECT name, order_type, deity_id
                    FROM ReligiousOrders
                    WHERE pantheon_id = $1
                """, pantheon_id)
                
                existing_names = {o['name'].lower() for o in existing_orders}
                existing_deity_orders = {}
                for o in existing_orders:
                    if o['deity_id']:
                        if o['deity_id'] not in existing_deity_orders:
                            existing_deity_orders[o['deity_id']] = []
                        existing_deity_orders[o['deity_id']].append(o['order_type'])
    
        if not pantheon:
            return []
    
        pantheon_data = dict(pantheon)
        deities_data = [dict(d) for d in deities]
        sites_data = [dict(s) for s in holy_sites]
    
        prompt = f"""
        Generate 3-4 religious orders for pantheon: {pantheon_data['name']}
        
        MATRIARCHAL ELEMENTS: {pantheon_data['matriarchal_elements']}
        
        DEITIES:
        {json.dumps(deities_data, indent=2)}
        
        HOLY SITES:
        {json.dumps(sites_data, indent=2)}
        
        EXISTING ORDERS (avoid duplicating these):
        {json.dumps([dict(o) for o in existing_orders], indent=2)}
        
        Create orders that:
        1. Have unique names not in the existing list
        2. Serve different purposes (militant, scholarly, charitable, mystic)
        3. Have clear gender compositions reflecting matriarchal hierarchy
        4. Are associated with specific deities when appropriate
        5. Have distinct practices and hierarchies
        
        Focus on orders that haven't been created yet.
        """
    
        run_config = RunConfig(workflow_name="OrderGeneration")
        result = await Runner.run(
            self.order_generator,
            prompt,
            context=ctx.context,
            run_config=run_config
        )
        
        orders = result.final_output_as(List[ReligiousOrderParams])
        created_orders = []
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for order in orders:
                    # Check for duplicates
                    if order.name.lower() in existing_names:
                        logger.info(f"Skipping duplicate order: {order.name}")
                        continue
                    
                    # Check if this deity already has this type of order
                    if order.deity_id and order.deity_id in existing_deity_orders:
                        if order.order_type in existing_deity_orders[order.deity_id]:
                            logger.info(f"Deity {order.deity_id} already has a {order.order_type} order")
                            continue
                    
                    # Use canon system for creation
                    from lore.core import canon
                    embed_text = f"{order.name} {order.order_type} {order.description} {order.gender_composition}"
                    
                    create_data = {
                        'name': order.name,
                        'order_type': order.order_type,
                        'description': order.description,
                        'gender_composition': order.gender_composition,
                        'founding_story': order.founding_story,
                        'headquarters': order.headquarters,
                        'hierarchy_structure': order.hierarchy_structure,
                        'vows': order.vows,
                        'practices': order.practices,
                        'deity_id': order.deity_id,
                        'pantheon_id': pantheon_id,
                        'special_abilities': order.special_abilities,
                        'notable_members': order.notable_members
                    }
                    
                    order_id = await canon.find_or_create_entity(
                        ctx=ctx,
                        conn=conn,
                        entity_type="religious_order",
                        entity_name=order.name,
                        search_fields={'name': order.name, 'pantheon_id': pantheon_id, 'name_field': 'name'},
                        create_data=create_data,
                        table_name="ReligiousOrders",
                        embedding_text=embed_text,
                        similarity_threshold=0.90
                    )
                    
                    created_dict = order.dict()
                    created_dict["id"] = order_id
                    created_dict["pantheon_id"] = pantheon_id
                    created_orders.append(created_dict)
                    
                    # Create order hierarchy positions
                    if order.hierarchy_structure:
                        await self._create_order_positions(conn, order_id, order.hierarchy_structure, order.gender_composition)
        
        return created_orders

    async def _create_order_positions(self, conn, order_id: int, hierarchy: List[str], gender_composition: str) -> None:
        """Create specific positions within a religious order."""
        for i, position in enumerate(hierarchy):
            # Determine gender requirements based on order composition
            if gender_composition == "female_only":
                gender_requirement = "female"
            elif gender_composition == "male_only":
                gender_requirement = "male"
            elif gender_composition == "female_led":
                gender_requirement = "female" if i < len(hierarchy) // 2 else "any"
            else:
                gender_requirement = "any"
            
            try:
                await conn.execute("""
                    INSERT INTO OrderPositions (
                        order_id, position_name, rank_level, 
                        gender_requirement, responsibilities
                    )
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (order_id, position_name) DO NOTHING
                """,
                    order_id, position, i + 1, gender_requirement,
                    f"Responsibilities of {position} in the order"
                )
            except Exception as e:
                logger.error(f"Error creating order position: {e}")

    async def _generate_religious_conflicts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Generate religious conflicts with sophisticated duplicate checking."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                orders = await conn.fetch("""
                    SELECT * FROM ReligiousOrders
                    WHERE pantheon_id = $1
                """, pantheon_id)
                
                texts = await conn.fetch("""
                    SELECT id, name, key_teachings
                    FROM ReligiousTexts
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
                
                # Get existing conflicts to avoid duplicates
                existing_conflicts = await conn.fetch("""
                    SELECT name, conflict_type, parties_involved, core_disagreement
                    FROM ReligiousConflicts
                    WHERE $1 = ANY(parties_involved) OR 
                          EXISTS (
                              SELECT 1 FROM ReligiousOrders ro
                              WHERE ro.pantheon_id = $1 
                              AND ro.name = ANY(ReligiousConflicts.parties_involved)
                          )
                """, pantheon['name'])
    
        if not pantheon or len(orders) < 2:
            return []
    
        pantheon_data = dict(pantheon)
        orders_data = [dict(o) for o in orders]
        texts_data = [dict(t) for t in texts]
        existing_data = [dict(c) for c in existing_conflicts]
    
        prompt = f"""
        Generate 2-3 religious conflicts for pantheon: {pantheon_data['name']}
        
        RELIGIOUS ORDERS:
        {json.dumps(orders_data, indent=2)}
        
        RELIGIOUS TEXTS:
        {json.dumps(texts_data, indent=2)}
        
        EXISTING CONFLICTS (avoid duplicating):
        {json.dumps(existing_data, indent=2)}
        
        Create conflicts that:
        1. Add theological depth without threatening matriarchal order
        2. Involve different parties than existing conflicts
        3. Have unique core disagreements
        4. Could believably arise from doctrinal interpretation
        5. Show realistic progression potential
        
        Focus on:
        - Interpretation disputes
        - Jurisdictional conflicts
        - Ritual disagreements
        - Succession disputes (within matriarchal framework)
        - Resource allocation conflicts
        """
    
        run_config = RunConfig(workflow_name="ConflictGeneration")
        result = await Runner.run(
            self.conflict_generator,
            prompt,
            context=ctx.context,
            run_config=run_config
        )
        
        conflicts = result.final_output_as(List[ReligiousConflictParams])
        created_conflicts = []
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for conflict in conflicts:
                    # Check for similar existing conflicts
                    is_duplicate = False
                    for existing in existing_conflicts:
                        # Check if same parties involved
                        existing_parties = set(existing['parties_involved'])
                        new_parties = set(conflict.parties_involved)
                        
                        if len(existing_parties & new_parties) >= len(new_parties) * 0.7:
                            # High overlap - check if same disagreement
                            if existing['conflict_type'] == conflict.conflict_type:
                                is_duplicate = True
                                break
                    
                    if is_duplicate:
                        logger.info(f"Skipping duplicate conflict: {conflict.name}")
                        continue
                    
                    # Create the conflict
                    from lore.core import canon
                    embed_text = f"{conflict.name} {conflict.conflict_type} {conflict.core_disagreement}"
                    
                    create_data = {
                        'name': conflict.name,
                        'conflict_type': conflict.conflict_type,
                        'description': conflict.description,
                        'parties_involved': conflict.parties_involved,
                        'core_disagreement': conflict.core_disagreement,
                        'beginning_date': conflict.beginning_date,
                        'resolution_date': conflict.resolution_date,
                        'status': conflict.status,
                        'casualties': conflict.casualties,
                        'historical_impact': conflict.historical_impact
                    }
                    
                    conflict_id = await canon.find_or_create_entity(
                        ctx=ctx,
                        conn=conn,
                        entity_type="religious_conflict",
                        entity_name=conflict.name,
                        search_fields={'name': conflict.name, 'name_field': 'name'},
                        create_data=create_data,
                        table_name="ReligiousConflicts",
                        embedding_text=embed_text,
                        similarity_threshold=0.85
                    )
                    
                    # Create conflict timeline entries
                    await self._create_conflict_timeline(conn, conflict_id, conflict)
                    
                    created_dict = conflict.dict()
                    created_dict["id"] = conflict_id
                    created_conflicts.append(created_dict)
        
        return created_conflicts

    async def _create_conflict_timeline(self, conn, conflict_id: int, conflict: ReligiousConflictParams) -> None:
        """Create timeline entries for a religious conflict."""
        # Initial outbreak
        await conn.execute("""
            INSERT INTO ReligiousConflictTimeline (
                conflict_id, event_date, event_description, 
                severity_change, key_figures
            )
            VALUES ($1, $2, $3, $4, $5)
        """,
            conflict_id,
            conflict.beginning_date or "Unknown date",
            f"Initial disagreement over {conflict.core_disagreement}",
            0,
            []
        )
        
        # If resolved, add resolution
        if conflict.status == "resolved" and conflict.resolution_date:
            await conn.execute("""
                INSERT INTO ReligiousConflictTimeline (
                    conflict_id, event_date, event_description,
                    severity_change, key_figures
                )
                VALUES ($1, $2, $3, $4, $5)
            """,
                conflict_id,
                conflict.resolution_date,
                f"Conflict resolved. Impact: {conflict.historical_impact or 'Unknown'}",
                -10,  # Severity drops to 0
                []
            )    

    # ===========================
    # Distribution and Evolution Methods
    # ===========================

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="distribute_religions",
        action_description="Distributing religions across nations",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        """Distribute religions across nations with canon checks."""
        nations = await self.geopolitical_manager.get_all_nations(ctx)
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheons = await conn.fetch("""
                    SELECT id, name, description, matriarchal_elements
                    FROM Pantheons
                """)
                
                # Check existing distributions
                existing = await conn.fetch("""
                    SELECT nation_id FROM NationReligion
                """)
                existing_nation_ids = {row['nation_id'] for row in existing}
        
        if not nations or not pantheons:
            return []
        
        distributions = []
        
        for nation in nations:
            # Skip if already has religion distribution
            if nation['id'] in existing_nation_ids:
                logger.info(f"Nation {nation['name']} already has religious distribution")
                continue
                
            # Generate distribution using agent
            prompt = f"""
            Determine religious distribution for nation: {nation['name']}
            
            NATION DATA:
            Government: {nation.get('government_type')}
            Matriarchy Level: {nation.get('matriarchy_level', 5)}/10
            Cultural Traits: {nation.get('cultural_traits', [])}
            
            AVAILABLE PANTHEONS:
            {json.dumps([dict(p) for p in pantheons], indent=2)}
            
            Create realistic distribution considering the nation's characteristics.
            Nation ID is: {nation['id']}
            """
            
            run_config = RunConfig(workflow_name="ReligiousDistribution")
            result = await Runner.run(
                self.distribution_agent,
                prompt,
                context=ctx.context,
                run_config=run_config
            )
            
            try:
                dist_data = result.final_output_as(NationReligionDistribution)
                dist_data.nation_id = nation["id"]
                
                # Store distribution
                embed_text = f"religion {nation['name']} {dist_data.religious_leadership}"
                
                create_data = {
                    'nation_id': dist_data.nation_id,
                    'state_religion': dist_data.state_religion,
                    'primary_pantheon_id': dist_data.primary_pantheon_id,
                    'pantheon_distribution': json.dumps(dist_data.pantheon_distribution),
                    'religiosity_level': dist_data.religiosity_level,
                    'religious_tolerance': dist_data.religious_tolerance,
                    'religious_leadership': dist_data.religious_leadership,
                    'religious_laws': json.dumps(dist_data.religious_laws),
                    'religious_holidays': dist_data.religious_holidays,
                    'religious_conflicts': dist_data.religious_conflicts,
                    'religious_minorities': dist_data.religious_minorities
                }
                
                async with await self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        distribution_id = await find_or_create_entity(
                            ctx=ctx,
                            conn=conn,
                            entity_type="religious_distribution",
                            entity_name=f"{nation['name']} religious system",
                            search_fields={'nation_id': nation["id"]},
                            create_data=create_data,
                            table_name="NationReligion",
                            embedding_text=embed_text,
                            similarity_threshold=0.95  # Very high threshold
                        )
                        
                        dist_dict = dist_data.dict()
                        dist_dict["id"] = distribution_id
                        distributions.append(dist_dict)
                        
                        # Generate regional practices
                        await self._generate_regional_practices(ctx, dist_dict)
                        
            except Exception as e:
                logger.error(f"Error distributing religion for nation {nation['id']}: {e}")
        
        return distributions

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_ritual",
        action_description="Generating detailed religious ritual",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def generate_ritual(
        self, 
        ctx, 
        pantheon_id: int,
        deity_id: Optional[int] = None,
        purpose: str = "blessing",
        formality_level: int = 5
    ) -> Dict[str, Any]:
        """Generate a detailed religious ritual."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                deity = None
                if deity_id:
                    deity = await conn.fetchrow("""
                        SELECT * FROM Deities 
                        WHERE id = $1 AND pantheon_id = $2
                    """, deity_id, pantheon_id)
                
                practices = await conn.fetch("""
                    SELECT * FROM ReligiousPractices
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
        
        if not pantheon:
            return {"error": "Pantheon not found"}
        
        deity_context = ""
        if deity:
            deity_context = f"""
            DEITY: {dict(deity)['name']}
            Domain: {dict(deity)['domain']}
            Description: {dict(deity)['description']}
            """
        
        prompt = f"""
        Generate a detailed {purpose} ritual for {dict(pantheon)['name']}.
        
        PANTHEON: {dict(pantheon)['description']}
        {deity_context}
        
        FORMALITY LEVEL: {formality_level}/10
        
        EXISTING PRACTICES FOR CONTEXT:
        {json.dumps([dict(p) for p in practices], indent=2)}
        
        Create a complete ritual with components, preparations, and variations
        that reflects matriarchal religious authority.
        """
        
        run_config = RunConfig(workflow_name="RitualGeneration")
        result = await Runner.run(
            self.ritual_generator,
            prompt,
            context=ctx.context,
            run_config=run_config
        )
        
        ritual = result.final_output_as(CompleteRitual)
        
        # Store the ritual in the database
        ritual_id = await self._store_ritual(pantheon_id, deity_id, ritual)
        
        return {
            "id": ritual_id,
            "pantheon_id": pantheon_id,
            "deity_id": deity_id,
            "ritual": ritual.dict()
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_theological_dispute",
        action_description="Simulating theological dispute",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def simulate_theological_dispute(
        self, 
        ctx, 
        pantheon_id: int, 
        dispute_topic: str
    ) -> Dict[str, Any]:
        """Simulate a theological dispute between religious factions."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, description 
                    FROM Deities WHERE pantheon_id = $1
                """, pantheon_id)
                
                religious_orders = await conn.fetch("""
                    SELECT id, name, order_type, description, gender_composition
                    FROM ReligiousOrders WHERE pantheon_id = $1
                """, pantheon_id)
        
        pantheon_data = dict(pantheon)
        deity_data = [dict(d) for d in deities]
        order_data = [dict(o) for o in religious_orders]
        
        # Generate theological positions if needed
        if not order_data or len(order_data) < 2:
            theological_positions = await self._generate_theological_positions(
                ctx, pantheon_data, deity_data, dispute_topic
            )
        else:
            theological_positions = await self._assign_theological_positions(
                ctx, order_data, dispute_topic
            )
        
        # Create debate agents for each position
        debate_agents = []
        for position in theological_positions:
            agent = Agent(
                name=f"{position['name']}DebateAgent",
                instructions=f"""You represent {position['name']} in theological debates.
                Your core belief: {position['core_belief']}
                Defend your interpretation using scripture and tradition.
                Maintain respect for the matriarchal religious order.""",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9)
            )
            debate_agents.append({"agent": agent, "position": position})
        
        # Run the dispute simulation
        dispute_sim = TheologicalDisputeSimulation(
            debate_agents,
            self.theological_arbiter,
            pantheon_data,
            dispute_topic,
            max_rounds=3
        )
        
        dispute_results = await dispute_sim.run(ctx.context)
        
        # Record the dispute
        dispute_id = await self._record_theological_dispute(
            ctx, pantheon_id, dispute_topic, theological_positions, dispute_results
        )
        
        return {
            "dispute_id": dispute_id,
            "topic": dispute_topic,
            "positions": theological_positions,
            "rounds": dispute_results["rounds"],
            "conclusion": dispute_results["conclusion"],
            "religious_implications": dispute_results["implications"]
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_religion_from_culture",
        action_description="Evolving religion through cultural interaction",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def evolve_religion_from_culture(
        self, 
        ctx, 
        pantheon_id: int, 
        nation_id: int, 
        years: int = 50
    ) -> Dict[str, Any]:
        """Evolve a religion based on cultural interaction over time."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                nation = await conn.fetchrow("""
                    SELECT * FROM Nations WHERE id = $1
                """, nation_id)
                
                if not pantheon or not nation:
                    return {"error": "Pantheon or nation not found"}
                
                culture_elements = await conn.fetch("""
                    SELECT * FROM CulturalElements 
                    WHERE $1 = ANY(practiced_by)
                """, nation["name"])
        
        pantheon_data = dict(pantheon)
        nation_data = dict(nation)
        cultural_data = [dict(c) for c in culture_elements]
        
        prompt = f"""
        Simulate how the religion of {pantheon_data['name']} evolves over {years} years
        through interaction with the culture of {nation_data['name']}.
        
        PANTHEON:
        {json.dumps(pantheon_data, indent=2)}
        
        NATION:
        Government: {nation_data.get('government_type')}
        Matriarchy Level: {nation_data.get('matriarchy_level')}/10
        
        CULTURAL ELEMENTS:
        {json.dumps(cultural_data, indent=2)}
        
        Determine evolutionary changes while maintaining matriarchal framework.
        """
        
        run_config = RunConfig(workflow_name="ReligiousEvolution")
        result = await Runner.run(
            self.evolution_agent,
            prompt,
            context=ctx.context,
            run_config=run_config
        )
        
        evolution_data = result.final_output_as(ReligiousEvolution)
        
        # Apply the religious evolution
        await self._apply_religious_evolution(
            ctx, pantheon_id, nation_id, evolution_data.dict()
        )
        
        return {
            "pantheon_id": pantheon_id,
            "nation_id": nation_id,
            "years_simulated": years,
            "evolution_results": evolution_data.dict()
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_sectarian_development",
        action_description="Creating sectarian split",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def generate_sectarian_development(
        self, 
        ctx, 
        pantheon_id: int, 
        trigger_event: str
    ) -> Dict[str, Any]:
        """Generate a sectarian split within a religion."""
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                religious_orders = await conn.fetch("""
                    SELECT * FROM ReligiousOrders WHERE pantheon_id = $1
                """, pantheon_id)
                
                religious_texts = await conn.fetch("""
                    SELECT * FROM ReligiousTexts WHERE pantheon_id = $1
                """, pantheon_id)
        
        pantheon_data = dict(pantheon)
        order_data = [dict(o) for o in religious_orders]
        text_data = [dict(t) for t in religious_texts]
        
        # Create sectarian agents
        orthodox_agent = Agent(
            name="OrthodoxSect",
            instructions=f"""You represent orthodox tradition of {pantheon_data['name']}.
            Value tradition, hierarchy, literal interpretation.
            Strongly matriarchal, preserving established order.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=SectarianPosition
        )
        
        reformist_agent = Agent(
            name="ReformistSect",
            instructions=f"""You represent reform movement in {pantheon_data['name']}.
            Seek adaptation while maintaining core beliefs.
            Still matriarchal but more flexible interpretation.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=SectarianPosition
        )
        
        mystic_agent = Agent(
            name="MysticSect",
            instructions=f"""You represent mystic tradition in {pantheon_data['name']}.
            Focus on direct divine experience over dogma.
            Strongly feminine-focused spirituality.""",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=SectarianPosition
        )
        
        # Generate responses for each sect
        base_prompt = f"""
        As {{}}, respond to this event: {trigger_event}
        
        PANTHEON: {pantheon_data['name']}
        TEXTS: {json.dumps(text_data[:3], indent=2)}
        
        Explain your theological interpretation and prescribed response.
        """
        
        run_config = RunConfig(workflow_name="SectarianDevelopment")
        
        orthodox_result = await Runner.run(
            orthodox_agent,
            base_prompt.format("the orthodox tradition"),
            context=ctx.context,
            run_config=run_config
        )
        
        reformist_result = await Runner.run(
            reformist_agent,
            base_prompt.format("a reformist movement"),
            context=ctx.context,
            run_config=run_config
        )
        
        mystic_result = await Runner.run(
            mystic_agent,
            base_prompt.format("a mystic tradition"),
            context=ctx.context,
            run_config=run_config
        )
        
        sects = [
            orthodox_result.final_output_as(SectarianPosition).dict(),
            reformist_result.final_output_as(SectarianPosition).dict(),
            mystic_result.final_output_as(SectarianPosition).dict()
        ]
        
        # Record the sectarian development
        for sect in sects:
            await self._create_religious_sect(ctx, pantheon_id, sect, trigger_event)
        
        return {
            "pantheon_id": pantheon_id,
            "trigger_event": trigger_event,
            "sects": sects
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_religion",
        action_description="Getting nation religious info",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool(strict_mode=True)
    async def get_nation_religion(self, ctx, nation_id: int) -> Dict[str, Any]:
        """Get comprehensive religious information about a nation."""
        cache_key = f"nation_religion_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = GLOBAL_LORE_CACHE.get(cache_key)
        if cached:
            return cached

        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return {"error": "Nation not found"}

                religion = await conn.fetchrow("""
                    SELECT * FROM NationReligion WHERE nation_id = $1
                """, nation_id)
                
                if not religion:
                    return {"error": "No religious data for this nation"}

                primary_pantheon = None
                if religion["primary_pantheon_id"]:
                    pantheon = await conn.fetchrow("""
                        SELECT id, name, description, matriarchal_elements
                        FROM Pantheons WHERE id = $1
                    """, religion["primary_pantheon_id"])
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
                    SELECT h.* FROM HolySites h
                    JOIN Locations l ON h.location_id = l.id
                    JOIN Nations n ON l.nation_id = n.id
                    WHERE n.id = $1
                """, nation_id)

                result = {
                    "nation": dict(nation),
                    "religion": dict(religion),
                    "primary_pantheon": primary_pantheon,
                    "regional_practices": [dict(pr) for pr in practices],
                    "holy_sites": [dict(hs) for hs in holy_sites]
                }

                # Parse JSON fields
                if result["religion"].get("pantheon_distribution"):
                    try:
                        result["religion"]["pantheon_distribution"] = json.loads(
                            result["religion"]["pantheon_distribution"]
                        )
                    except:
                        pass

                if result["religion"].get("religious_laws"):
                    try:
                        result["religion"]["religious_laws"] = json.loads(
                            result["religion"]["religious_laws"]
                        )
                    except:
                        pass

                GLOBAL_LORE_CACHE.set(cache_key, result)
                return result

    # Helper methods for complex operations
    async def _generate_regional_practices(
        self, 
        ctx, 
        distribution_data: Dict[str, Any]
    ) -> None:
        """Generate regional variations of religious practices."""
        nation_id = distribution_data.get("nation_id")
        primary_pantheon_id = distribution_data.get("primary_pantheon_id")
        
        if not primary_pantheon_id:
            return

        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practices = await conn.fetch("""
                    SELECT id, name, practice_type, description, purpose
                    FROM ReligiousPractices WHERE pantheon_id = $1
                """, primary_pantheon_id)

                nation = await conn.fetchrow("""
                    SELECT name, government_type, matriarchy_level, cultural_traits
                    FROM Nations WHERE id = $1
                """, nation_id)

        if not practices or not nation:
            return

        for practice in practices:
            prompt = f"""
            Create regional variation of religious practice for {dict(nation)['name']}.
            
            PRACTICE: {dict(practice)['name']} - {dict(practice)['description']}
            Practice ID: {practice['id']}
            NATION: Matriarchy level {dict(nation).get('matriarchy_level')}/10
            RELIGIOSITY: {distribution_data.get('religiosity_level')}/10
            
            Adapt the practice to local culture while maintaining core theology.
            """
            
            run_config = RunConfig(workflow_name="RegionalPracticeVariation")
            result = await Runner.run(
                self.regional_practice_agent,
                prompt,
                context=ctx.context,
                run_config=run_config
            )
            
            try:
                var_data = result.final_output_as(RegionalPracticeVariation)
                var_data.practice_id = practice['id']  # Ensure correct practice ID
                
                embed_text = f"practice {dict(practice)['name']} {var_data.regional_variation}"
                embedding = await generate_embedding(embed_text)
                
                async with await self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO RegionalReligiousPractice (
                                nation_id, practice_id, regional_variation,
                                importance, frequency, local_additions,
                                gender_differences, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        nation_id, var_data.practice_id, var_data.regional_variation,
                        var_data.importance, var_data.frequency, 
                        var_data.local_additions, var_data.gender_differences,
                        embedding)
                        
            except Exception as e:
                logger.error(f"Error generating regional practice variation: {e}")

    async def _generate_theological_positions(
        self, 
        ctx, 
        pantheon_data: Dict[str, Any], 
        deity_data: List[Dict[str, Any]], 
        dispute_topic: str
    ) -> List[Dict[str, Any]]:
        """Generate theological positions for a dispute."""
        prompt = f"""
        Generate 3 theological positions on: {dispute_topic}
        
        PANTHEON: {pantheon_data['name']}
        DEITIES: {json.dumps(deity_data[:5], indent=2)}
        
        Create diverse but orthodox positions that maintain matriarchal order.
        
        Return JSON array with each position having:
        - name: school of thought name
        - core_belief: central theological claim
        - scriptural_basis: supporting texts/traditions
        - key_arguments: list of main arguments
        """
        
        # Use a specialized theological generator if needed
        position_generator = Agent(
            name="TheologicalPositionGenerator",
            instructions="Generate diverse theological positions for religious disputes.",
            model="gpt-4.1-nano",
            output_type=List[TheologicalPosition]
        )
        
        result = await Runner.run(
            position_generator,
            prompt,
            context=ctx.context
        )
        
        positions = result.final_output_as(List[TheologicalPosition])
        return [pos.dict() for pos in positions]

    async def _assign_theological_positions(
        self, 
        ctx, 
        order_data: List[Dict[str, Any]], 
        dispute_topic: str
    ) -> List[Dict[str, Any]]:
        """Assign theological positions to existing religious orders."""
        positions = []
        
        for order in order_data[:3]:  # Use up to 3 orders
            position = {
                "name": order['name'],
                "core_belief": f"{order['name']}'s interpretation of {dispute_topic}",
                "scriptural_basis": f"Based on {order['name']}'s traditions",
                "key_arguments": [
                    f"Argument based on {order['order_type']} tradition",
                    f"Gender composition ({order['gender_composition']}) influences interpretation"
                ]
            }
            positions.append(position)
        
        return positions

    async def _record_theological_dispute(
        self, 
        ctx,
        pantheon_id: int, 
        dispute_topic: str, 
        positions: List[Dict[str, Any]], 
        results: Dict[str, Any]
    ) -> int:
        """Record a theological dispute in the database."""
        parties = [pos['name'] for pos in positions]
        
        conflict_params = ReligiousConflictParams(
            name=f"Theological Dispute: {dispute_topic}",
            conflict_type="theological_dispute",
            description=f"Dispute over {dispute_topic} involving {len(positions)} positions",
            parties_involved=parties,
            core_disagreement=dispute_topic,
            status="resolved" if results.get("conclusion") else "ongoing",
            historical_impact=results.get("implications", "Impact still being assessed")
        )
        
        return await self.add_religious_conflict(ctx, conflict_params)

    async def _apply_religious_evolution(
        self, 
        ctx,
        pantheon_id: int, 
        nation_id: int, 
        evolution_data: Dict[str, Any]
    ) -> None:
        """Apply evolutionary changes to a religion."""
        # This would update various religious elements based on evolution_data
        # For example, creating new practices, modifying existing ones, etc.
        
        if "new_practices" in evolution_data:
            for practice_data in evolution_data["new_practices"]:
                practice_params = ReligiousPracticeParams(
                    name=practice_data.get("name", "Evolved Practice"),
                    practice_type=practice_data.get("type", "ritual"),
                    description=practice_data.get("description", ""),
                    purpose=practice_data.get("purpose", "cultural_synthesis"),
                    pantheon_id=pantheon_id
                )
                await self.add_religious_practice(ctx, practice_params)
        
        # Update nation's religious data to reflect evolution
        if "religious_changes" in evolution_data:
            async with await self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE NationReligion 
                        SET religious_laws = religious_laws || $1::jsonb
                        WHERE nation_id = $2
                    """, 
                    json.dumps(evolution_data["religious_changes"]), 
                    nation_id)

    async def _create_religious_sect(
        self, 
        ctx,
        pantheon_id: int, 
        sect_data: Dict[str, Any], 
        trigger_event: str
    ) -> int:
        """Create a new religious sect/order based on sectarian development."""
        order_params = ReligiousOrderParams(
            name=sect_data.get("sect_name", "New Sect"),
            order_type="sectarian",
            description=f"Formed in response to: {trigger_event}. {sect_data.get('interpretation', '')}",
            gender_composition=sect_data.get("gender_composition", "mixed"),
            founding_story=f"Split from mainstream faith due to: {trigger_event}",
            practices=sect_data.get("prescribed_practices", []),
            pantheon_id=pantheon_id
        )
        
        return await self.add_religious_order(ctx, order_params)

    async def _store_ritual(
        self, 
        pantheon_id: int, 
        deity_id: Optional[int], 
        ritual: CompleteRitual
    ) -> int:
        """Store a generated ritual as a religious practice."""
        # Convert the ritual to a religious practice
        components_desc = "\n".join([
            f"{comp.name}: {comp.description}" 
            for comp in ritual.components
        ])
        
        practice_params = ReligiousPracticeParams(
            name=ritual.name,
            practice_type="ritual",
            description=f"{ritual.purpose}. {ritual.theological_significance}\n\nComponents:\n{components_desc}",
            purpose=ritual.purpose,
            frequency=ritual.occasion,
            required_elements=[comp.name for comp in ritual.components],
            performed_by=[],  # Could extract from components
            restricted_to=ritual.restrictions,
            deity_id=deity_id,
            pantheon_id=pantheon_id
        )
        
        # Note: We need ctx here but don't have it in this helper
        # In practice, you'd pass ctx to this method
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                embed_text = f"ritual {ritual.name} {ritual.purpose}"
                embedding = await generate_embedding(embed_text)
                
                ritual_id = await conn.fetchval("""
                    INSERT INTO ReligiousPractices (
                        name, practice_type, description, purpose,
                        frequency, required_elements, performed_by,
                        restricted_to, deity_id, pantheon_id, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """,
                practice_params.name, practice_params.practice_type,
                practice_params.description, practice_params.purpose,
                practice_params.frequency, practice_params.required_elements,
                practice_params.performed_by, practice_params.restricted_to,
                practice_params.deity_id, practice_params.pantheon_id, embedding)
                
                return ritual_id

    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="religion_manager",
            directive_text="Create and manage faith systems emphasizing feminine divine superiority.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        logger.info(f"ReligionManager registered with governance")


# ===========================
# Helper Classes
# ===========================

class TheologicalDisputeSimulation:
    """Manages simulated theological disputes between religious factions."""
    
    def __init__(
        self, 
        debate_agents, 
        arbiter_agent, 
        pantheon_data, 
        dispute_topic, 
        max_rounds=3
    ):
        self.debate_agents = debate_agents
        self.arbiter_agent = arbiter_agent
        self.pantheon_data = pantheon_data
        self.dispute_topic = dispute_topic
        self.max_rounds = max_rounds
    
    async def run(self, context):
        """Run the theological dispute simulation."""
        rounds = []
        
        run_config = RunConfig(
            workflow_name="TheologicalDispute",
            trace_metadata={
                "pantheon": self.pantheon_data["name"], 
                "topic": self.dispute_topic
            }
        )
        
        # First round - initial positions
        first_round = []
        for agent_data in self.debate_agents:
            prompt = f"""
            As {agent_data['position']['name']}, state your position on: {self.dispute_topic}
            
            Explain:
            1. Your theological interpretation
            2. Scriptural basis
            3. Why your position maintains proper order
            
            Be concise but thorough.
            """
            
            result = await Runner.run(
                agent_data["agent"], 
                prompt, 
                context=context,
                run_config=run_config
            )
            
            first_round.append({
                "position": agent_data["position"]["name"],
                "argument": result.final_output
            })
        
        rounds.append({"round": 1, "arguments": first_round})
        
        # Subsequent rounds - responses
        for round_num in range(2, self.max_rounds + 1):
            round_arguments = []
            
            for i, agent_data in enumerate(self.debate_agents):
                # Get other positions from previous round
                prev_arguments = [
                    arg for arg in rounds[-1]["arguments"] 
                    if arg["position"] != agent_data["position"]["name"]
                ]
                
                counter_positions = "\n\n".join([
                    f"{arg['position']}: {arg['argument']}" 
                    for arg in prev_arguments
                ])
                
                prompt = f"""
                As {agent_data['position']['name']}, respond to these arguments:
                
                {counter_positions}
                
                Topic: {self.dispute_topic}
                
                Defend your position while addressing their points.
                Show why your interpretation best serves the faith.
                """
                
                result = await Runner.run(
                    agent_data["agent"], 
                    prompt, 
                    context=context,
                    run_config=run_config
                )
                
                round_arguments.append({
                    "position": agent_data["position"]["name"],
                    "argument": result.final_output
                })
            
            rounds.append({"round": round_num, "arguments": round_arguments})
        
        # Arbiter judgment
        all_arguments = "\n\n".join([
            f"ROUND {r['round']}:\n" + "\n".join([
                f"{arg['position']}: {arg['argument']}" 
                for arg in r["arguments"]
            ])
            for r in rounds
        ])
        
        arbiter_prompt = f"""
        As theological arbiter for {self.pantheon_data['name']}, judge this dispute:
        
        TOPIC: {self.dispute_topic}
        
        ARGUMENTS:
        {all_arguments}
        
        Provide your judgment maintaining orthodox matriarchal principles.
        """
        
        arbiter_result = await Runner.run(
            self.arbiter_agent, 
            arbiter_prompt, 
            context=context,
            run_config=run_config
        )
        
        conclusion = arbiter_result.final_output_as(DisputeConclusion)
        
        return {
            "rounds": rounds,
            "conclusion": conclusion.conclusion,
            "implications": conclusion.implications
        }
