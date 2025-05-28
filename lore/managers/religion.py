# lore/managers/religion.py

import logging
import json
import random
from typing import Dict, List, Any, Optional

# Agents SDK imports
from agents import Agent, function_tool, Runner, ModelSettings
from agents.run_context import RunContextWrapper
from agents.run import RunConfig
from agents import handoff
from agents import InputGuardrail, GuardrailFunctionOutput

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.utils.theming import MatriarchalThemingUtils
from lore.core.cache import GLOBAL_LORE_CACHE

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MatriarchalThemeGuardrail(BaseModel):
    """Output schema for matriarchal theme validation."""
    is_matriarchal: bool
    reasoning: str
    suggestions: List[str] = []

class ReligionManager(BaseLoreManager):
    """
    Comprehensive system for managing religions, faiths, and belief systems
    within the matriarchal society, including both creation and distribution.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.initialized = False
        
        # Initialize agents with proper SDK pattern
        self.theme_guardrail_agent = Agent(
            name="MatriarchalThemeAgent",
            instructions="You verify that all religious content maintains matriarchal themes. You identify elements that might contradict a female-dominant religious structure.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            output_type=MatriarchalThemeGuardrail
        )

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
                    gender TEXT NOT NULL, -- e.g. female, male, non-binary
                    domain TEXT[] NOT NULL, -- e.g. love, war, knowledge
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
            # Additional tables as in original code...
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

    # Define matriarchal guardrail
    async def matriarchal_theme_guardrail(self, ctx: RunContextWrapper, agent, input_data: str) -> GuardrailFunctionOutput:
        """Guardrail to ensure content maintains matriarchal themes."""
        result = await Runner.run(self.theme_guardrail_agent, input_data, context=ctx.context)
        validation_result = result.final_output
        
        return GuardrailFunctionOutput(
            output_info=validation_result,
            tripwire_triggered=not validation_result.is_matriarchal
        )

    # ------------------------------------------------------------------------
    # Core Faith System Methods (Create/Update)
    # ------------------------------------------------------------------------

    async def _add_deity_impl(
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
        await self.ensure_initialized()
    
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
                """,
                name, gender, domain, description, pantheon_id,
                iconography, holy_symbol, sacred_animals, sacred_colors,
                json.dumps(relationships), rank, worshippers)
    
                embed_text = f"{name} {gender} {' '.join(domain)} {description}"
                await self.generate_and_store_embedding(embed_text, conn, "Deities", "id", deity_id)
                GLOBAL_LORE_CACHE.invalidate_pattern("deity")
                return deity_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_deity",
        action_description="Adding deity: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
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
        """Add a deity to the world's religious system."""
        return await self._add_deity_impl(
            ctx, name, gender, domain, description, pantheon_id, iconography, holy_symbol,
            sacred_animals, sacred_colors, relationships, rank, worshippers
        )



    async def _add_pantheon_impl(
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
        await self.ensure_initialized()
    
        major_holy_days = major_holy_days or []
        geographical_spread = geographical_spread or []
        dominant_nations = dominant_nations or []
        primary_worshippers = primary_worshippers or []
        taboos = taboos or []
    
        embed_text = f"{name} {description} {origin_story} {matriarchal_elements}"
        embedding = await generate_embedding(embed_text)
    
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
                """,
                name, description, origin_story, matriarchal_elements,
                creation_myth, afterlife_beliefs, cosmic_structure,
                major_holy_days, geographical_spread, dominant_nations,
                primary_worshippers, taboos, embedding)
    
                GLOBAL_LORE_CACHE.invalidate_pattern("pantheon")
                return pantheon_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_pantheon",
        action_description="Adding pantheon: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
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
        """Add a pantheon to the world's religious system."""
        return await self._add_pantheon_impl(
            ctx, name, description, origin_story, matriarchal_elements,
            creation_myth, afterlife_beliefs, cosmic_structure, major_holy_days,
            geographical_spread, dominant_nations, primary_worshippers, taboos,
        )


    async def _add_religious_practice_impl(
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
        await self.initialize_tables()
    
        required_elements = required_elements or []
        performed_by = performed_by or []
        restricted_to = restricted_to or []
    
        embed_text = f"{name} {practice_type} {description} {purpose}"
        embedding = await generate_embedding(embed_text)
    
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
                """,
                name, practice_type, description, purpose,
                frequency, required_elements, performed_by,
                restricted_to, deity_id, pantheon_id, embedding)
    
                GLOBAL_LORE_CACHE.invalidate_pattern("practice")
                return practice_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_practice",
        action_description="Adding religious practice: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
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
        return await self._add_religious_practice_impl(
            ctx, name, practice_type, description, purpose, frequency, required_elements,
            performed_by, restricted_to, deity_id, pantheon_id
        )


    async def _add_holy_site_impl(
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
        await self.initialize_tables()
    
        clergy_hierarchy = clergy_hierarchy or []
        miracles_reported = miracles_reported or []
        restrictions = restrictions or []
    
        embed_text = f"{name} {site_type} {description} {clergy_type}"
        embedding = await generate_embedding(embed_text)
    
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
                """,
                name, site_type, description, clergy_type,
                location_id, location_description, deity_id,
                pantheon_id, clergy_hierarchy, pilgrimage_info,
                miracles_reported, restrictions, architectural_features,
                embedding)
    
                GLOBAL_LORE_CACHE.invalidate_pattern("site")
                return site_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_holy_site",
        action_description="Adding holy site: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
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
        return await self._add_holy_site_impl(
            ctx, name, site_type, description, clergy_type,
            location_id, location_description, deity_id, pantheon_id,
            clergy_hierarchy, pilgrimage_info, miracles_reported, restrictions, architectural_features
        )


    async def _add_religious_text_impl(
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
        await self.initialize_tables()
    
        restricted_to = restricted_to or []
        notable_passages = notable_passages or []
    
        embed_text = f"{name} {text_type} {description} {' '.join(key_teachings)}"
        embedding = await generate_embedding(embed_text)
    
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
                """,
                name, text_type, description, key_teachings,
                authorship, restricted_to, deity_id,
                pantheon_id, notable_passages, age_description,
                embedding)
    
                GLOBAL_LORE_CACHE.invalidate_pattern("text")
                return text_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_text",
        action_description="Adding religious text: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
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
        return await self._add_religious_text_impl(
            ctx, name, text_type, description, key_teachings, authorship,
            restricted_to, deity_id, pantheon_id, notable_passages, age_description
        )

    async def _add_religious_order_impl(
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
        await self.initialize_tables()
    
        hierarchy_structure = hierarchy_structure or []
        vows = vows or []
        practices = practices or []
        special_abilities = special_abilities or []
        notable_members = notable_members or []
    
        embed_text = f"{name} {order_type} {description} {gender_composition}"
        embedding = await generate_embedding(embed_text)
    
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
                """,
                name, order_type, description, gender_composition,
                founding_story, headquarters, hierarchy_structure,
                vows, practices, deity_id, pantheon_id,
                special_abilities, notable_members, embedding)
    
                GLOBAL_LORE_CACHE.invalidate_pattern("order")
                return order_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_order",
        action_description="Adding religious order: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
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
        return await self._add_religious_order_impl(
            ctx, name, order_type, description, gender_composition, founding_story, headquarters,
            hierarchy_structure, vows, practices, deity_id, pantheon_id, special_abilities,
            notable_members
        )

    async def _add_religious_conflict_impl(
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
        await self.initialize_tables()
    
        embed_text = f"{name} {conflict_type} {description} {core_disagreement}"
        embedding = await generate_embedding(embed_text)
    
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
                """,
                name, conflict_type, description, parties_involved,
                core_disagreement, beginning_date, resolution_date,
                status, casualties, historical_impact, embedding)
    
                GLOBAL_LORE_CACHE.invalidate_pattern("conflict")
                return conflict_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_religious_conflict",
        action_description="Adding religious conflict: {name}",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
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
        return await self._add_religious_conflict_impl(
            ctx, name, conflict_type, description, parties_involved, core_disagreement,
            beginning_date, resolution_date, status, casualties, historical_impact
        )

    # ------------------------------------------------------------------------
    # LLM-Based Generation
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_pantheon",
        action_description="Generating pantheon for the world",
        id_from_context=lambda ctx: "religion_manager"
    )
    @function_tool
    async def generate_pantheon(self, ctx) -> Dict[str, Any]:
        """Generate a complete pantheon for the world with governance oversight."""
        run_ctx = RunContextWrapper(context=ctx.context)

        # Gather context from DB
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                foundation_lore = await conn.fetch("""
                    SELECT category, description FROM WorldLore
                    WHERE category in ('cosmology', 'magic_system', 'social_structure')
                """)
                foundation_context = {}
                for row in foundation_lore:
                    foundation_context[row['category']] = row['description']

                # Some regions
                regions = await conn.fetch("""
                    SELECT name FROM GeographicRegions
                    LIMIT 5
                """)
                region_names = [r['name'] for r in regions]

                # Nations
                nations = await conn.fetch("""
                    SELECT name, matriarchy_level FROM Nations
                    ORDER BY matriarchy_level DESC
                    LIMIT 5
                """)
                nation_context = ""
                for row in nations:
                    nation_context += f"{row['name']} (matriarchy level: {row['matriarchy_level']}), "

        # Create pantheon generation agent
        pantheon_agent = Agent(
            name="PantheonGenerationAgent",
            instructions="You create religious pantheons for matriarchal fantasy worlds. Focus on feminine divine power structures.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            # Add input guardrail to ensure matriarchal themes
            input_guardrails=[
                InputGuardrail(guardrail_function=self.matriarchal_theme_guardrail)
            ]
        )

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

        Return JSON:
        {{
          "pantheon": {{
             "name": ...,
             "description": ...,
             "origin_story": ...,
             "matriarchal_elements": ...,
             "creation_myth": ...,
             "afterlife_beliefs": ...,
             "cosmic_structure": ...,
             "major_holy_days": [...],
             "geographical_spread": [...],
             "dominant_nations": [...],
             "primary_worshippers": [...],
             "taboos": [...]
          }},
          "deities": [
             {{
                "name": ...,
                "gender": ...,
                "domain": [...],
                "description": ...,
                "iconography": ...,
                "holy_symbol": ...,
                "sacred_animals": [...],
                "sacred_colors": [...],
                "rank": ...,
                "worshippers": [...],
                "relationships": {{ ... }}
             }},
             ...
          ]
        }}
        """

        run_config = RunConfig(workflow_name="PantheonGeneration")
        result = await Runner.run(pantheon_agent, prompt, context=run_ctx.context, run_config=run_config)
        response_text = result.final_output

        try:
            pantheon_data = json.loads(response_text)

            if not isinstance(pantheon_data, dict):
                raise ValueError("Response was not a JSON object.")

            if "pantheon" not in pantheon_data or "deities" not in pantheon_data:
                raise ValueError("Missing 'pantheon' or 'deities' in LLM response")

            p_info = pantheon_data["pantheon"]
            d_info = pantheon_data["deities"]

            # Create the pantheon
            pantheon_id = await self.add_pantheon(
                run_ctx,
                name=p_info.get("name","The Pantheon"),
                description=p_info.get("description",""),
                origin_story=p_info.get("origin_story",""),
                matriarchal_elements=p_info.get("matriarchal_elements",""),
                creation_myth=p_info.get("creation_myth"),
                afterlife_beliefs=p_info.get("afterlife_beliefs"),
                cosmic_structure=p_info.get("cosmic_structure"),
                major_holy_days=p_info.get("major_holy_days"),
                geographical_spread=p_info.get("geographical_spread"),
                dominant_nations=p_info.get("dominant_nations"),
                primary_worshippers=p_info.get("primary_worshippers"),
                taboos=p_info.get("taboos")
            )

            created_deities = []
            for deity in d_info:
                try:
                    deity_id = await self.add_deity(
                        run_ctx,
                        name=deity.get("name","Unnamed Deity"),
                        gender=deity.get("gender","female"),
                        domain=deity.get("domain",[]),
                        description=deity.get("description",""),
                        pantheon_id=pantheon_id,
                        iconography=deity.get("iconography"),
                        holy_symbol=deity.get("holy_symbol"),
                        sacred_animals=deity.get("sacred_animals"),
                        sacred_colors=deity.get("sacred_colors"),
                        relationships=deity.get("relationships",{}),
                        rank=deity.get("rank",5),
                        worshippers=deity.get("worshippers",[])
                    )
                    deity["id"] = deity_id
                    created_deities.append(deity)
                except Exception as e:
                    logger.error(f"Error creating deity {deity.get('name')}: {e}")

            return {
                "pantheon": {**p_info, "id": pantheon_id},
                "deities": created_deities
            }

        except Exception as e:
            logger.error(f"Error generating pantheon: {e}")
            return {"error": str(e)}

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_religious_practices",
        action_description="Generating religious practices for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_religious_practices(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate religious practices for a pantheon.
        """
        run_ctx = RunContextWrapper(context=ctx.context)
        # gather pantheon/deities
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons
                    WHERE id = $1
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
        Generate religious practices for this pantheon:

        PANTHEON: {pantheon_data.get('name')}
        DESCRIPTION: {pantheon_data.get('description')}

        DEITIES:
        {json.dumps(deities_data, indent=2)}

        Create 5-7 religious practices that:
        1. Reinforce matriarchal dominance
        2. Varied practice types
        3. Provide specific details
        4. Show which deities they're associated with

        Return JSON array with each practice:
        - name
        - practice_type
        - description
        - purpose
        - frequency
        - required_elements
        - performed_by
        - restricted_to
        - deity_id
        """

        practice_agent = Agent(
            name="ReligiousPracticeAgent",
            instructions="You create religious practices for fantasy pantheons.",
            model="gpt-4.1-nano"
        )

        run_config = RunConfig(workflow_name="PracticeGeneration")
        result = await Runner.run(practice_agent, prompt, context=run_ctx.context, run_config=run_config)
        response_text = result.final_output

        try:
            practices = json.loads(response_text)
            if not isinstance(practices, list):
                if isinstance(practices, dict):
                    practices = [practices]
                else:
                    practices = []

            created_practices = []
            for p in practices:
                try:
                    p_id = await self.add_religious_practice(
                        run_ctx,
                        name=p.get("name","Unnamed Practice"),
                        practice_type=p.get("practice_type","ritual"),
                        description=p.get("description",""),
                        purpose=p.get("purpose","worship"),
                        frequency=p.get("frequency"),
                        required_elements=p.get("required_elements"),
                        performed_by=p.get("performed_by"),
                        restricted_to=p.get("restricted_to"),
                        deity_id=p.get("deity_id"),
                        pantheon_id=pantheon_id
                    )
                    p["id"] = p_id
                    created_practices.append(p)
                except Exception as e:
                    logger.error(f"Error creating religious practice {p.get('name')}: {e}")

            return created_practices
        except Exception as e:
            logger.error(f"Error generating religious practices: {e}")
            return []

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_holy_sites",
        action_description="Generating holy sites for pantheon: {pantheon_id}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_holy_sites(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Generate holy sites for a pantheon.
        """
        run_ctx = RunContextWrapper(context=ctx.context)
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT name, description, geographical_spread, dominant_nations
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                if not pantheon:
                    return [{"error": "Pantheon not found"}]

                # get major deities
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain
                    FROM Deities
                    WHERE pantheon_id = $1 AND rank >= 6
                    ORDER BY rank DESC
                """, pantheon_id)

                # get potential locations
                locations = await conn.fetch("""
                    SELECT id, location_name, description
                    FROM Locations
                    LIMIT 10
                """)

                pantheon_data = dict(pantheon)
                deities_data = [dict(d) for d in deities]
                location_data = [dict(loc) for loc in locations]

        prompt = f"""
        Generate holy sites for pantheon:
        {pantheon_data.get('name')} - {pantheon_data.get('description')}

        MAJOR DEITIES:
        {json.dumps(deities_data, indent=2)}

        POTENTIAL LOCATIONS:
        {json.dumps(location_data, indent=2)}

        Create 3-5 holy sites. Return JSON array with each site:
        - name
        - site_type
        - description
        - clergy_type
        - location_id
        - location_description
        - deity_id
        - clergy_hierarchy
        - pilgrimage_info
        - miracles_reported
        - restrictions
        - architectural_features
        """

        site_agent = Agent(
            name="HolySiteAgent",
            instructions="You create holy sites for fantasy pantheons.",
            model="gpt-4.1-nano"
        )

        run_config = RunConfig(workflow_name="HolySiteGeneration")
        result = await Runner.run(site_agent, prompt, context=run_ctx.context, run_config=run_config)
        response_text = result.final_output

        try:
            sites = json.loads(response_text)
            if not isinstance(sites, list):
                if isinstance(sites, dict):
                    sites = [sites]
                else:
                    sites = []

            created_sites = []
            for s in sites:
                try:
                    s_id = await self.add_holy_site(
                        run_ctx,
                        name=s.get("name","Unnamed Site"),
                        site_type=s.get("site_type","temple"),
                        description=s.get("description",""),
                        clergy_type=s.get("clergy_type","priestesses"),
                        location_id=s.get("location_id"),
                        location_description=s.get("location_description"),
                        deity_id=s.get("deity_id"),
                        pantheon_id=pantheon_id,
                        clergy_hierarchy=s.get("clergy_hierarchy"),
                        pilgrimage_info=s.get("pilgrimage_info"),
                        miracles_reported=s.get("miracles_reported"),
                        restrictions=s.get("restrictions"),
                        architectural_features=s.get("architectural_features")
                    )
                    s["id"] = s_id
                    created_sites.append(s)
                except Exception as e:
                    logger.error(f"Error creating holy site {s.get('name')}: {e}")

            return created_sites
        except Exception as e:
            logger.error(f"Error generating holy sites: {e}")
            return []

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_faith_system",
        action_description="Generating complete faith system for the world",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """
        Generate a complete faith system: pantheon, practices, holy sites, texts,
        orders, conflicts.
        """
        run_ctx = RunContextWrapper(context=ctx.context)
        # 1) Generate pantheon
        pantheon_data = await self.generate_pantheon(run_ctx)
        if "error" in pantheon_data:
            return pantheon_data

        pantheon_id = pantheon_data["pantheon"]["id"]

        # 2) Generate practices
        practices = await self.generate_religious_practices(run_ctx, pantheon_id)

        # 3) Holy sites
        holy_sites = await self.generate_holy_sites(run_ctx, pantheon_id)

        # 4) Religious texts
        religious_texts = await self._generate_religious_texts(run_ctx, pantheon_id)

        # 5) Religious orders
        religious_orders = await self._generate_religious_orders(run_ctx, pantheon_id)

        # 6) Religious conflicts
        religious_conflicts = await self._generate_religious_conflicts(run_ctx, pantheon_id)

        return {
            "pantheon": pantheon_data["pantheon"],
            "deities": pantheon_data["deities"],
            "practices": practices,
            "holy_sites": holy_sites,
            "religious_texts": religious_texts,
            "religious_orders": religious_orders,
            "religious_conflicts": religious_conflicts
        }

    # --- "Private" Helper methods for sub-generation ---
    async def _generate_religious_texts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Helper method to generate religious texts for a pantheon.
        """
        run_ctx = RunContextWrapper(context=ctx.context)
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT name, description, creation_myth
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                if not pantheon:
                    return []

                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)

                pantheon_data = dict(pantheon)
                deities_data = [dict(d) for d in deities]

        prompt = f"""
        Generate religious texts for pantheon: {pantheon_data.get('name')}

        DESCRIPTION: {pantheon_data.get('description')}
        CREATION MYTH: {pantheon_data.get('creation_myth')}

        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}

        Create 3-5 religious texts that:
        1. Emphasize matriarchal dominance
        2. Varied text types (scripture, commentary, hymns)
        3. Describe who has access
        4. Include key teachings

        Return JSON array:
        - name
        - text_type
        - description
        - key_teachings
        - authorship
        - restricted_to
        - deity_id
        - notable_passages
        - age_description
        """

        text_agent = Agent(
            name="ReligiousTextAgent",
            instructions="You create religious texts for fantasy pantheons.",
            model="gpt-4.1-nano"
        )

        run_config = RunConfig(workflow_name="TextGeneration")
        result = await Runner.run(text_agent, prompt, context=ctx.context, run_config=run_config)
        response_text = result.final_output

        try:
            texts = json.loads(response_text)
            if not isinstance(texts, list):
                if isinstance(texts, dict):
                    texts = [texts]
                else:
                    texts = []

            created_texts = []
            for t in texts:
                try:
                    t_id = await self.add_religious_text(
                        run_ctx,
                        name=t.get("name","Unnamed Text"),
                        text_type=t.get("text_type","scripture"),
                        description=t.get("description",""),
                        key_teachings=t.get("key_teachings",[]),
                        authorship=t.get("authorship"),
                        restricted_to=t.get("restricted_to"),
                        deity_id=t.get("deity_id"),
                        pantheon_id=pantheon_id,
                        notable_passages=t.get("notable_passages"),
                        age_description=t.get("age_description")
                    )
                    t["id"] = t_id
                    created_texts.append(t)
                except Exception as e:
                    logger.error(f"Error creating religious text {t.get('name')}: {e}")

            return created_texts
        except Exception as e:
            logger.error(f"Error generating religious texts: {e}")
            return []

    async def _generate_religious_orders(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Helper method to generate religious orders for a pantheon.
        """
        run_ctx = RunContextWrapper(context=ctx.context)
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                if not pantheon:
                    return []

                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, rank
                    FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)

                holy_sites = await conn.fetch("""
                    SELECT id, name, site_type
                    FROM HolySites
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)

                pantheon_data = dict(pantheon)
                deities_data = [dict(d) for d in deities]
                site_data = [dict(s) for s in holy_sites]

        prompt = f"""
        Generate religious orders for pantheon: {pantheon_data.get('name')}

        DESCRIPTION: {pantheon_data.get('description')}
        MATRIARCHAL ELEMENTS: {pantheon_data.get('matriarchal_elements')}

        DEITIES:
        {json.dumps(deities_data[:5], indent=2)}

        HOLY SITES:
        {json.dumps(site_data, indent=2)}

        Create 3-4 orders that:
        1. Emphasize female leadership
        2. Varied order types
        3. Clear gender compositions
        4. Hierarchies and practices

        Return JSON array:
        - name
        - order_type
        - description
        - gender_composition
        - founding_story
        - headquarters
        - hierarchy_structure
        - vows
        - practices
        - deity_id
        - special_abilities
        - notable_members
        """

        order_agent = Agent(
            name="ReligiousOrderAgent",
            instructions="You create religious orders for fantasy pantheons.",
            model="gpt-4.1-nano"
        )

        run_config = RunConfig(workflow_name="OrderGeneration")
        result = await Runner.run(order_agent, prompt, context=ctx.context, run_config=run_config)
        response_text = result.final_output

        try:
            orders = json.loads(response_text)
            if not isinstance(orders, list):
                if isinstance(orders, dict):
                    orders = [orders]
                else:
                    orders = []

            created_orders = []
            for o in orders:
                try:
                    o_id = await self.add_religious_order(
                        run_ctx,
                        name=o.get("name","Unnamed Order"),
                        order_type=o.get("order_type","monastic"),
                        description=o.get("description",""),
                        gender_composition=o.get("gender_composition","female-only"),
                        founding_story=o.get("founding_story"),
                        headquarters=o.get("headquarters"),
                        hierarchy_structure=o.get("hierarchy_structure"),
                        vows=o.get("vows"),
                        practices=o.get("practices"),
                        deity_id=o.get("deity_id"),
                        pantheon_id=pantheon_id,
                        special_abilities=o.get("special_abilities"),
                        notable_members=o.get("notable_members")
                    )
                    o["id"] = o_id
                    created_orders.append(o)
                except Exception as e:
                    logger.error(f"Error creating religious order {o.get('name')}: {e}")

            return created_orders
        except Exception as e:
            logger.error(f"Error generating religious orders: {e}")
            return []

    async def _generate_religious_conflicts(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Helper method to generate religious conflicts for a pantheon.
        """
        run_ctx = RunContextWrapper(context=ctx.context)
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT name, description, matriarchal_elements
                    FROM Pantheons
                    WHERE id = $1
                """, pantheon_id)
                if not pantheon:
                    return []

                orders = await conn.fetch("""
                    SELECT id, name, order_type, gender_composition
                    FROM ReligiousOrders
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)

                nations = await conn.fetch("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    LIMIT 5
                """)

                pantheon_data = dict(pantheon)
                order_data = [dict(o) for o in orders]
                nation_data = [dict(n) for n in nations]

        prompt = f"""
        Generate 2-3 religious conflicts for pantheon: {pantheon_data.get('name')}.

        DESCRIPTION: {pantheon_data.get('description')}
        MATRIARCHAL ELEMENTS: {pantheon_data.get('matriarchal_elements')}

        RELIGIOUS ORDERS:
        {json.dumps(order_data, indent=2)}

        NATIONS:
        {json.dumps(nation_data, indent=2)}

        Return JSON array each conflict:
        - name
        - conflict_type
        - description
        - parties_involved
        - core_disagreement
        - beginning_date
        - resolution_date
        - status
        - casualties
        - historical_impact
        """

        conflict_agent = Agent(
            name="ReligiousConflictAgent",
            instructions="You create religious conflicts for fantasy pantheons.",
            model="gpt-4.1-nano"
        )

        run_config = RunConfig(workflow_name="ReligiousConflictGeneration")
        result = await Runner.run(conflict_agent, prompt, context=ctx.context, run_config=run_config)
        response_text = result.final_output

        try:
            conflicts = json.loads(response_text)
            if not isinstance(conflicts, list):
                if isinstance(conflicts, dict):
                    conflicts = [conflicts]
                else:
                    conflicts = []

            created_conflicts = []
            for c in conflicts:
                try:
                    c_id = await self.add_religious_conflict(
                        run_ctx,
                        name=c.get("name","Unnamed Conflict"),
                        conflict_type=c.get("conflict_type","schism"),
                        description=c.get("description",""),
                        parties_involved=c.get("parties_involved",[]),
                        core_disagreement=c.get("core_disagreement",""),
                        beginning_date=c.get("beginning_date"),
                        resolution_date=c.get("resolution_date"),
                        status=c.get("status","ongoing"),
                        casualties=c.get("casualties"),
                        historical_impact=c.get("historical_impact")
                    )
                    c["id"] = c_id
                    created_conflicts.append(c)
                except Exception as e:
                    logger.error(f"Error creating religious conflict {c.get('name')}: {e}")

            return created_conflicts
        except Exception as e:
            logger.error(f"Error generating religious conflicts: {e}")
            return []

    # ------------------------------------------------------------------------
    # Distribution Methods
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="distribute_religions",
        action_description="Distributing religions across nations",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        """
        Distribute religions across nations with governance oversight.
        """
        run_ctx = RunContextWrapper(context=ctx.context)
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)

        # get pantheons
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheons = await conn.fetch("""
                    SELECT id, name, description, matriarchal_elements
                    FROM Pantheons
                """)
                pantheon_data = [dict(p) for p in pantheons]

        if not nations or not pantheon_data:
            return []

        distribution_agent = Agent(
            name="ReligiousDistributionAgent",
            instructions="You distribute religious pantheons across nations in a matriarchal world.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )

        distributions = []
        for nation in nations:
            prompt = f"""
            Determine religious distribution for this nation:

            NATION:
            {json.dumps(nation, indent=2)}

            AVAILABLE PANTHEONS:
            {json.dumps(pantheon_data, indent=2)}

            Create a realistic distribution that:
            1. Considers matriarchy_level {nation.get("matriarchy_level",5)}
            2. Possibly state religion?
            3. Distributes pantheons in percentages
            4. Religious laws & matriarchal leadership

            Return JSON:
            {{
              "nation_id": {nation["id"]},
              "state_religion": <bool>,
              "primary_pantheon_id": <int or null>,
              "pantheon_distribution": {{ pantheon_id: % }}, 
              "religiosity_level": <1-10>,
              "religious_tolerance": <1-10>,
              "religious_leadership": "...",
              "religious_laws": {{...}},
              "religious_holidays": [...],
              "religious_conflicts": [...],
              "religious_minorities": [...]
            }}
            """

            run_config = RunConfig(workflow_name="ReligiousDistribution")
            result = await Runner.run(distribution_agent, prompt, context=run_ctx.context, run_config=run_config)

            try:
                dist_data = json.loads(result.final_output)
                if "nation_id" not in dist_data or "religiosity_level" not in dist_data:
                    continue

                embed_text = f"religion {nation['name']} {dist_data.get('religious_leadership','')} {dist_data.get('religious_tolerance',5)}"
                embedding = await generate_embedding(embed_text)

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
                        dist_data["nation_id"],
                        dist_data.get("state_religion", False),
                        dist_data.get("primary_pantheon_id"),
                        json.dumps(dist_data.get("pantheon_distribution", {})),
                        dist_data.get("religiosity_level",5),
                        dist_data.get("religious_tolerance",5),
                        dist_data.get("religious_leadership"),
                        json.dumps(dist_data.get("religious_laws", {})),
                        dist_data.get("religious_holidays",[]),
                        dist_data.get("religious_conflicts",[]),
                        dist_data.get("religious_minorities",[]),
                        embedding)

                        dist_data["id"] = distribution_id
                        distributions.append(dist_data)

                        # generate regional religious practices
                        await self._generate_regional_practices(run_ctx, dist_data)
            except Exception as e:
                logger.error(f"Error distributing religion for nation {nation['id']}: {e}")

        return distributions

    async def _generate_regional_practices(self, ctx, distribution_data: Dict[str, Any]) -> None:
        """Generate regional variations of religious practices for a given nation/pantheon distribution."""
        nation_id = distribution_data.get("nation_id")
        primary_pantheon_id = distribution_data.get("primary_pantheon_id")
        if not primary_pantheon_id:
            return

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practices = await conn.fetch("""
                    SELECT id, name, practice_type, description, purpose
                    FROM ReligiousPractices
                    WHERE pantheon_id = $1
                """, primary_pantheon_id)

                practice_data = [dict(pr) for pr in practices]

                nation = await conn.fetchrow("""
                    SELECT name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                nation_data = dict(nation) if nation else {}

        if not practice_data or not nation_data:
            return

        reg_practice_agent = Agent(
            name="RegionalPracticeAgent",
            instructions="You create regional practice variations for matriarchal societies.",
            model="gpt-4.1-nano"
        )

        for pr in practice_data:
            prompt = f"""
            Create a regional variation of this religious practice for nation {nation_data.get('name','Unknown')}.
            
            NATION:
            {json.dumps(nation_data, indent=2)}

            RELIGIOUS PRACTICE:
            {json.dumps(pr, indent=2)}

            RELIGIOUS CONTEXT:
            Religiosity level: {distribution_data.get("religiosity_level",5)}
            Tolerance: {distribution_data.get("religious_tolerance",5)}

            Return JSON:
            {{
              "practice_id": {pr["id"]},
              "regional_variation": "...",
              "importance": <1-10>,
              "frequency": "...",
              "local_additions": "...",
              "gender_differences": "..."
            }}
            """

            run_config = RunConfig(workflow_name="RegionalPracticeGeneration")
            result = await Runner.run(reg_practice_agent, prompt, context=ctx.context, run_config=run_config)

            try:
                var_data = json.loads(result.final_output)
                if "practice_id" not in var_data or "regional_variation" not in var_data:
                    continue

                embed_text = f"practice {pr['name']} {var_data['regional_variation']}"
                embedding = await generate_embedding(embed_text)

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
                        var_data["practice_id"],
                        var_data.get("regional_variation",""),
                        var_data.get("importance",5),
                        var_data.get("frequency"),
                        var_data.get("local_additions"),
                        var_data.get("gender_differences"),
                        embedding)
            except Exception as e:
                logger.error(f"Error generating regional practice variation: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_theological_dispute",
        action_description="Simulating theological disputes between religious factions",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def simulate_theological_dispute(self, ctx, pantheon_id: int, dispute_topic: str) -> Dict[str, Any]:
        """Simulate a theological dispute with competing religious interpretations."""
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon data
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                deities = await conn.fetch("""
                    SELECT id, name, gender, domain, description 
                    FROM Deities
                    WHERE pantheon_id = $1
                """, pantheon_id)
                
                religious_orders = await conn.fetch("""
                    SELECT id, name, order_type, description, gender_composition
                    FROM ReligiousOrders
                    WHERE pantheon_id = $1
                """, pantheon_id)
        
        pantheon_data = dict(pantheon)
        deity_data = [dict(d) for d in deities]
        order_data = [dict(o) for o in religious_orders]
        
        if not order_data or len(order_data) < 2:
            # Create theological schools of thought if none exist
            theological_positions = await self._generate_theological_positions(
                run_ctx, pantheon_data, deity_data, dispute_topic
            )
        else:
            # Use existing religious orders as dispute participants
            theological_positions = await self._assign_theological_positions(
                run_ctx, order_data, dispute_topic
            )
        
        # Create theological debate agents
        debate_agents = []
        for position in theological_positions:
            agent = Agent(
                name=f"{position['name']}Agent",
                instructions=f"You represent {position['name']}, holding this theological position: {position['core_belief']}. Defend your interpretation using scripture and tradition.",
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.9)
            )
            debate_agents.append({"agent": agent, "position": position})
        
        # Create a theological arbiter agent
        arbiter_agent = Agent(
            name="TheologicalArbiterAgent",
            instructions=f"You are a senior religious authority in the pantheon of {pantheon_data['name']}. Your role is to evaluate theological arguments and determine their validity within tradition.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
        
        # Simulate the dispute
        dispute_simulation = TheologicalDispute(
            debate_agents,
            arbiter_agent,
            pantheon_data,
            dispute_topic,
            max_rounds=3
        )
        
        dispute_results = await dispute_simulation.run(run_ctx.context)
        
        # Record the dispute outcome
        dispute_id = await self._record_theological_dispute(
            pantheon_id, dispute_topic, theological_positions, dispute_results
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
        action_description="Evolving religion based on cultural interactions",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def evolve_religion_from_culture(self, ctx, pantheon_id: int, nation_id: int, years: int = 50) -> Dict[str, Any]:
        """Evolve a religion based on its interaction with a culture over time."""
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon and nation data
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                nation = await conn.fetchrow("""
                    SELECT * FROM Nations WHERE id = $1
                """, nation_id)
                
                if not pantheon or not nation:
                    return {"error": "Pantheon or nation not found"}
                
                # Get cultural elements
                culture_elements = await conn.fetch("""
                    SELECT * FROM CulturalElements 
                    WHERE $1 = ANY(practiced_by)
                """, nation["name"])
        
        pantheon_data = dict(pantheon)
        nation_data = dict(nation)
        cultural_data = [dict(c) for c in culture_elements]
        
        # Create religious evolution agent
        evolution_agent = Agent(
            name="ReligiousEvolutionAgent",
            instructions="You simulate how religion evolves through cultural influence over decades or centuries. Maintain matriarchal themes.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        prompt = f"""
        Simulate how the religion of {pantheon_data['name']} evolves over {years} years
        through interaction with the culture of {nation_data['name']}.
        
        PANTHEON:
        {json.dumps(pantheon_data, indent=2)}
        
        NATION:
        {json.dumps(nation_data, indent=2)}
        
        CULTURAL ELEMENTS:
        {json.dumps(cultural_data, indent=2)}
        
        Determine:
        1. How rituals change
        2. New interpretations of deities
        3. Modified cultural practices
        4. Syncretic elements that emerge
        5. New religious roles or hierarchies
        
        Return detailed JSON with these changes, ensuring a matriarchal framework remains.
        """
        
        result = await Runner.run(evolution_agent, prompt, context=run_ctx.context)
        try:
            evolution_data = json.loads(result.final_output)
            
            # Apply the religious evolution
            await self._apply_religious_evolution(
                pantheon_id, nation_id, evolution_data
            )
            
            return {
                "pantheon_id": pantheon_id,
                "nation_id": nation_id,
                "years_simulated": years,
                "evolution_results": evolution_data
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse evolution data", "raw_output": result.final_output}    

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_sectarian_development",
        action_description="Creating sectarian development with divergent beliefs",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_sectarian_development(self, ctx, pantheon_id: int, trigger_event: str) -> Dict[str, Any]:
        """Generate a sectarian split within a religion based on a triggering event."""
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon data and religious orders
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                religious_orders = await conn.fetch("""
                    SELECT * FROM ReligiousOrders
                    WHERE pantheon_id = $1
                """, pantheon_id)
                
                religious_texts = await conn.fetch("""
                    SELECT * FROM ReligiousTexts
                    WHERE pantheon_id = $1
                """, pantheon_id)
        
        pantheon_data = dict(pantheon)
        order_data = [dict(o) for o in religious_orders]
        text_data = [dict(t) for t in religious_texts]
        
        # Create sectarian development agents with divergent instructions
        orthodox_agent = Agent(
            name="OrthodoxSectAgent",
            instructions=f"You represent the orthodox tradition of {pantheon_data['name']}. You value tradition, established hierarchy, and literal interpretation of sacred texts. Strongly matriarchal.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        reformist_agent = Agent(
            name="ReformistSectAgent",
            instructions=f"You represent a reformist movement within {pantheon_data['name']}. You seek to adapt traditions while maintaining core beliefs. Still matriarchal but more flexible.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        mystic_agent = Agent(
            name="MysticSectAgent",
            instructions=f"You represent a mystic tradition within {pantheon_data['name']}. You focus on direct divine experience over dogma. Strongly feminine-focused spirituality.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Generate sectarian responses to the trigger event
        orthodox_prompt = f"""
        As the orthodox tradition of {pantheon_data['name']}, respond to this event:
        {trigger_event}
        
        PANTHEON DETAILS:
        {json.dumps(pantheon_data, indent=2)}
        
        RELIGIOUS TEXTS:
        {json.dumps(text_data, indent=2)}
        
        Explain your interpretation of this event, what it means for the faith,
        and how believers should respond. Return JSON with:
        - sect_name
        - interpretation
        - doctrinal_position
        - prescribed_practices
        - view_of_other_sects
        """
        
        reformist_prompt = f"""
        As a reformist movement within {pantheon_data['name']}, respond to this event:
        {trigger_event}
        
        PANTHEON DETAILS:
        {json.dumps(pantheon_data, indent=2)}
        
        RELIGIOUS TEXTS:
        {json.dumps(text_data, indent=2)}
        
        Explain your interpretation of this event, what it means for the faith,
        and how believers should respond. Return JSON with:
        - sect_name
        - interpretation
        - doctrinal_position
        - prescribed_practices
        - view_of_other_sects
        """
        
        mystic_prompt = f"""
        As a mystic tradition within {pantheon_data['name']}, respond to this event:
        {trigger_event}
        
        PANTHEON DETAILS:
        {json.dumps(pantheon_data, indent=2)}
        
        RELIGIOUS TEXTS:
        {json.dumps(text_data, indent=2)}
        
        Explain your interpretation of this event, what it means for the faith,
        and how believers should respond. Return JSON with:
        - sect_name
        - interpretation
        - doctrinal_position
        - prescribed_practices
        - view_of_other_sects
        """
        
        orthodox_result = await Runner.run(orthodox_agent, orthodox_prompt, context=run_ctx.context)
        reformist_result = await Runner.run(reformist_agent, reformist_prompt, context=run_ctx.context)
        mystic_result = await Runner.run(mystic_agent, mystic_prompt, context=run_ctx.context)
        
        try:
            orthodox_data = json.loads(orthodox_result.final_output)
            reformist_data = json.loads(reformist_result.final_output)
            mystic_data = json.loads(mystic_result.final_output)
            
            sects = [orthodox_data, reformist_data, mystic_data]
            
            # Record the sectarian development
            for sect in sects:
                await self._create_religious_sect(pantheon_id, sect, trigger_event)
            
            return {
                "pantheon_id": pantheon_id,
                "trigger_event": trigger_event,
                "sects": sects
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse sectarian data"}


    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_religion",
        action_description="Getting religious info for nation {nation_id}",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def get_nation_religion(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive religious information about a nation.
        """
        cache_key = f"nation_religion_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = GLOBAL_LORE_CACHE.get(cache_key)
        if cached:
            return cached

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                if not nation:
                    return {"error": "Nation not found"}

                religion = await conn.fetchrow("""
                    SELECT *
                    FROM NationReligion
                    WHERE nation_id = $1
                """, nation_id)
                if not religion:
                    return {"error": "No religious data for this nation"}

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

                # regional practices
                practices = await conn.fetch("""
                    SELECT r.*, p.name as practice_name, p.practice_type, p.purpose
                    FROM RegionalReligiousPractice r
                    JOIN ReligiousPractices p ON r.practice_id = p.id
                    WHERE r.nation_id = $1
                """, nation_id)

                # possibly get holy sites in this nation
                # code might vary depending on how your locations link to nations
                # We'll skip for brevity or do a direct approach.

                result = {
                    "nation": dict(nation),
                    "religion": dict(religion),
                    "primary_pantheon": primary_pantheon,
                    "regional_practices": [dict(pr) for pr in practices],
                    "holy_sites": []
                }

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

                GLOBAL_LORE_CACHE.set(cache_key, result)
                return result

    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="religion_manager",
            directive_text="Create and manage faith systems emphasizing feminine divine superiority.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
        logger.info(f"ReligionManager registered with governance for user {self.user_id}, conversation {self.conversation_id}")

class RitualComponent(BaseModel):
    """Model for a component of a religious ritual."""
    name: str
    description: str
    purpose: str
    participants: List[str]
    required_items: List[str]
    symbolic_meaning: str

class CompleteRitual(BaseModel):
    """Model for a complete religious ritual."""
    name: str
    purpose: str
    occasion: str
    duration: str
    preparation: str
    components: List[RitualComponent]
    variations: Dict[str, str]
    restrictions: List[str]
    theological_significance: str

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_ritual",
        action_description="Generating detailed religious ritual with structured outputs",
        id_from_context=lambda ctx: "religion_manager"
    )
    async def generate_ritual(
        self, 
        ctx, 
        pantheon_id: int, 
        deity_id: Optional[int] = None,
        purpose: str = "blessing",
        formality_level: int = 5
    ) -> Dict[str, Any]:
        """Generate a detailed religious ritual with structured outputs."""
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get pantheon and deity data
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheon = await conn.fetchrow("""
                    SELECT * FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                if not pantheon:
                    return {"error": "Pantheon not found"}
                
                deity = None
                if deity_id:
                    deity = await conn.fetchrow("""
                        SELECT * FROM Deities WHERE id = $1 AND pantheon_id = $2
                    """, deity_id, pantheon_id)
                
                # Get existing practices for context
                practices = await conn.fetch("""
                    SELECT * FROM ReligiousPractices
                    WHERE pantheon_id = $1
                    LIMIT 5
                """, pantheon_id)
        
        pantheon_data = dict(pantheon)
        deity_data = dict(deity) if deity else None
        practice_data = [dict(p) for p in practices]
        
        # Create ritual generation agent
        ritual_agent = Agent(
            name="RitualGenerationAgent",
            instructions="You create detailed religious rituals for matriarchal fantasy religions. Focus on symbolism, required components, and theological significance.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9),
            output_type=CompleteRitual
        )
        
        deity_str = ""
        if deity_data:
            deity_str = f"DEITY:\n{json.dumps(deity_data, indent=2)}"
        
        prompt = f"""
            Generate a detailed {purpose} ritual for the religion of {pantheon_data['name']}.
            
            PANTHEON:
            {json.dumps(pantheon_data, indent=2)}
            
            {deity_str}
            
            FORMALITY LEVEL: {formality_level}/10
            
            EXISTING PRACTICES:
            {json.dumps(practice_data, indent=2)}
            
            Create a CompleteRitual object with detailed components, ensuring it reflects
            matriarchal power structures and feminine divine authority.
            """
        
        result = await Runner.run(ritual_agent, prompt, context=run_ctx.context)
        ritual_data = result.final_output
        
        # Store the ritual in the database
        ritual_id = await self._store_ritual(pantheon_id, deity_id, ritual_data)
        
        ritual_dict = ritual_data.dict()
        ritual_dict["id"] = ritual_id
        
        return ritual_dict
class TheologicalDispute:
    """
    Manages simulated theological disputes between religious factions.
    """
    
    def __init__(self, debate_agents, arbiter_agent, pantheon_data, dispute_topic, max_rounds=3):
        self.debate_agents = debate_agents
        self.arbiter_agent = arbiter_agent
        self.pantheon_data = pantheon_data
        self.dispute_topic = dispute_topic
        self.max_rounds = max_rounds
    
    async def run(self, context):
        """Run the full theological dispute simulation."""
        rounds = []
        
        # Setup the run configuration
        run_config = RunConfig(
            workflow_name="TheologicalDispute",
            trace_metadata={"pantheon": self.pantheon_data["name"], "topic": self.dispute_topic}
        )
        
        # First round - initial positions
        first_round = []
        for agent_data in self.debate_agents:
            prompt = f"""
            As {agent_data['position']['name']}, state your initial position on:
            {self.dispute_topic}
            
            Clearly explain:
            1. Your theological interpretation
            2. Your scriptural basis
            3. Why your position is correct
            
            Keep your response concise but thorough.
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
        
        # Subsequent rounds - responses to other positions
        for round_num in range(2, self.max_rounds + 1):
            round_arguments = []
            
            for i, agent_data in enumerate(self.debate_agents):
                # Get arguments from previous round
                prev_arguments = [
                    arg for arg in rounds[-1]["arguments"] 
                    if arg["position"] != agent_data["position"]["name"]
                ]
                
                counter_positions = "\n\n".join([
                    f"{arg['position']} argued: {arg['argument']}" 
                    for arg in prev_arguments
                ])
                
                prompt = f"""
                As {agent_data['position']['name']}, respond to these theological arguments:
                
                {counter_positions}
                
                Regarding the topic: {self.dispute_topic}
                
                Defend your position while addressing their key points.
                Point out flaws in their reasoning while reinforcing your interpretation.
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
        
        # Arbiter summary and implications
        all_arguments = ""
        for r in rounds:
            all_arguments += f"ROUND {r['round']}:\n"
            for arg in r["arguments"]:
                all_arguments += f"{arg['position']}: {arg['argument']}\n\n"
        
        arbiter_prompt = f"""
        As the theological arbiter for {self.pantheon_data['name']}, review this theological dispute:
        
        TOPIC: {self.dispute_topic}
        
        DEBATE SUMMARY:
        {all_arguments}
        
        Provide:
        1. An impartial analysis of each position's theological merit
        2. Your conclusion on the most sound interpretation
        3. The religious implications for the faith's future
        
        Return as JSON with "conclusion" and "implications" fields.
        """
        
        arbiter_result = await Runner.run(
            self.arbiter_agent, 
            arbiter_prompt, 
            context=context,
            run_config=run_config
        )
        
        try:
            arbiter_judgment = json.loads(arbiter_result.final_output)
        except json.JSONDecodeError:
            arbiter_judgment = {
                "conclusion": "The arbiter could not reach a definitive conclusion.",
                "implications": "The theological dispute remains unresolved."
            }
        
        return {
            "rounds": rounds,
            "conclusion": arbiter_judgment.get("conclusion", ""),
            "implications": arbiter_judgment.get("implications", "")
        }
