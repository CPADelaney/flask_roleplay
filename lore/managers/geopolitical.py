# lore/managers/geopolitical.py

import logging
import json
import random
from typing import Dict, List, Any, Optional

# Agents SDK imports
from agents import Agent, function_tool, Runner
from agents.run_context import RunContextWrapper
from agents.run import RunConfig
from agents.models import ModelSettings

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.core.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

logger = logging.getLogger(__name__)

# We optionally create an agent to decide how many "nations" or "regions" to create, 
# if you want to move that logic from random to agent-based. 
# (You can skip if you prefer.)
distribution_agent = Agent(
    name="GeopoliticalDistributionAgent",
    instructions=(
        "Given context about the world, you decide how many geopolitical items to generate, or how to distribute them. "
        "Return JSON like: { \"count\": 5 }, or something relevant. "
    ),
    model="o3-mini",
    model_settings=ModelSettings(temperature=0.0)  # Generally straightforward logic
)

class GeopoliticalSystemManager(BaseLoreManager):
    """
    Manager for geopolitical content, including regions, nations, and political entities.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "geopolitical"

    async def _initialize_tables(self):
        """Initialize necessary tables."""
        table_definitions = {
            "GeographicRegions": """
                CREATE TABLE GeographicRegions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    region_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    climate TEXT,
                    resources TEXT[],
                    governing_faction TEXT,
                    population_density TEXT,
                    major_settlements TEXT[],
                    cultural_traits TEXT[],
                    dangers TEXT[],
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_geographicregions_embedding 
                ON GeographicRegions USING ivfflat (embedding vector_cosine_ops);
            """,
            "PoliticalEntities": """
                CREATE TABLE PoliticalEntities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    region_id INTEGER,
                    governance_style TEXT,
                    leadership_structure TEXT,
                    population_scale TEXT,
                    cultural_identity TEXT,
                    economic_focus TEXT[],
                    political_values TEXT[],
                    matriarchy_level INTEGER CHECK (matriarchy_level BETWEEN 1 AND 10),
                    relations JSONB,
                    embedding VECTOR(1536),
                    FOREIGN KEY (region_id) REFERENCES GeographicRegions(id) ON DELETE SET NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_politicalentities_embedding 
                ON PoliticalEntities USING ivfflat (embedding vector_cosine_ops);
            """
        }
        await self.initialize_tables_for_class(table_definitions)

    async def ensure_initialized(self):
        """Ensure system is fully initialized with necessary tables."""
        if not self.initialized:
            await super().ensure_initialized()
            await self._initialize_tables()

    # ------------------------------------------------------------------------
    # 1) Add geographic region
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_geographic_region",
        action_description="Adding geographic region: {name}",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    @function_tool
    async def add_geographic_region(
        self, 
        ctx,
        name: str,
        region_type: str,
        description: str,
        climate: Optional[str] = None,
        resources: Optional[List[str]] = None,
        governing_faction: Optional[str] = None,
        population_density: Optional[str] = None,
        major_settlements: Optional[List[str]] = None,
        cultural_traits: Optional[List[str]] = None,
        dangers: Optional[List[str]] = None
    ) -> int:
        """
        Add a geographic region to the database (as a function tool).
        """

        await self.ensure_initialized()

        resources = resources or []
        major_settlements = major_settlements or []
        cultural_traits = cultural_traits or []
        dangers = dangers or []

        # Apply matriarchal theming
        description = MatriarchalThemingUtils.apply_matriarchal_theme("region", description)

        # Generate embedding
        embedding_text = f"{name} {region_type} {description} {climate or ''}"
        embedding = await generate_embedding(embedding_text)

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                region_id = await conn.fetchval("""
                    INSERT INTO GeographicRegions (
                        name, region_type, description, climate, resources,
                        governing_faction, population_density, major_settlements,
                        cultural_traits, dangers, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING id
                """,
                name, region_type, description, climate, resources,
                governing_faction, population_density, major_settlements,
                cultural_traits, dangers, embedding)
                
                return region_id

    # ------------------------------------------------------------------------
    # 2) Generate world nations
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world_nations",
        action_description="Generating world nations",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    async def generate_world_nations(self, ctx, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a set of nations for the world (with LLM-based approach).
        By default, we accept a 'count', but we can also let the LLM override it if we want.
        """
        run_ctx = RunContextWrapper(context=ctx.context)

        # (Optional) Let an agent decide how many nations to generate
        dist_prompt = (
            "We want to create some matriarchal nations. We proposed a default of {count}, but you can override. "
            "Return JSON with a 'count' field. e.g. {\"count\": 5}"
        ).format(count=count)

        dist_config = RunConfig(workflow_name="NationDistribution")
        dist_result = await Runner.run(distribution_agent, dist_prompt, context=run_ctx.context, run_config=dist_config)

        try:
            dist_data = json.loads(dist_result.final_output)
            count = dist_data.get("count", count)
        except json.JSONDecodeError:
            pass  # fallback to the existing 'count'

        # Gather context from DB
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                regions = await conn.fetch("""
                    SELECT id, name, region_type, climate, resources
                    FROM GeographicRegions
                    LIMIT 10
                """)
                region_data = [dict(r) for r in regions]

                # Some cultural elements
                cultures = await conn.fetch("""
                    SELECT name, element_type, description
                    FROM CulturalElements
                    LIMIT 8
                """)
                culture_data = [dict(c) for c in cultures]

        # Create agent for nation generation
        nation_agent = Agent(
            name="NationGenerationAgent",
            instructions="You create detailed nations for matriarchal fantasy worlds.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )

        generated_nations = []
        for i in range(count):
            # We'll keep your partial random logic for matriarchy range
            if i < count // 3:
                matriarchy_range = (8, 10)
            elif i < 2 * count // 3:
                matriarchy_range = (6, 8)
            else:
                matriarchy_range = (4, 7)
            matriarchy_level = random.randint(*matriarchy_range)

            # Build prompt
            prompt = f"""
            Generate a detailed nation for a fantasy world with predominantly matriarchal societies.

            GEOGRAPHIC CONTEXT (sample):
            {json.dumps(random.sample(region_data, min(3, len(region_data))), indent=2)}

            CULTURAL CONTEXT (sample):
            {json.dumps(random.sample(culture_data, min(2, len(culture_data))), indent=2)}

            This nation has a matriarchy level of {matriarchy_level}/10.

            Provide:
            - name
            - government_type
            - description (emphasizing feminine power)
            - relative_power (1-10)
            - matriarchy_level = {matriarchy_level}
            - population_scale
            - major_resources
            - major_cities
            - cultural_traits
            - notable_features
            """

            run_config = RunConfig(workflow_name="NationGen")
            result = await Runner.run(nation_agent, prompt, context=run_ctx.context, run_config=run_config)
            try:
                nation_data = json.loads(result.final_output)
                nation_data["matriarchy_level"] = matriarchy_level

                # Thematic
                if "description" in nation_data:
                    emphasis = matriarchy_level // 3
                    nation_data["description"] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "nation", nation_data["description"], emphasis_level=emphasis
                    )

                # Integrate with your politics manager
                from lore.managers.politics import WorldPoliticsManager
                politics_manager = WorldPoliticsManager(self.user_id, self.conversation_id)
                await politics_manager.ensure_initialized()

                # Add the nation
                nation_id = await politics_manager.add_nation(
                    run_ctx,
                    name=nation_data.get("name", f"Nation_{i+1}"),
                    government_type=nation_data.get("government_type", "matriarchy"),
                    description=nation_data.get("description",""),
                    relative_power=nation_data.get("relative_power", random.randint(3,8)),
                    matriarchy_level=nation_data["matriarchy_level"],
                    population_scale=nation_data.get("population_scale"),
                    major_resources=nation_data.get("major_resources", []),
                    major_cities=nation_data.get("major_cities", []),
                    cultural_traits=nation_data.get("cultural_traits", []),
                    notable_features=nation_data.get("notable_features", [])
                )
                nation_data["id"] = nation_id
                generated_nations.append(nation_data)

            except Exception as e:
                logger.error(f"Error generating nation: {e}")

        return generated_nations

    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="geopolitical_manager",
            directive_text=(
                "Create and manage geographic regions and political entities "
                "for the matriarchal world."
            ),
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
