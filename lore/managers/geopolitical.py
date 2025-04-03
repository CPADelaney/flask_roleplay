# lore/managers/geopolitical.py

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
from lore.utils.theming import MatriarchalThemingUtils

class GeopoliticalSystemManager(BaseLoreManager):
    """
    Manager for geopolitical content, including regions, nations, and political entities.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "geopolitical"
    
    async def _initialize_tables(self):
        """Initialize necessary tables"""
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_geographic_region",
        action_description="Adding geographic region: {name}",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
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
        Add a geographic region to the database
        
        Args:
            name: Name of the region
            region_type: Type of region (mountain range, forest, etc.)
            description: Detailed description
            climate: Climate description
            resources: Available resources
            governing_faction: Who controls the region
            population_density: How populated the region is
            major_settlements: Significant settlements in the region
            cultural_traits: Cultural characteristics of the region
            dangers: Hazards in the region
            
        Returns:
            ID of the created region
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        resources = resources or []
        major_settlements = major_settlements or []
        cultural_traits = cultural_traits or []
        dangers = dangers or []
        
        # Apply matriarchal theming 
        description = MatriarchalThemingUtils.apply_matriarchal_theme("region", description)
        
        # Generate embedding
        embedding_text = f"{name} {region_type} {description} {climate or ''}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
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
                """, name, region_type, description, climate, resources,
                     governing_faction, population_density, major_settlements,
                     cultural_traits, dangers, embedding)
                
                return region_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world_nations",
        action_description="Generating world nations",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    async def generate_world_nations(self, ctx, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a set of nations for the world
        
        Args:
            count: Number of nations to generate
            
        Returns:
            List of generated nations
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Get context data
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get geographic regions for context
                regions = await conn.fetch("""
                    SELECT id, name, region_type, climate, resources
                    FROM GeographicRegions
                    LIMIT 10
                """)
                
                region_data = [dict(region) for region in regions]
                
                # Get cultural elements for context
                cultures = await conn.fetch("""
                    SELECT name, element_type, description
                    FROM CulturalElements
                    LIMIT 8
                """)
                
                culture_data = [dict(culture) for culture in cultures]
        
        # Create agent for nation generation
        nation_agent = Agent(
            name="NationGenerationAgent",
            instructions="You create detailed nations for matriarchal fantasy worlds.",
            model="o3-mini"
        )
        
        generated_nations = []
        for i in range(count):
            # Determine matriarchy level - ensure spectrum of levels but bias toward high
            if i < count // 3:  # First third - very high matriarchy
                matriarchy_range = (8, 10)
            elif i < 2 * count // 3:  # Middle third - high matriarchy
                matriarchy_range = (6, 8)
            else:  # Final third - moderate to high matriarchy
                matriarchy_range = (4, 7)
                
            matriarchy_level = random.randint(*matriarchy_range)
            
            # Create prompt for the agent
            prompt = f"""
            Generate a detailed nation for a fantasy world with predominantly matriarchal societies.
            
            GEOGRAPHIC CONTEXT:
            {json.dumps(random.sample(region_data, min(3, len(region_data))), indent=2)}
            
            CULTURAL CONTEXT:
            {json.dumps(random.sample(culture_data, min(2, len(culture_data))), indent=2)}
            
            This nation should have a matriarchy level of {matriarchy_level}/10, where 10 is complete female dominance
            and 1 would be equal gender roles (no nation is below 4).
            
            Create a nation that:
            1. Has a unique identity and government structure
            2. Reflects its matriarchy level in its social systems
            3. Has distinctive resources and cultural traits
            4. Could realistically exist in the world context
            
            Return a JSON object with:
            - name: Name of the nation
            - government_type: Type of government system
            - description: Detailed description emphasizing feminine power structures
            - relative_power: Military/economic power (1-10)
            - matriarchy_level: {matriarchy_level}
            - population_scale: Rough population size
            - major_resources: Array of key resources
            - major_cities: Array of important settlements
            - cultural_traits: Array of defining cultural elements
            - notable_features: Any other distinctive elements
            """
            
            # Get response from agent
            result = await Runner.run(nation_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse the JSON response
                nation_data = json.loads(result.final_output)
                
                # Set matriarchy level explicitly
                nation_data["matriarchy_level"] = matriarchy_level
                
                # Apply matriarchal theming
                if "description" in nation_data:
                    nation_data["description"] = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "nation", nation_data["description"], emphasis_level=matriarchy_level // 3
                    )
                
                # Add to world_politics manager
                from lore.managers.politics import WorldPoliticsManager
                politics_manager = WorldPoliticsManager(self.user_id, self.conversation_id)
                await politics_manager.ensure_initialized()
                
                # Add the nation
                nation_id = await politics_manager.add_nation(
                    run_ctx,
                    name=nation_data.get("name", f"Nation {i+1}"),
                    government_type=nation_data.get("government_type", "matriarchy"),
                    description=nation_data.get("description", ""),
                    relative_power=nation_data.get("relative_power", random.randint(3, 8)),
                    matriarchy_level=nation_data.get("matriarchy_level", matriarchy_level),
                    population_scale=nation_data.get("population_scale"),
                    major_resources=nation_data.get("major_resources"),
                    major_cities=nation_data.get("major_cities"),
                    cultural_traits=nation_data.get("cultural_traits"),
                    notable_features=nation_data.get("notable_features")
                )
                
                # Add ID to data and add to results
                nation_data["id"] = nation_id
                generated_nations.append(nation_data)
                
            except Exception as e:
                logging.error(f"Error generating nation: {e}")
        
        return generated_nations
        
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="geopolitical_manager",
            directive_text="Create and manage geographic regions and political entities for the matriarchal world.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
