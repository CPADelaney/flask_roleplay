# lore/managers/geopolitical.py

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Union, Tuple
from pydantic import BaseModel, Field

# Agents SDK imports
from agents import (
    Agent, function_tool, Runner, trace, RunResultStreaming,
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, handoff, ModelSettings
)
from agents.run_context import RunContextWrapper
from agents.run import RunConfig

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.core.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils
from lore.core.tool_registry import registered_tool, tool_registry

logger = logging.getLogger(__name__)

# Pydantic models for structured outputs
class GeographicRegion(BaseModel):
    """Model for geographic regions."""
    name: str
    region_type: str
    description: str
    climate: Optional[str] = None
    resources: List[str] = []
    governing_faction: Optional[str] = None
    population_density: Optional[str] = None
    major_settlements: List[str] = []
    cultural_traits: List[str] = []
    dangers: List[str] = []
    
    # New fields for enhanced functionality
    terrain_features: List[str] = []
    defensive_characteristics: Optional[str] = None
    strategic_value: int = Field(5, ge=1, le=10)
    matriarchal_influence: int = Field(5, ge=1, le=10)

class PoliticalEntity(BaseModel):
    """Model for political entities."""
    name: str
    entity_type: str
    description: str
    region_id: Optional[int] = None
    governance_style: str
    leadership_structure: str
    population_scale: str
    cultural_identity: str
    economic_focus: List[str] = []
    political_values: List[str] = []
    matriarchy_level: int = Field(..., ge=1, le=10)
    relations: Dict[str, Any] = {}
    
    # New fields for enhanced functionality
    military_strength: int = Field(5, ge=1, le=10)
    diplomatic_stance: str
    internal_conflicts: List[str] = []
    power_centers: List[Dict[str, Any]] = []

class BorderDispute(BaseModel):
    """Model for border disputes between regions/entities."""
    region1_id: int
    region2_id: int
    dispute_type: str
    description: str
    severity: int = Field(..., ge=1, le=10)
    duration: str
    causal_factors: List[str]
    status: str
    resolution_attempts: List[Dict[str, Any]] = []
    strategic_implications: str
    
    # Matriarchal elements
    female_leaders_involved: List[str]
    gender_dynamics: str

class ConflictSimulation(BaseModel):
    """Model for conflict simulation results."""
    conflict_type: str
    primary_actors: List[Dict[str, Any]]
    timeline: List[Dict[str, str]]
    intensity_progression: List[int]
    diplomatic_events: List[Dict[str, Any]]
    military_events: List[Dict[str, Any]]
    civilian_impact: Dict[str, Any]
    resolution_scenarios: List[Dict[str, Any]]
    most_likely_outcome: Dict[str, Any]
    
    # Simulation metadata
    duration_months: int
    confidence_level: int = Field(..., ge=1, le=10)
    simulation_basis: str

class GeopoliticalSystemManager(BaseLoreManager):
    """
    Enhanced manager for geopolitical content with conflict simulation,
    region-specific handoffs, time evolution, and border dispute resolution.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "geopolitical"
        
        # Initialize specialized agents
        self._init_specialized_agents()
    
    def _init_specialized_agents(self):
        """Initialize specialized agents for different tasks."""
        # Distribution agent for determining counts
        self.distribution_agent = Agent(
            name="GeopoliticalDistributionAgent",
            instructions=(
                "Given context about the world, you decide how many geopolitical items to generate, or how to distribute them. "
                "Return JSON like: { \"count\": 5 }, or something relevant. "
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.0)
        )
        
        # Specialized agents for different region types
        self.mountainous_region_agent = Agent(
            name="MountainousRegionAgent",
            instructions=(
                "You specialize in creating mountainous regions with all their unique characteristics. "
                "Consider defensibility, resources, isolation factors, and how matriarchal societies "
                "would develop in such terrain."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        self.coastal_region_agent = Agent(
            name="CoastalRegionAgent",
            instructions=(
                "You specialize in creating coastal regions with their unique characteristics. "
                "Consider trade, naval power, resources, and how matriarchal societies "
                "would leverage maritime advantages."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        self.plains_region_agent = Agent(
            name="PlainsRegionAgent",
            instructions=(
                "You specialize in creating plains and lowland regions with their unique characteristics. "
                "Consider agriculture, mobility, defensibility challenges, and how matriarchal societies "
                "would organize in open territories."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Conflict simulation agent
        self.conflict_simulation_agent = Agent(
            name="ConflictSimulationAgent",
            instructions=(
                "You simulate realistic conflicts between political entities or regions. "
                "Consider military capabilities, terrain, diplomatic factors, leadership quality, "
                "and social cohesion. Produce detailed timelines with multiple possible outcomes."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7)
        )
        
        # Border dispute resolution agent
        self.border_resolution_agent = Agent(
            name="BorderDisputeResolutionAgent",
            instructions=(
                "You specialize in analyzing and resolving border disputes between regions or nations. "
                "Consider historical claims, resources, strategic value, cultural factors, and diplomatic options. "
                "Suggest multiple resolution paths with their implications."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7)
        )
        
        # Time evolution prediction agent
        self.time_evolution_agent = Agent(
            name="GeopoliticalEvolutionAgent",
            instructions=(
                "You predict how geopolitical situations will evolve over time. "
                "Consider trends, leadership shifts, resource pressures, cultural factors, and external influences. "
                "Show how matriarchal power structures might develop or respond to changes."
            ),
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        # Create a region agent with handoffs
        self.region_agent = Agent(
            name="RegionCreationAgent",
            instructions="You create detailed geographic regions, adapting to terrain types.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.9),
            handoffs=[
                handoff(
                    self.mountainous_region_agent,
                    tool_name_override="create_mountainous_region",
                    tool_description_override="Create a detailed mountainous region with all characteristics"
                ),
                handoff(
                    self.coastal_region_agent,
                    tool_name_override="create_coastal_region",
                    tool_description_override="Create a detailed coastal region with all characteristics"
                ),
                handoff(
                    self.plains_region_agent,
                    tool_name_override="create_plains_region",
                    tool_description_override="Create a detailed plains region with all characteristics"
                )
            ]
        )
        self.trade_modeling_agent = Agent(
            name="EconomicTradeModelingAgent",
            instructions="Simulate trade relations between two nations, considering trade routes, goods, and economic impact.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7),
            tools=[simulate_trade]  # Add the function tool directly
        )
        
        self.geography_effect_agent = Agent(
            name="ClimateGeographyEffectAgent",
            instructions="Simulate the effects of climate and geography on political development.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.7),
            tools=[simulate_geography_impact]  # Add the function tool directly
        )
        
        self.covert_operations_agent = Agent(
            name="CovertOperationsSimulator",
            instructions="Simulate espionage and covert operations between nations.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8),
            tools=[simulate_espionage]  # Add the function tool directly
        )


    async def _initialize_tables(self):
        """Initialize necessary tables with enhanced fields."""
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
                    terrain_features TEXT[],
                    defensive_characteristics TEXT,
                    strategic_value INTEGER CHECK (strategic_value BETWEEN 1 AND 10),
                    matriarchal_influence INTEGER CHECK (matriarchal_influence BETWEEN 1 AND 10),
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
                    military_strength INTEGER CHECK (military_strength BETWEEN 1 AND 10),
                    diplomatic_stance TEXT,
                    internal_conflicts TEXT[],
                    power_centers JSONB,
                    embedding VECTOR(1536),
                    FOREIGN KEY (region_id) REFERENCES GeographicRegions(id) ON DELETE SET NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_politicalentities_embedding 
                ON PoliticalEntities USING ivfflat (embedding vector_cosine_ops);
            """,
            "BorderDisputes": """
                CREATE TABLE BorderDisputes (
                    id SERIAL PRIMARY KEY,
                    region1_id INTEGER NOT NULL,
                    region2_id INTEGER NOT NULL,
                    dispute_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                    duration TEXT NOT NULL,
                    causal_factors TEXT[],
                    status TEXT NOT NULL,
                    resolution_attempts JSONB,
                    strategic_implications TEXT,
                    female_leaders_involved TEXT[],
                    gender_dynamics TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (region1_id) REFERENCES GeographicRegions(id) ON DELETE CASCADE,
                    FOREIGN KEY (region2_id) REFERENCES GeographicRegions(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_borderdisputes_embedding 
                ON BorderDisputes USING ivfflat (embedding vector_cosine_ops);
            """,
            "ConflictSimulations": """
                CREATE TABLE ConflictSimulations (
                    id SERIAL PRIMARY KEY,
                    conflict_type TEXT NOT NULL,
                    primary_actors JSONB NOT NULL,
                    timeline JSONB NOT NULL,
                    intensity_progression INTEGER[],
                    diplomatic_events JSONB,
                    military_events JSONB,
                    civilian_impact JSONB,
                    resolution_scenarios JSONB,
                    most_likely_outcome JSONB,
                    duration_months INTEGER,
                    confidence_level INTEGER CHECK (confidence_level BETWEEN 1 AND 10),
                    simulation_basis TEXT,
                    simulation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_conflictsimulations_embedding 
                ON ConflictSimulations USING ivfflat (embedding vector_cosine_ops);
            """
        }
        await self.initialize_tables_from_definitions(table_definitions)

    async def ensure_initialized(self):
        """Ensure system is fully initialized with necessary tables."""
        if not self.initialized:
            await super().ensure_initialized()
            await self._initialize_tables()

    # ------------------------------------------------------------------------
    # 1) Add geographic region with specialized handoffs
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_geographic_region",
        action_description="Adding geographic region: {name}",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    @registered_tool(category="geopolitical")
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
        dangers: Optional[List[str]] = None,
        terrain_features: Optional[List[str]] = None,
        defensive_characteristics: Optional[str] = None,
        strategic_value: int = 5,
        matriarchal_influence: int = 5
    ) -> int:
        """
        Add a geographic region to the database with enhanced characteristics.
        
        Args:
            ctx: Context object
            name: Region name
            region_type: Type of region
            description: Description text
            climate: Climate details
            resources: Available resources
            governing_faction: Who governs the region
            population_density: Density description
            major_settlements: Major settlements in the region
            cultural_traits: Cultural characteristics
            dangers: Dangers in the region
            terrain_features: Notable terrain features
            defensive_characteristics: Defensive advantages/disadvantages
            strategic_value: Strategic importance (1-10)
            matriarchal_influence: Level of matriarchal influence (1-10)
            
        Returns:
            ID of the created region
        """
        with trace(
            "AddGeographicRegion", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "region_name": name}
        ):
            await self.ensure_initialized()

            resources = resources or []
            major_settlements = major_settlements or []
            cultural_traits = cultural_traits or []
            dangers = dangers or []
            terrain_features = terrain_features or []

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
                            cultural_traits, dangers, terrain_features,
                            defensive_characteristics, strategic_value,
                            matriarchal_influence, embedding
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        RETURNING id
                    """,
                    name, region_type, description, climate, resources,
                    governing_faction, population_density, major_settlements,
                    cultural_traits, dangers, terrain_features,
                    defensive_characteristics, strategic_value,
                    matriarchal_influence, embedding)
                    
                    return region_id

    # ------------------------------------------------------------------------
    # 2) Generate world nations with agent-based distribution
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world_nations",
        action_description="Generating world nations",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    @registered_tool(category="geopolitical")
    async def generate_world_nations(self, ctx, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a set of nations for the world with political simulation capabilities.
        
        Args:
            ctx: Context object
            count: Number of nations to generate
            
        Returns:
            List of generated nations
        """
        with trace(
            "GenerateWorldNations", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "count": count}
        ):
            run_ctx = self.create_run_context(ctx)

            # (Optional) Let an agent decide how many nations to generate
            dist_prompt = (
                "We want to create some matriarchal nations. We proposed a default of {count}, but you can override. "
                "Return JSON with a 'count' field. e.g. {\"count\": 5}"
            ).format(count=count)

            dist_config = RunConfig(workflow_name="NationDistribution")
            dist_result = await Runner.run(
                self.distribution_agent, 
                dist_prompt, 
                context=run_ctx.context, 
                run_config=dist_config
            )

            try:
                dist_data = json.loads(dist_result.final_output)
                count = dist_data.get("count", count)
            except json.JSONDecodeError:
                pass  # fallback to the existing 'count'

            # Gather context from DB
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    regions = await conn.fetch("""
                        SELECT id, name, region_type, climate, resources,
                               terrain_features, strategic_value, matriarchal_influence
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

            # Create agent for nation generation with structured output
            nation_agent = Agent(
                name="NationGenerationAgent",
                instructions="You create detailed nations for matriarchal fantasy worlds.",
                model="o3-mini",
                model_settings=ModelSettings(temperature=0.9),
                output_type=List[PoliticalEntity]
            )

            # Build prompt
            prompt = f"""
            Generate {count} detailed nations for a fantasy world with predominantly matriarchal societies.

            GEOGRAPHIC CONTEXT (sample):
            {json.dumps(region_data[:3], indent=2)}

            CULTURAL CONTEXT (sample):
            {json.dumps(culture_data[:2], indent=2)}

            Create nations with:
            1. Varied matriarchy levels (higher = more female-dominated)
            2. Distinct governance styles and leadership structures
            3. Different military capabilities and diplomatic stances
            4. Internal power centers and conflicts
            5. Economic and cultural characteristics
            
            Return {count} PoliticalEntity objects with all required fields.
            """

            run_config = RunConfig(workflow_name="NationGen")
            result = await Runner.run(nation_agent, prompt, context=run_ctx.context, run_config=run_config)
            
            nations = result.final_output
            generated_nations = []
            
            # Insert nations into database
            for nation in nations:
                try:
                    # Assign to a region if possible
                    region_id = None
                    if regions:
                        region = random.choice(regions)
                        region_id = region["id"]
                        nation.region_id = region_id
                    
                    # Apply matriarchal theming to description
                    emphasis = nation.matriarchy_level // 3
                    nation.description = MatriarchalThemingUtils.apply_matriarchal_theme(
                        "nation", nation.description, emphasis_level=emphasis
                    )
                    
                    # Insert into DB
                    embed_text = f"{nation.name} {nation.entity_type} {nation.description}"
                    embedding = await generate_embedding(embed_text)
                    
                    async with self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            nation_id = await conn.fetchval("""
                                INSERT INTO PoliticalEntities (
                                    name, entity_type, description, region_id,
                                    governance_style, leadership_structure, population_scale,
                                    cultural_identity, economic_focus, political_values,
                                    matriarchy_level, relations, military_strength,
                                    diplomatic_stance, internal_conflicts, power_centers,
                                    embedding
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                                RETURNING id
                            """,
                            nation.name,
                            nation.entity_type,
                            nation.description,
                            nation.region_id,
                            nation.governance_style,
                            nation.leadership_structure,
                            nation.population_scale,
                            nation.cultural_identity,
                            nation.economic_focus,
                            nation.political_values,
                            nation.matriarchy_level,
                            json.dumps(nation.relations),
                            nation.military_strength,
                            nation.diplomatic_stance,
                            nation.internal_conflicts,
                            json.dumps(nation.power_centers),
                            embedding
                            )
                            
                            # Add to result
                            nation_dict = nation.dict()
                            nation_dict["id"] = nation_id
                            generated_nations.append(nation_dict)
                            
                except Exception as e:
                    logger.error(f"Error generating nation: {e}")

            return generated_nations

    # ------------------------------------------------------------------------
    # 3) Conflict simulation between nations/regions
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_conflict",
        action_description="Simulating conflict between entities",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    @registered_tool(category="geopolitical_conflict")
    async def simulate_conflict(
        self,
        ctx,
        entity1_id: int,
        entity2_id: int,
        conflict_type: str,
        duration_months: int = 12
    ) -> ConflictSimulation:
        """
        Simulate a conflict between two political entities or regions.
        
        Args:
            ctx: Context object
            entity1_id: ID of first entity
            entity2_id: ID of second entity
            conflict_type: Type of conflict (war, border_dispute, trade_war, etc.)
            duration_months: Duration to simulate in months
            
        Returns:
            A ConflictSimulation object with detailed simulation results
        """
        with trace(
            "SimulateConflict", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "conflict_type": conflict_type
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Fetch entity details
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    entity1 = await conn.fetchrow("""
                        SELECT * FROM PoliticalEntities WHERE id = $1
                    """, entity1_id)
                    
                    entity2 = await conn.fetchrow("""
                        SELECT * FROM PoliticalEntities WHERE id = $2
                    """, entity2_id)
                    
                    if not entity1 or not entity2:
                        raise ValueError("One or both entities not found")
                    
                    # Fetch regions
                    region1 = None
                    if entity1["region_id"]:
                        region1 = await conn.fetchrow("""
                            SELECT * FROM GeographicRegions WHERE id = $1
                        """, entity1["region_id"])
                    
                    region2 = None
                    if entity2["region_id"]:
                        region2 = await conn.fetchrow("""
                            SELECT * FROM GeographicRegions WHERE id = $1
                        """, entity2["region_id"])
            
            # Parse relations and other JSON fields
            entity1_data = dict(entity1)
            entity2_data = dict(entity2)
            
            for entity in [entity1_data, entity2_data]:
                if "relations" in entity and entity["relations"]:
                    try:
                        entity["relations"] = json.loads(entity["relations"])
                    except:
                        entity["relations"] = {}
                
                if "power_centers" in entity and entity["power_centers"]:
                    try:
                        entity["power_centers"] = json.loads(entity["power_centers"])
                    except:
                        entity["power_centers"] = []
            
            # Prepare region data
            region1_data = dict(region1) if region1 else {}
            region2_data = dict(region2) if region2 else {}
            
            # Determine relative advantages
            military_advantage = entity1_data.get("military_strength", 5) - entity2_data.get("military_strength", 5)
            diplomatic_advantage = 0
            if entity1_data.get("diplomatic_stance") == "aggressive" and entity2_data.get("diplomatic_stance") == "peaceful":
                diplomatic_advantage = 2
            elif entity1_data.get("diplomatic_stance") == "peaceful" and entity2_data.get("diplomatic_stance") == "aggressive":
                diplomatic_advantage = -2
            
            # Calculate terrain advantages if regions are present
            terrain_advantage = 0
            if region1 and region2:
                if region1_data.get("defensive_characteristics") and "easily defensible" in region1_data["defensive_characteristics"].lower():
                    terrain_advantage += 1
                if region2_data.get("defensive_characteristics") and "easily defensible" in region2_data["defensive_characteristics"].lower():
                    terrain_advantage -= 1
            
            # Create simulation prompt
            simulation_prompt = f"""
            Simulate a {conflict_type} conflict between these two political entities over {duration_months} months:
            
            ENTITY 1:
            {json.dumps(entity1_data, indent=2)}
            
            ENTITY 2:
            {json.dumps(entity2_data, indent=2)}
            
            REGION 1 (ENTITY 1):
            {json.dumps(region1_data, indent=2) if region1_data else "No specific region"}
            
            REGION 2 (ENTITY 2):
            {json.dumps(region2_data, indent=2) if region2_data else "No specific region"}
            
            ADVANTAGES:
            Military advantage: {military_advantage} (positive = Entity 1 advantage)
            Diplomatic advantage: {diplomatic_advantage} (positive = Entity 1 advantage)
            Terrain advantage: {terrain_advantage} (positive = Entity 1 advantage)
            
            Simulate the conflict progression with:
            1. Monthly timeline of key events
            2. Diplomatic and military developments
            3. Civilian impacts
            4. Multiple possible resolution scenarios
            5. Most likely outcome
            
            Consider matriarchal leadership dynamics and gender factors in the simulation.
            
            Return a ConflictSimulation object with all required fields.
            """
            
            # Run the simulation
            conflict_agent = self.conflict_simulation_agent.clone(
                output_type=ConflictSimulation
            )
            
            run_config = RunConfig(
                workflow_name="ConflictSimulation",
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "entity1_id": str(entity1_id),
                    "entity2_id": str(entity2_id)
                }
            )
            
            result = await Runner.run(
                conflict_agent, 
                simulation_prompt, 
                context=run_ctx.context,
                run_config=run_config
            )
            
            simulation = result.final_output
            
            # Store the simulation in the database
            try:
                embed_text = f"{conflict_type} {entity1_data['name']} {entity2_data['name']} {simulation.most_likely_outcome.get('description', '')}"
                embedding = await generate_embedding(embed_text)
                
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        sim_id = await conn.fetchval("""
                            INSERT INTO ConflictSimulations (
                                conflict_type, primary_actors, timeline, intensity_progression,
                                diplomatic_events, military_events, civilian_impact,
                                resolution_scenarios, most_likely_outcome, duration_months,
                                confidence_level, simulation_basis, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                            RETURNING id
                        """,
                        simulation.conflict_type,
                        json.dumps(simulation.primary_actors),
                        json.dumps(simulation.timeline),
                        simulation.intensity_progression,
                        json.dumps(simulation.diplomatic_events),
                        json.dumps(simulation.military_events),
                        json.dumps(simulation.civilian_impact),
                        json.dumps(simulation.resolution_scenarios),
                        json.dumps(simulation.most_likely_outcome),
                        simulation.duration_months,
                        simulation.confidence_level,
                        simulation.simulation_basis,
                        embedding)
                        
                        # If this was a border dispute and involves regions, update border dispute table
                        if conflict_type.lower() in ["border_dispute", "territorial_conflict"] and entity1["region_id"] and entity2["region_id"]:
                            await self._record_border_dispute(
                                conn,
                                entity1["region_id"],
                                entity2["region_id"],
                                conflict_type,
                                simulation.most_likely_outcome.get("description", ""),
                                simulation.intensity_progression[-1] if simulation.intensity_progression else 5,
                                f"{duration_months} months",
                                [f.get("description", "") for f in simulation.primary_actors],
                                simulation.most_likely_outcome.get("status", "active"),
                                [{"attempt": r.get("type", ""), "outcome": r.get("result", "")} for r in simulation.resolution_scenarios],
                                simulation.most_likely_outcome.get("strategic_implications", ""),
                                [],  # female_leaders_involved
                                "Matriarchal leadership directs the conflict"
                            )
            except Exception as e:
                logger.error(f"Error storing conflict simulation: {e}")
            
            return simulation
    
    async def _record_border_dispute(
        self,
        conn,
        region1_id: int,
        region2_id: int,
        dispute_type: str,
        description: str,
        severity: int,
        duration: str,
        causal_factors: List[str],
        status: str,
        resolution_attempts: List[Dict[str, Any]],
        strategic_implications: str,
        female_leaders_involved: List[str],
        gender_dynamics: str
    ) -> int:
        """Helper method to record a border dispute in the database."""
        embed_text = f"{dispute_type} {description} {strategic_implications}"
        embedding = await generate_embedding(embed_text)
        
        dispute_id = await conn.fetchval("""
            INSERT INTO BorderDisputes (
                region1_id, region2_id, dispute_type, description,
                severity, duration, causal_factors, status,
                resolution_attempts, strategic_implications,
                female_leaders_involved, gender_dynamics, embedding
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
        """,
        region1_id, region2_id, dispute_type, description,
        severity, duration, causal_factors, status,
        json.dumps(resolution_attempts), strategic_implications,
        female_leaders_involved, gender_dynamics, embedding)
        
        return dispute_id

    # ------------------------------------------------------------------------
    # 4) Border dispute resolution
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="resolve_border_dispute",
        action_description="Resolving border dispute between regions",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    @registered_tool(category="geopolitical_conflict")
    async def resolve_border_dispute(
        self,
        ctx,
        dispute_id: int,
        resolution_approach: str
    ) -> Dict[str, Any]:
        """
        Analyze and resolve a border dispute between regions.
        
        Args:
            ctx: Context object
            dispute_id: ID of the border dispute
            resolution_approach: Approach to resolution (negotiation, arbitration, force, etc.)
            
        Returns:
            Dictionary with resolution details
        """
        with trace(
            "ResolveBorderDispute", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "dispute_id": dispute_id,
                "approach": resolution_approach
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Fetch dispute details
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    dispute = await conn.fetchrow("""
                        SELECT * FROM BorderDisputes WHERE id = $1
                    """, dispute_id)
                    
                    if not dispute:
                        return {"error": "Border dispute not found"}
                    
                    # Fetch involved regions
                    region1 = await conn.fetchrow("""
                        SELECT * FROM GeographicRegions WHERE id = $1
                    """, dispute["region1_id"])
                    
                    region2 = await conn.fetchrow("""
                        SELECT * FROM GeographicRegions WHERE id = $1
                    """, dispute["region2_id"])
                    
                    if not region1 or not region2:
                        return {"error": "One or both regions not found"}
                    
                    # Fetch political entities in these regions
                    entities1 = await conn.fetch("""
                        SELECT * FROM PoliticalEntities WHERE region_id = $1
                    """, dispute["region1_id"])
                    
                    entities2 = await conn.fetch("""
                        SELECT * FROM PoliticalEntities WHERE region_id = $1
                    """, dispute["region2_id"])
            
            # Parse JSON fields
            dispute_data = dict(dispute)
            if "resolution_attempts" in dispute_data and dispute_data["resolution_attempts"]:
                try:
                    dispute_data["resolution_attempts"] = json.loads(dispute_data["resolution_attempts"])
                except:
                    dispute_data["resolution_attempts"] = []
            
            # Prepare context data
            region1_data = dict(region1)
            region2_data = dict(region2)
            entities1_data = [dict(e) for e in entities1]
            entities2_data = [dict(e) for e in entities2]
            
            # Create resolution prompt
            resolution_prompt = f"""
            Resolve this border dispute using a {resolution_approach} approach:
            
            DISPUTE:
            {json.dumps(dispute_data, indent=2)}
            
            REGION 1:
            {json.dumps(region1_data, indent=2)}
            
            REGION 2:
            {json.dumps(region2_data, indent=2)}
            
            POLITICAL ENTITIES IN REGION 1:
            {json.dumps(entities1_data, indent=2)}
            
            POLITICAL ENTITIES IN REGION 2:
            {json.dumps(entities2_data, indent=2)}
            
            Analyze this dispute and propose a detailed resolution that:
            1. Addresses the core causes
            2. Considers strategic and resource factors
            3. Reflects matriarchal leadership styles and motivations
            4. Details immediate and long-term outcomes
            5. Evaluates stability of the resolution
            
            Return structured JSON with:
            - resolution_method: specific approach used
            - resolution_details: step-by-step process
            - territory_outcome: who gets what
            - concessions: what each side gives up
            - enforcement_mechanisms: how the resolution is enforced
            - stability_rating: 1-10 score for how stable this resolution is
            - matriarchal_factors: how feminine leadership affected the outcome
            """
            
            # Run the resolution
            result = await Runner.run(
                self.border_resolution_agent, 
                resolution_prompt, 
                context=run_ctx.context
            )
            
            try:
                resolution_data = json.loads(result.final_output)
            except json.JSONDecodeError:
                resolution_data = {"raw_output": result.final_output}
            
            # Update the dispute status in the database
            try:
                new_status = "resolved"
                
                if "stability_rating" in resolution_data:
                    stability = resolution_data["stability_rating"]
                    if stability <= 3:
                        new_status = "temporarily_resolved"
                    elif stability >= 8:
                        new_status = "permanently_resolved"
                
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        # Get current resolution attempts
                        current_attempts = dispute_data.get("resolution_attempts", [])
                        
                        # Add new attempt
                        new_attempt = {
                            "date": datetime.now().isoformat(),
                            "approach": resolution_approach,
                            "outcome": resolution_data.get("resolution_method", resolution_approach),
                            "details": resolution_data.get("resolution_details", ""),
                            "stability": resolution_data.get("stability_rating", 5)
                        }
                        
                        current_attempts.append(new_attempt)
                        
                        # Update the dispute
                        await conn.execute("""
                            UPDATE BorderDisputes
                            SET status = $1,
                                resolution_attempts = $2
                            WHERE id = $3
                        """, new_status, json.dumps(current_attempts), dispute_id)
                        
                        # Also update the strategic implications if available
                        if "strategic_implications" in resolution_data:
                            await conn.execute("""
                                UPDATE BorderDisputes
                                SET strategic_implications = $1
                                WHERE id = $2
                            """, resolution_data["strategic_implications"], dispute_id)
                
                resolution_data["status"] = new_status
                resolution_data["dispute_id"] = dispute_id
                return resolution_data
                
            except Exception as e:
                logger.error(f"Error updating border dispute resolution: {e}")
                resolution_data["error"] = str(e)
                return resolution_data

    # ------------------------------------------------------------------------
    # 5) Time evolution prediction
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="predict_geopolitical_evolution",
        action_description="Predicting geopolitical evolution over time",
        id_from_context=lambda ctx: "geopolitical_manager"
    )
    @registered_tool(category="geopolitical_prediction")
    async def predict_geopolitical_evolution(
        self,
        ctx,
        entity_id: int,
        years_forward: int = 5,
        include_events: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Predict the geopolitical evolution of an entity over time, streaming results.
        
        Args:
            ctx: Context object
            entity_id: ID of the political entity
            years_forward: Number of years to predict
            include_events: Whether to include significant events
            
        Yields:
            Evolution updates as they are predicted
        """
        with trace(
            "PredictGeopoliticalEvolution", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "entity_id": entity_id,
                "years_forward": years_forward
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Fetch entity details
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    entity = await conn.fetchrow("""
                        SELECT * FROM PoliticalEntities WHERE id = $1
                    """, entity_id)
                    
                    if not entity:
                        yield {"error": "Political entity not found"}
                        return
                    
                    # Fetch region
                    region = None
                    if entity["region_id"]:
                        region = await conn.fetchrow("""
                            SELECT * FROM GeographicRegions WHERE id = $1
                        """, entity["region_id"])
                    
                    # Fetch related conflicts
                    conflicts = await conn.fetch("""
                        SELECT c.* 
                        FROM ConflictSimulations c
                        JOIN jsonb_array_elements(c.primary_actors) actors ON TRUE
                        WHERE actors->>'id' = $1::text OR actors->>'name' = $2
                        ORDER BY c.simulation_date DESC
                        LIMIT 3
                    """, str(entity_id), entity["name"])
                    
                    # Get neighboring entities
                    neighbors = []
                    if region:
                        neighbors = await conn.fetch("""
                            SELECT e.* 
                            FROM PoliticalEntities e
                            JOIN GeographicRegions g ON e.region_id = g.id
                            WHERE g.id != $1 AND e.id != $2
                            LIMIT 5
                        """, region["id"], entity_id)
            
            # Parse entity data
            entity_data = dict(entity)
            
            for field in ["relations", "power_centers"]:
                if field in entity_data and entity_data[field]:
                    try:
                        entity_data[field] = json.loads(entity_data[field])
                    except:
                        entity_data[field] = {}
            
            # Prepare context data
            region_data = dict(region) if region else {}
            conflict_data = [dict(c) for c in conflicts]
            neighbor_data = [dict(n) for n in neighbors]
            
            # Create a stream for yielding results
            stream = StreamingResponse()
            
            # Initial yield with basic information
            yield {
                "entity_name": entity_data["name"],
                "current_state": {
                    "governance": entity_data["governance_style"],
                    "matriarchy_level": entity_data["matriarchy_level"],
                    "military_strength": entity_data["military_strength"],
                    "diplomatic_stance": entity_data["diplomatic_stance"]
                },
                "years_to_predict": years_forward,
                "prediction_status": "starting"
            }
            
            # Allow a small delay for better streaming experience
            await asyncio.sleep(0.5)
            
            # Create evolution prompt
            evolution_prompt = f"""
            Predict the geopolitical evolution of this political entity over {years_forward} years:
            
            ENTITY:
            {json.dumps(entity_data, indent=2)}
            
            REGION:
            {json.dumps(region_data, indent=2)}
            
            RECENT CONFLICTS:
            {json.dumps(conflict_data, indent=2)}
            
            NEIGHBORING ENTITIES:
            {json.dumps(neighbor_data, indent=2)}
            
            {f'Include significant events and their impacts.' if include_events else ''}
            
            Predict evolution in these areas:
            1. Governance style and leadership
            2. Matriarchal power dynamics
            3. Military capabilities
            4. Diplomatic relationships
            5. Internal stability and conflicts
            6. Economic development
            7. Cultural shifts
            
            Return a year-by-year prediction with key changes and events.
            Format as structured JSON with each year as a separate object.
            """
            
# Create a streaming run configuration
            streaming_config = RunConfig(
                workflow_name="GeopoliticalEvolution",
                chunk_size=150,  # Characters per chunk
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "entity_id": str(entity_id)
                }
            )
            
            # Run the time evolution prediction with streaming
            async for chunk in Runner.stream(
                self.time_evolution_agent,
                evolution_prompt,
                context=run_ctx.context,
                run_config=streaming_config
            ):
                # Process the streamed chunk
                try:
                    # Try to parse any JSON that might be in the chunk
                    # This is challenging as chunks might split JSON
                    # For simplicity, we'll just accumulate and periodically try to parse
                    if "year" in chunk.lower() and ":" in chunk:
                        # This appears to be a year marker
                        year_match = re.search(r'year (\d+)', chunk.lower())
                        if year_match:
                            year = int(year_match.group(1))
                            
                            year_data = {
                                "year": year,
                                "prediction": chunk,
                                "entity_name": entity_data["name"],
                                "prediction_status": "in_progress"
                            }
                            
                            yield year_data
                            
                            # Small delay for better streaming experience
                            await asyncio.sleep(0.2)
                except Exception as e:
                    # Just continue on parsing errors
                    pass
            
            # Final attempt to parse the complete result
            try:
                # Fetch the complete result (not just the streamed chunks)
                complete_evolution = await Runner.run(
                    self.time_evolution_agent, 
                    evolution_prompt, 
                    context=run_ctx.context
                )
                
                # Parse the complete evolution data
                try:
                    evolution_data = json.loads(complete_evolution.final_output)
                    
                    # Final yield with complete data
                    yield {
                        "entity_name": entity_data["name"],
                        "complete_evolution": evolution_data,
                        "prediction_status": "complete"
                    }
                except json.JSONDecodeError:
                    # If we can't parse as JSON, yield the raw text
                    yield {
                        "entity_name": entity_data["name"],
                        "evolution_text": complete_evolution.final_output,
                        "prediction_status": "complete_raw"
                    }
            except Exception as e:
                yield {
                    "entity_name": entity_data["name"],
                    "error": str(e),
                    "prediction_status": "error"
                }

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
# Keep the model classes for data structure
class EconomicTradeModelingAgent(BaseModel):
    """Model for economic trade simulation between nations."""
    nation1: str
    nation2: str
    trade_goods: List[str]
    trade_value: float
    trade_route: str
    impact_on_economy: float  # Positive/negative impact on the economy

class ClimateGeographyEffectAgent(BaseModel):
    """Simulate the political impact of terrain features and climate."""
    region_name: str
    terrain_features: List[str]
    climate_type: str
    resource_availability: List[str]
    political_stability: float

class CovertOperationsSimulator(BaseModel):
    """Model covert operations between political entities."""
    agent_name: str
    target_nation: str
    operation_type: str
    mission_outcome: str
    secrecy_level: int

# Move function tools outside the classes
@function_tool
async def simulate_trade(ctx: RunContextWrapper, model_data: dict) -> str:
    """Simulate the economic impact of trade between two nations."""
    # Create a model instance from the data or use the data directly
    trade_info = {
        "nation1": model_data["nation1"],
        "nation2": model_data["nation2"],
        "trade_goods": model_data["trade_goods"],
        "trade_value": model_data["trade_value"],
        "trade_route": model_data["trade_route"],
        "impact_on_economy": model_data["impact_on_economy"]
    }
    return json.dumps(trade_info)

@function_tool
async def simulate_geography_impact(ctx: RunContextWrapper, region_name: str, terrain_features: List[str], 
                                   climate_type: str, resource_availability: List[str] = None) -> str:
    """Simulate how terrain features and climate affect political stability."""
    if resource_availability is None:
        resource_availability = []
        
    climate_effects = {
        "mountainous": 2,
        "coastal": 1,
        "desert": -1
    }
    terrain_effects = {feature: climate_effects.get(feature, 0) for feature in terrain_features}
    stability_impact = sum(terrain_effects.values()) + (1 if climate_type == "temperate" else 0)
    return f"Political stability in {region_name} is influenced by: {stability_impact}"

@function_tool
async def simulate_espionage(ctx: RunContextWrapper, agent_name: str, target_nation: str,
                            operation_type: str, mission_outcome: str, secrecy_level: int) -> str:
    """Simulate covert operations between nations."""
    espionage_details = {
        "agent_name": agent_name,
        "target_nation": target_nation,
        "operation_type": operation_type,
        "mission_outcome": mission_outcome,
        "secrecy_level": secrecy_level
    }
    return json.dumps(espionage_details)
                                
tool_registry.register_tool(simulate_trade, "economic_trade")
tool_registry.register_tool(simulate_geography_impact, "climate_geography_effect")
tool_registry.register_tool(simulate_espionage, "covert_operations")
