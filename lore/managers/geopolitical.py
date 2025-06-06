# lore/managers/geopolitical.py

import logging
import json
import random
import asyncio
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Union, Tuple
from pydantic import BaseModel, Field

# OpenAI Agents SDK imports
from agents import (
    Agent, function_tool, Runner, trace, RunResultStreaming,
    GuardrailFunctionOutput, InputGuardrail, handoff, ModelSettings
)
from agents.run_context import RunContextWrapper
from agents.run import RunConfig

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager

logger = logging.getLogger(__name__)

async def _analyze_multi_party_dynamics(
    entities_data: Dict[int, Dict[str, Any]], 
    regions_data: Dict[int, Dict[str, Any]],
    alliances: Optional[Dict[str, List[int]]]
) -> Dict[str, Any]:
    """
    Analyze power dynamics for multi-party conflicts.
    
    Returns dict with:
    - power_rankings: Sorted list of entities by total power
    - alliance_strengths: Power of each alliance
    - geographic_clusters: Which entities are geographically close
    - likely_battlefronts: Where conflicts will occur
    """
    power_scores = {}
    
    # Calculate individual power scores
    for entity_id, entity_data in entities_data.items():
        military = entity_data.get("military_strength", 5)
        matriarchy = entity_data.get("matriarchy_level", 5)
        
        # Get terrain bonus from region
        terrain_bonus = 0
        if entity_id in regions_data:
            region = regions_data[entity_id]
            strategic_value = region.get("strategic_value", 5)
            defensive_chars = region.get("defensive_characteristics", "")
            
            terrain_bonus = strategic_value / 10
            if "easily defensible" in defensive_chars.lower():
                terrain_bonus += 0.2
        
        # Calculate total power
        power = military + (matriarchy * 0.3) + terrain_bonus
        power_scores[entity_id] = {
            "entity_name": entity_data["name"],
            "total_power": power,
            "military": military,
            "terrain_advantage": terrain_bonus,
            "government_stability": 10 - len(entity_data.get("internal_conflicts", []))
        }
    
    # Calculate alliance strengths
    alliance_strengths = {}
    if alliances:
        for alliance_name, members in alliances.items():
            total_strength = sum(power_scores[m]["total_power"] for m in members if m in power_scores)
            alliance_strengths[alliance_name] = {
                "members": members,
                "combined_strength": total_strength,
                "member_count": len(members)
            }
    
    # Identify geographic clusters
    geographic_clusters = []
    if regions_data:
        # Group entities by geographic proximity (simplified)
        for entity_id, region in regions_data.items():
            # Find other entities in same or adjacent regions
            cluster = [entity_id]
            for other_id, other_region in regions_data.items():
                if entity_id != other_id:
                    # Simple proximity check - in real implementation would use actual geography
                    if region.get("region_type") == other_region.get("region_type"):
                        cluster.append(other_id)
            
            if len(cluster) > 1:
                geographic_clusters.append(cluster)
    
    return {
        "power_rankings": sorted(power_scores.items(), key=lambda x: x[1]["total_power"], reverse=True),
        "alliance_strengths": alliance_strengths,
        "geographic_clusters": geographic_clusters,
        "unaligned_entities": [
            eid for eid in entities_data.keys()
            if not any(eid in alliance_members for alliance_members in (alliances or {}).values())
        ]
    }

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

class EconomicTradeSimulation(BaseModel):
    """Model for economic trade simulation between nations."""
    nation1: str
    nation2: str
    trade_goods: List[str]
    trade_value: float
    trade_route: str
    impact_on_economy: float  # Positive/negative impact on the economy

class ClimateGeographyEffect(BaseModel):
    """Model for simulating the political impact of terrain features and climate."""
    region_name: str
    terrain_features: List[str]
    climate_type: str
    resource_availability: List[str]
    political_stability: float

class CovertOperation(BaseModel):
    """Model for covert operations between political entities."""
    agent_name: str
    target_nation: str
    operation_type: str
    mission_outcome: str
    secrecy_level: int

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
            model="gpt-4.1-nano",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        self.coastal_region_agent = Agent(
            name="CoastalRegionAgent",
            instructions=(
                "You specialize in creating coastal regions with their unique characteristics. "
                "Consider trade, naval power, resources, and how matriarchal societies "
                "would leverage maritime advantages."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        self.plains_region_agent = Agent(
            name="PlainsRegionAgent",
            instructions=(
                "You specialize in creating plains and lowland regions with their unique characteristics. "
                "Consider agriculture, mobility, defensibility challenges, and how matriarchal societies "
                "would organize in open territories."
            ),
            model="gpt-4.1-nano",
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
            model="gpt-4.1-nano",
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
            model="gpt-4.1-nano",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        # Create a region agent with handoffs
        self.region_agent = Agent(
            name="RegionCreationAgent",
            instructions="You create detailed geographic regions, adapting to terrain types.",
            model="gpt-4.1-nano",
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
        
        # Specialized simulation agents
        self.trade_modeling_agent = Agent(
            name="EconomicTradeModelingAgent",
            instructions="Simulate trade relations between two nations, considering trade routes, goods, and economic impact.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            output_type=EconomicTradeSimulation
        )
        
        self.geography_effect_agent = Agent(
            name="ClimateGeographyEffectAgent",
            instructions="Simulate the effects of climate and geography on political development.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            output_type=ClimateGeographyEffect
        )
        
        self.covert_operations_agent = Agent(
            name="CovertOperationsSimulator",
            instructions="Simulate espionage and covert operations between nations.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
            output_type=CovertOperation
        )

    async def ensure_initialized(self):
        """Ensure system is fully initialized with necessary tables."""
        if not self.initialized:
            await super().ensure_initialized()
            await self._initialize_tables()

    async def _initialize_tables(self):
        """Initialize necessary tables with enhanced fields."""
        table_definitions = {
            "GeographicRegions": """
                CREATE TABLE IF NOT EXISTS GeographicRegions (
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
                CREATE TABLE IF NOT EXISTS PoliticalEntities (
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
                CREATE TABLE IF NOT EXISTS BorderDisputes (
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
                CREATE TABLE IF NOT EXISTS ConflictSimulations (
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
        await self._initialize_tables_for_class_impl(table_definitions)

    # ------------------------------------------------------------------------
    # 1) Add geographic region with specialized handoffs
    # ------------------------------------------------------------------------
    async def _add_geographic_region_impl(
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
        Actual business logic and DB insert for a geographic region.
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
    
            # Apply matriarchal theming if utility available
            try:
                from lore.utils.theming import MatriarchalThemingUtils
                description = MatriarchalThemingUtils.apply_matriarchal_theme("region", description)
            except ImportError:
                # If theming utils not available, continue without theming
                pass
    
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


    @staticmethod
    @function_tool
    async def add_geographic_region(
        ctx: RunContextWrapper,
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
        Add a geographic region to the database using canon system.
        
        Args:
            ctx: Context object
            name: Name of the region
            region_type: Type of region (mountains, coast, desert, etc.)
            description: Description of the region
            [other args...]
            
        Returns:
            ID of the created region
        """
        manager = ctx.context.get("manager")
        if not manager:
            from lore.managers.geopolitical import GeopoliticalSystemManager
            manager = GeopoliticalSystemManager(
                ctx.context.get("user_id"), 
                ctx.context.get("conversation_id")
            )
            await manager.ensure_initialized()
        
        # Get LoreSystem for canon operations
        from lore.core.lore_system import LoreSystem
        lore_system = await LoreSystem.get_instance(
            ctx.context.get("user_id"), 
            ctx.context.get("conversation_id")
        )
        
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor, AgentType
            governor = await NyxUnifiedGovernor(
                ctx.context.get("user_id"), 
                ctx.context.get("conversation_id")
            ).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="add_geographic_region",
                action_details={"name": name, "type": region_type}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            pass
        
        with trace(
            "AddGeographicRegion", 
            group_id=manager.trace_group_id,
            metadata={**manager.trace_metadata, "region_name": name}
        ):
            await manager.ensure_initialized()
            
            # Check for semantic duplicates
            async with get_db_connection_context() as conn:
                # Generate embedding for the new region
                embedding_text = f"{name} {region_type} {description} {climate or ''}"
                search_vector = await generate_embedding(embedding_text)
                
                # Search for similar regions
                similar_region = await conn.fetchrow("""
                    SELECT id, name, region_type, 1 - (embedding <=> $1) AS similarity
                    FROM GeographicRegions
                    WHERE 1 - (embedding <=> $1) > 0.85
                    ORDER BY embedding <=> $1
                    LIMIT 1
                """, search_vector)
                
                if similar_region:
                    # Use validation to check if it's truly a duplicate
                    from lore.core.validation import CanonValidationAgent
                    validation_agent = CanonValidationAgent()
                    
                    existing_region = await conn.fetchrow("""
                        SELECT * FROM GeographicRegions WHERE id = $1
                    """, similar_region['id'])
                    
                    is_duplicate = await validation_agent.confirm_is_duplicate_region(
                        conn,
                        proposal={
                            "name": name,
                            "region_type": region_type,
                            "description": description
                        },
                        existing_region=dict(existing_region)
                    )
                    
                    if is_duplicate:
                        logger.warning(f"Region '{name}' is a duplicate of existing region ID {similar_region['id']}")
                        # Update the existing region with any new information
                        await lore_system.propose_and_enact_change(
                            ctx=ctx,
                            entity_type="GeographicRegions",
                            entity_identifier={"id": similar_region['id']},
                            updates={
                                "resources": resources,
                                "strategic_value": strategic_value,
                                "matriarchal_influence": matriarchal_influence
                            },
                            reason=f"Updating existing region with new information from duplicate submission"
                        )
                        return similar_region['id']
            
            # Canonically establish governing faction if specified
            if governing_faction:
                async with get_db_connection_context() as conn:
                    from lore.core import canon
                    faction_id = await canon.find_or_create_faction(
                        ctx, conn, governing_faction, faction_type="regional_authority"
                    )
            
            # Canonically establish major settlements
            settlement_ids = []
            if major_settlements:
                async with get_db_connection_context() as conn:
                    from lore.core import canon
                    for settlement in major_settlements:
                        location_id = await canon.find_or_create_location(ctx, conn, settlement)
                        settlement_ids.append(location_id)
            
            resources = resources or []
            cultural_traits = cultural_traits or []
            dangers = dangers or []
            terrain_features = terrain_features or []
            
            # Apply matriarchal theming
            try:
                from lore.utils.theming import MatriarchalThemingUtils
                description = MatriarchalThemingUtils.apply_matriarchal_theme("region", description)
            except ImportError:
                pass
            
            # Generate embedding
            embedding = await generate_embedding(embedding_text)
            
            # Create the region
            async with manager.get_connection_pool() as pool:
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
                    
                    # Log canonical event
                    from lore.core import canon
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"Geographic region '{name}' established as {region_type} with strategic value {strategic_value}",
                        tags=["geography", "region", "canon"],
                        significance=8
                    )
                    
                    return region_id
    
    # ------------------------------------------------------------------------
    # 2) Generate world nations with agent-based distribution
    # ------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def generate_world_nations(ctx: RunContextWrapper, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a set of nations for the world with political simulation capabilities.
        
        Args:
            ctx: Context object
            count: Number of nations to generate
            
        Returns:
            List of generated nations
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="generate_world_nations",
                action_details={"count": count}
            )
            
            if not permission.get("approved", True):
                return [{"error": permission.get("reasoning", "Action not permitted by governance")}]
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "GenerateWorldNations", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "count": count}
        ):
            run_ctx = self.create_run_context(ctx)

            # (Optional) Let an agent decide how many nations to generate
            dist_prompt = (
                f"We want to create some matriarchal nations. We proposed a default of {count}, but you can override. "
                "Return JSON with a 'count' field. e.g. {\"count\": 5}"
            )

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

                    # Some cultural elements - try to fetch if table exists
                    try:
                        cultures = await conn.fetch("""
                            SELECT name, element_type, description
                            FROM CulturalElements
                            LIMIT 8
                        """)
                        culture_data = [dict(c) for c in cultures]
                    except:
                        # If table doesn't exist, use default data
                        culture_data = [
                            {"name": "Maternal Lineage", "element_type": "social", 
                             "description": "Society traces lineage through the maternal line"},
                            {"name": "Women's Councils", "element_type": "governance",
                             "description": "Councils of elder women make important decisions"}
                        ]

            # Create agent for nation generation with structured output
            nation_agent = Agent(
                name="NationGenerationAgent",
                instructions="You create detailed nations for matriarchal fantasy worlds.",
                model="gpt-4.1-nano",
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
                    
                    # Apply matriarchal theming to description if utility available
                    try:
                        from lore.utils.theming import MatriarchalThemingUtils
                        emphasis = nation.matriarchy_level // 3
                        nation.description = MatriarchalThemingUtils.apply_matriarchal_theme(
                            "nation", nation.description, emphasis_level=emphasis
                        )
                    except ImportError:
                        # If theming utils not available, continue without theming
                        pass
                    
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
    @staticmethod
    @function_tool(strict=False)  # Disable strict schema to allow flexible alliance structure
    async def simulate_conflict(
        ctx: RunContextWrapper,
        entity_ids: List[int],
        conflict_type: str,
        alliances: Optional[Dict[str, List[int]]] = None,
        duration_months: int = 12
    ) -> Dict[str, Any]:
        """
        Simulate a conflict between multiple political entities or regions.
        Uses canon system to ensure consistency.
        
        Args:
            ctx: Context object
            entity_ids: List of entity IDs involved in the conflict
            conflict_type: Type of conflict (war, border_dispute, trade_war, civil_war, proxy_war, etc.)
            alliances: Optional dict mapping alliance names to lists of entity IDs
                      e.g., {"Northern Alliance": [1, 3], "Southern Pact": [2, 4, 5]}
            duration_months: Duration to simulate in months
            
        Returns:
            A ConflictSimulation object with detailed simulation results
        """
        if len(entity_ids) < 2:
            return {"error": "Conflict requires at least 2 entities"}
        
        manager = ctx.context.get("manager")
        if not manager:
            from lore.managers.geopolitical import GeopoliticalSystemManager
            manager = GeopoliticalSystemManager(
                ctx.context.get("user_id"), 
                ctx.context.get("conversation_id")
            )
            await manager.ensure_initialized()
        
        # Get LoreSystem for canon operations
        from lore.core.lore_system import LoreSystem
        lore_system = await LoreSystem.get_instance(
            ctx.context.get("user_id"), 
            ctx.context.get("conversation_id")
        )
        
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor, AgentType
            governor = await NyxUnifiedGovernor(
                ctx.context.get("user_id"), 
                ctx.context.get("conversation_id")
            ).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="simulate_conflict",
                action_details={
                    "entity_ids": entity_ids, 
                    "conflict_type": conflict_type,
                    "num_parties": len(entity_ids)
                }
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            pass
            
        with trace(
            "SimulateMultiPartyConflict", 
            group_id=manager.trace_group_id,
            metadata={
                **manager.trace_metadata, 
                "entity_ids": entity_ids,
                "num_parties": len(entity_ids),
                "conflict_type": conflict_type,
                "has_alliances": alliances is not None
            }
        ):
            run_ctx = manager.create_run_context(ctx)
            
            # Fetch all entity details
            entities_data = {}
            regions_data = {}
            
            async with manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    for entity_id in entity_ids:
                        entity = await conn.fetchrow("""
                            SELECT * FROM PoliticalEntities WHERE id = $1
                        """, entity_id)
                        
                        if not entity:
                            return {"error": f"Entity {entity_id} not found"}
                        
                        entity_data = dict(entity)
                        
                        # Parse JSON fields
                        if "relations" in entity_data and entity_data["relations"]:
                            try:
                                entity_data["relations"] = json.loads(entity_data["relations"])
                            except:
                                entity_data["relations"] = {}
                        
                        if "power_centers" in entity_data and entity_data["power_centers"]:
                            try:
                                entity_data["power_centers"] = json.loads(entity_data["power_centers"])
                            except:
                                entity_data["power_centers"] = []
                        
                        entities_data[entity_id] = entity_data
                        
                        # Fetch region if exists
                        if entity["region_id"]:
                            region = await conn.fetchrow("""
                                SELECT * FROM GeographicRegions WHERE id = $1
                            """, entity["region_id"])
                            
                            if region:
                                regions_data[entity_id] = dict(region)
            
            # Calculate power dynamics and relationships
            power_analysis = await _analyze_multi_party_dynamics(
                entities_data, regions_data, alliances
            )
            
            # Create simulation prompt
            entities_prompt = "\n\n".join([
                f"ENTITY {eid} - {edata['name']}:\n{json.dumps(edata, indent=2)}"
                for eid, edata in entities_data.items()
            ])
            
            regions_prompt = "\n\n".join([
                f"REGION FOR ENTITY {eid}:\n{json.dumps(rdata, indent=2)}"
                for eid, rdata in regions_data.items()
            ])
            
            alliances_prompt = ""
            if alliances:
                alliances_prompt = "\nALLIANCES:\n" + json.dumps(alliances, indent=2)
            
            simulation_prompt = f"""
            Simulate a {conflict_type} conflict involving {len(entity_ids)} political entities over {duration_months} months:
            
            {entities_prompt}
            
            {regions_prompt}
            
            {alliances_prompt}
            
            POWER DYNAMICS:
            {json.dumps(power_analysis, indent=2)}
            
            Simulate the conflict progression with:
            1. Monthly timeline showing multi-party interactions
            2. Alliance formations and betrayals
            3. Shifting fronts and multiple theaters of war
            4. Diplomatic maneuvers between all parties
            5. Cascading effects on non-aligned entities
            6. Multiple possible resolution scenarios
            7. Most likely outcome for all parties
            
            Consider:
            - Some entities may switch sides
            - New alliances may form during conflict
            - Some parties may seek separate peace
            - Proxy conflicts and client states
            - Matriarchal leadership dynamics
            
            The most_likely_outcome should include:
            - description: Overall description of the outcome
            - status: Complex status for multi-party conflict
            - winner_entities: List of entity IDs that achieved objectives
            - loser_entities: List of entity IDs that failed objectives
            - neutral_outcomes: List of entity IDs with mixed results
            - territorial_changes: List of {region_id, old_controller, new_controller}
            - political_changes: List of {entity_id, change_type, details}
            - alliance_changes: New or broken alliances
            - power_balance_shift: How regional power dynamics changed
            
            Return a ConflictSimulation object with all required fields.
            """
            
            # Run the simulation
            conflict_agent = manager.conflict_simulation_agent.clone(
                output_type=ConflictSimulation
            )
            
            run_config = RunConfig(
                workflow_name="MultiPartyConflictSimulation",
                trace_metadata={
                    "user_id": str(manager.user_id),
                    "conversation_id": str(manager.conversation_id),
                    "entity_ids": str(entity_ids),
                    "num_parties": str(len(entity_ids))
                }
            )
            
            result = await Runner.run(
                conflict_agent, 
                simulation_prompt, 
                context=run_ctx.context,
                run_config=run_config
            )
            
            simulation = result.final_output
            
            # Store the simulation
            try:
                # Create embedding text from all entity names
                entity_names = [entities_data[eid]['name'] for eid in entity_ids]
                embed_text = f"{conflict_type} involving {', '.join(entity_names)} - {simulation.most_likely_outcome.get('description', '')}"
                embedding = await generate_embedding(embed_text)
                
                async with manager.get_connection_pool() as pool:
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
                        conflict_type + f" (Multi-party: {len(entity_ids)} entities)",
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
                
                # Apply canonical changes based on outcome
                if simulation and simulation.most_likely_outcome:
                    outcome = simulation.most_likely_outcome
                    
                    # Apply territorial changes
                    if outcome.get("territorial_changes"):
                        for change in outcome["territorial_changes"]:
                            if change.get("region_id") and change.get("new_controller"):
                                # Verify the region exists
                                async with manager.get_connection_pool() as pool:
                                    async with pool.acquire() as conn:
                                        region_exists = await conn.fetchval("""
                                            SELECT EXISTS(SELECT 1 FROM GeographicRegions WHERE id = $1)
                                        """, change["region_id"])
                                        
                                        if region_exists:
                                            old_controller = change.get("old_controller", "Unknown")
                                            
                                            # Apply the territorial change
                                            await lore_system.propose_and_enact_change(
                                                ctx=ctx,
                                                entity_type="GeographicRegions",
                                                entity_identifier={"id": change["region_id"]},
                                                updates={"governing_faction": change["new_controller"]},
                                                reason=f"Territory transferred from {old_controller} to {change['new_controller']} in multi-party {conflict_type}"
                                            )
                                            
                                            # Log territorial change
                                            from lore.core import canon
                                            await canon.log_canonical_event(
                                                ctx, conn,
                                                f"Region {change['region_id']} control transferred from {old_controller} to {change['new_controller']} following multi-party conflict",
                                                tags=["conflict", "territorial_change", "multi_party", "canon"],
                                                significance=8
                                            )
                    
                    # Apply political changes to multiple entities
                    if outcome.get("political_changes"):
                        for change in outcome["political_changes"]:
                            entity_id = change.get("entity_id")
                            change_type = change.get("change_type", "leadership")
                            details = change.get("details", {})
                            
                            if change_type == "leadership" and details.get("new_leader"):
                                # Create new leader
                                async with get_db_connection_context() as conn:
                                    from lore.core import canon
                                    leader_id = await canon.find_or_create_npc(
                                        ctx, conn,
                                        npc_name=details["new_leader"],
                                        role="Political Leader",
                                        affiliations=[entities_data[entity_id]['name']]
                                    )
                                    
                                    # Update entity
                                    await lore_system.propose_and_enact_change(
                                        ctx=ctx,
                                        entity_type="PoliticalEntities",
                                        entity_identifier={"id": entity_id},
                                        updates={"leader_npc_id": leader_id},
                                        reason=f"Leadership change in {entities_data[entity_id]['name']} due to multi-party conflict"
                                    )
                            
                            elif change_type == "government":
                                # Change government type
                                await lore_system.propose_and_enact_change(
                                    ctx=ctx,
                                    entity_type="PoliticalEntities",
                                    entity_identifier={"id": entity_id},
                                    updates={
                                        "governance_style": details.get("new_style", "transitional"),
                                        "internal_conflicts": [f"Post-conflict transition from {conflict_type}"]
                                    },
                                    reason=f"Government change in {entities_data[entity_id]['name']} following conflict"
                                )
                            
                            elif change_type == "dissolution":
                                # Entity dissolves/splits
                                await lore_system.propose_and_enact_change(
                                    ctx=ctx,
                                    entity_type="PoliticalEntities",
                                    entity_identifier={"id": entity_id},
                                    updates={
                                        "entity_type": "defunct",
                                        "description": entities_data[entity_id]['description'] + f" [Dissolved following {conflict_type}]"
                                    },
                                    reason=f"{entities_data[entity_id]['name']} dissolved in multi-party conflict"
                                )
                    
                    # Update alliance structures
                    if outcome.get("alliance_changes"):
                        for alliance_change in outcome["alliance_changes"]:
                            if alliance_change.get("type") == "new_alliance":
                                members = alliance_change.get("members", [])
                                alliance_name = alliance_change.get("name", "Post-Conflict Alliance")
                                
                                # Create new faction for alliance
                                async with get_db_connection_context() as conn:
                                    from lore.core import canon
                                    alliance_id = await canon.find_or_create_faction(
                                        ctx, conn,
                                        faction_name=alliance_name,
                                        faction_type="military_alliance"
                                    )
                                    
                                    # Update each member's relations
                                    for member_id in members:
                                        if member_id in entities_data:
                                            entity_relations = entities_data[member_id].get("relations", {})
                                            
                                            # Add alliance members to relations
                                            for other_member in members:
                                                if other_member != member_id:
                                                    entity_relations[str(other_member)] = {
                                                        "status": "allied",
                                                        "alliance": alliance_name,
                                                        "since": datetime.now().isoformat()
                                                    }
                                            
                                            await lore_system.propose_and_enact_change(
                                                ctx=ctx,
                                                entity_type="PoliticalEntities",
                                                entity_identifier={"id": member_id},
                                                updates={"relations": json.dumps(entity_relations)},
                                                reason=f"{entities_data[member_id]['name']} joins {alliance_name}"
                                            )
                    
                    # Update power balance for all entities
                    winners = outcome.get("winner_entities", [])
                    losers = outcome.get("loser_entities", [])
                    
                    for entity_id in entity_ids:
                        if entity_id in winners:
                            # Increase military strength and influence
                            new_strength = min(10, entities_data[entity_id].get("military_strength", 5) + 1)
                            await lore_system.propose_and_enact_change(
                                ctx=ctx,
                                entity_type="PoliticalEntities",
                                entity_identifier={"id": entity_id},
                                updates={
                                    "military_strength": new_strength,
                                    "diplomatic_stance": "confident"
                                },
                                reason=f"{entities_data[entity_id]['name']} emerged victorious from conflict"
                            )
                        
                        elif entity_id in losers:
                            # Decrease military strength
                            new_strength = max(1, entities_data[entity_id].get("military_strength", 5) - 2)
                            await lore_system.propose_and_enact_change(
                                ctx=ctx,
                                entity_type="PoliticalEntities",
                                entity_identifier={"id": entity_id},
                                updates={
                                    "military_strength": new_strength,
                                    "diplomatic_stance": "defensive",
                                    "internal_conflicts": entities_data[entity_id].get("internal_conflicts", []) + ["Post-war instability"]
                                },
                                reason=f"{entities_data[entity_id]['name']} weakened by conflict"
                            )
                    
                    # Log the overall conflict as a canonical event
                    async with get_db_connection_context() as conn:
                        from lore.core import canon
                        
                        conflict_summary = (
                            f"Multi-party {conflict_type} involving {', '.join(entity_names)} concluded. "
                            f"Winners: {', '.join([entities_data[w]['name'] for w in winners])}. "
                            f"Losers: {', '.join([entities_data[l]['name'] for l in losers])}."
                        )
                        
                        await canon.log_canonical_event(
                            ctx, conn,
                            conflict_summary + f" {outcome.get('description', '')}",
                            tags=["conflict", "multi_party", "geopolitical", "canon"],
                            significance=10  # Multi-party conflicts are highly significant
                        )
                
                # Return simulation with ID
                sim_dict = simulation.dict()
                sim_dict["id"] = sim_id
                sim_dict["entity_count"] = len(entity_ids)
                sim_dict["alliance_structure"] = alliances
                return sim_dict
                
            except Exception as e:
                logger.error(f"Error storing multi-party conflict simulation: {e}")
                return simulation.dict()
        
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
    @staticmethod
    @function_tool
    async def resolve_border_dispute(
        ctx: RunContextWrapper,
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
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="resolve_border_dispute",
                action_details={"dispute_id": dispute_id}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
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
    @staticmethod
    @function_tool
    async def predict_geopolitical_evolution(
        ctx: RunContextWrapper,
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
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="predict_geopolitical_evolution",
                action_details={"entity_id": entity_id}
            )
            
            if not permission.get("approved", True):
                yield {"error": permission.get("reasoning", "Action not permitted by governance")}
                return
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
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
                trace_metadata={
                    "user_id": str(self.user_id),
                    "conversation_id": str(self.conversation_id),
                    "entity_id": str(entity_id)
                }
            )
            
            # Run the time evolution prediction with streaming
            streaming_result = Runner.run_streamed(
                self.time_evolution_agent,
                evolution_prompt,
                context=run_ctx.context,
                run_config=streaming_config
            )
            
            # Buffer to accumulate content between "Year X" markers
            current_year = None
            buffer = ""
            
            # Process the streamed events
            async for event in streaming_result.stream_events():
                if event.type == "run_item_stream_event":
                    if event.item.type == "message_output_item":
                        # Extract text from the message output
                        from agents.items import ItemHelpers
                        message_text = ItemHelpers.text_message_output(event.item)
                        
                        # Check for year markers
                        year_match = re.search(r'year (\d+)', message_text.lower())
                        if year_match and buffer:
                            # If we're starting a new year, yield the previous year's content
                            if current_year is not None:
                                year_data = {
                                    "year": current_year,
                                    "prediction": buffer.strip(),
                                    "entity_name": entity_data["name"],
                                    "prediction_status": "in_progress"
                                }
                                yield year_data
                                
                                # Small delay for better streaming experience
                                await asyncio.sleep(0.2)
                            
                            # Start accumulating for the new year
                            current_year = int(year_match.group(1))
                            buffer = message_text
                        else:
                            # Continue accumulating
                            buffer += message_text
            
            # Yield any remaining content
            if current_year is not None and buffer:
                year_data = {
                    "year": current_year,
                    "prediction": buffer.strip(),
                    "entity_name": entity_data["name"],
                    "prediction_status": "in_progress"
                }
                yield year_data
            
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

    # ------------------------------------------------------------------------
    # 6) Specialized simulations
    # ------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def simulate_trade(
        ctx: RunContextWrapper,
        nation1: str,
        nation2: str,
        trade_goods: List[str],
        trade_route: str
    ) -> Dict[str, Any]:
        """
        Simulate the economic impact of trade between two nations.
        
        Args:
            ctx: Context object
            nation1: First nation name
            nation2: Second nation name
            trade_goods: List of goods being traded
            trade_route: Route of trade
            
        Returns:
            Economic trade simulation results
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="simulate_trade",
                action_details={"nation1": nation1, "nation2": nation2}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "SimulateTrade",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "nation1": nation1,
                "nation2": nation2
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Get nation information from DB if available
            nations_data = {}
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    for name in [nation1, nation2]:
                        nation = await conn.fetchrow("""
                            SELECT * FROM PoliticalEntities WHERE name ILIKE $1
                        """, f"%{name}%")
                        
                        if nation:
                            nation_data = dict(nation)
                            # Parse JSON fields
                            for field in ["relations", "power_centers"]:
                                if field in nation_data and nation_data[field]:
                                    try:
                                        nation_data[field] = json.loads(nation_data[field])
                                    except:
                                        nation_data[field] = {}
                            
                            nations_data[name] = nation_data
            
            # Prepare prompt
            prompt = f"""
            Simulate trade relations between these two nations:
            
            NATION 1: {nation1}
            {json.dumps(nations_data.get(nation1, {}), indent=2) if nation1 in nations_data else "No additional data available"}
            
            NATION 2: {nation2}
            {json.dumps(nations_data.get(nation2, {}), indent=2) if nation2 in nations_data else "No additional data available"}
            
            TRADE GOODS: {', '.join(trade_goods)}
            TRADE ROUTE: {trade_route}
            
            Analyze:
            1. Economic impact on both nations
            2. Balance of trade (who benefits more)
            3. Strategic implications
            4. Cultural exchange aspects
            5. Impact on matriarchal power structures
            
            Return an EconomicTradeSimulation object with all fields.
            """
            
            # Run the trade simulation
            result = await Runner.run(self.trade_modeling_agent, prompt, context=run_ctx.context)
            trade_sim = result.final_output
            
            return trade_sim.dict()

    @staticmethod
    @function_tool
    async def simulate_geography_impact(
        ctx: RunContextWrapper,
        region_name: str,
        terrain_features: List[str],
        climate_type: str
    ) -> Dict[str, Any]:
        """
        Simulate how terrain features and climate affect political stability.
        
        Args:
            ctx: Context object
            region_name: Name of the region
            terrain_features: List of terrain features
            climate_type: Type of climate
            
        Returns:
            Geography effect simulation results
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="simulate_geography_impact",
                action_details={"region_name": region_name}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "SimulateGeographyImpact",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "region_name": region_name,
                "climate_type": climate_type
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Get region information from DB if available
            region_data = {}
            resource_availability = []
            
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    region = await conn.fetchrow("""
                        SELECT * FROM GeographicRegions WHERE name ILIKE $1
                    """, f"%{region_name}%")
                    
                    if region:
                        region_data = dict(region)
                        resource_availability = region_data.get("resources", [])
            
            # Prepare prompt
            prompt = f"""
            Simulate the impact of geography and climate on political development:
            
            REGION: {region_name}
            {json.dumps(region_data, indent=2) if region_data else "No additional data available"}
            
            TERRAIN FEATURES: {', '.join(terrain_features)}
            CLIMATE TYPE: {climate_type}
            RESOURCE AVAILABILITY: {', '.join(resource_availability)}
            
            Analyze:
            1. Defensive characteristics of the terrain
            2. Impact on resource accessibility
            3. Settlement patterns influenced by geography
            4. Cultural adaptations to the environment
            5. How matriarchal structures might be strengthened or challenged by geography
            
            Return a ClimateGeographyEffect object with all fields.
            """
            
            # Run the geography impact simulation
            result = await Runner.run(self.geography_effect_agent, prompt, context=run_ctx.context)
            geography_effect = result.final_output
            
            return geography_effect.dict()

    @staticmethod
    @function_tool
    async def simulate_espionage(
        ctx: RunContextWrapper,
        agent_name: str,
        target_nation: str,
        operation_type: str,
        secrecy_level: int
    ) -> Dict[str, Any]:
        """
        Simulate covert operations between nations.
        
        Args:
            ctx: Context object
            agent_name: Name of the agent conducting the operation
            target_nation: Target nation
            operation_type: Type of operation (intelligence gathering, sabotage, etc.)
            secrecy_level: Level of secrecy (1-10)
            
        Returns:
            Covert operation simulation results
        """
        # Apply governance if available
        try:
            from nyx.nyx_governance import NyxUnifiedGovernor
            governor = await NyxUnifiedGovernor(ctx.context.get("user_id"), ctx.context.get("conversation_id")).initialize()
            permission = await governor.check_action_permission(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="geopolitical_manager",
                action_type="simulate_espionage",
                action_details={"target_nation": target_nation}
            )
            
            if not permission.get("approved", True):
                return {"error": permission.get("reasoning", "Action not permitted by governance")}
        except (ImportError, Exception):
            # Governance optional - continue if not available
            pass
            
        with trace(
            "SimulateEspionage",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "agent_name": agent_name,
                "target_nation": target_nation
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Get target nation information from DB if available
            nation_data = {}
            
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    nation = await conn.fetchrow("""
                        SELECT * FROM PoliticalEntities WHERE name ILIKE $1
                    """, f"%{target_nation}%")
                    
                    if nation:
                        nation_data = dict(nation)
                        # Parse JSON fields
                        for field in ["relations", "power_centers"]:
                            if field in nation_data and nation_data[field]:
                                try:
                                    nation_data[field] = json.loads(nation_data[field])
                                except:
                                    nation_data[field] = {}
            
            # Prepare prompt
            prompt = f"""
            Simulate a covert operation:
            
            AGENT: {agent_name}
            TARGET NATION: {target_nation}
            {json.dumps(nation_data, indent=2) if nation_data else "No additional data available"}
            
            OPERATION TYPE: {operation_type}
            SECRECY LEVEL: {secrecy_level} (1-10)
            
            Analyze:
            1. Likelihood of success based on target nation's security
            2. Potential international ramifications if discovered
            3. Strategic value of the intelligence gathered
            4. Matriarchal leadership's response if discovered
            5. Unexpected outcomes or complications
            
            Determine a realistic mission outcome (success, partial success, failure, catastrophic).
            Return a CovertOperation object with all fields.
            """
            
            # Run the espionage simulation
            result = await Runner.run(self.covert_operations_agent, prompt, context=run_ctx.context)
            espionage_result = result.final_output
            
            return espionage_result.dict()
