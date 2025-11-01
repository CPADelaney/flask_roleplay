# lore/managers/local_lore.py
"""
Local Lore Manager for comprehensive world-building in a matriarchal society.

This module provides advanced narrative crafting capabilities including:
- Urban myth creation and evolution
- Historical event tracking
- Landmark management
- Narrative connections between lore elements
- Myth transmission simulation
- Tourism development planning
- Oral vs written tradition analysis

All lore elements emphasize matriarchal themes and feminine authority.

External API:
    The decorated function-tool wrappers at the bottom of this module
    (add_urban_myth, add_local_history, etc.) are the public API.
    Import and use these functions, not the class methods directly.
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple, TypedDict, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from datetime import datetime

# Agents SDK imports
from agents import (
    Agent, function_tool, trace, RunResultStreaming,
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, handoff, RunContextWrapper, RunConfig
)

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils
from lore.cache_version import get_lore_db_connection_context as get_db_connection_context
from nyx.gateway import llm_gateway
from lore.utils.llm_gateway import build_llm_request

if TYPE_CHECKING:
    from lore.core.registry import ManagerRegistry

logger = logging.getLogger(__name__)

# ===== HELPER TO GET MANAGER FROM CONTEXT =====
async def _get_local_lore_manager_from_ctx(ctx) -> "LocalLoreManager":
    """
    Resolve the LocalLoreManager linked to this RunContextWrapper or dict.

    Args:
        ctx: RunContextWrapper or dict containing user_id and conversation_id

    Returns:
        LocalLoreManager instance for the given context

    Raises:
        ValueError: If context doesn't contain required IDs
    """
    if hasattr(ctx, "context"):
        c = ctx.context
    else:
        c = ctx

    user_id = c.get("user_id")
    conv_id = c.get("conversation_id")

    if not user_id or not conv_id:
        raise ValueError("Context must contain user_id and conversation_id")

    # Use registry to get/create the manager
    from lore.core.registry import ManagerRegistry
    registry = ManagerRegistry(user_id, conv_id)

    # Ensure manager is initialized
    await registry.get_manager("local_lore")

    # Return the LocalLoreManager instance
    return await registry.get_local_lore_manager()

# ===== ENUMS =====
class NarrativeStyle(str, Enum):
    """Narrative styles for myth evolution."""
    FOLKLORE = "folklore"
    HISTORICAL = "historical"
    SUPERNATURAL = "supernatural"
    COMMERCIAL = "commercial"
    ACADEMIC = "academic"

class EvolutionType(str, Enum):
    """Types of narrative evolution."""
    NATURAL = "natural"
    CULTURAL = "cultural"
    POLITICAL = "political"
    RELIGIOUS = "religious"
    TECHNOLOGICAL = "technological"

class ConnectionType(str, Enum):
    """Types of narrative connections."""
    MYTH_TO_HISTORY = "myth_to_history"
    HISTORY_TO_LANDMARK = "history_to_landmark"
    MYTH_TO_LANDMARK = "myth_to_landmark"
    DERIVATIVE = "derivative"
    CONTRADICTORY = "contradictory"
    REINFORCING = "reinforcing"

# ===== INPUT MODELS =====
class LocationDataInput(BaseModel):
    """Input model for location data."""
    id: int
    location_name: str
    location_type: Optional[str] = "settlement"
    description: Optional[str] = ""

    model_config = ConfigDict(extra='forbid')

class MythCreationInput(BaseModel):
    """Input for creating urban myths."""
    name: str = Field(..., description="Name of the urban myth")
    description: str = Field(..., description="Detailed description of the myth")
    origin_location: Optional[str] = Field(None, description="Where the myth originated")
    origin_event: Optional[str] = Field(None, description="Event that spawned the myth")
    believability: int = Field(6, ge=1, le=10, description="How believable the myth is")
    spread_rate: int = Field(5, ge=1, le=10, description="How quickly the myth spreads")
    regions_known: List[str] = Field(default_factory=list, description="Regions where myth is known")
    narrative_style: NarrativeStyle = Field(NarrativeStyle.FOLKLORE, description="Narrative style")
    themes: List[str] = Field(default_factory=lambda: ["mystery", "caution"])
    matriarchal_elements: List[str] = Field(default_factory=lambda: ["female authority"])

class HistoryCreationInput(BaseModel):
    """Input for creating historical events."""
    location_id: int
    event_name: str
    description: str
    date_description: str = "Some time ago"
    significance: int = Field(5, ge=1, le=10)
    impact_type: str = "cultural"
    notable_figures: List[str] = Field(default_factory=list)
    current_relevance: Optional[str] = None
    commemoration: Optional[str] = None
    connected_myths: List[int] = Field(default_factory=list)
    related_landmarks: List[int] = Field(default_factory=list)
    narrative_category: str = "historical"

class LandmarkCreationInput(BaseModel):
    """Input for creating landmarks."""
    name: str
    location_id: int
    landmark_type: str = Field("structure", description="Type of landmark")
    description: str
    historical_significance: Optional[str] = None
    current_use: Optional[str] = None
    controlled_by: Optional[str] = None
    legends: List[str] = Field(default_factory=list)
    connected_histories: List[int] = Field(default_factory=list)
    architectural_style: Optional[str] = None
    symbolic_meaning: Optional[str] = None
    matriarchal_significance: Literal["low", "moderate", "high"] = "moderate"

# ===== OUTPUT MODELS =====
class UrbanMyth(BaseModel):
    """Model for urban myths."""
    id: Optional[int] = None
    name: str
    description: str
    origin_location: Optional[str] = None
    origin_event: Optional[str] = None
    believability: int = Field(6, ge=1, le=10)
    spread_rate: int = Field(5, ge=1, le=10)
    regions_known: List[str] = Field(default_factory=list)
    narrative_style: str = "folklore"
    themes: List[str] = Field(default_factory=list)
    variations: List[Dict[str, str]] = Field(default_factory=list)
    matriarchal_elements: List[str] = Field(default_factory=list)

class LocalHistory(BaseModel):
    """Model for local historical events."""
    id: Optional[int] = None
    location_id: int
    event_name: str
    description: str
    date_description: str = "Some time ago"
    significance: int = Field(5, ge=1, le=10)
    impact_type: str = "cultural"
    notable_figures: List[str] = Field(default_factory=list)
    current_relevance: Optional[str] = None
    commemoration: Optional[str] = None
    connected_myths: List[int] = Field(default_factory=list)
    related_landmarks: List[int] = Field(default_factory=list)
    narrative_category: str = "historical"

class Landmark(BaseModel):
    """Model for landmarks."""
    id: Optional[int] = None
    name: str
    location_id: int
    landmark_type: str
    description: str
    historical_significance: Optional[str] = None
    current_use: Optional[str] = None
    controlled_by: Optional[str] = None
    legends: List[str] = Field(default_factory=list)
    connected_histories: List[int] = Field(default_factory=list)
    architectural_style: Optional[str] = None
    symbolic_meaning: Optional[str] = None
    matriarchal_significance: str = "moderate"

class NarrativeEvolution(BaseModel):
    """Model for narrative evolution results."""
    original_element_id: int
    element_type: Literal["myth", "history", "landmark"]
    before_description: str
    after_description: str
    evolution_type: str
    causal_factors: List[str]
    believability_change: int = 0
    significance_change: int = 0
    matriarchal_impact: Optional[str] = None

class MythTransmissionResult(BaseModel):
    """Result of myth transmission simulation."""
    myth_id: int
    myth_name: str
    original_regions: List[str]
    new_regions: List[str]
    transmission_path: List[Dict[str, Any]]
    transformation_details: List[Dict[str, Any]]
    final_believability: int = Field(ge=1, le=10)
    final_spread_rate: int = Field(ge=1, le=10)
    variants_created: int = 0
    cultural_adaptations: List[str] = Field(default_factory=list)

class NarrativeConnection(BaseModel):
    """Model for connections between narrative elements."""
    element1_type: str
    element1_id: int
    element2_type: str
    element2_id: int
    connection_type: ConnectionType
    connection_description: str
    connection_strength: int = Field(5, ge=1, le=10)
    suggested_modifications: List[str] = Field(default_factory=list)

class ConsistencyCheckResult(BaseModel):
    """Result of narrative consistency check."""
    location_id: int
    location_name: str
    inconsistencies: List[str]
    suggested_fixes: List[Dict[str, Any]]
    consistency_rating: int = Field(5, ge=1, le=10)
    potential_new_connections: List[NarrativeConnection]
    matriarchal_coherence: int = Field(5, ge=1, le=10)

class TouristDevelopment(BaseModel):
    """Tourist attraction development plan."""
    marketing_angle: str
    proposed_activities: List[str]
    merchandise_ideas: List[str]
    matriarchal_highlights: str
    economic_impact_estimate: str
    visitor_demographics: List[str] = Field(default_factory=list)
    seasonal_considerations: Optional[str] = None

class TraditionDynamics(BaseModel):
    """Oral vs written tradition comparison."""
    oral_version: str
    written_version: str
    key_differences: List[str]
    matriarchal_comparison: str
    preservation_challenges: List[str] = Field(default_factory=list)
    cultural_significance: Optional[str] = None

class LegendVariant(BaseModel):
    """Contradictory legend variant."""
    title: str
    variant_description: str
    inconsistency_explanation: str
    believability_impact: int = Field(0, ge=-5, le=5)
    cultural_origin: Optional[str] = None

class LocationLoreResult(BaseModel):
    """Complete lore for a location."""
    location: Dict[str, Any]
    histories: List[LocalHistory]
    landmarks: List[Landmark]
    myths: List[UrbanMyth]
    connections: List[NarrativeConnection] = Field(default_factory=list)
    total_elements: int = 0

class LoreEvolutionResult(BaseModel):
    """Result of location lore evolution."""
    event: str
    location_id: int
    location_name: str
    new_history: Optional[LocalHistory] = None
    new_landmark: Optional[Landmark] = None
    updated_landmark: Optional[Dict[str, Any]] = None
    new_myth: Optional[UrbanMyth] = None
    narrative_impact: Optional[str] = None

# ===== SPECIALIZED AGENTS =====
class SpecializedAgents:
    """Container for all specialized agents used by LocalLoreManager."""

    def __init__(self, theme_guardrail: InputGuardrail, coherence_guardrail: OutputGuardrail):
        """Initialize specialized agents with guardrails."""
        # Evolution specialists
        self.folklore_agent = Agent(
            name="FolkloreEvolutionAgent",
            instructions=(
                "You specialize in evolving folklore-style urban myths. "
                "Create poetic, metaphorical narratives with moral lessons "
                "that center matriarchal values and feminine wisdom. "
                "Always maintain the core identity while enhancing folkloric qualities."
            ),
            model="gpt-5-nano",
            output_type=str,
            input_guardrails=[theme_guardrail],
            output_guardrails=[coherence_guardrail]
        )

        self.historical_agent = Agent(
            name="HistoricalEvolutionAgent",
            instructions=(
                "You specialize in evolving myths into pseudo-historical narratives. "
                "Add specific dates, notable female leaders, and concrete impacts. "
                "Make myths seem like real historical events while preserving core elements."
            ),
            model="gpt-5-nano",
            output_type=str,
            input_guardrails=[theme_guardrail],
            output_guardrails=[coherence_guardrail]
        )

        self.supernatural_agent = Agent(
            name="SupernaturalEvolutionAgent",
            instructions=(
                "You specialize in adding supernatural elements to myths. "
                "Introduce divine feminine power, magical occurrences, and mystical beings. "
                "Emphasize matriarchal spiritual authority and cosmic feminine forces."
            ),
            model="gpt-5-nano",
            output_type=str,
            input_guardrails=[theme_guardrail],
            output_guardrails=[coherence_guardrail]
        )

        # Connection specialists
        self.myth_history_connector = Agent(
            name="MythHistoryConnector",
            instructions=(
                "You create plausible narrative links between myths and historical events. "
                "Show how myths might originate from real occurrences or vice versa. "
                "Always highlight matriarchal power dynamics in connections."
            ),
            model="gpt-5-nano",
            output_type=NarrativeConnection,
            input_guardrails=[theme_guardrail],
            output_guardrails=[coherence_guardrail],
        )

        self.history_landmark_connector = Agent(
            name="HistoryLandmarkConnector",
            instructions=(
                "You connect historical events with physical landmarks. "
                "Explain how events shaped places or how landmarks witnessed history. "
                "Emphasize the role of female leaders and matriarchal institutions."
            ),
            model="gpt-5-nano",
            output_type=NarrativeConnection,
            input_guardrails=[theme_guardrail],
            output_guardrails=[coherence_guardrail],
        )

        # Analysis specialists
        self.consistency_agent = Agent(
            name="NarrativeConsistencyAgent",
            instructions=(
                "You ensure consistency across connected lore elements. "
                "Identify contradictions, timeline issues, and thematic conflicts. "
                "Suggest fixes that strengthen matriarchal narrative coherence."
            ),
            model="gpt-5-nano",
            output_type=ConsistencyCheckResult,
            input_guardrails=[theme_guardrail]
        )

        self.transmission_agent = Agent(
            name="MythTransmissionAgent",
            instructions=(
                "You simulate how myths spread between regions and cultures. "
                "Track transformations, distortions, and local adaptations. "
                "Show how matriarchal elements persist or transform across cultures."
            ),
            model="gpt-5-nano",
            output_type=MythTransmissionResult,
            input_guardrails=[theme_guardrail]
        )

        # Creative specialists
        self.variant_agent = Agent(
            name="LegendVariantAgent",
            instructions=(
                "You create contradictory versions of myths with intentional inconsistencies. "
                "Each variant must have believable cultural origins for its differences. "
                "Maintain some core elements while introducing compelling contradictions."
            ),
            model="gpt-5-nano",
            output_type=List[LegendVariant],
            input_guardrails=[theme_guardrail]
        )

        self.tourism_agent = Agent(
            name="TouristAttractionAgent",
            instructions=(
                "You transform myths into commercial tourist attractions. "
                "Create compelling marketing that highlights matriarchal themes. "
                "Balance authenticity with visitor appeal and economic viability."
            ),
            model="gpt-5-nano",
            output_type=TouristDevelopment,
            input_guardrails=[theme_guardrail]
        )

        self.tradition_agent = Agent(
            name="TraditionDynamicsAgent",
            instructions=(
                "You analyze how myths change between oral and written traditions. "
                "Highlight differences in detail, consistency, and local variations. "
                "Show how matriarchal elements evolve in different mediums."
            ),
            model="gpt-5-nano",
            output_type=TraditionDynamics,
            input_guardrails=[theme_guardrail]
        )

        # Master evolution agent with handoffs
        self.myth_evolution_agent = Agent(
            name="MythEvolutionCoordinator",
            instructions=(
                "You coordinate myth evolution using appropriate narrative styles. "
                "Analyze the context and select the best specialist for each evolution. "
                "Ensure matriarchal themes are preserved and enhanced."
            ),
            model="gpt-5-nano",
            handoffs=[
                handoff(
                    self.folklore_agent,
                    tool_name_override="evolve_as_folklore",
                    tool_description_override="Evolve myth using folklore narrative style"
                ),
                handoff(
                    self.historical_agent,
                    tool_name_override="evolve_as_historical",
                    tool_description_override="Evolve myth into pseudo-historical narrative"
                ),
                handoff(
                    self.supernatural_agent,
                    tool_name_override="evolve_as_supernatural",
                    tool_description_override="Evolve myth with supernatural elements"
                )
            ],
            input_guardrails=[theme_guardrail],
            output_guardrails=[coherence_guardrail],
        )

# ===== GUARDRAILS =====
async def matriarchal_theme_guardrail(context: RunContextWrapper, agent: Agent) -> GuardrailFunctionOutput:
    """Ensures all lore respects matriarchal themes."""
    # Check if the input contains anti-matriarchal themes
    input_text = str(context.messages[-1].content if context.messages else "")

    problematic_terms = [
        "patriarchal dominance", "male supremacy", "women's subordination"
    ]

    for term in problematic_terms:
        if term.lower() in input_text.lower():
            return GuardrailFunctionOutput(
                should_block=True,
                message="Content must respect matriarchal themes and feminine authority."
            )

    return GuardrailFunctionOutput(should_block=False)

async def content_coherence_guardrail(context: RunContextWrapper, agent: Agent, output: Any) -> GuardrailFunctionOutput:
    """Ensures generated content maintains narrative coherence."""
    # Basic coherence check - ensure output isn't empty or nonsensical
    if isinstance(output, str) and len(output.strip()) < 20:
        return GuardrailFunctionOutput(
            should_block=True,
            message="Output too short or incoherent. Please regenerate."
        )

    return GuardrailFunctionOutput(should_block=False)


# ===== MAIN MANAGER CLASS =====
class LocalLoreManager(BaseLoreManager):
    """
    Enhanced manager for local lore elements with specialized agents,
    narrative evolution, and comprehensive world-building capabilities.

    Note: Use the function-tool wrappers (add_urban_myth, etc.) for external access,
    not the class methods directly.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "locallore"

        # Initialize guardrails as InputGuardrail/OutputGuardrail instances
        self.theme_guardrail = InputGuardrail(matriarchal_theme_guardrail)
        self.coherence_guardrail = OutputGuardrail(content_coherence_guardrail)

        # Initialize specialized agents with guardrails
        self.specialized_agents = SpecializedAgents(
            self.theme_guardrail,
            self.coherence_guardrail
        )

        # Empty agents dict - will be populated in initialize_agents
        self.agents = {}

    async def initialize_agents(self):
        """
        Override agent initialization to handle both dictionary and specialized agents.
        """
        try:
            # Call base class first to get any foundation agents
            await super().initialize_agents()

            # Now merge in our specialized agents
            self.agents.update({
                # Evolution specialists
                "folklore": self.specialized_agents.folklore_agent,
                "historical": self.specialized_agents.historical_agent,
                "supernatural": self.specialized_agents.supernatural_agent,

                # Connection specialists
                "myth_history_connector": self.specialized_agents.myth_history_connector,
                "history_landmark_connector": self.specialized_agents.history_landmark_connector,

                # Analysis specialists
                "consistency": self.specialized_agents.consistency_agent,
                "transmission": self.specialized_agents.transmission_agent,

                # Creative specialists
                "variant": self.specialized_agents.variant_agent,
                "tourism": self.specialized_agents.tourism_agent,
                "tradition": self.specialized_agents.tradition_agent,

                # Master coordinator
                "myth_evolution": self.specialized_agents.myth_evolution_agent
            })

            logger.info(f"All agents initialized for {self.__class__.__name__} user {self.user_id}")

        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise


    async def _initialize_tables(self):
        """
        Initialize database tables with enhanced fields.
        Override from base class to handle LocalLoreManager specific tables.
        """
        table_definitions = {
            "UrbanMyths": """
                CREATE TABLE IF NOT EXISTS UrbanMyths (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    origin_location TEXT,
                    origin_event TEXT,
                    believability INTEGER CHECK (believability BETWEEN 1 AND 10),
                    spread_rate INTEGER CHECK (spread_rate BETWEEN 1 AND 10),
                    regions_known TEXT[],
                    variations TEXT[],
                    narrative_style TEXT,
                    themes TEXT[],
                    matriarchal_elements TEXT[],
                    versions_json JSONB DEFAULT '{}',
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_evolution TIMESTAMP,
                    embedding VECTOR(1536)
                );

                CREATE INDEX IF NOT EXISTS idx_urbanmyths_embedding
                ON UrbanMyths USING ivfflat (embedding vector_cosine_ops);

                CREATE INDEX IF NOT EXISTS idx_urbanmyths_style
                ON UrbanMyths(narrative_style);
            """,
            "LocalHistories": """
                CREATE TABLE IF NOT EXISTS LocalHistories (
                    id SERIAL PRIMARY KEY,
                    location_id INTEGER NOT NULL,
                    event_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    date_description TEXT,
                    significance INTEGER CHECK (significance BETWEEN 1 AND 10),
                    impact_type TEXT,
                    notable_figures TEXT[],
                    current_relevance TEXT,
                    commemoration TEXT,
                    connected_myths INTEGER[],
                    related_landmarks INTEGER[],
                    narrative_category TEXT,
                    matriarchal_impact TEXT,
                    embedding VECTOR(1536),
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_localhistories_embedding
                ON LocalHistories USING ivfflat (embedding vector_cosine_ops);

                CREATE INDEX IF NOT EXISTS idx_localhistories_location
                ON LocalHistories(location_id);

                CREATE INDEX IF NOT EXISTS idx_localhistories_significance
                ON LocalHistories(significance DESC);
            """,
            "Landmarks": """
                CREATE TABLE IF NOT EXISTS Landmarks (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_id INTEGER NOT NULL,
                    landmark_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    historical_significance TEXT,
                    current_use TEXT,
                    controlled_by TEXT,
                    legends TEXT[],
                    connected_histories INTEGER[],
                    architectural_style TEXT,
                    symbolic_meaning TEXT,
                    matriarchal_significance TEXT,
                    visitor_count INTEGER DEFAULT 0,
                    embedding VECTOR(1536),
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_landmarks_embedding
                ON Landmarks USING ivfflat (embedding vector_cosine_ops);

                CREATE INDEX IF NOT EXISTS idx_landmarks_location
                ON Landmarks(location_id);

                CREATE INDEX IF NOT EXISTS idx_landmarks_type
                ON Landmarks(landmark_type);
            """,
            "NarrativeConnections": """
                CREATE TABLE IF NOT EXISTS NarrativeConnections (
                    id SERIAL PRIMARY KEY,
                    element1_type TEXT NOT NULL,
                    element1_id INTEGER NOT NULL,
                    element2_type TEXT NOT NULL,
                    element2_id INTEGER NOT NULL,
                    connection_type TEXT NOT NULL,
                    connection_description TEXT NOT NULL,
                    connection_strength INTEGER CHECK (connection_strength BETWEEN 1 AND 10),
                    validated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536)
                );

                CREATE INDEX IF NOT EXISTS idx_narrativeconnections_embedding
                ON NarrativeConnections USING ivfflat (embedding vector_cosine_ops);

                CREATE INDEX IF NOT EXISTS idx_narrativeconnections_elements
                ON NarrativeConnections(element1_type, element1_id, element2_type, element2_id);

                CREATE INDEX IF NOT EXISTS idx_narrativeconnections_strength
                ON NarrativeConnections(connection_strength DESC);
            """,
            "MythEvolutions": """
                CREATE TABLE IF NOT EXISTS MythEvolutions (
                    id SERIAL PRIMARY KEY,
                    myth_id INTEGER NOT NULL,
                    previous_version TEXT NOT NULL,
                    new_version TEXT NOT NULL,
                    evolution_type TEXT NOT NULL,
                    evolution_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    causal_factors TEXT[],
                    believability_before INTEGER,
                    believability_after INTEGER,
                    spread_rate_before INTEGER,
                    spread_rate_after INTEGER,
                    regions_known_before TEXT[],
                    regions_known_after TEXT[],
                    matriarchal_impact TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (myth_id) REFERENCES UrbanMyths(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_mythevolutions_embedding
                ON MythEvolutions USING ivfflat (embedding vector_cosine_ops);

                CREATE INDEX IF NOT EXISTS idx_mythevolutions_myth
                ON MythEvolutions(myth_id);

                CREATE INDEX IF NOT EXISTS idx_mythevolutions_date
                ON MythEvolutions(evolution_date DESC);
            """
        }

        await self.initialize_tables_for_class(table_definitions)

    def get_agent(self, agent_name: str) -> Agent:
        """
        Get an agent by name, supporting both dictionary and attribute access.

        Args:
            agent_name: Name of the agent to retrieve

        Returns:
            Agent instance

        Raises:
            KeyError: If agent not found
        """
        # First check the dictionary
        if agent_name in self.agents:
            return self.agents[agent_name]

        # Then check if it's accessible via specialized_agents
        if hasattr(self.specialized_agents, agent_name):
            return getattr(self.specialized_agents, agent_name)

        raise KeyError(f"Agent '{agent_name}' not found")

    # ===== URBAN MYTH OPERATIONS - IMPLEMENTATION =====

    async def _add_urban_myth_impl(self, ctx, input: MythCreationInput) -> int:
        """Add an urban myth using canon system. (Implementation)"""
        with trace("AddUrbanMyth", metadata={"myth_name": input.name}):
            await self.ensure_initialized()

            # Apply matriarchal theming
            themed_description = MatriarchalThemingUtils.apply_matriarchal_theme(
                "myth", input.description
            )

            # Ensure regions_known has at least the origin
            if not input.regions_known and input.origin_location:
                input.regions_known = [input.origin_location]
            elif not input.regions_known:
                input.regions_known = ["local area"]

            # Prepare data for canon
            myth_data = {
                "name": input.name,
                "description": themed_description,
                "origin_location": input.origin_location,
                "origin_event": input.origin_event,
                "believability": input.believability,
                "spread_rate": input.spread_rate,
                "regions_known": input.regions_known,
                "narrative_style": input.narrative_style.value,
                "themes": input.themes,
                "matriarchal_elements": input.matriarchal_elements
            }

            # Create via canon
            async with get_db_connection_context() as conn:
                from lore.core import canon

                # Establish origin location if specified
                if input.origin_location:
                    location_id = await canon.find_or_create_location(
                        ctx, conn, input.origin_location
                    )

                # Create the myth
                myth_id = await canon.find_or_create_urban_myth(ctx, conn, **myth_data)

            logger.info(f"Created urban myth '{input.name}' with ID {myth_id}")
            return myth_id

    # ===== LOCAL HISTORY OPERATIONS - IMPLEMENTATION =====

    async def _add_local_history_impl(self, ctx, input: HistoryCreationInput) -> int:
        """Add local history using canon system. (Implementation)"""
        with trace("AddLocalHistory", metadata={"event_name": input.event_name}):
            await self.ensure_initialized()

            # Apply matriarchal theming
            themed_description = MatriarchalThemingUtils.apply_matriarchal_theme(
                "history", input.description
            )

            # Prepare data for canon
            history_data = {
                "location_id": input.location_id,
                "event_name": input.event_name,
                "description": themed_description,
                "date_description": input.date_description,
                "significance": input.significance,
                "impact_type": input.impact_type,
                "notable_figures": input.notable_figures,
                "current_relevance": input.current_relevance,
                "commemoration": input.commemoration,
                "connected_myths": input.connected_myths,
                "related_landmarks": input.related_landmarks,
                "narrative_category": input.narrative_category
            }

            # Create via canon
            async with get_db_connection_context() as conn:
                from lore.core import canon

                event_id = await canon.create_local_history(ctx, conn, **history_data)

                # Create narrative connections
                for myth_id in input.connected_myths:
                    await self._create_narrative_connection(
                        conn, "myth", myth_id, "history", event_id,
                        ConnectionType.MYTH_TO_HISTORY,
                        "Historical basis for myth", 7
                    )

                for landmark_id in input.related_landmarks:
                    await self._create_narrative_connection(
                        conn, "history", event_id, "landmark", landmark_id,
                        ConnectionType.HISTORY_TO_LANDMARK,
                        "Event occurred at landmark", 8
                    )

            # Invalidate cache
            self.invalidate_cache_pattern(f"local_history_{input.location_id}")

            logger.info(f"Created local history '{input.event_name}' with ID {event_id}")
            return event_id

    # ===== LANDMARK OPERATIONS - IMPLEMENTATION =====

    async def _add_landmark_impl(self, ctx, input: LandmarkCreationInput) -> int:
        """Add landmark using canon system. (Implementation)"""
        with trace("AddLandmark", metadata={"landmark_name": input.name}):
            await self.ensure_initialized()

            # Verify location exists
            async with get_db_connection_context() as conn:
                location = await conn.fetchrow(
                    """
                        SELECT * FROM Locations
                        WHERE COALESCE(id, location_id) = $1
                    """,
                    input.location_id,
                )
                if not location:
                    raise ValueError(f"Location with ID {input.location_id} not found")

            # Apply matriarchal theming
            themed_description = MatriarchalThemingUtils.apply_matriarchal_theme(
                "landmark", input.description
            )

            # Prepare data for canon
            landmark_data = {
                "name": input.name,
                "location_id": input.location_id,
                "landmark_type": input.landmark_type,
                "description": themed_description,
                "historical_significance": input.historical_significance,
                "current_use": input.current_use,
                "controlled_by": input.controlled_by,
                "legends": input.legends,
                "connected_histories": input.connected_histories,
                "architectural_style": input.architectural_style,
                "symbolic_meaning": input.symbolic_meaning,
                "matriarchal_significance": input.matriarchal_significance
            }

            # Create via canon
            async with get_db_connection_context() as conn:
                from lore.core import canon

                # Establish controlling faction if specified
                if input.controlled_by:
                    controller_id = await canon.find_or_create_faction(
                        ctx, conn, input.controlled_by, faction_type="landmark_controller"
                    )

                # Create the landmark
                landmark_id = await canon.find_or_create_landmark(ctx, conn, **landmark_data)

                # Update location if high significance
                if input.matriarchal_significance == "high":
                    from lore.core.lore_system import LoreSystem
                    lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)

                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="Locations",
                        entity_identifier={"id": input.location_id},
                        updates={"has_significant_landmarks": True},
                        reason=f"High significance landmark '{input.name}' established"
                    )

                # Create narrative connections
                for history_id in input.connected_histories:
                    await self._create_narrative_connection(
                        conn, "history", history_id, "landmark", landmark_id,
                        ConnectionType.HISTORY_TO_LANDMARK,
                        "Historical event at landmark", 8
                    )

            # Invalidate cache
            self.invalidate_cache_pattern(f"landmarks_{input.location_id}")

            logger.info(f"Created landmark '{input.name}' with ID {landmark_id}")
            return landmark_id

    # ===== MYTH EVOLUTION - IMPLEMENTATION =====

    async def _evolve_myth_impl(
        self,
        ctx,
        myth_id: int,
        evolution_type: EvolutionType,
        causal_factors: Optional[List[str]] = None
    ) -> NarrativeEvolution:
        """Evolve myth using canon system for updates. (Implementation)"""
        with trace(
            "EvolveMythWithAgent",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "myth_id": myth_id,
                "evolution_type": evolution_type.value
            }
        ):
            run_ctx = self.create_run_context(ctx)
            causal_factors = causal_factors or ["natural evolution"]

            # Fetch the myth
            async with self.db_connection() as conn:
                myth = await conn.fetchrow("""
                SELECT * FROM UrbanMyths WHERE id = $1
                """, myth_id)

                if not myth:
                    raise ValueError(f"Myth with ID {myth_id} not found")

            myth_data = dict(myth)

            # Prepare evolution prompt
            evolution_prompt = f"""
    Evolve this urban myth based on the following context:

    MYTH: {myth_data['name']}
    CURRENT DESCRIPTION: {myth_data['description']}
    BELIEVABILITY: {myth_data['believability']}/10
    SPREAD RATE: {myth_data['spread_rate']}/10
    REGIONS KNOWN: {', '.join(myth_data['regions_known'] or [])}

    EVOLUTION TYPE: {evolution_type.value}
    CAUSAL FACTORS: {', '.join(causal_factors)}

    Transform this myth according to the evolution type while:
    1. Maintaining the core narrative identity
    2. Enhancing matriarchal themes appropriately
    3. Making the changes feel natural and believable
    4. Adjusting believability and spread rate as appropriate

    Return only the evolved description.
    """

            # Select agent based on evolution type
            agent_map = {
                EvolutionType.CULTURAL: self.specialized_agents.folklore_agent,
                EvolutionType.POLITICAL: self.specialized_agents.historical_agent,
                EvolutionType.RELIGIOUS: self.specialized_agents.supernatural_agent,
                EvolutionType.TECHNOLOGICAL: self.specialized_agents.historical_agent,
                EvolutionType.NATURAL: self.specialized_agents.myth_evolution_agent
            }

            selected_agent = agent_map.get(evolution_type, self.specialized_agents.myth_evolution_agent)

            # Run the evolution
            result = (await llm_gateway.execute(build_llm_request(selected_agent,
                evolution_prompt,
                context=run_ctx.context
            ))).raw

            new_description = result.final_output

            # Apply additional theming
            new_description = MatriarchalThemingUtils.apply_matriarchal_theme(
                "myth", new_description
            )

            # Calculate stat changes based on evolution type
            stat_changes = self._calculate_evolution_stats(evolution_type)

            new_believability = max(1, min(10,
                myth_data['believability'] + stat_changes['believability']))
            new_spread_rate = max(1, min(10,
                myth_data['spread_rate'] + stat_changes['spread_rate']))

            # Store the evolution and update via canon
            async with get_db_connection_context() as conn:
                from lore.core import canon

                # Record the evolution in MythEvolutions table
                evo_id = await conn.fetchval("""
                    INSERT INTO MythEvolutions (
                        myth_id, previous_version, new_version, evolution_type,
                        causal_factors, believability_before, believability_after,
                        spread_rate_before, spread_rate_after, regions_known_before,
                        regions_known_after, matriarchal_impact
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    RETURNING id
                """,
                myth_id, myth_data['description'], new_description, evolution_type.value,
                causal_factors, myth_data['believability'], new_believability,
                myth_data['spread_rate'], new_spread_rate, myth_data['regions_known'],
                myth_data['regions_known'], stat_changes.get('matriarchal_impact'))

                # Update the myth via canon
                updates = {
                    "description": new_description,
                    "believability": new_believability,
                    "spread_rate": new_spread_rate,
                    "narrative_style": evolution_type.value,
                    "last_evolution": datetime.now()
                }

                await canon.update_urban_myth(ctx, conn, myth_id, updates)

            return NarrativeEvolution(
                original_element_id=myth_id,
                element_type="myth",
                before_description=myth_data['description'],
                after_description=new_description,
                evolution_type=evolution_type.value,
                causal_factors=causal_factors,
                believability_change=new_believability - myth_data['believability'],
                significance_change=0,
                matriarchal_impact=stat_changes.get('matriarchal_impact')
            )

    def _attach_guardrail_once(self, agent: Agent, *, input_gr=None, output_gr=None):
        if input_gr and input_gr not in agent.input_guardrails:
            agent.input_guardrails.append(input_gr)
        if output_gr and output_gr not in agent.output_guardrails:
            agent.output_guardrails.append(output_gr)

    def _calculate_evolution_stats(self, evolution_type: EvolutionType) -> Dict[str, Any]:
        """Calculate stat changes based on evolution type."""
        stat_map = {
            EvolutionType.CULTURAL: {
                'believability': random.randint(0, 2),
                'spread_rate': random.randint(1, 3),
                'matriarchal_impact': "Cultural reinforcement of feminine wisdom"
            },
            EvolutionType.POLITICAL: {
                'believability': random.randint(1, 3),
                'spread_rate': random.randint(0, 2),
                'matriarchal_impact': "Political legitimization of matriarchal authority"
            },
            EvolutionType.RELIGIOUS: {
                'believability': random.randint(-2, 0),
                'spread_rate': random.randint(1, 3),
                'matriarchal_impact': "Spiritual elevation of feminine divine"
            },
            EvolutionType.TECHNOLOGICAL: {
                'believability': random.randint(-1, 1),
                'spread_rate': random.randint(2, 4),
                'matriarchal_impact': "Modern adaptation preserving ancient wisdom"
            },
            EvolutionType.NATURAL: {
                'believability': random.randint(-1, 1),
                'spread_rate': random.randint(0, 2),
                'matriarchal_impact': "Organic evolution of matriarchal themes"
            }
        }
        return stat_map.get(evolution_type, stat_map[EvolutionType.NATURAL])

    # ===== NARRATIVE CONNECTIONS - IMPLEMENTATION =====

    async def _create_narrative_connection(
        self,
        conn,
        element1_type: str,
        element1_id: int,
        element2_type: str,
        element2_id: int,
        connection_type: ConnectionType,
        connection_description: str,
        connection_strength: int
    ) -> int:
        """Create a narrative connection between elements."""
        embedding_text = (
            f"{element1_type} {element1_id} {connection_type.value} "
            f"{element2_type} {element2_id} {connection_description}"
        )
        embedding = await generate_embedding(embedding_text)

        connection_id = await conn.fetchval("""
            INSERT INTO NarrativeConnections (
                element1_type, element1_id, element2_type, element2_id,
                connection_type, connection_description, connection_strength, embedding
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """,
        element1_type, element1_id, element2_type, element2_id,
        connection_type.value, connection_description, connection_strength, embedding)

        return connection_id

    async def _connect_myth_history_impl(self, ctx, myth_id: int, history_id: int) -> NarrativeConnection:
        with trace("ConnectMythHistory", group_id=self.trace_group_id,
                   metadata={**self.trace_metadata, "myth_id": myth_id, "history_id": history_id}):
            run_ctx = self.create_run_context(ctx)

            async with self.db_connection() as conn:
                myth = await conn.fetchrow("SELECT * FROM UrbanMyths WHERE id=$1", myth_id)
                history = await conn.fetchrow("SELECT * FROM LocalHistories WHERE id=$1", history_id)
                if not myth or not history:
                    raise ValueError("Myth or history not found")

            # Create connection prompt
            connection_prompt = f"""
Analyze the connection between this myth and historical event:

MYTH: {myth['name']}
{myth['description']}

HISTORICAL EVENT: {history['event_name']}
{history['description']}
Date: {history['date_description']}

Create a narrative connection that:
1. Explains how they might be related
2. Preserves the integrity of both narratives
3. Highlights matriarchal elements in the connection
4. Suggests how one might have influenced the other

Be specific and compelling in your connection.
"""

            # Use the connector agent via specialized_agents
            result = (await llm_gateway.execute(build_llm_request(self.specialized_agents.myth_history_connector,
                connection_prompt,
                context=run_ctx.context
            ))).raw

            connection: NarrativeConnection = result.final_output

            # Store the connection
            async with self.db_connection() as conn:      # <── re-use same connection context
                await self._create_narrative_connection(
                conn, "myth", myth_id, "history", history_id,
                connection.connection_type, connection.connection_description,
                connection.connection_strength
                )
                await self._update_cross_references(conn, "myth", myth_id, "history", history_id)
            return connection

    async def _connect_history_landmark_impl(self, ctx, history_id: int, landmark_id: int) -> NarrativeConnection:
        with trace("ConnectHistoryLandmark", group_id=self.trace_group_id,
                   metadata={**self.trace_metadata, "history_id": history_id, "landmark_id": landmark_id}):
            run_ctx = self.create_run_context(ctx)

            async with self.db_connection() as conn:
                history = await conn.fetchrow("SELECT * FROM LocalHistories WHERE id=$1", history_id)
                landmark = await conn.fetchrow("SELECT * FROM Landmarks WHERE id=$1", landmark_id)
                if not history or not landmark:
                    raise ValueError("History or landmark not found")

                # Verify same location
                if history["location_id"] != landmark["location_id"]:
                    logger.warning(
                        f"History and landmark in different locations: "
                        f"{history['location_id']} vs {landmark['location_id']}"
                    )

            # Create connection prompt
            connection_prompt = f"""
Analyze the connection between this historical event and landmark:

HISTORICAL EVENT: {history['event_name']}
{history['description']}
Date: {history['date_description']}
Significance: {history['significance']}/10

LANDMARK: {landmark['name']} ({landmark['landmark_type']})
{landmark['description']}
Historical Significance: {landmark['historical_significance'] or 'Unknown'}

Create a narrative connection that:
1. Explains how the event and place are linked
2. Shows how the landmark witnessed or was shaped by the event
3. Emphasizes matriarchal power dynamics
4. Suggests how the landmark commemorates the event

Be historically plausible and culturally sensitive.
"""

            # Use the connector agent via specialized_agents
            result = (await llm_gateway.execute(build_llm_request(self.specialized_agents.history_landmark_connector,
                connection_prompt,
                context=run_ctx.context
            ))).raw

            connection: NarrativeConnection = result.final_output

            # Store the connection
            async with self.db_connection() as conn:
                await self._create_narrative_connection(
                conn, "history", history_id, "landmark", landmark_id,
                connection.connection_type, connection.connection_description,
                connection.connection_strength
                )
                await self._update_cross_references(conn, "history", history_id, "landmark", landmark_id)
            return connection

    async def _update_cross_references(
        self, conn, type1: str, id1: int, type2: str, id2: int
    ):
        """Update cross-reference arrays in the database."""
        # Update references based on types
        if type1 == "myth" and type2 == "history":
            # Add history to myth's connected histories
            await conn.execute("""
                UPDATE LocalHistories
                SET connected_myths = array_append(
                    COALESCE(connected_myths, ARRAY[]::integer[]), $1
                )
                WHERE id = $2 AND NOT ($1 = ANY(COALESCE(connected_myths, ARRAY[]::integer[])))
            """, id1, id2)

        elif type1 == "history" and type2 == "landmark":
            # Update both directions
            await conn.execute("""
                UPDATE LocalHistories
                SET related_landmarks = array_append(
                    COALESCE(related_landmarks, ARRAY[]::integer[]), $1
                )
                WHERE id = $2 AND NOT ($1 = ANY(COALESCE(related_landmarks, ARRAY[]::integer[])))
            """, id2, id1)

            await conn.execute("""
                UPDATE Landmarks
                SET connected_histories = array_append(
                    COALESCE(connected_histories, ARRAY[]::integer[]), $1
                )
                WHERE id = $2 AND NOT ($1 = ANY(COALESCE(connected_histories, ARRAY[]::integer[])))
            """, id1, id2)

    # ===== NARRATIVE CONSISTENCY - IMPLEMENTATION =====

    async def _ensure_narrative_consistency_impl(
        self,
        ctx,
        location_id: int,
        auto_fix: bool = True
    ) -> ConsistencyCheckResult:
        """Analyze and optionally fix narrative inconsistencies for a location. (Implementation)"""
        with trace(
            "EnsureNarrativeConsistency",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "location_id": location_id,
                "auto_fix": auto_fix
            }
        ):
            run_ctx = self.create_run_context(ctx)

            # Get all lore for the location
            location_lore = await self._get_location_lore_impl(run_ctx, location_id)

            # Get all connections (passing already fetched IDs for efficiency)
            connections = await self._get_location_connections_efficient(
                location_id,
                [m.id for m in location_lore.myths if m.id],
                [h.id for h in location_lore.histories if h.id],
                [l.id for l in location_lore.landmarks if l.id]
            )

            # Prepare consistency check prompt
            consistency_prompt = f"""
            Analyze narrative consistency for this location's lore:

            LOCATION: {location_lore.location['location_name']}

            MYTHS ({len(location_lore.myths)} total):
            {self._summarize_elements(location_lore.myths, 'myth')}

            HISTORIES ({len(location_lore.histories)} total):
            {self._summarize_elements(location_lore.histories, 'history')}

            LANDMARKS ({len(location_lore.landmarks)} total):
            {self._summarize_elements(location_lore.landmarks, 'landmark')}

            CONNECTIONS ({len(connections)} total):
            {self._summarize_connections(connections)}

            Identify:
            1. Timeline inconsistencies
            2. Contradictory facts
            3. Thematic conflicts
            4. Missing connections
            5. Creates regional variants

            Include specific transmission paths and cultural adaptations.

            Provide specific, actionable fixes for each issue.
            Rate overall consistency and matriarchal coherence (1-10).
            """

            # Run consistency check using specialized agent
            result = (await llm_gateway.execute(build_llm_request(self.specialized_agents.consistency_agent,
                consistency_prompt,
                context=run_ctx.context
            ))).raw

            consistency_result: ConsistencyCheckResult = result.final_output

            # Apply fixes if requested
            fixes_applied = []
            if auto_fix and consistency_result.suggested_fixes:
                fixes_applied = await self._apply_consistency_fixes(
                    consistency_result.suggested_fixes
                )

            # Create new connections if suggested
            new_connections = []
            if consistency_result.potential_new_connections:
                new_connections = await self._create_suggested_connections(
                    consistency_result.potential_new_connections
                )

            # Update the result with applied changes
            consistency_result.suggested_fixes = [
                fix for fix in consistency_result.suggested_fixes
                if fix not in fixes_applied
            ]

            # Invalidate cache
            self.invalidate_cache_pattern(f"location_lore_{location_id}")

            return consistency_result

    # ===== LOCATION LORE OPERATIONS - IMPLEMENTATION =====

    async def _get_location_lore_impl(self, ctx, location_id: int) -> LocationLoreResult:
        """
        Get all lore associated with a location. (Implementation)

        Args:
            location_id: ID of the location

        Returns:
            LocationLoreResult with all location lore
        """
        cache_key = f"location_lore_{location_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return LocationLoreResult(**cached)

        async with self.db_connection() as conn:
            # Get location
            location = await conn.fetchrow("""
            SELECT COALESCE(id, location_id) AS id,
                   location_name,
                   location_type,
                   description
            FROM Locations
            WHERE COALESCE(id, location_id) = $1
            """, location_id)

            if not location:
                raise ValueError(f"Location with ID {location_id} not found")

            location_data = dict(location)
            if "location_id" not in location_data:
                location_data["location_id"] = location_data.get("id")
            if "id" not in location_data and "location_id" in location_data:
                location_data["id"] = location_data["location_id"]

            # Get histories
            histories = await conn.fetch("""
            SELECT * FROM LocalHistories
            WHERE location_id = $1
            ORDER BY significance DESC, creation_date DESC
            """, location_id)

            # Get landmarks
            landmarks = await conn.fetch("""
            SELECT * FROM Landmarks
            WHERE location_id = $1
            ORDER BY matriarchal_significance DESC, creation_date DESC
            """, location_id)

            # Get myths (by location name or in regions_known)
            location_name = location_data["location_name"]
            myths = await conn.fetch("""
            SELECT * FROM UrbanMyths
            WHERE origin_location = $1
               OR $1 = ANY(regions_known)
            ORDER BY believability DESC, spread_rate DESC
            """, location_name)

            # Get connections
            connections = await self._get_location_connections(location_id)

            # Convert to models
            result = LocationLoreResult(
            location=location_data,
            histories=[LocalHistory(**dict(h)) for h in histories],
            landmarks=[Landmark(**dict(l)) for l in landmarks],
            myths=[UrbanMyth(**dict(m)) for m in myths],
            connections=[
                NarrativeConnection(
                    element1_type=c['element1_type'],
                    element1_id=c['element1_id'],
                    element2_type=c['element2_type'],
                    element2_id=c['element2_id'],
                    connection_type=ConnectionType(c['connection_type']),
                    connection_description=c['connection_description'],
                    connection_strength=c['connection_strength']
                ) for c in connections
            ],
            total_elements=len(histories) + len(landmarks) + len(myths)
            )

            self.set_cache(cache_key, result.model_dump())
            return result

    async def _generate_location_lore_impl(
        self,
        ctx,
        location_data: LocationDataInput
    ) -> Dict[str, Any]:
        """
        Generate comprehensive lore for a location using specialized agents. (Implementation)

        Args:
            location_data: LocationDataInput with location information

        Returns:
            Dictionary with generated lore statistics
        """
        run_ctx = self.create_run_context(ctx)

        with trace(
            "GenerateLocationLore",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "location_id": location_data.id,
                "location_name": location_data.location_name
            }
        ):
            # Generate different types of lore in parallel
            myths_task = self._generate_myths_for_location(run_ctx, location_data)
            histories_task = self._generate_histories_for_location(run_ctx, location_data)
            landmarks_task = self._generate_landmarks_for_location(run_ctx, location_data)

            # Wait for all generation tasks
            generated_myths, generated_histories, generated_landmarks = await asyncio.gather(
                myths_task, histories_task, landmarks_task
            )

            # Create connections between generated elements
            connections_created = await self._generate_initial_connections(
                run_ctx,
                generated_myths,
                generated_histories,
                generated_landmarks
            )

            # Invalidate cache
            self.invalidate_cache(f"location_lore_{location_data.id}")

            return {
                "location": location_data.model_dump(),
                "generated": {
                    "myths": len(generated_myths),
                    "histories": len(generated_histories),
                    "landmarks": len(generated_landmarks),
                    "connections": connections_created
                },
                "summary": (
                    f"Generated {len(generated_myths)} myths, "
                    f"{len(generated_histories)} historical events, "
                    f"{len(generated_landmarks)} landmarks, and "
                    f"{connections_created} narrative connections for "
                    f"{location_data.location_name}"
                )
            }

    async def _evolve_location_lore_impl(
        self,
        ctx,
        location_id: int,
        event_description: str
    ) -> LoreEvolutionResult:
        """
        Evolve location lore based on a significant event. (Implementation)
        Uses canon system to ensure changes are consistent.

        Args:
            location_id: ID of the location
            event_description: Description of the event causing evolution

        Returns:
            LoreEvolutionResult with changes made
        """
        # Get LoreSystem for canon operations
        from lore.core.lore_system import LoreSystem
        lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)

        run_ctx = self.create_run_context(ctx)

        with trace(
            "EvolveLocationLore",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "location_id": location_id,
                "event": event_description[:100]
            }
        ):
            # Get current lore
            location_lore = await self._get_location_lore_impl(ctx, location_id)

            # Apply matriarchal theming to the event
            themed_event = MatriarchalThemingUtils.apply_matriarchal_theme(
                "event", event_description, emphasis_level=1
            )

            # Create evolution agent with context
            evolution_agent = Agent(
                name="LoreEvolutionCoordinator",
                instructions=(
                    "You analyze how major events impact local lore. "
                    "Create realistic consequences that respect existing narratives."
                ),
                model="gpt-5-nano",
            )

            # Determine what type of lore to create/modify
            evolution_prompt = f"""
A significant event has occurred in {location_lore.location['location_name']}:

EVENT: {themed_event}

CURRENT LORE SUMMARY:
- {len(location_lore.myths)} myths
- {len(location_lore.histories)} historical events
- {len(location_lore.landmarks)} landmarks

Based on this event, determine:
1. Should we create a new historical record? (yes/no and why)
2. Should we create a new myth about it? (yes/no and why)
3. Should we create or modify a landmark? (yes/no and why)
4. Which existing elements might be affected?

Return a JSON object with your analysis and specific recommendations.
"""

            result = (await llm_gateway.execute(build_llm_request(evolution_agent,
                evolution_prompt,
                context=run_ctx.context
            ))).raw

            # Parse recommendations
            try:
                recommendations = json.loads(result.final_output)
            except:
                recommendations = {
                    "new_history": True,
                    "new_myth": False,
                    "landmark_change": False
                }

            evolution_result = LoreEvolutionResult(
                event=themed_event,
                location_id=location_id,
                location_name=location_lore.location['location_name']
            )

            # Create new history if recommended
            if recommendations.get('new_history'):
                # Generate a proper event title
                event_title = f"The {self._generate_event_title(themed_event)}"

                # Check if this event already exists to avoid duplicates
                async with get_db_connection_context() as conn:
                    existing_event = await conn.fetchrow("""
                        SELECT id FROM LocalHistories
                        WHERE location_id = $1
                        AND (event_name = $2 OR description ILIKE $3)
                        AND date_description = 'Recently'
                    """, location_id, event_title, f"%{themed_event[:50]}%")

                    if not existing_event:
                        history_input = HistoryCreationInput(
                            location_id=location_id,
                            event_name=event_title,
                            description=themed_event,
                            date_description="Recently",
                            significance=7,
                            impact_type="transformative",
                            current_relevance="Still unfolding",
                            narrative_category="contemporary"
                        )

                        history_id = await self._add_local_history_impl(run_ctx, history_input)
                        evolution_result.new_history = LocalHistory(
                            id=history_id,
                            **history_input.model_dump()
                        )

                        # Log as canonical event
                        from lore.core import canon
                        await canon.log_canonical_event(
                            ctx, conn,
                            f"{event_title} occurred at {location_lore.location['location_name']}",
                            tags=["history", "location", "evolution", "canon"],
                            significance=7
                        )

            # Create new myth if recommended
            if recommendations.get('new_myth') and random.random() > 0.5:
                # Check myth doesn't already exist
                myth_name = f"The {self._generate_myth_title(themed_event)}"

                async with get_db_connection_context() as conn:
                    existing_myth = await conn.fetchrow("""
                        SELECT id FROM UrbanMyths
                        WHERE name = $1 AND origin_location = $2
                    """, myth_name, location_lore.location['location_name'])

                    if not existing_myth:
                        myth_input = MythCreationInput(
                            name=myth_name,
                            description=self._mythologize_event(themed_event),
                            origin_location=location_lore.location['location_name'],
                            origin_event=themed_event,
                            believability=4,
                            spread_rate=7,
                            themes=['contemporary', 'transformation']
                        )

                        myth_id = await self._add_urban_myth_impl(run_ctx, myth_input)
                        evolution_result.new_myth = UrbanMyth(
                            id=myth_id,
                            **myth_input.model_dump()
                        )

            # Handle landmark changes if recommended
            if recommendations.get('landmark_change'):
                # Check if we're modifying existing or creating new
                if location_lore.landmarks and random.random() > 0.3:
                    # Modify existing landmark
                    landmark_to_modify = random.choice(location_lore.landmarks)

                    # Use the canon system to record the change
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="Landmarks",
                        entity_identifier={"id": landmark_to_modify.id},
                        updates={
                            "description": (
                                f"{landmark_to_modify.description} "
                                f"Recent events have left their mark here - {themed_event.lower()}"
                            ),
                            "historical_significance": f"Site of recent events: {themed_event}"
                        },
                        reason=f"Landmark affected by: {themed_event}"
                    )

                    evolution_result.updated_landmark = {
                        "id": landmark_to_modify.id,
                        "name": landmark_to_modify.name,
                        "change": "Updated due to recent events"
                    }
                else:
                    # Create new landmark only if it doesn't exist
                    memorial_name = f"Memorial of {self._generate_event_title(themed_event)}"

                    async with get_db_connection_context() as conn:
                        existing_memorial = await conn.fetchrow("""
                            SELECT id FROM Landmarks
                            WHERE name = $1 AND location_id = $2
                        """, memorial_name, location_id)

                        if not existing_memorial:
                            landmark_input = LandmarkCreationInput(
                                name=memorial_name,
                                location_id=evolution_result.location_id,
                                landmark_type="monument",
                                description=(
                                    f"A newly erected monument commemorating the recent events. {themed_event}"
                                ),
                                current_use="Memorial and gathering place",
                                controlled_by="The Council of Matriarchs",
                                matriarchal_significance="high"
                            )

                            landmark_id = await self._add_landmark_impl(ctx, landmark_input)
                            evolution_result.new_landmark = Landmark(
                                id=landmark_id,
                                **landmark_input.model_dump()
                            )

            # Create connections between new elements
            if evolution_result.new_history and evolution_result.new_myth:
                await self._connect_myth_history_impl(
                    run_ctx,
                    evolution_result.new_myth.id,
                    evolution_result.new_history.id
                )

            # Invalidate cache
            self.invalidate_cache(f"location_lore_{location_id}")

            return evolution_result

    # ===== SPECIALIZED FEATURES - IMPLEMENTATION =====

    async def _generate_legend_variants_impl(
        self,
        ctx,
        myth_id: int,
        variant_count: int = 3
    ) -> Dict[str, Any]:
        """Create contradictory versions of a myth. (Implementation)"""
        run_ctx = self.create_run_context(ctx)

        with trace(
            "GenerateLegendVariants",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "myth_id": myth_id,
                "variant_count": variant_count
            }
        ):
            # Fetch the myth
            async with self.db_connection() as conn:
                myth = await conn.fetchrow(
                "SELECT * FROM UrbanMyths WHERE id = $1", myth_id
                )
                if not myth:
                    raise ValueError(f"Myth with ID {myth_id} not found")

            myth_data = dict(myth)

            # Create variants prompt
            variants_prompt = f"""
Create {variant_count} contradictory versions of this myth:

ORIGINAL MYTH: {myth_data['name']}
DESCRIPTION: {myth_data['description']}

Each variant must:
1. Maintain some core elements
2. Have at least one major contradiction
3. Feel culturally authentic
4. Have a plausible reason for the variation
5. Preserve matriarchal themes differently

Make the contradictions meaningful - different moral lessons,
different antagonists, different outcomes, etc.
"""

            # Run variant generation using specialized agent
            result = (await llm_gateway.execute(build_llm_request(self.specialized_agents.variant_agent,
                variants_prompt,
                context=run_ctx.context
            ))).raw

            variants: List[LegendVariant] = result.final_output

            # Store variants in versions_json
            async with self.db_connection() as conn:
                versions_json = myth_data.get('versions_json') or {}
                if 'contradictory_variants' not in versions_json:
                    versions_json['contradictory_variants'] = []

                # Add new variants
                for variant in variants:
                    versions_json['contradictory_variants'].append(
                        variant.model_dump()
                    )

                await conn.execute("""
                UPDATE UrbanMyths
                SET versions_json = $1
                WHERE id = $2
                """, json.dumps(versions_json), myth_id)

            return {
                "myth_id": myth_id,
                "myth_name": myth_data['name'],
                "variants_created": len(variants),
                "variants": [v.model_dump() for v in variants]
            }

    async def _develop_tourist_attraction_impl(
        self,
        ctx,
        myth_id: int
    ) -> TouristDevelopment:
        """Transform a myth into a tourist attraction plan. (Implementation)"""
        run_ctx = self.create_run_context(ctx)

        with trace(
            "DevelopTouristAttraction",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "myth_id": myth_id}
        ):
            # Fetch the myth
            async with self.db_connection() as conn:
                myth = await conn.fetchrow(
                "SELECT * FROM UrbanMyths WHERE id = $1", myth_id
                )
                if not myth:
                    raise ValueError(f"Myth with ID {myth_id} not found")

            myth_data = dict(myth)

            # Create tourism prompt
            tourism_prompt = f"""
Develop a tourist attraction based on this myth:

MYTH: {myth_data['name']}
DESCRIPTION: {myth_data['description']}
ORIGIN: {myth_data.get('origin_location', 'Unknown')}
BELIEVABILITY: {myth_data['believability']}/10

Create a comprehensive tourism plan that:
1. Respects the cultural significance
2. Highlights matriarchal themes as unique selling points
3. Includes interactive experiences
4. Suggests authentic merchandise
5. Estimates economic impact
6. Identifies target demographics
7. Considers seasonal factors

Balance commercialization with cultural preservation.
"""

            # Run tourism development using specialized agent
            result = (await llm_gateway.execute(build_llm_request(self.specialized_agents.tourism_agent,
                tourism_prompt,
                context=run_ctx.context
            ))).raw

            tourism_plan: TouristDevelopment = result.final_output

            # Store in versions_json
            async with self.db_connection() as conn:
                versions_json = myth_data.get('versions_json') or {}
                versions_json['tourist_development'] = tourism_plan.model_dump()

                await conn.execute("""
                UPDATE UrbanMyths
                SET versions_json = $1
                WHERE id = $2
                """, json.dumps(versions_json), myth_id)

            return tourism_plan

    async def _simulate_tradition_dynamics_impl(
        self,
        ctx,
        myth_id: int
    ) -> TraditionDynamics:
        """Compare oral vs written tradition versions of a myth. (Implementation)"""
        run_ctx = self.create_run_context(ctx)

        with trace(
            "SimulateTraditionDynamics",
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "myth_id": myth_id}
        ):
            # Fetch the myth
            async with self.db_connection() as conn:
                myth = await conn.fetchrow(
                "SELECT * FROM UrbanMyths WHERE id = $1", myth_id
                )
                if not myth:
                    raise ValueError(f"Myth with ID {myth_id} not found")

            myth_data = dict(myth)

            # Create tradition comparison prompt
            tradition_prompt = f"""
Analyze how this myth differs between oral and written traditions:

MYTH: {myth_data['name']}
ORIGINAL VERSION: {myth_data['description']}
THEMES: {', '.join(myth_data.get('themes', []))}

Compare:
1. How details change in oral retellings
2. What gets emphasized in written records
3. How matriarchal elements evolve in each medium
4. Preservation challenges for each tradition
5. Cultural significance of each version

Show specific differences in language, detail, and emphasis.
"""

            # Run tradition analysis using specialized agent
            result = (await llm_gateway.execute(build_llm_request(self.specialized_agents.tradition_agent,
                tradition_prompt,
                context=run_ctx.context
            ))).raw

            tradition_analysis: TraditionDynamics = result.final_output

            # Store in versions_json
            async with self.db_connection() as conn:
                versions_json = myth_data.get('versions_json') or {}
                versions_json['tradition_dynamics'] = tradition_analysis.model_dump()

                await conn.execute("""
                UPDATE UrbanMyths
                SET versions_json = $1
                WHERE id = $2
                """, json.dumps(versions_json), myth_id)

            return tradition_analysis

    async def _simulate_myth_transmission_impl(
        self,
        ctx,
        myth_id: int,
        target_regions: List[str],
        transmission_steps: int = 3
    ) -> MythTransmissionResult:
        """Simulate how a myth spreads and transforms across regions. (Implementation)"""
        with trace(
            "SimulateMythTransmission",
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata,
                "myth_id": myth_id,
                "regions": target_regions
            }
        ):
            run_ctx = self.create_run_context(ctx)

            # Fetch the myth
            async with self.db_connection() as conn:
                myth = await conn.fetchrow(
                "SELECT * FROM UrbanMyths WHERE id = $1", myth_id
                )

                if not myth:
                    raise ValueError(f"Myth with ID {myth_id} not found")

                # Get cultural context
                cultural_elements = await conn.fetch("""
                SELECT name, element_type, description
                FROM CulturalElements
                ORDER BY RANDOM()
                LIMIT 5
                """)

            myth_data = dict(myth)
            original_regions = myth_data.get("regions_known", [])
            new_regions = [r for r in target_regions if r not in original_regions]

            if not new_regions:
                # No new regions to spread to
                return MythTransmissionResult(
                    myth_id=myth_id,
                    myth_name=myth_data["name"],
                    original_regions=original_regions,
                    new_regions=[],
                    transmission_path=[],
                    transformation_details=[],
                    final_believability=myth_data["believability"],
                    final_spread_rate=myth_data["spread_rate"],
                    variants_created=0
                )

            # Create transmission prompt
            transmission_prompt = f"""
            Simulate the transmission of this myth across regions:

            MYTH: {myth_data['name']}
            DESCRIPTION: {myth_data['description']}
            ORIGINAL REGIONS: {', '.join(original_regions)}
            TARGET REGIONS: {', '.join(new_regions)}
            TRANSMISSION STEPS: {transmission_steps}

            CULTURAL CONTEXT:
            {self._format_cultural_elements(cultural_elements)}

            Show how the myth transforms as it spreads, including:
            1. Transmission paths between regions
            2. Cultural adaptations in each region
            3. Changes in believability and spread rate
            4. Creation of regional variants
            5. Preservation of matriarchal elements

            Include specific transmission paths and cultural adaptations.
            """

            # Run transmission simulation using specialized agent
            result = (await llm_gateway.execute(build_llm_request(self.specialized_agents.transmission_agent,
                transmission_prompt,
                context=run_ctx.context
            ))).raw

            transmission_result: MythTransmissionResult = result.final_output

            # Apply transmission results
            await self._apply_transmission_results(
                myth_id, myth_data, transmission_result
            )

            return transmission_result

    # ===== HELPER METHODS =====

    def _format_cultural_elements(self, elements: List[Any]) -> str:
        """Format cultural elements for prompts."""
        if not elements:
            return "No cultural context available"

        formatted = []
        for elem in elements:
            formatted.append(
                f"- {elem['name']} ({elem['element_type']}): {elem['description']}"
            )
        return "\n".join(formatted)

    async def _apply_transmission_results(
        self,
        myth_id: int,
        original_myth: Dict[str, Any],
        transmission_result: MythTransmissionResult
    ):
        """Apply the results of myth transmission simulation."""
        async with self.db_connection() as conn:
            # Update the original myth
            updated_regions = (
            transmission_result.original_regions +
            transmission_result.new_regions
            )

            await conn.execute("""
            UPDATE UrbanMyths
            SET regions_known = $1,
                believability = $2,
                spread_rate = $3
            WHERE id = $4
            """, updated_regions, transmission_result.final_believability,
            transmission_result.final_spread_rate, myth_id)

            # Create regional variants if specified
            for i, transformation in enumerate(
                transmission_result.transformation_details[:transmission_result.variants_created]
            ):
                if 'variant_description' in transformation:
                    variant_name = (
                        f"{original_myth['name']} "
                        f"({transmission_result.new_regions[i % len(transmission_result.new_regions)]} Variant)"
                    )

                # Create variant as a new myth
                variant_input = MythCreationInput(
                    name=variant_name,
                    description=transformation['variant_description'],
                    origin_location=transmission_result.new_regions[
                        i % len(transmission_result.new_regions)
                    ],
                    origin_event=f"Transmission of '{original_myth['name']}'",
                    believability=transmission_result.final_believability,
                    spread_rate=transmission_result.final_spread_rate,
                    regions_known=[transmission_result.new_regions[
                        i % len(transmission_result.new_regions)
                    ]],
                    narrative_style=NarrativeStyle(
                        original_myth.get('narrative_style', 'folklore')
                    ),
                    themes=original_myth.get('themes', []),
                    matriarchal_elements=original_myth.get('matriarchal_elements', [])
                )

                variant_id = await self._add_urban_myth_impl(
                    RunContextWrapper(context={}), variant_input
                )

                # Create connection between original and variant
                await self._create_narrative_connection(
                    conn, "myth", myth_id, "myth", variant_id,
                    ConnectionType.DERIVATIVE,
                    f"Regional variant from transmission to {variant_input.origin_location}",
                    6
                )

    async def _generate_myths_for_location(
        self,
        ctx,
        location_data: LocationDataInput
    ) -> List[int]:
        """Generate urban myths for a location."""
        prompt = f"""
        Generate 2-3 compelling urban myths for this location:

        LOCATION: {location_data.location_name} ({location_data.location_type})
        DESCRIPTION: {location_data.description}

        Create myths that:
        1. Feel authentic to the location
        2. Have varying believability (some more plausible than others)
        3. Include strong matriarchal themes
        4. Connect to local fears, hopes, or mysteries
        5. Could spread to neighboring regions

        Return a JSON array of myth objects with these fields:
        - name: string
        - description: string (detailed, 2-3 sentences)
        - believability: number (1-10)
        - spread_rate: number (1-10)
        - themes: array of strings
        - origin_event: string (optional)
        """

        # Create a specialized myth generation agent
        myth_gen_agent = Agent(
            name="LocationMythGenerator",
            instructions="You create authentic urban myths for specific locations.",
            model="gpt-5-nano",
            output_type=List[Dict[str, Any]]
        )

        result = (await llm_gateway.execute(build_llm_request(myth_gen_agent,
            prompt,
            context=ctx.context,
            run_config=RunConfig(workflow_name="GenerateLocationMyths")
        ))).raw

        generated_ids = []
        for myth_data in result.final_output:
            try:
                myth_input = MythCreationInput(
                    name=myth_data['name'],
                    description=myth_data['description'],
                    origin_location=location_data.location_name,
                    origin_event=myth_data.get('origin_event'),
                    believability=myth_data.get('believability', 6),
                    spread_rate=myth_data.get('spread_rate', 5),
                    regions_known=[location_data.location_name],
                    themes=myth_data.get('themes', ['mystery']),
                    matriarchal_elements=['feminine wisdom', 'matriarchal power']
                )

                myth_id = await self._add_urban_myth_impl(ctx, myth_input)
                generated_ids.append(myth_id)

            except Exception as e:
                logger.error(f"Error creating myth: {e}")

        return generated_ids

    async def _generate_histories_for_location(
        self,
        ctx,
        location_data: LocationDataInput
    ) -> List[int]:
        """Generate historical events for a location."""
        prompt = f"""
        Generate 2-3 significant historical events for this location:

        LOCATION: {location_data.location_name} ({location_data.location_type})
        DESCRIPTION: {location_data.description}

        Create events that:
        1. Show the location's development over time
        2. Include at least one event centered on female leadership
        3. Have lasting impacts on the location
        4. Vary in time periods (ancient, recent, etc.)
        5. Could connect to local myths or landmarks

        Return a JSON array of event objects with these fields:
        - event_name: string
        - description: string (detailed, 2-3 sentences)
        - date_description: string (e.g., "Three centuries ago", "Last decade")
        - significance: number (1-10)
        - impact_type: string (cultural, political, economic, religious)
        - notable_figures: array of strings (names)
        - current_relevance: string (how it affects the location today)
        """

        history_gen_agent = Agent(
            name="LocationHistoryGenerator",
            instructions="You create compelling historical events for locations.",
            model="gpt-5-nano",
            output_type=List[Dict[str, Any]]
        )

        result = (await llm_gateway.execute(build_llm_request(history_gen_agent,
            prompt,
            context=ctx.context,
            run_config=RunConfig(workflow_name="GenerateLocationHistory")
        ))).raw

        generated_ids = []
        for event_data in result.final_output:
            try:
                history_input = HistoryCreationInput(
                    location_id=location_data.id,
                    event_name=event_data['event_name'],
                    description=event_data['description'],
                    date_description=event_data.get('date_description', 'Some time ago'),
                    significance=event_data.get('significance', 5),
                    impact_type=event_data.get('impact_type', 'cultural'),
                    notable_figures=event_data.get('notable_figures', []),
                    current_relevance=event_data.get('current_relevance')
                )

                history_id = await self._add_local_history_impl(ctx, history_input)
                generated_ids.append(history_id)

            except Exception as e:
                logger.error(f"Error creating history: {e}")

        return generated_ids

    async def _generate_landmarks_for_location(
        self,
        ctx,
        location_data: LocationDataInput
    ) -> List[int]:
        """Generate landmarks for a location."""
        prompt = f"""
        Generate 2-3 significant landmarks for this location:

        LOCATION: {location_data.location_name} ({location_data.location_type})
        DESCRIPTION: {location_data.description}

        Create landmarks that:
        1. Include both natural and constructed features
        2. Have at least one with high matriarchal significance
        3. Could be settings for myths or historical events
        4. Have interesting architectural or natural features
        5. Serve current purposes in the community

        Return a JSON array of landmark objects with these fields:
        - name: string
        - landmark_type: string (natural, structure, monument, sacred_site)
        - description: string (detailed, 2-3 sentences)
        - historical_significance: string (optional)
        - current_use: string
        - controlled_by: string (who maintains/owns it)
        - architectural_style: string (if applicable)
        - matriarchal_significance: string (low, moderate, high)
        """

        landmark_gen_agent = Agent(
            name="LocationLandmarkGenerator",
            instructions="You create memorable landmarks for locations.",
            model="gpt-5-nano",
            output_type=List[Dict[str, Any]]
        )

        result = (await llm_gateway.execute(build_llm_request(landmark_gen_agent,
            prompt,
            context=ctx.context,
            run_config=RunConfig(workflow_name="GenerateLocationLandmarks")
        ))).raw

        generated_ids = []
        for landmark_data in result.final_output:
            try:
                landmark_input = LandmarkCreationInput(
                    name=landmark_data['name'],
                    location_id=location_data.id,
                    landmark_type=landmark_data.get('landmark_type', 'structure'),
                    description=landmark_data['description'],
                    historical_significance=landmark_data.get('historical_significance'),
                    current_use=landmark_data.get('current_use'),
                    controlled_by=landmark_data.get('controlled_by'),
                    architectural_style=landmark_data.get('architectural_style'),
                    matriarchal_significance=landmark_data.get(
                        'matriarchal_significance', 'moderate'
                    )
                )

                landmark_id = await self._add_landmark_impl(ctx, landmark_input)
                generated_ids.append(landmark_id)

            except Exception as e:
                logger.error(f"Error creating landmark: {e}")

        return generated_ids

    async def _generate_initial_connections(
        self,
        ctx,
        myth_ids: List[int],
        history_ids: List[int],
        landmark_ids: List[int]
    ) -> int:
        """Generate initial narrative connections between newly created elements."""
        connections_created = 0

        # Connect some myths to histories
        if myth_ids and history_ids:
            for i in range(min(2, len(myth_ids), len(history_ids))):
                try:
                    await self._connect_myth_history_impl(
                        ctx, myth_ids[i], history_ids[i % len(history_ids)]
                    )
                    connections_created += 1
                except Exception as e:
                    logger.error(f"Error connecting myth to history: {e}")

        # Connect some histories to landmarks
        if history_ids and landmark_ids:
            for i in range(min(2, len(history_ids), len(landmark_ids))):
                try:
                    await self._connect_history_landmark_impl(
                        ctx, history_ids[i], landmark_ids[i % len(landmark_ids)]
                    )
                    connections_created += 1
                except Exception as e:
                    logger.error(f"Error connecting history to landmark: {e}")

        return connections_created

    def _generate_event_title(self, event: str) -> str:
        """Generate a title for a historical event."""
        # Simple heuristic - take first few significant words
        words = event.split()[:5]
        return ' '.join(w for w in words if len(w) > 3).title()

    def _generate_myth_title(self, event: str) -> str:
        """Generate a title for a myth based on an event."""
        titles = [
            "Legend of", "Tale of", "Mystery of", "Whispers of", "Shadow of"
        ]
        return f"{random.choice(titles)} {self._generate_event_title(event)}"

    def _mythologize_event(self, event: str) -> str:
        """Transform an event into mythological language."""
        return (
            f"They say that when {event.lower()}, the very spirits of the land "
            f"stirred with ancient power. Some claim to have seen signs and portents, "
            f"while others speak of the Matriarch's blessing upon those who witnessed it."
        )

    def _summarize_elements(self, elements: List[Any], element_type: str) -> str:
        """Create a summary of lore elements for the consistency check."""
        if not elements:
            return "None"

        summaries = []
        for elem in elements[:5]:  # Limit to first 5 for brevity
            if element_type == 'myth':
                summaries.append(
                    f"- {elem.name}: {elem.description[:100]}... "
                    f"(Believability: {elem.believability}/10)"
                )
            elif element_type == 'history':
                summaries.append(
                    f"- {elem.event_name}: {elem.description[:100]}... "
                    f"(Date: {elem.date_description})"
                )
            elif element_type == 'landmark':
                summaries.append(
                    f"- {elem.name}: {elem.description[:100]}... "
                    f"(Type: {elem.landmark_type})"
                )

        if len(elements) > 5:
            summaries.append(f"... and {len(elements) - 5} more")

        return "\n".join(summaries)

    def _summarize_connections(self, connections: List[Dict[str, Any]]) -> str:
        if not connections:
            return "None"

        lines = []
        for c in connections[:5]:
            lines.append(
                f"- {c['element1_type']} #{c['element1_id']} → "
                f"{c['element2_type']} #{c['element2_id']}: "
                f"{str(c['connection_type'])} (Strength: {c['connection_strength']}/10)"
            )

        if len(connections) > 5:
            lines.append(f"... and {len(connections) - 5} more")

        return "\n".join(lines)

    async def _get_location_connections(
        self, location_id: int
    ) -> List[Dict[str, Any]]:
        """
        Return *all* NarrativeConnections touching any lore element
        that belongs to the given location – without calling
        `_get_location_lore_impl()` (avoids infinite recursion).
        """
        async with self.db_connection() as conn:
            # grab ids directly – no other helper calls
            myth_ids = [
            r["id"]
            for r in await conn.fetch(
                """
                SELECT id
                FROM UrbanMyths
                WHERE origin_location = (
                        SELECT location_name FROM Locations
                        WHERE COALESCE(id, location_id) = $1
                     )
                   OR $1 = ANY(regions_known)
                """,
                location_id,
            )
            ]
            history_ids = [
            r["id"]
            for r in await conn.fetch(
                "SELECT id FROM LocalHistories WHERE location_id = $1", location_id
            )
            ]
            landmark_ids = [
            r["id"]
            for r in await conn.fetch(
                "SELECT id FROM Landmarks WHERE location_id = $1", location_id
            )
            ]

        # delegate to the efficient helper
        return await self._get_location_connections_efficient(
            location_id, myth_ids, history_ids, landmark_ids
        )

    async def _get_location_connections_efficient(
        self,
        location_id: int,
        myth_ids: List[int],
        history_ids: List[int],
        landmark_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Get all narrative connections for a location (efficient version)."""
        async with self.db_connection() as conn:
            all_connections = []

            # Fetch connections for each type
            for element_type, ids in [
                ("myth", myth_ids),
                ("history", history_ids),
                ("landmark", landmark_ids)
            ]:
                if ids:
                    connections = await conn.fetch("""
                        SELECT * FROM NarrativeConnections
                        WHERE (element1_type = $1 AND element1_id = ANY($2::int[]))
                       OR (element2_type = $1 AND element2_id = ANY($2::int[]))
                """, element_type, ids)

                all_connections.extend([dict(c) for c in connections])

            # Deduplicate
            seen = set()
            unique_connections = []
            for conn in all_connections:
                if conn['id'] not in seen:
                    seen.add(conn['id'])
                    unique_connections.append(conn)

            return unique_connections

    async def _apply_consistency_fixes(self, suggested_fixes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply consistency fixes using canon system."""
        applied_fixes = []

        async with get_db_connection_context() as conn:
            from lore.core import canon

            for fix in suggested_fixes:
                try:
                    if 'element_type' in fix and 'element_id' in fix:
                        if fix['element_type'] == 'myth' and 'new_description' in fix:
                            await canon.update_urban_myth(
                                self.create_run_context({}), conn,
                                fix['element_id'],
                                {"description": fix['new_description']}
                            )
                            applied_fixes.append(fix)

                        elif fix['element_type'] == 'history' and 'new_description' in fix:
                            # Update history via direct SQL since it's less complex
                            await conn.execute("""
                                UPDATE LocalHistories SET description = $1 WHERE id = $2
                            """, fix['new_description'], fix['element_id'])
                            applied_fixes.append(fix)

                        elif fix['element_type'] == 'landmark' and 'new_description' in fix:
                            await canon.update_landmark(
                                self.create_run_context({}), conn,
                                fix['element_id'],
                                {"description": fix['new_description']}
                            )
                            applied_fixes.append(fix)

                    elif 'connection_id' in fix and 'new_description' in fix:
                        # Update connection
                        await conn.execute("""
                            UPDATE NarrativeConnections
                            SET connection_description = $1, validated = TRUE
                            WHERE id = $2
                        """, fix['new_description'], fix['connection_id'])
                        applied_fixes.append(fix)

                except Exception as e:
                    logger.error(f"Error applying consistency fix: {e}")

        return applied_fixes

    async def _create_suggested_connections(
        self, suggested_connections: List[NarrativeConnection]
    ) -> List[int]:
        """Create suggested narrative connections."""
        created_ids = []

        async with self.db_connection() as conn:
            for connection in suggested_connections:
                try:
                    connection_id = await self._create_narrative_connection(
                        conn,
                        connection.element1_type,
                        connection.element1_id,
                        connection.element2_type,
                        connection.element2_id,
                        connection.connection_type,
                        connection.connection_description,
                        connection.connection_strength
                    )

                    created_ids.append(connection_id)
                    logger.info(
                        f"Created suggested connection #{connection_id} between "
                        f"{connection.element1_type} #{connection.element1_id} and "
                        f"{connection.element2_type} #{connection.element2_id}"
                    )

                except Exception as e:
                    logger.error(f"Error creating suggested connection: {e}")

        return created_ids

    # ===== GOVERNANCE REGISTRATION =====

    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="local_lore_manager",
            directive_text=(
                "Create and manage comprehensive local lore including myths, "
                "histories, and landmarks with strong matriarchal themes and "
                "rich narrative connections."
            ),
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )

    # ===== UTILITY METHODS =====

    def create_run_context(self, ctx) -> RunContextWrapper:
        """Create a run context from the provided context."""
        if isinstance(ctx, RunContextWrapper):
            return ctx
        return RunContextWrapper(context=ctx.context if hasattr(ctx, 'context') else {})

    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return super().get_cache(f"{self.cache_namespace}:{key}")

    def set_cache(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value."""
        return super().set_cache(f"{self.cache_namespace}:{key}", value, ttl)

    def invalidate_cache(self, key: str):
        """Invalidate specific cache key."""
        return super().invalidate_cache(f"{self.cache_namespace}:{key}")

    def invalidate_cache_pattern(self, pattern: str):
        """
        Fire-and-forget invalidation so existing non-awaited calls keep working.
        Use invalidate_cache_pattern_async(...) if you need to await completion.
        """
        return asyncio.create_task(super().invalidate_cache_pattern(f"{self.cache_namespace}:{pattern}"))

    async def invalidate_cache_pattern_async(self, pattern: str):
        """
        Awaitable invalidation for use in new code.
        """
        await super().invalidate_cache_pattern(f"{self.cache_namespace}:{pattern}")

# ===== FUNCTION TOOL WRAPPERS =====
# These are the decorated wrappers that external code will call

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="add_urban_myth",
    action_description="Adding urban myth: {input.name}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def add_urban_myth(ctx, input: MythCreationInput) -> int:
    """
    Add an urban myth to the lore system.

    Args:
        ctx: Context containing user and conversation IDs
        input: MythCreationInput with myth details

    Returns:
        ID of the created myth
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._add_urban_myth_impl(ctx, input)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="add_local_history",
    action_description="Adding local history: {input.event_name}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def add_local_history(ctx, input: HistoryCreationInput) -> int:
    """
    Add a historical event to a location.

    Args:
        ctx: Context containing user and conversation IDs
        input: HistoryCreationInput with event details

    Returns:
        ID of the created history
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._add_local_history_impl(ctx, input)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="add_landmark",
    action_description="Adding landmark: {input.name}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def add_landmark(ctx, input: LandmarkCreationInput) -> int:
    """
    Add a landmark to a location.

    Args:
        ctx: Context containing user and conversation IDs
        input: LandmarkCreationInput with landmark details

    Returns:
        ID of the created landmark
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._add_landmark_impl(ctx, input)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="evolve_myth",
    action_description="Evolving urban myth: {myth_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def evolve_myth(
    ctx,
    myth_id: int,
    evolution_type: EvolutionType,
    causal_factors: Optional[List[str]] = None
) -> NarrativeEvolution:
    """
    Evolve an urban myth based on specified factors.

    Args:
        ctx: Context containing user and conversation IDs
        myth_id: ID of the myth to evolve
        evolution_type: Type of evolution to apply
        causal_factors: Optional list of factors causing evolution

    Returns:
        NarrativeEvolution with before/after states
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._evolve_myth_impl(ctx, myth_id, evolution_type, causal_factors)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="connect_myth_history",
    action_description="Connecting myth {myth_id} to history {history_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def connect_myth_history(ctx, myth_id: int, history_id: int) -> NarrativeConnection:
    """
    Create a narrative connection between a myth and historical event.

    Args:
        ctx: Context containing user and conversation IDs
        myth_id: ID of the myth
        history_id: ID of the historical event

    Returns:
        NarrativeConnection describing the link
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._connect_myth_history_impl(ctx, myth_id, history_id)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="connect_history_landmark",
    action_description="Connecting history {history_id} to landmark {landmark_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def connect_history_landmark(ctx, history_id: int, landmark_id: int) -> NarrativeConnection:
    """
    Create a narrative connection between a historical event and landmark.

    Args:
        ctx: Context containing user and conversation IDs
        history_id: ID of the historical event
        landmark_id: ID of the landmark

    Returns:
        NarrativeConnection describing the link
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._connect_history_landmark_impl(ctx, history_id, landmark_id)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="ensure_narrative_consistency",
    action_description="Ensuring narrative consistency for location {location_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def ensure_narrative_consistency(
    ctx,
    location_id: int,
    auto_fix: bool = True
) -> ConsistencyCheckResult:
    """
    Check and optionally fix narrative consistency for a location.

    Args:
        ctx: Context containing user and conversation IDs
        location_id: ID of the location to check
        auto_fix: Whether to apply suggested fixes automatically

    Returns:
        ConsistencyCheckResult with issues and fixes
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._ensure_narrative_consistency_impl(ctx, location_id, auto_fix)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="get_location_lore",
    action_description="Getting all lore for location: {location_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def get_location_lore(ctx, location_id: int) -> LocationLoreResult:
    """
    Get all lore associated with a location.

    Args:
        ctx: Context containing user and conversation IDs
        location_id: ID of the location

    Returns:
        LocationLoreResult with all myths, histories, landmarks, and connections
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._get_location_lore_impl(ctx, location_id)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_location_lore",
    action_description="Generating lore for location: {location_data.id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def generate_location_lore(ctx, location_data: LocationDataInput) -> Dict[str, Any]:
    """
    Generate comprehensive lore for a location.

    Args:
        ctx: Context containing user and conversation IDs
        location_data: LocationDataInput with location details

    Returns:
        Dictionary with generation statistics
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._generate_location_lore_impl(ctx, location_data)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="evolve_location_lore",
    action_description="Evolving lore for location: {location_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def evolve_location_lore(
    ctx,
    location_id: int,
    event_description: str
) -> LoreEvolutionResult:
    """
    Evolve location lore based on a significant event.

    Args:
        ctx: Context containing user and conversation IDs
        location_id: ID of the location
        event_description: Description of the event causing evolution

    Returns:
        LoreEvolutionResult with changes made
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._evolve_location_lore_impl(ctx, location_id, event_description)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_legend_variants",
    action_description="Creating legend variants for myth {myth_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def generate_legend_variants(
    ctx,
    myth_id: int,
    variant_count: int = 3
) -> Dict[str, Any]:
    """
    Create contradictory versions of a myth.

    Args:
        ctx: Context containing user and conversation IDs
        myth_id: ID of the myth
        variant_count: Number of variants to create

    Returns:
        Dictionary with created variants
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._generate_legend_variants_impl(ctx, myth_id, variant_count)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="develop_tourist_attraction",
    action_description="Developing tourism for myth {myth_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def develop_tourist_attraction(ctx, myth_id: int) -> TouristDevelopment:
    """
    Transform a myth into a tourist attraction plan.

    Args:
        ctx: Context containing user and conversation IDs
        myth_id: ID of the myth

    Returns:
        TouristDevelopment plan
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._develop_tourist_attraction_impl(ctx, myth_id)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="simulate_tradition_dynamics",
    action_description="Simulating tradition dynamics for myth {myth_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def simulate_tradition_dynamics(ctx, myth_id: int) -> TraditionDynamics:
    """
    Compare oral vs written tradition versions of a myth.

    Args:
        ctx: Context containing user and conversation IDs
        myth_id: ID of the myth

    Returns:
        TraditionDynamics comparison
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._simulate_tradition_dynamics_impl(ctx, myth_id)

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="simulate_myth_transmission",
    action_description="Simulating transmission of myth {myth_id}",
    id_from_context=lambda ctx: "local_lore_manager"
)
async def simulate_myth_transmission(
    ctx,
    myth_id: int,
    target_regions: List[str],
    transmission_steps: int = 3
) -> MythTransmissionResult:
    """
    Simulate how a myth spreads and transforms across regions.

    Args:
        ctx: Context containing user and conversation IDs
        myth_id: ID of the myth
        target_regions: List of regions to spread to
        transmission_steps: Number of transmission steps to simulate

    Returns:
        MythTransmissionResult with transformation details
    """
    mgr = await _get_local_lore_manager_from_ctx(ctx)
    return await mgr._simulate_myth_transmission_impl(ctx, myth_id, target_regions, transmission_steps)

# ===== DEPRECATED COMPATIBILITY SHIMS =====
# These are provided to guide users to the new API

def _deprecated_method_notice():
    """Helper to create deprecation notice."""
    return (
        "This method has been moved to a function-tool wrapper. "
        "Import and use the function directly instead of calling the class method. "
        "Example: from lore.managers.local_lore import add_urban_myth"
    )

# If needed, you can add deprecation warnings to the class methods
# LocalLoreManager.add_urban_myth = lambda self, *args: (_ for _ in ()).throw(
#     DeprecationWarning(_deprecated_method_notice())
# )
