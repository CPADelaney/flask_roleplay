# lore/managers/local_lore.py

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from pydantic import BaseModel, Field

# Agents SDK imports
from agents import (
    Agent, function_tool, Runner, trace, RunResultStreaming, ModelSettings,
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, handoff, RunContextWrapper, RunConfig
)

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.managers.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class UrbanMyth(BaseModel):
    """Model for urban myths."""
    name: str
    description: str
    origin_location: Optional[str] = None
    origin_event: Optional[str] = None
    believability: int = Field(6, ge=1, le=10)
    spread_rate: int = Field(5, ge=1, le=10)
    regions_known: List[str] = []
    
    # New fields for narrative evolution
    narrative_style: str = "folklore"
    themes: List[str] = []
    variations: List[Dict[str, str]] = []
    matriarchal_elements: List[str] = []

class LocalHistory(BaseModel):
    """Model for local historical events."""
    location_id: int
    event_name: str
    description: str
    date_description: str = "Some time ago"
    significance: int = Field(5, ge=1, le=10)
    impact_type: str = "cultural"
    notable_figures: List[str] = []
    current_relevance: Optional[str] = None
    commemoration: Optional[str] = None
    
    # New fields for narrative coherence
    connected_myths: List[int] = []
    related_landmarks: List[int] = []
    narrative_category: str = "historical"

class Landmark(BaseModel):
    """Model for landmarks."""
    name: str
    location_id: int
    landmark_type: str
    description: str
    historical_significance: Optional[str] = None
    current_use: Optional[str] = None
    controlled_by: Optional[str] = None
    legends: List[str] = []
    
    # New fields for narrative connections
    connected_histories: List[int] = []
    architectural_style: Optional[str] = None
    symbolic_meaning: Optional[str] = None
    matriarchal_significance: str = "moderate"

class NarrativeEvolution(BaseModel):
    """Model for narrative evolution results."""
    original_element_id: int
    element_type: str
    before_description: str
    after_description: str
    evolution_type: str
    causal_factors: List[str]
    believability_change: int = 0
    significance_change: int = 0

class MythTransmissionSimulation(BaseModel):
    """Model for myth transmission simulation results."""
    myth_id: int
    myth_name: str
    original_regions: List[str]
    new_regions: List[str]
    transmission_path: List[Dict[str, Any]]
    transformation_details: List[Dict[str, Any]]
    final_believability: int
    final_spread_rate: int
    variants_created: int

class LocalLoreManager(BaseLoreManager):
    """
    Enhanced manager for local lore elements with myth evolution agents,
    myth-history-landmark handoff chains, and narrative consistency.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "locallore"
        
        # Initialize specialized agents
        self._init_specialized_agents()
        
    
    def _init_specialized_agents(self):
        """Initialize specialized agents for different lore tasks."""
        # Myth evolution agents
        self.folklore_agent = Agent(
            name="FolkloreEvolutionAgent",
            instructions=(
                "You specialize in evolving folklore-style urban myths. "
                "Create poetic, metaphorical narratives with moral lessons "
                "that center matriarchal values and feminine wisdom."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        self.historical_agent = Agent(
            name="HistoricalEvolutionAgent",
            instructions=(
                "You specialize in evolving myths into pseudo-historical narratives. "
                "Add details that make myths seem like real historical events with dates, "
                "specific people (especially female leaders), and concrete impacts."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        self.supernatural_agent = Agent(
            name="SupernaturalEvolutionAgent",
            instructions=(
                "You specialize in adding supernatural elements to myths. "
                "Introduce divine intervention, magical occurrences, and mystical beings. "
                "Emphasize feminine divine power and matriarchal spiritual authority."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        # Combined myth evolution agent with handoffs
        self.myth_evolution_agent = Agent(
            name="MythEvolutionAgent",
            instructions=(
                "You evolve urban myths over time using different narrative styles. "
                "You select the appropriate specialist based on the desired evolution type."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8),
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
            ]
        )
        
        # Myth-history connector agent
        self.myth_history_connector = Agent(
            name="MythHistoryConnectorAgent",
            instructions=(
                "You specialize in connecting myths with historical events. "
                "Create plausible narrative links between folklore and actual events, "
                "showing how myths might have originated from real occurrences."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        # History-landmark connector agent
        self.history_landmark_connector = Agent(
            name="HistoryLandmarkConnectorAgent",
            instructions=(
                "You specialize in connecting historical events with landmarks. "
                "Create narrative links between events and physical places, "
                "explaining how events led to landmarks or how landmarks witnessed events."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        # Narrative consistency agent
        self.narrative_consistency_agent = Agent(
            name="NarrativeConsistencyAgent",
            instructions=(
                "You ensure consistency across connected local lore elements. "
                "Analyze myths, histories, and landmarks for contradictions or gaps. "
                "Suggest modifications to maintain a cohesive narrative."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
        
        # Myth transmission simulation agent
        self.transmission_agent = Agent(
            name="MythTransmissionAgent",
            instructions=(
                "You simulate how myths spread between regions and populations. "
                "Track changes that occur during transmission, including distortions, "
                "adaptations to local contexts, and shifts in central elements."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )

        # Additional specialized agents
        self.variant_agent = Agent(
            name="LegendVariantAgent",
            instructions=(
                "You create multiple contradictory versions of the same myth, "
                "each with at least one intentional inconsistency."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.9)
        )
        
        self.tourism_agent = Agent(
            name="TouristAttractionAgent",
            instructions=(
                "You transform local myths into commercial tourist attractions, "
                "including marketing angles, attractions, merchandise, etc. "
                "Focus on how matriarchal elements can draw in visitors."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        
        self.oral_written_agent = Agent(
            name="OralWrittenTraditionAgent",
            instructions=(
                "You simulate how a myth changes when recorded in writing vs. shared orally. "
                "Highlight differences in detail, consistency, and local variations. "
                "Always emphasize matriarchal elements in both mediums."
            ),
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )


    async def initialize_tables(self):
        """Ensure all local lore tables exist with enhanced fields."""
        table_definitions = {
            "UrbanMyths": """
                CREATE TABLE UrbanMyths (
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
                    versions_json JSONB,
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_urbanmyths_embedding 
                ON UrbanMyths USING ivfflat (embedding vector_cosine_ops);
            """,
            "LocalHistories": """
                CREATE TABLE LocalHistories (
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
                    embedding VECTOR(1536),
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_localhistories_embedding 
                ON LocalHistories USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_localhistories_location
                ON LocalHistories(location_id);
            """,
            "Landmarks": """
                CREATE TABLE Landmarks (
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
                    embedding VECTOR(1536),
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_landmarks_embedding 
                ON Landmarks USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_landmarks_location
                ON Landmarks(location_id);
            """,
            "NarrativeConnections": """
                CREATE TABLE NarrativeConnections (
                    id SERIAL PRIMARY KEY,
                    element1_type TEXT NOT NULL,
                    element1_id INTEGER NOT NULL,
                    element2_type TEXT NOT NULL,
                    element2_id INTEGER NOT NULL,
                    connection_type TEXT NOT NULL,
                    connection_description TEXT NOT NULL,
                    connection_strength INTEGER CHECK (connection_strength BETWEEN 1 AND 10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_narrativeconnections_embedding 
                ON NarrativeConnections USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_narrativeconnections_elements
                ON NarrativeConnections(element1_id, element2_id);
            """,
            "MythEvolutions": """
                CREATE TABLE MythEvolutions (
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
                    embedding VECTOR(1536),
                    FOREIGN KEY (myth_id) REFERENCES UrbanMyths(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_mythevolutions_embedding 
                ON MythEvolutions USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_mythevolutions_myth
                ON MythEvolutions(myth_id);
            """
        }
        
        await self.initialize_tables_for_class(table_definitions)

    async def _add_urban_myth_impl(
        self,
        ctx,
        name: str,
        description: str,
        origin_location: Optional[str] = None,
        origin_event: Optional[str] = None,
        believability: int = 6,
        spread_rate: int = 5,
        regions_known: List[str] = None,
        narrative_style: str = "folklore",
        themes: List[str] = None,
        matriarchal_elements: List[str] = None
    ) -> int:
        with trace(
            "AddUrbanMyth", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "myth_name": name}
        ):
            await self.ensure_initialized()
            regions_known = regions_known or ["local area"]
            themes = themes or ["mystery", "caution"]
            matriarchal_elements = matriarchal_elements or ["female authority"]
    
            # Apply matriarchal theming
            description = MatriarchalThemingUtils.apply_matriarchal_theme("myth", description)
    
            # Embedding
            embedding_text = f"{name} {description} {narrative_style} {' '.join(themes)}"
            embedding = await generate_embedding(embedding_text)
    
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    myth_id = await conn.fetchval("""
                        INSERT INTO UrbanMyths (
                            name, description, origin_location, origin_event,
                            believability, spread_rate, regions_known, narrative_style,
                            themes, matriarchal_elements, embedding
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        RETURNING id
                    """,
                    name, description, origin_location, origin_event,
                    believability, spread_rate, regions_known, narrative_style,
                    themes, matriarchal_elements, embedding)
    
                    return myth_id

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_urban_myth",
        action_description="Adding urban myth: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def add_urban_myth(
        self,
        ctx,
        name: str,
        description: str,
        origin_location: Optional[str] = None,
        origin_event: Optional[str] = None,
        believability: int = 6,
        spread_rate: int = 5,
        regions_known: List[str] = None,
        narrative_style: str = "folklore",
        themes: List[str] = None,
        matriarchal_elements: List[str] = None
    ) -> int:
        """
        Add an urban myth to the database.
        
        Args:
            name: Name of the urban myth
            description: Description of the urban myth
            origin_location: Location where the myth originated
            origin_event: Event that spawned the myth
            believability: How believable the myth is (1-10)
            spread_rate: How quickly the myth spreads (1-10)
            regions_known: List of regions where the myth is known
            narrative_style: Style of the narrative (folklore, historical, etc.)
            themes: Themes of the myth
            matriarchal_elements: Elements related to matriarchal power
            
        Returns:
            ID of the created urban myth
        """
        return await self._add_urban_myth_impl(
            ctx, name, description, origin_location, origin_event, believability,
            spread_rate, regions_known, narrative_style, themes, matriarchal_elements
        )
    
    async def _add_local_history_impl(
        self,
        ctx,
        location_id: int,
        event_name: str,
        description: str,
        date_description: str = "Some time ago",
        significance: int = 5,
        impact_type: str = "cultural",
        notable_figures: List[str] = None,
        current_relevance: str = None,
        commemoration: str = None,
        connected_myths: List[int] = None,
        related_landmarks: List[int] = None,
        narrative_category: str = "historical"
    ) -> int:
        with trace(
            "AddLocalHistory", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "event_name": event_name}
        ):
            await self.ensure_initialized()
            notable_figures = notable_figures or []
            connected_myths = connected_myths or []
            related_landmarks = related_landmarks or []
    
            description = MatriarchalThemingUtils.apply_matriarchal_theme("history", description)
    
            embedding_text = f"{event_name} {description} {date_description} {narrative_category}"
            embedding = await generate_embedding(embedding_text)
    
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    event_id = await conn.fetchval("""
                        INSERT INTO LocalHistories (
                            location_id, event_name, description, date_description,
                            significance, impact_type, notable_figures,
                            current_relevance, commemoration, connected_myths,
                            related_landmarks, narrative_category, embedding
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        RETURNING id
                    """,
                    location_id, event_name, description, date_description,
                    significance, impact_type, notable_figures,
                    current_relevance, commemoration, connected_myths,
                    related_landmarks, narrative_category, embedding)
    
                    # Create narrative connections if provided
                    if connected_myths:
                        for myth_id in connected_myths:
                            await self._create_narrative_connection(
                                conn, "myth", myth_id, "history", event_id, 
                                "myth_to_history", "Myth has historical basis", 7
                            )
                    
                    if related_landmarks:
                        for landmark_id in related_landmarks:
                            await self._create_narrative_connection(
                                conn, "history", event_id, "landmark", landmark_id, 
                                "history_to_landmark", "Event occurred at landmark", 8
                            )
    
                    # Invalidate relevant cache
                    self.invalidate_cache_pattern(f"local_history_{location_id}")
                    return event_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_local_history",
        action_description="Adding local history event: {event_name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def add_local_history(
        self,
        ctx,
        location_id: int,
        event_name: str,
        description: str,
        date_description: str = "Some time ago",
        significance: int = 5,
        impact_type: str = "cultural",
        notable_figures: List[str] = None,
        current_relevance: str = None,
        commemoration: str = None,
        connected_myths: List[int] = None,
        related_landmarks: List[int] = None,
        narrative_category: str = "historical"
    ) -> int:
        """
        Add a local historical event to the database.
        
        Args:
            location_id: ID of the location
            event_name: Name of the historical event
            description: Description of the event
            date_description: Text description of when the event occurred
            significance: How significant the event was (1-10)
            impact_type: Type of impact (cultural, political, etc.)
            notable_figures: List of notable figures involved
            current_relevance: How the event is still relevant today
            commemoration: How the event is commemorated
            connected_myths: List of myth IDs connected to this event
            related_landmarks: List of landmark IDs related to this event
            narrative_category: Category of the narrative
            
        Returns:
            ID of the created historical event
        """
        return await self._add_local_history_impl(
            ctx, location_id, event_name, description, date_description, significance,
            impact_type, notable_figures, current_relevance, commemoration, 
            connected_myths, related_landmarks, narrative_category
        )
       
    async def _add_landmark_impl(
        self,
        ctx,
        name: str,
        location_id: int,
        landmark_type: str,
        description: str,
        historical_significance: str = None,
        current_use: str = None,
        controlled_by: str = None,
        legends: List[str] = None,
        connected_histories: List[int] = None,
        architectural_style: str = None,
        symbolic_meaning: str = None,
        matriarchal_significance: str = "moderate"
    ) -> int:
        with trace(
            "AddLandmark", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "landmark_name": name}
        ):
            await self.ensure_initialized()
            legends = legends or []
            connected_histories = connected_histories or []
    
            description = MatriarchalThemingUtils.apply_matriarchal_theme("landmark", description)
    
            embedding_text = f"{name} {landmark_type} {description} {matriarchal_significance}"
            embedding = await generate_embedding(embedding_text)
    
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    landmark_id = await conn.fetchval("""
                        INSERT INTO Landmarks (
                            name, location_id, landmark_type, description,
                            historical_significance, current_use, controlled_by,
                            legends, connected_histories, architectural_style,
                            symbolic_meaning, matriarchal_significance, embedding
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        RETURNING id
                    """,
                    name, location_id, landmark_type, description,
                    historical_significance, current_use, controlled_by,
                    legends, connected_histories, architectural_style,
                    symbolic_meaning, matriarchal_significance, embedding)
    
                    # Create narrative connections if provided
                    if connected_histories:
                        for history_id in connected_histories:
                            await self._create_narrative_connection(
                                conn, "history", history_id, "landmark", landmark_id, 
                                "history_to_landmark", "Event occurred at landmark", 8
                            )
    
                    self.invalidate_cache_pattern(f"landmarks_{location_id}")
                    return landmark_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_landmark",
        action_description="Adding landmark: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def add_landmark(
        self,
        ctx,
        name: str,
        location_id: int,
        landmark_type: str,
        description: str,
        historical_significance: str = None,
        current_use: str = None,
        controlled_by: str = None,
        legends: List[str] = None,
        connected_histories: List[int] = None,
        architectural_style: str = None,
        symbolic_meaning: str = None,
        matriarchal_significance: str = "moderate"
    ) -> int:
        """
        Add a landmark to the database.
        
        Args:
            name: Name of the landmark
            location_id: ID of the location
            landmark_type: Type of landmark (natural, building, etc.)
            description: Description of the landmark
            historical_significance: Historical significance of the landmark
            current_use: Current use of the landmark
            controlled_by: Who controls the landmark
            legends: List of legends associated with the landmark
            connected_histories: List of history IDs connected to this landmark
            architectural_style: Architectural style of the landmark
            symbolic_meaning: Symbolic meaning of the landmark
            matriarchal_significance: Significance to matriarchal power (low/moderate/high)
            
        Returns:
            ID of the created landmark
        """
        return await self._add_landmark_impl(
            ctx, name, location_id, landmark_type, description, historical_significance,
            current_use, controlled_by, legends, connected_histories, architectural_style,
            symbolic_meaning, matriarchal_significance
        )

    async def _create_narrative_connection(
        self,
        conn,
        element1_type: str,
        element1_id: int,
        element2_type: str,
        element2_id: int,
        connection_type: str,
        connection_description: str,
        connection_strength: int
    ) -> int:
        """Helper method to create a narrative connection between elements."""
        embedding_text = f"{element1_type} {element1_id} {connection_type} {element2_type} {element2_id} {connection_description}"
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
        connection_type, connection_description, connection_strength, embedding)
        
        return connection_id

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_myth",
        action_description="Evolving urban myth: {myth_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def evolve_myth(
        self,
        ctx,
        myth_id: int,
        evolution_type: str,
        causal_factors: List[str] = None
    ) -> NarrativeEvolution:
        """
        Evolve an urban myth using a specialized narrative style agent.
        
        Args:
            myth_id: ID of the myth to evolve
            evolution_type: Type of evolution (folklore, historical, supernatural)
            causal_factors: Factors causing the evolution
            
        Returns:
            NarrativeEvolution object with before/after details
        """
        with trace(
            "EvolveMythWithAgent", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "myth_id": myth_id,
                "evolution_type": evolution_type
            }
        ):
            run_ctx = self.create_run_context(ctx)
            causal_factors = causal_factors or ["natural evolution"]
            
            # Fetch the myth
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    myth = await conn.fetchrow("""
                        SELECT * FROM UrbanMyths WHERE id = $1
                    """, myth_id)
                    
                    if not myth:
                        raise ValueError(f"Myth with ID {myth_id} not found")
            
            # Convert to dictionary
            myth_data = dict(myth)
            
            # Select the appropriate agent based on evolution type
            if evolution_type.lower() == "folklore":
                # Use folklore evolution agent
                evolution_prompt = f"""
                Evolve this urban myth using folklore narrative style:
                
                MYTH: {myth_data['name']}
                CURRENT DESCRIPTION: {myth_data['description']}
                BELIEVABILITY: {myth_data['believability']}/10
                SPREAD RATE: {myth_data['spread_rate']}/10
                
                CAUSAL FACTORS:
                {', '.join(causal_factors)}
                
                Transform this myth into a more folkloric version with:
                - Moral lessons
                - Metaphorical elements
                - Cultural wisdom
                - Matriarchal values
                
                Return the evolved myth description that maintains the core identity
                but enhances the folkloric qualities.
                """
                
                result = await Runner.run(
                    self.folklore_agent, 
                    evolution_prompt, 
                    context=run_ctx.context
                )
                
                # Expect believability to increase slightly, spread rate to increase moderately
                believability_change = random.randint(0, 2)
                spread_rate_change = random.randint(1, 3)
                
            elif evolution_type.lower() == "historical":
                # Use historical evolution agent
                evolution_prompt = f"""
                Evolve this urban myth into a pseudo-historical narrative:
                
                MYTH: {myth_data['name']}
                CURRENT DESCRIPTION: {myth_data['description']}
                BELIEVABILITY: {myth_data['believability']}/10
                SPREAD RATE: {myth_data['spread_rate']}/10
                
                CAUSAL FACTORS:
                {', '.join(causal_factors)}
                
                Transform this myth into a more historical version with:
                - Specific dates or time periods
                - Named individuals (especially female figures)
                - Concrete locations
                - Cause-and-effect relationships
                
                Return the evolved myth description that sounds like a historical account
                while maintaining the core narrative elements.
                """
                
                result = await Runner.run(
                    self.historical_agent, 
                    evolution_prompt, 
                    context=run_ctx.context
                )
                
                # Expect believability to increase substantially, spread rate moderate
                believability_change = random.randint(1, 3)
                spread_rate_change = random.randint(0, 2)
                
            elif evolution_type.lower() == "supernatural":
                # Use supernatural evolution agent
                evolution_prompt = f"""
                Evolve this urban myth by adding supernatural elements:
                
                MYTH: {myth_data['name']}
                CURRENT DESCRIPTION: {myth_data['description']}
                BELIEVABILITY: {myth_data['believability']}/10
                SPREAD RATE: {myth_data['spread_rate']}/10
                
                CAUSAL FACTORS:
                {', '.join(causal_factors)}
                
                Transform this myth by adding:
                - Divine or magical intervention
                - Supernatural beings or forces
                - Mystical consequences
                - Feminine divine power
                
                Return the evolved myth description with these new supernatural elements
                while maintaining the core narrative identity.
                """
                
                result = await Runner.run(
                    self.supernatural_agent, 
                    evolution_prompt, 
                    context=run_ctx.context
                )
                
                # Expect believability to decrease slightly, spread rate to increase
                believability_change = random.randint(-2, 0)
                spread_rate_change = random.randint(1, 3)
                
            else:
                # Use the general myth evolution agent with handoffs
                evolution_prompt = f"""
                Evolve this urban myth using the most appropriate narrative style:
                
                MYTH: {myth_data['name']}
                CURRENT DESCRIPTION: {myth_data['description']}
                BELIEVABILITY: {myth_data['believability']}/10
                SPREAD RATE: {myth_data['spread_rate']}/10
                
                CAUSAL FACTORS:
                {', '.join(causal_factors)}
                
                Choose the most appropriate evolution style (folklore, historical, supernatural)
                and transform the myth accordingly.
                
                Return the evolved myth description.
                """
                
                result = await Runner.run(
                    self.myth_evolution_agent, 
                    evolution_prompt, 
                    context=run_ctx.context
                )
                
                # Default changes
                believability_change = random.randint(-1, 1)
                spread_rate_change = random.randint(0, 2)
            
            # Get the evolved description
            new_description = result.final_output
            
            # Apply matriarchal theming to ensure consistency
            new_description = MatriarchalThemingUtils.apply_matriarchal_theme("myth", new_description)
            
            # Calculate new values
            new_believability = max(1, min(10, myth_data['believability'] + believability_change))
            new_spread_rate = max(1, min(10, myth_data['spread_rate'] + spread_rate_change))
            
            # Store the evolution
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Record the evolution
                    evo_id = await conn.fetchval("""
                        INSERT INTO MythEvolutions (
                            myth_id, previous_version, new_version, evolution_type,
                            causal_factors, believability_before, believability_after,
                            spread_rate_before, spread_rate_after, regions_known_before,
                            regions_known_after
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        RETURNING id
                    """,
                    myth_id, myth_data['description'], new_description, evolution_type,
                    causal_factors, myth_data['believability'], new_believability,
                    myth_data['spread_rate'], new_spread_rate, myth_data['regions_known'],
                    myth_data['regions_known'])
                    
                    # Update the myth itself
                    await conn.execute("""
                        UPDATE UrbanMyths
                        SET description = $1,
                            believability = $2,
                            spread_rate = $3,
                            narrative_style = $4
                        WHERE id = $5
                    """, new_description, new_believability, new_spread_rate, evolution_type, myth_id)
                    
                    # Update the embedding
                    embedding_text = f"{myth_data['name']} {new_description} {evolution_type}"
                    new_embedding = await generate_embedding(embedding_text)
                    
                    await conn.execute("""
                        UPDATE UrbanMyths
                        SET embedding = $1
                        WHERE id = $2
                    """, new_embedding, myth_id)
            
            # Return the evolution details
            return NarrativeEvolution(
                original_element_id=myth_id,
                element_type="myth",
                before_description=myth_data['description'],
                after_description=new_description,
                evolution_type=evolution_type,
                causal_factors=causal_factors,
                believability_change=believability_change,
                significance_change=0  # Myths don't have significance, but histories do
            )

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="connect_myth_history",
        action_description="Connecting myth {myth_id} to history {history_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def connect_myth_history(
        self,
        ctx,
        myth_id: int,
        history_id: int
    ) -> Dict[str, Any]:
        """
        Create a narrative connection between a myth and a historical event.
        
        Args:
            myth_id: ID of the myth
            history_id: ID of the historical event
            
        Returns:
            Dictionary with connection details
        """
        with trace(
            "ConnectMythHistory", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "myth_id": myth_id,
                "history_id": history_id
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Fetch the myth and history
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    myth = await conn.fetchrow("""
                        SELECT * FROM UrbanMyths WHERE id = $1
                    """, myth_id)
                    
                    history = await conn.fetchrow("""
                        SELECT * FROM LocalHistories WHERE id = $1
                    """, history_id)
                    
                    if not myth or not history:
                        return {"error": "Myth or history not found"}
            
            # Create prompt for connection
            myth_data = dict(myth)
            history_data = dict(history)
            
            connection_prompt = f"""
            Create a narrative connection between this myth and historical event:
            
            MYTH:
            {json.dumps(myth_data, indent=2)}
            
            HISTORICAL EVENT:
            {json.dumps(history_data, indent=2)}
            
            Analyze how these two narratives might be connected. Consider:
            1. How the myth might have originated from the historical event
            2. How the historical event might reinforce or be explained by the myth
            3. Shared thematic elements or symbols
            4. How feminine or matriarchal power appears in both
            
            Return JSON with:
            - connection_type: brief label for the connection
            - connection_description: detailed explanation
            - connection_strength: 1-10 how strongly they connect
            - suggested_modifications: changes to either that would strengthen the connection
            """
            
            result = await Runner.run(
                self.myth_history_connector, 
                connection_prompt, 
                context=run_ctx.context
            )
            
            try:
                connection_data = json.loads(result.final_output)
            except json.JSONDecodeError:
                connection_data = {
                    "connection_type": "myth_history_link",
                    "connection_description": "Narrative link between myth and history",
                    "connection_strength": 5,
                    "suggested_modifications": ["No specific modifications suggested"]
                }
            
            # Create the connection in the database
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Create the narrative connection
                    connection_id = await self._create_narrative_connection(
                        conn, 
                        "myth", myth_id, 
                        "history", history_id, 
                        connection_data.get("connection_type", "myth_history_link"),
                        connection_data.get("connection_description", "Narrative link between myth and history"),
                        connection_data.get("connection_strength", 5)
                    )
                    
                    # Update the related fields in both tables
                    # First check if history already has this myth connected
                    history_myths = history_data.get("connected_myths", [])
                    if myth_id not in history_myths:
                        history_myths.append(myth_id)
                        await conn.execute("""
                            UPDATE LocalHistories
                            SET connected_myths = $1
                            WHERE id = $2
                        """, history_myths, history_id)
            
            # Return the connection data
            return {
                "connection_id": connection_id,
                "myth_id": myth_id,
                "myth_name": myth_data["name"],
                "history_id": history_id,
                "history_name": history_data["event_name"],
                "connection_type": connection_data.get("connection_type", "myth_history_link"),
                "connection_description": connection_data.get("connection_description", "Narrative link between myth and history"),
                "connection_strength": connection_data.get("connection_strength", 5),
                "suggested_modifications": connection_data.get("suggested_modifications", [])
            }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="connect_history_landmark",
        action_description="Connecting history {history_id} to landmark {landmark_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def connect_history_landmark(
        self,
        ctx,
        history_id: int,
        landmark_id: int
    ) -> Dict[str, Any]:
        """
        Create a narrative connection between a historical event and a landmark.
        
        Args:
            history_id: ID of the historical event
            landmark_id: ID of the landmark
            
        Returns:
            Dictionary with connection details
        """
        with trace(
            "ConnectHistoryLandmark", 
            group_id=self.trace_group_id,
            metadata={
                **self.trace_metadata, 
                "history_id": history_id,
                "landmark_id": landmark_id
            }
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Fetch the history and landmark
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    history = await conn.fetchrow("""
                        SELECT * FROM LocalHistories WHERE id = $1
                    """, history_id)
                    
                    landmark = await conn.fetchrow("""
                        SELECT * FROM Landmarks WHERE id = $1
                    """, landmark_id)
                    
                    if not history or not landmark:
                        return {"error": "History or landmark not found"}
                    
                    # Check if they belong to the same location
                    if history["location_id"] != landmark["location_id"]:
                        location1 = await conn.fetchrow("""
                            SELECT location_name FROM Locations WHERE id = $1
                        """, history["location_id"])
                        
                        location2 = await conn.fetchrow("""
                            SELECT location_name FROM Locations WHERE id = $1
                        """, landmark["location_id"])
                        
                        location_warning = (
                            f"Warning: History belongs to {location1['location_name']} "
                            f"while landmark belongs to {location2['location_name']}"
                        )
                    else:
                        location_warning = None
            
            # Create prompt for connection
            history_data = dict(history)
            landmark_data = dict(landmark)
            
            connection_prompt = f"""
            Create a narrative connection between this historical event and landmark:
            
            HISTORICAL EVENT:
            {json.dumps(history_data, indent=2)}
            
            LANDMARK:
            {json.dumps(landmark_data, indent=2)}
            
            {location_warning or ""}
            
            Analyze how these two elements might be connected. Consider:
            1. How the event might have occurred at or impacted the landmark
            2. How the landmark might commemorate or be changed by the event
            3. Shared symbolic meaning or importance
            4. How matriarchal authority is reflected in both
            
            Return JSON with these fields:
            - connection_type: brief label for the connection
            - connection_description: detailed explanation
            - connection_strength: 1-10 how strongly they connect
            - suggested_modifications: changes to either that would strengthen the connection
            """
            
            result = await Runner.run(
                self.history_landmark_connector, 
                connection_prompt, 
                context=run_ctx.context
            )
            
            try:
                connection_data = json.loads(result.final_output)
            except json.JSONDecodeError:
                connection_data = {
                    "connection_type": "history_landmark_link",
                    "connection_description": "Narrative link between history and landmark",
                    "connection_strength": 5,
                    "suggested_modifications": ["No specific modifications suggested"]
                }
            
            # Create the connection in the database
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Create the narrative connection
                    connection_id = await self._create_narrative_connection(
                        conn, 
                        "history", history_id, 
                        "landmark", landmark_id, 
                        connection_data.get("connection_type", "history_landmark_link"),
                        connection_data.get("connection_description", "Narrative link between history and landmark"),
                        connection_data.get("connection_strength", 5)
                    )
                    
                    # Update the related fields in both tables
                    # First check if history already has this landmark connected
                    related_landmarks = history_data.get("related_landmarks", [])
                    if landmark_id not in related_landmarks:
                        related_landmarks.append(landmark_id)
                        await conn.execute("""
                            UPDATE LocalHistories
                            SET related_landmarks = $1
                            WHERE id = $2
                        """, related_landmarks, history_id)
                    
                    # Then check if landmark already has this history connected
                    connected_histories = landmark_data.get("connected_histories", [])
                    if history_id not in connected_histories:
                        connected_histories.append(history_id)
                        await conn.execute("""
                            UPDATE Landmarks
                            SET connected_histories = $1
                            WHERE id = $2
                        """, connected_histories, landmark_id)
            
            # Return the connection data
            return {
                "connection_id": connection_id,
                "history_id": history_id,
                "history_name": history_data["event_name"],
                "landmark_id": landmark_id,
                "landmark_name": landmark_data["name"],
                "connection_type": connection_data.get("connection_type", "history_landmark_link"),
                "connection_description": connection_data.get("connection_description", "Narrative link between history and landmark"),
                "connection_strength": connection_data.get("connection_strength", 5),
                "suggested_modifications": connection_data.get("suggested_modifications", []),
                "location_warning": location_warning
            }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="ensure_narrative_consistency",
        action_description="Ensuring narrative consistency for location {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def ensure_narrative_consistency(
        self,
        ctx,
        location_id: int
    ) -> Dict[str, Any]:
        """
        Analyze and fix inconsistencies in the narrative web for a location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with consistency check results
        """
        with trace(
            "EnsureNarrativeConsistency", 
            group_id=self.trace_group_id,
            metadata={**self.trace_metadata, "location_id": location_id}
        ):
            run_ctx = self.create_run_context(ctx)
            
            # Get all lore for the location
            location_lore = await self.get_location_lore(run_ctx, location_id)
            if "error" in location_lore:
                return location_lore
            
            # Get all narrative connections
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Fetch connections involving myths
                    myth_connections = []
                    for myth in location_lore.get("myths", []):
                        connections = await conn.fetch("""
                            SELECT * FROM NarrativeConnections
                            WHERE (element1_type = 'myth' AND element1_id = $1)
                               OR (element2_type = 'myth' AND element2_id = $1)
                        """, myth["id"])
                        
                        myth_connections.extend([dict(c) for c in connections])
                    
                    # Fetch connections involving histories
                    history_connections = []
                    for history in location_lore.get("histories", []):
                        connections = await conn.fetch("""
                            SELECT * FROM NarrativeConnections
                            WHERE (element1_type = 'history' AND element1_id = $1)
                               OR (element2_type = 'history' AND element2_id = $1)
                        """, history["id"])
                        
                        history_connections.extend([dict(c) for c in connections])
                    
                    # Fetch connections involving landmarks
                    landmark_connections = []
                    for landmark in location_lore.get("landmarks", []):
                        connections = await conn.fetch("""
                            SELECT * FROM NarrativeConnections
                            WHERE (element1_type = 'landmark' AND element1_id = $1)
                               OR (element2_type = 'landmark' AND element2_id = $1)
                        """, landmark["id"])
                        
                        landmark_connections.extend([dict(c) for c in connections])
            
            # Deduplicate connections
            all_connections = {}
            for conn in myth_connections + history_connections + landmark_connections:
                conn_id = conn["id"]
                if conn_id not in all_connections:
                    all_connections[conn_id] = conn
            
            # Prepare the consistency check prompt
            consistency_prompt = f"""
            Analyze narrative consistency for this location:
            
            LOCATION: {location_lore['location']['location_name']}
            
            MYTHS:
            {json.dumps(location_lore.get('myths', []), indent=2)}
            
            HISTORICAL EVENTS:
            {json.dumps(location_lore.get('histories', []), indent=2)}
            
            LANDMARKS:
            {json.dumps(location_lore.get('landmarks', []), indent=2)}
            
            NARRATIVE CONNECTIONS:
            {json.dumps(list(all_connections.values()), indent=2)}
            
            Identify inconsistencies, gaps, or contradictions between connected elements.
            Focus on:
            1. Timeline inconsistencies
            2. Contradictory facts or descriptions
            3. Thematic contradictions
            4. Logical flaws in connections
            5. Matriarchal power representation inconsistencies
            
            Return JSON with:
            - inconsistencies: array of specific issues
            - suggested_fixes: array of specific changes to resolve issues
            - consistency_rating: 1-10 overall narrative coherence
            - potential_new_connections: suggestions for new connections to improve coherence
            """
            
            result = await Runner.run(
                self.narrative_consistency_agent, 
                consistency_prompt, 
                context=run_ctx.context
            )
            
            try:
                consistency_data = json.loads(result.final_output)
            except json.JSONDecodeError:
                consistency_data = {
                    "inconsistencies": ["Unable to parse consistency check results"],
                    "suggested_fixes": [],
                    "consistency_rating": 5,
                    "potential_new_connections": []
                }
            
            # Apply suggested fixes if any
            fixes_applied = []
            if consistency_data.get("suggested_fixes"):
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        for fix in consistency_data["suggested_fixes"]:
                            try:
                                if fix.get("element_type") == "myth" and fix.get("element_id") and fix.get("new_description"):
                                    await conn.execute("""
                                        UPDATE UrbanMyths
                                        SET description = $1
                                        WHERE id = $2
                                    """, fix["new_description"], fix["element_id"])
                                    
                                    fixes_applied.append(f"Updated myth {fix['element_id']}")
                                
                                elif fix.get("element_type") == "history" and fix.get("element_id") and fix.get("new_description"):
                                    await conn.execute("""
                                        UPDATE LocalHistories
                                        SET description = $1
                                        WHERE id = $2
                                    """, fix["new_description"], fix["element_id"])
                                    
                                    fixes_applied.append(f"Updated history {fix['element_id']}")
                                
                                elif fix.get("element_type") == "landmark" and fix.get("element_id") and fix.get("new_description"):
                                    await conn.execute("""
                                        UPDATE Landmarks
                                        SET description = $1
                                        WHERE id = $2
                                    """, fix["new_description"], fix["element_id"])
                                    
                                    fixes_applied.append(f"Updated landmark {fix['element_id']}")
                                
                                elif fix.get("connection_id") and fix.get("new_description"):
                                    await conn.execute("""
                                        UPDATE NarrativeConnections
                                        SET connection_description = $1
                                        WHERE id = $2
                                    """, fix["new_description"], fix["connection_id"])
                                    
                                    fixes_applied.append(f"Updated connection {fix['connection_id']}")
                            except Exception as e:
                                logging.error(f"Error applying fix: {e}")
            
            # Create suggested connections if any
            new_connections = []
            if consistency_data.get("potential_new_connections"):
                async with self.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        for new_conn in consistency_data["potential_new_connections"]:
                            try:
                                if all(k in new_conn for k in ["element1_type", "element1_id", "element2_type", "element2_id"]):
                                    conn_id = await self._create_narrative_connection(
                                        conn,
                                        new_conn["element1_type"],
                                        new_conn["element1_id"],
                                        new_conn["element2_type"],
                                        new_conn["element2_id"],
                                        new_conn.get("connection_type", "narrative_link"),
                                        new_conn.get("connection_description", "Generated connection for narrative consistency"),
                                        new_conn.get("connection_strength", 5)
                                    )
                                    
                                    new_connections.append(f"Created connection {conn_id}")
                            except Exception as e:
                                logging.error(f"Error creating connection: {e}")
            
            # Invalidate cache
            self.invalidate_cache_pattern(f"location_lore_{location_id}")
            
            # Return results
            return {
                "location_id": location_id,
                "location_name": location_lore['location']['location_name'],
                "consistency_check": consistency_data,
                "fixes_applied": fixes_applied,
                "new_connections": new_connections
            }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_myth_transmission",
        action_description="Simulating transmission of myth {myth_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def simulate_myth_transmission(
        self,
        ctx,
        myth_id: int,
        target_regions: List[str],
        transmission_steps: int = 3
    ) -> MythTransmissionSimulation:
        """
        Simulate how a myth spreads and changes as it moves between regions.
        
        Args:
            myth_id: ID of the myth to transmit
            target_regions: Regions where the myth will spread
            transmission_steps: Number of transmission steps to simulate
            
        Returns:
            MythTransmissionSimulation object with transmission details
        """
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
            
            # Fetch the original myth
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    myth = await conn.fetchrow("""
                        SELECT * FROM UrbanMyths WHERE id = $1
                    """, myth_id)
                    
                    if not myth:
                        raise ValueError(f"Myth with ID {myth_id} not found")
                    
                    # Get cultural elements for context
                    cultures = await conn.fetch("""
                        SELECT name, element_type, description
                        FROM CulturalElements
                        LIMIT 5
                    """)
                    cultural_context = [dict(c) for c in cultures]
            
            # Convert to dictionary
            myth_data = dict(myth)
            original_regions = myth_data.get("regions_known", [])
            new_regions = [r for r in target_regions if r not in original_regions]
            
            # If no new regions, return early
            if not new_regions:
                return MythTransmissionSimulation(
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
            
            # Create transmission simulation prompt
            transmission_prompt = f"""
            Simulate how this myth spreads and transforms as it moves between regions:
            
            MYTH:
            {json.dumps(myth_data, indent=2)}
            
            ORIGINAL REGIONS: {', '.join(original_regions)}
            TARGET REGIONS: {', '.join(new_regions)}
            
            CULTURAL CONTEXT:
            {json.dumps(cultural_context, indent=2)}
            
            Simulate {transmission_steps} steps of transmission, showing:
            1. How the myth travels (trade routes, travelers, etc.)
            2. How each new culture modifies the myth
            3. How matriarchal elements are reinforced or adapted
            4. Changes in believability and spread rate
            
            Return a MythTransmissionSimulation object with all required fields.
            """
            
            # Use the transmission agent with structured output
            transmission_agent = self.transmission_agent.clone(
                output_type=MythTransmissionSimulation
            )
            
            result = await Runner.run(
                transmission_agent, 
                transmission_prompt, 
                context=run_ctx.context
            )
            
            transmission_data = result.final_output
            
            # Apply the transmission results to the database
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Update the original myth with new regions and possibly changed stats
                    updated_regions = original_regions + new_regions
                    await conn.execute("""
                        UPDATE UrbanMyths
                        SET regions_known = $1,
                            believability = $2,
                            spread_rate = $3
                        WHERE id = $4
                    """, updated_regions, transmission_data.final_believability, 
                        transmission_data.final_spread_rate, myth_id)
                    
                    # Create variants if needed
                    for i, transformation in enumerate(transmission_data.transformation_details):
                        if i >= transmission_data.variants_created:
                            break
                            
                        if "variant_description" in transformation:
                            # Create a new variant
                            variant_name = f"{myth_data['name']} ({new_regions[i % len(new_regions)]} Variant)"
                            variant_description = transformation["variant_description"]
                            
                            # Apply matriarchal theming
                            variant_description = MatriarchalThemingUtils.apply_matriarchal_theme("myth", variant_description)
                            
                            # Create the variant
                            await self.add_urban_myth(
                                run_ctx,
                                name=variant_name,
                                description=variant_description,
                                origin_location=new_regions[i % len(new_regions)],
                                origin_event=f"Transmission of myth '{myth_data['name']}'",
                                believability=transmission_data.final_believability,
                                spread_rate=transmission_data.final_spread_rate,
                                regions_known=[new_regions[i % len(new_regions)]],
                                narrative_style=myth_data.get("narrative_style", "folklore"),
                                themes=myth_data.get("themes", []),
                                matriarchal_elements=myth_data.get("matriarchal_elements", [])
                            )
            
            return transmission_data

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_location_lore",
        action_description="Getting all lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def get_location_lore(self, ctx, location_id: int) -> Dict[str, Any]:
        """
        Get all lore associated with a location (myths, history, landmarks).
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with all location lore
        """
        cache_key = f"location_lore_{location_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                location = await conn.fetchrow("""
                    SELECT id, location_name
                    FROM Locations
                    WHERE id = $1
                """, location_id)
                if not location:
                    return {"error": "Location not found"}

                location_name = location["location_name"]

                # Histories
                histories = await conn.fetch("""
                    SELECT id, event_name, description, date_description,
                           significance, impact_type, notable_figures,
                           current_relevance, commemoration
                    FROM LocalHistories
                    WHERE location_id = $1
                    ORDER BY significance DESC
                """, location_id)

                # Landmarks
                landmarks = await conn.fetch("""
                    SELECT id, name, landmark_type, description,
                           historical_significance, current_use,
                           controlled_by, legends
                    FROM Landmarks
                    WHERE location_id = $1
                """, location_id)

                # Myths
                myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate
                    FROM UrbanMyths
                    WHERE origin_location = $1 OR $1 = ANY(regions_known)
                """, location_name)

                result = {
                    "location": dict(location),
                    "histories": [dict(h) for h in histories],
                    "landmarks": [dict(l) for l in landmarks],
                    "myths": [dict(m) for m in myths]
                }
                self.set_cache(cache_key, result)
                return result

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_location_lore",
        action_description="Generating lore for location: {location_data['id']}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def generate_location_lore(self, ctx, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive lore for a location using LLM calls 
        for myths, local histories, and landmarks.
        
        Args:
            location_data: Dictionary with location information
            
        Returns:
            Dictionary with generated lore
        """
        run_ctx = self.create_run_context(ctx)

        if not location_data or "id" not in location_data:
            return {"error": "Invalid location data"}

        location_id = location_data["id"]

        # Generate myths
        myths = await self._generate_myths_for_location(run_ctx, location_data)
        # Generate local histories
        histories = await self._generate_local_history(run_ctx, location_data)
        # Generate landmarks
        landmarks = await self._generate_landmarks(run_ctx, location_data)

        self.invalidate_cache(f"location_lore_{location_id}")

        return {
            "location": location_data,
            "generated_myths": myths,
            "generated_histories": histories,
            "generated_landmarks": landmarks
        }

    async def _generate_myths_for_location(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate urban myths for a location."""
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        desc = location_data.get('description', '')

        prompt = f"""
        Generate 2-3 urban myths or local legends associated with this location:

        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {desc}

        Requirements:
        1. Somewhat believable but with fantastical elements
        2. Reflect local concerns or history
        3. Ties to matriarchal power structures

        Return JSON array with:
        - name
        - description
        - believability (1-10)
        - spread_rate (1-10)
        - origin
        """

        myth_agent = Agent(
            name="UrbanMythAgent",
            instructions="You create urban myths and local legends for locations.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        run_config = RunConfig(workflow_name="GenerateMyths")
        result = await Runner.run(myth_agent, prompt, context=ctx.context, run_config=run_config)

        saved_myths = []
        try:
            myths = json.loads(result.final_output)
            if not isinstance(myths, list):
                myths = [myths] if isinstance(myths, dict) else []

            for myth in myths:
                name = myth.get('name')
                description = myth.get('description')
                believability = myth.get('believability', random.randint(4,8))
                spread_rate = myth.get('spread_rate', random.randint(3,7))

                if not name or not description:
                    continue

                try:
                    myth_id = await self.add_urban_myth(
                        ctx,
                        name=name,
                        description=description,
                        origin_location=location_name,
                        believability=believability,
                        spread_rate=spread_rate,
                        regions_known=[location_name]
                    )
                    myth['id'] = myth_id
                    saved_myths.append(myth)
                except Exception as e:
                    logger.error(f"Error saving urban myth '{name}': {e}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse myths JSON: {result.final_output}")

        return saved_myths

    async def _generate_local_history(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate local historical events for a location."""
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        desc = location_data.get('description', '')

        prompt = f"""
        Generate 2-3 local historical events for this location:

        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {desc}

        Requirements:
        - Reflect local development, conflict, or cultural shifts
        - At least one event about matriarchal power
        - Different time frames

        Return JSON array with:
        - event_name
        - description
        - date_description
        - significance (1-10)
        - impact_type
        - notable_figures
        - current_relevance
        """

        history_agent = Agent(
            name="LocalHistoryAgent",
            instructions="You create local historical events for locations.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        run_config = RunConfig(workflow_name="GenerateLocalHistory")
        result = await Runner.run(history_agent, prompt, context=ctx.context, run_config=run_config)

        saved_events = []
        try:
            events = json.loads(result.final_output)
            if not isinstance(events, list):
                events = [events] if isinstance(events, dict) else []

            for evt in events:
                event_name = evt.get('event_name')
                description = evt.get('description')
                date_description = evt.get('date_description', 'Some time ago')
                significance = evt.get('significance', 5)
                impact_type = evt.get('impact_type', 'cultural')
                notable_figures = evt.get('notable_figures', [])
                current_relevance = evt.get('current_relevance')

                if not event_name or not description:
                    continue

                try:
                    event_id = await self.add_local_history(
                        ctx,
                        location_id=location_id,
                        event_name=event_name,
                        description=description,
                        date_description=date_description,
                        significance=significance,
                        impact_type=impact_type,
                        notable_figures=notable_figures,
                        current_relevance=current_relevance
                    )
                    evt['id'] = event_id
                    saved_events.append(evt)
                except Exception as e:
                    logger.error(f"Error saving local history '{event_name}': {e}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse local history JSON: {result.final_output}")

        return saved_events

    async def _generate_landmarks(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate landmarks for a location."""
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        desc = location_data.get('description', '')

        prompt = f"""
        Generate 2-3 landmarks for this location:

        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {desc}

        Requirements:
        1. At least one natural landmark (if relevant)
        2. At least one architectural/built landmark
        3. At least one linked to matriarchal power

        Return JSON array with:
        - name
        - landmark_type
        - description
        - historical_significance
        - current_use
        - controlled_by
        - legends (array)
        """

        landmark_agent = Agent(
            name="LandmarkAgent",
            instructions="You create landmarks for specific locations.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )
        run_config = RunConfig(workflow_name="GenerateLandmarks")
        result = await Runner.run(landmark_agent, prompt, context=ctx.context, run_config=run_config)

        saved_landmarks = []
        try:
            landmarks = json.loads(result.final_output)
            if not isinstance(landmarks, list):
                landmarks = [landmarks] if isinstance(landmarks, dict) else []

            for lm in landmarks:
                name = lm.get('name')
                lm_type = lm.get('landmark_type', 'building')
                description = lm.get('description')
                hist_signif = lm.get('historical_significance')
                current_use = lm.get('current_use')
                controlled_by = lm.get('controlled_by')
                legends = lm.get('legends', [])

                if not name or not description:
                    continue

                try:
                    landmark_id = await self.add_landmark(
                        ctx,
                        name=name,
                        location_id=location_id,
                        landmark_type=lm_type,
                        description=description,
                        historical_significance=hist_signif,
                        current_use=current_use,
                        controlled_by=controlled_by,
                        legends=legends
                    )
                    lm['id'] = landmark_id
                    saved_landmarks.append(lm)
                except Exception as e:
                    logger.error(f"Error saving landmark '{name}': {e}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse landmark JSON: {result.final_output}")

        return saved_landmarks

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_location_lore",
        action_description="Evolving lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def evolve_location_lore(self, ctx, location_id: int, event_description: str) -> Dict[str, Any]:
        """
        Evolve the lore of a location based on an event, 
        using agent calls to produce new or updated content.
        
        Args:
            location_id: ID of the location
            event_description: Description of the event causing evolution
            
        Returns:
            Dictionary with evolution results
        """
        run_ctx = self.create_run_context(ctx)
        location_lore = await self.get_location_lore(ctx, location_id)
        if "error" in location_lore:
            return location_lore

        # Theming the event
        themed_event = MatriarchalThemingUtils.apply_matriarchal_theme("event", event_description, emphasis_level=1)

        location_name = location_lore.get('location', {}).get('location_name', 'Unknown')
        evolution_agent = Agent(
            name="LoreEvolutionAgent",
            instructions="You evolve location lore based on an event that occurs.",
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.8)
        )

        # We'll do a triple-prompt approach as before
        history_prompt = f"""
        The location is: {location_name}
        EVENT: {themed_event}
        
        Create one new historical entry in JSON:
        "new_history": {{
          "event_name": "...",
          "description": "...",
          "date_description": "Recently",
          "significance": int (1-10),
          "impact_type": "...",
          "notable_figures": [...],
          "current_relevance": "..."
        }}
        """

        landmark_prompt = f"""
        The location is: {location_name}
        EVENT: {themed_event}

        CURRENT LANDMARKS:
        {json.dumps(location_lore.get('landmarks', [])[:2], indent=2)}

        Decide if we add or modify a landmark. Return JSON:
        - "new_landmark": ... (same structure as add_landmark call)
        - "modified_landmark_id": ...
        - "landmark_update": "New description if modifying"
        """

        myth_prompt = f"""
        The location is: {location_name}
        EVENT: {themed_event}

        Possibly create a new myth. Return JSON with "new_myth": {{
          "name": "...",
          "description": "...",
          "believability": int,
          "spread_rate": int
        }}
        """

        # Run them
        history_result = await Runner.run(evolution_agent, history_prompt, context=run_ctx.context)
        landmark_result = await Runner.run(evolution_agent, landmark_prompt, context=run_ctx.context)
        myth_result = await Runner.run(evolution_agent, myth_prompt, context=run_ctx.context)

        # Process results
        new_history = None
        new_landmark = None
        updated_landmark = None
        new_myth = None

        try:
            # 1) Add new history
            history_data = json.loads(history_result.final_output)
            if "new_history" in history_data:
                h = history_data["new_history"]
                try:
                    hist_id = await self.add_local_history(
                        run_ctx,
                        location_id=location_id,
                        event_name=h.get("event_name","Recent Event"),
                        description=h.get("description",""),
                        date_description=h.get("date_description","Recently"),
                        significance=h.get("significance",5),
                        impact_type=h.get("impact_type","cultural"),
                        notable_figures=h.get("notable_figures",[]),
                        current_relevance=h.get("current_relevance")
                    )
                    h["id"] = hist_id
                    new_history = h
                except Exception as e:
                    logger.error(f"Error adding new history: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed parsing new_history: {history_result.final_output}")

        try:
            # 2) Landmark changes
            landmark_data = json.loads(landmark_result.final_output)
            if "new_landmark" in landmark_data and landmark_data["new_landmark"]:
                nl = landmark_data["new_landmark"]
                try:
                    lm_id = await self.add_landmark(
                        run_ctx,
                        name=nl.get("name","New Landmark"),
                        location_id=location_id,
                        landmark_type=nl.get("landmark_type","structure"),
                        description=nl.get("description",""),
                        historical_significance=nl.get("historical_significance"),
                        current_use=nl.get("current_use"),
                        controlled_by=nl.get("controlled_by"),
                        legends=nl.get("legends",[])
                    )
                    nl["id"] = lm_id
                    new_landmark = nl
                except Exception as e:
                    logger.error(f"Error adding new landmark: {e}")

            if "modified_landmark_id" in landmark_data and "landmark_update" in landmark_data:
                mod_id = landmark_data["modified_landmark_id"]
                mod_desc = landmark_data["landmark_update"]
                if mod_id and mod_desc:
                    try:
                        async with self.get_connection_pool() as pool:
                            async with pool.acquire() as conn:
                                # verify
                                existing = await conn.fetchrow("""
                                    SELECT * FROM Landmarks WHERE id=$1 AND location_id=$2
                                """, mod_id, location_id)
                                if existing:
                                    await conn.execute("""
                                        UPDATE Landmarks SET description=$1
                                        WHERE id=$2
                                    """, mod_desc, mod_id)
                                    updated_landmark = {
                                        "id": mod_id,
                                        "name": existing["name"],
                                        "old_description": existing["description"],
                                        "new_description": mod_desc
                                    }
                    except Exception as e:
                        logger.error(f"Error updating landmark {mod_id}: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed parsing new_landmark or modification: {landmark_result.final_output}")

        try:
            # 3) Myth changes
            myth_data = json.loads(myth_result.final_output)
            if "new_myth" in myth_data and myth_data["new_myth"]:
                nm = myth_data["new_myth"]
                try:
                    myth_id = await self.add_urban_myth(
                        run_ctx,
                        name=nm.get("name","New Myth"),
                        description=nm.get("description",""),
                        origin_location=location_lore["location"].get("location_name"),
                        origin_event=themed_event,
                        believability=nm.get("believability",5),
                        spread_rate=nm.get("spread_rate",3),
                        regions_known=[location_lore["location"].get("location_name")]
                    )
                    nm["id"] = myth_id
                    new_myth = nm
                except Exception as e:
                    logger.error(f"Error adding new myth: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed parsing new_myth: {myth_result.final_output}")

        self.invalidate_cache(f"location_lore_{location_id}")
        return {
            "event": themed_event,
            "location_id": location_id,
            "location_name": location_lore["location"].get("location_name","Unknown"),
            "new_history": new_history,
            "new_landmark": new_landmark,
            "updated_landmark": updated_landmark,
            "new_myth": new_myth
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_legend_variants",
        action_description="Creating contradictory legend variants for myth {myth_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def generate_legend_variants(
        self,
        ctx,
        myth_id: int,
        variant_count: int = 3
    ) -> Dict[str, Any]:
        """
        Create multiple contradictory versions of the same myth, storing them in versions_json.
        Each variant has at least one intentional inconsistency.
        
        Args:
            myth_id: ID of the myth to create variants for
            variant_count: Number of variants to create
            
        Returns:
            Dictionary with variant information
        """
        run_ctx = self.create_run_context(ctx)

        # Fetch the myth record
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                myth = await conn.fetchrow("SELECT * FROM UrbanMyths WHERE id=$1", myth_id)
                if not myth:
                    return {"error": "Myth not found"}

                versions_json = myth["versions_json"] or {}
                if "contradictory_variants" not in versions_json:
                    versions_json["contradictory_variants"] = []

        myth_data = dict(myth)
        prompt = f"""
        Create {variant_count} contradictory legend variants of the following myth. 
        Each variant must have at least one deliberate inconsistency or twist.
        MYTH:
        {json.dumps(myth_data, indent=2)}

        Return a JSON array of contradictory variants, each with fields:
        - 'title'
        - 'variant_description'
        - 'inconsistency_explanation'
        """

        result = await Runner.run(
            self.variant_agent,
            prompt,
            context=run_ctx.context,
            run_config=RunConfig(workflow_name="ContradictoryLegendVariants")
        )

        try:
            variants = json.loads(result.final_output)
            if not isinstance(variants, list):
                variants = [variants]

            # Save them to versions_json->contradictory_variants
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Append the new variants to existing ones if any
                    all_variants = versions_json["contradictory_variants"]
                    all_variants.extend(variants)

                    versions_json["contradictory_variants"] = all_variants
                    await conn.execute("""
                        UPDATE UrbanMyths
                        SET versions_json = $1
                        WHERE id = $2
                    """, json.dumps(versions_json), myth_id)

            return {
                "myth_id": myth_id,
                "myth_name": myth_data["name"],
                "new_variants": variants
            }

        except json.JSONDecodeError:
            return {"error": "Failed to parse contradictory variants"}

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="develop_tourist_attraction",
        action_description="Developing tourism for myth {myth_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def develop_tourist_attraction(
        self,
        ctx,
        myth_id: int
    ) -> Dict[str, Any]:
        """
        Model how a local myth is turned into a commercial tourist attraction.
        Store the results in the myth's versions_json under 'tourist_development'.
        
        Args:
            myth_id: ID of the myth to develop for tourism
            
        Returns:
            Dictionary with tourist development details
        """
        run_ctx = self.create_run_context(ctx)

        # Load the myth
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                myth = await conn.fetchrow("SELECT * FROM UrbanMyths WHERE id=$1", myth_id)
                if not myth:
                    return {"error": "Myth not found"}

                versions_json = myth["versions_json"] or {}

        myth_data = dict(myth)
        prompt = f"""
        Transform this myth into a tourist attraction. Provide details on:
        - Marketing angle & branding
        - Proposed tours or events for visitors
        - Possible merchandise or souvenirs
        - How to highlight matriarchal themes to attract visitors
        - Potential economic impact on the local region

        MYTH:
        {json.dumps(myth_data, indent=2)}

        Return JSON with fields:
        {{
          "marketing_angle": "...",
          "proposed_activities": ["...", "..."],
          "merchandise_ideas": ["...", "..."],
          "matriarchal_highlights": "...",
          "economic_impact_estimate": "..."
        }}
        """

        result = await Runner.run(
            self.tourism_agent,
            prompt,
            context=run_ctx.context,
            run_config=RunConfig(workflow_name="TouristAttractionDev")
        )

    ### NEW ###
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_tradition_dynamics",
        action_description="Simulating oral vs. written tradition for myth {myth_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool 
    async def simulate_tradition_dynamics(
        self,
        ctx,
        myth_id: int
    ) -> Dict[str, Any]:
        """
        Simulate how a myth changes when recorded in writing vs. shared orally.
        Store results in UrbanMyths.versions_json["tradition_dynamics"].
        """
        run_ctx = self.create_run_context(ctx)

        # Load the myth
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                myth = await conn.fetchrow("SELECT * FROM UrbanMyths WHERE id=$1", myth_id)
                if not myth:
                    return {"error": "Myth not found"}

                versions_json = myth["versions_json"] or {}

        myth_data = dict(myth)
        prompt = f"""
        Compare how this myth evolves when transmitted orally vs. written form.
        Show changes in detail, consistency, local variations, and matriarchal elements.

        MYTH:
        {json.dumps(myth_data, indent=2)}

        Return JSON with keys:
        {{
          "oral_version": "...",
          "written_version": "...",
          "key_differences": ["...", "..."],
          "matriarchal_comparison": "..."
        }}
        """

        result = await Runner.run(
            self.oral_written_agent,
            prompt,
            context=run_ctx.context,
            run_config=RunConfig(workflow_name="OralVsWritten")
        )

        try:
            tradition_data = json.loads(result.final_output)
            versions_json["tradition_dynamics"] = tradition_data

            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE UrbanMyths
                        SET versions_json = $1
                        WHERE id = $2
                    """, json.dumps(versions_json), myth_id)

            return {
                "myth_id": myth_id,
                "myth_name": myth_data["name"],
                "tradition_dynamics": tradition_data
            }

        except json.JSONDecodeError:
            return {"error": "Failed to parse oral vs. written tradition dynamics"}

    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="local_lore_manager",
            directive_text="Create and manage local lore, myths, and histories with matriarchal influences.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
