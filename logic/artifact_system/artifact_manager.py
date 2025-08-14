"""
Artifact Management System with Lazy Loading

This module provides sophisticated artifact management capabilities including
dynamic LLM-based discovery, analysis, and integration with the game's
lore, canon, and inventory systems.

Uses lazy loading to prevent circular dependencies.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
import asyncio
from datetime import datetime
import random
import hashlib

from pydantic import BaseModel, Field

# Type checking imports (these don't actually import at runtime)
if TYPE_CHECKING:
    from agents import RunContextWrapper, Agent, Runner
    from nyx.integrate import CentralGovernance
    from logic.conflict_system.conflict_synthesizer import ConflictSynthesizer
    from lore.lore_generator import DynamicLoreGenerator
    from lore.core.context import CanonicalContext

logger = logging.getLogger(__name__)

# =====================================================
# Pydantic Models for Structured Artifact Generation
# =====================================================

class ArtifactProperties(BaseModel):
    """Properties that define an artifact's characteristics"""
    power_level: int = Field(..., ge=1, le=10, description="Power level from 1-10")
    rarity: str = Field(..., description="Rarity tier: legendary, epic, rare, uncommon")
    origin_story: str = Field(..., description="The artifact's origin and history")
    cultural_significance: str = Field(..., description="Cultural meaning and importance")
    hidden_properties: List[str] = Field(default_factory=list, description="Properties not immediately apparent")
    activation_requirements: List[str] = Field(default_factory=list, description="What's needed to use the artifact")
    current_state: str = Field(default="dormant", description="Current state: dormant, active, corrupted, etc")
    bonded_to: Optional[str] = Field(None, description="Character it's bonded to, if any")
    
class ArtifactDiscoveryContext(BaseModel):
    """Context for artifact discovery"""
    location: str = Field(..., description="Where the artifact is found")
    discovery_method: str = Field(..., description="How it was discovered")
    environmental_clues: List[str] = Field(..., description="Clues in the environment")
    required_conditions: List[str] = Field(default_factory=list, description="Conditions that had to be met")
    discovery_narrative: str = Field(..., description="Narrative description of the discovery")
    
class ArtifactAnalysisResult(BaseModel):
    """Result of analyzing an artifact"""
    identified_properties: Dict[str, Any] = Field(..., description="Properties discovered through analysis")
    power_assessment: str = Field(..., description="Assessment of the artifact's power")
    historical_context: str = Field(..., description="Historical information uncovered")
    potential_uses: List[str] = Field(..., description="Potential applications")
    warnings: List[str] = Field(default_factory=list, description="Dangers or warnings")
    compatibility: Dict[str, float] = Field(default_factory=dict, description="Compatibility with characters/factions")
    
class ArtifactIntegrationPlan(BaseModel):
    """Plan for integrating an artifact with a conflict or story element"""
    integration_type: str = Field(..., description="Type: power_shift, revelation, catalyst, resolution")
    target_elements: List[str] = Field(..., description="Story elements affected")
    expected_outcomes: List[str] = Field(..., description="Expected narrative outcomes")
    risk_assessment: str = Field(..., description="Risks of the integration")
    narrative_impact: int = Field(..., ge=1, le=10, description="Expected narrative impact 1-10")

class GeneratedArtifact(BaseModel):
    """Complete generated artifact data"""
    name: str = Field(..., description="Artifact name")
    description: str = Field(..., description="Physical description")
    artifact_type: str = Field(..., description="Type: relic, weapon, tome, jewelry, tool, vestige")
    properties: ArtifactProperties = Field(..., description="Artifact properties")
    discovery_context: Optional[ArtifactDiscoveryContext] = Field(None, description="Discovery context if found")
    lore_connections: List[str] = Field(default_factory=list, description="Connections to world lore")
    faction_affiliations: List[str] = Field(default_factory=list, description="Related factions")

# =====================================================
# Core Artifact Manager with Lazy Loading
# =====================================================

class ArtifactManager:
    """
    Advanced artifact management system with sophisticated discovery,
    analysis, and integration capabilities.
    
    Uses lazy loading to prevent circular dependencies.
    """
    
    # Rarity weights for generation (lower = rarer)
    RARITY_WEIGHTS = {
        "legendary": 0.02,  # 2% chance
        "epic": 0.08,       # 8% chance  
        "rare": 0.20,       # 20% chance
        "uncommon": 0.70    # 70% chance
    }
    
    # Minimum power thresholds by rarity
    RARITY_POWER_THRESHOLDS = {
        "legendary": 8,
        "epic": 6,
        "rare": 4,
        "uncommon": 2
    }
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the artifact management system."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.is_initialized = False
        self.active_artifacts = {}
        self.artifact_history = []
        self.analysis_cache = {}
        
        # Lazy-loaded components (initialized on first use)
        self._conflict_resolution = None
        self._governor = None
        self._lore_generator = None
        
        # Agent components (will be initialized later)
        self._discovery_agent = None
        self._analysis_agent = None
        self._integration_agent = None
        
        # Agentic components
        self.agent_context = None
        self.agent_performance = {}
        self.agent_learning = {}
        self.agent_coordination = {}
    
    # =====================================================
    # Lazy Loading Properties
    # =====================================================
    
    @property
    def conflict_resolution(self):
        """Lazy load conflict resolution system."""
        if self._conflict_resolution is None:
            from logic.conflict_system.conflict_resolution import ConflictResolutionSystem
            self._conflict_resolution = ConflictResolutionSystem(self.user_id, self.conversation_id)
        return self._conflict_resolution
    
    @property
    def governor(self):
        """Lazy load governance system."""
        if self._governor is None:
            # Import here to avoid circular dependency
            from nyx.integrate import get_central_governance
            # This will need to be awaited when used
            self._governor = get_central_governance(self.user_id, self.conversation_id)
        return self._governor
    
    @property
    def lore_generator(self):
        """Lazy load lore generator."""
        if self._lore_generator is None:
            from lore.lore_generator import DynamicLoreGenerator
            # Note: This will need the governor, so it's created in initialize()
            self._lore_generator = None  # Will be set in initialize()
        return self._lore_generator
    
    @property
    def discovery_agent(self):
        """Lazy load discovery agent."""
        if self._discovery_agent is None:
            raise RuntimeError("Agents not initialized. Call initialize() first.")
        return self._discovery_agent
    
    @property
    def analysis_agent(self):
        """Lazy load analysis agent."""
        if self._analysis_agent is None:
            raise RuntimeError("Agents not initialized. Call initialize() first.")
        return self._analysis_agent
    
    @property
    def integration_agent(self):
        """Lazy load integration agent."""
        if self._integration_agent is None:
            raise RuntimeError("Agents not initialized. Call initialize() first.")
        return self._integration_agent
        
    async def initialize(self):
        """Initialize the artifact management system."""
        if not self.is_initialized:
            try:
                # Initialize governance (lazy loaded)
                from nyx.integrate import get_central_governance
                self._governor = await get_central_governance(self.user_id, self.conversation_id)
                
                # Initialize conflict resolution (lazy loaded)
                from logic.conflict_system.conflict_resolution import ConflictResolutionSystem
                self._conflict_resolution = ConflictResolutionSystem(self.user_id, self.conversation_id)
                await self._conflict_resolution.initialize()
                
                # Initialize lore generator with the governor
                from lore.lore_generator import DynamicLoreGenerator
                self._lore_generator = DynamicLoreGenerator.get_instance(
                    self.user_id, 
                    self.conversation_id,
                    self._governor
                )
                await self._lore_generator.initialize()
                
                # Initialize agents
                await self._initialize_agents()
                
                self.is_initialized = True
                logger.info(f"Artifact management system initialized for user {self.user_id}")
            except Exception as e:
                logger.error(f"Failed to initialize artifact system: {e}")
                raise
        return self
        
    async def _initialize_agents(self):
        """Initialize the artifact system agents using proper OpenAI SDK."""
        try:
            # Lazy import agent dependencies
            from agents import Agent, ModelSettings
            from agents.models.openai_responses import OpenAIResponsesModel
            from logic.chatgpt_integration import get_async_openai_client
            
            # Create discovery agent
            self._discovery_agent = Agent(
                name="ArtifactDiscoveryAgent",
                instructions=(
                    "You are an expert at discovering ancient and powerful artifacts in a femdom fantasy world. "
                    "You understand the rarity and significance of true artifacts. "
                    "When discovering artifacts, consider the location, current narrative context, "
                    "and ensure each artifact has deep lore connections and cultural significance. "
                    "Artifacts should feel earned and meaningful, not random loot. "
                    "Return structured JSON matching the GeneratedArtifact schema."
                ),
                model=OpenAIResponsesModel(
                    model="gpt-5-nano",
                    openai_client=get_async_openai_client()
                ),
                model_settings=ModelSettings(temperature=0.8),
                output_type=GeneratedArtifact
            )
            
            # Create analysis agent  
            self._analysis_agent = Agent(
                name="ArtifactAnalysisAgent",
                instructions=(
                    "You analyze artifacts to uncover their properties, history, and potential. "
                    "Consider both obvious and hidden properties, cultural context, "
                    "and how the artifact might interact with the current narrative. "
                    "Assess compatibility with different characters and factions. "
                    "Return structured JSON matching the ArtifactAnalysisResult schema."
                ),
                model=OpenAIResponsesModel(
                    model="gpt-5-nano",
                    openai_client=get_async_openai_client()
                ),
                model_settings=ModelSettings(temperature=0.6),
                output_type=ArtifactAnalysisResult
            )
            
            # Create integration agent
            self._integration_agent = Agent(
                name="ArtifactIntegrationAgent",
                instructions=(
                    "You plan how artifacts integrate with ongoing conflicts and narratives. "
                    "Consider power dynamics, story impact, and unintended consequences. "
                    "Ensure integrations feel natural and enhance rather than break the narrative. "
                    "Return structured JSON matching the ArtifactIntegrationPlan schema."
                ),
                model=OpenAIResponsesModel(
                    model="gpt-5-nano",
                    openai_client=get_async_openai_client()
                ),
                model_settings=ModelSettings(temperature=0.7),
                output_type=ArtifactIntegrationPlan
            )
            
            # Initialize agent context
            self.agent_context = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "active_artifacts": self.active_artifacts,
                "artifact_history": self.artifact_history,
                "analysis_cache": self.analysis_cache,
                "performance_metrics": self.agent_performance,
                "learning_state": self.agent_learning,
                "coordination_state": self.agent_coordination
            }
            
            logger.info("Artifact system agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing artifact system agents: {e}")
            raise
            
    async def discover_artifact(
        self,
        location: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Discover a new artifact in a location.
        WITH GOVERNANCE TRACKING.
        
        Args:
            location: Location to search
            context: Additional context for discovery
            
        Returns:
            Discovered artifact details
        """
        try:
            # Lazy import dependencies
            from agents import Runner
            from nyx.nyx_governance import AgentType
            from nyx.governance_helpers import with_governance
            
            # Apply governance decorator to inner function
            @with_governance(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                action_type="discover_artifact",
                action_description="Discovering a new artifact in the world",
                id_from_context=lambda ctx: "artifact_discovery"
            )
            async def _discover_with_governance():
                # Check if discovery should happen (rarity check)
                if not await self._should_discover_artifact(context):
                    return {
                        "discovered": False,
                        "reason": "No artifact presence detected in this location at this time."
                    }
                
                # Get world lore context
                world_lore = await self._get_relevant_lore(location, context)
                
                # Build discovery prompt
                discovery_prompt = f"""
                Location: {location}
                Current Narrative Context: {json.dumps(context, indent=2)}
                Relevant World Lore: {json.dumps(world_lore, indent=2)}
                
                Discover a significant artifact in this location. The artifact should:
                1. Have deep connections to the world's lore and history
                2. Be appropriately rare and powerful
                3. Fit naturally into the current narrative
                4. Have both obvious and hidden properties
                5. Require specific conditions or knowledge to fully utilize
                
                Generate a complete artifact with all required properties.
                """
                
                # Use discovery agent to generate artifact
                result = await Runner.run(
                    self.discovery_agent,
                    discovery_prompt,
                    context=self.agent_context
                )
                
                artifact_data = result.final_output_as(GeneratedArtifact)
                
                # Store in canon system
                artifact_id = await self._store_artifact_canonically(artifact_data)
                
                # Add to active artifacts
                self.active_artifacts[artifact_id] = artifact_data.dict()
                
                # Update history
                self.artifact_history.append({
                    "id": artifact_id,
                    "artifact": artifact_data.dict(),
                    "discovered_at": location,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Update agent performance
                self._update_agent_performance("artifact_discovery", True)
                
                # Report to governance if available
                if self._governor:
                    await self._governor.process_agent_action_report(
                        agent_type=AgentType.NARRATIVE_CRAFTER,
                        agent_id="artifact_manager",
                        action={
                            "type": "artifact_discovery",
                            "artifact_name": artifact_data.name,
                            "location": location,
                            "rarity": artifact_data.properties.rarity
                        },
                        result={"artifact_id": artifact_id, "success": True}
                    )
                
                return {
                    "discovered": True,
                    "artifact_id": artifact_id,
                    "artifact": artifact_data.dict(),
                    "narrative": artifact_data.discovery_context.discovery_narrative if artifact_data.discovery_context else ""
                }
            
            # Execute the governed function
            return await _discover_with_governance()
            
        except Exception as e:
            logger.error(f"Error discovering artifact: {e}")
            self._update_agent_performance("artifact_discovery", False)
            return {"discovered": False, "error": str(e)}
    
    async def _should_discover_artifact(self, context: Dict[str, Any]) -> bool:
        """
        Determine if an artifact should be discovered based on rarity and context.
        
        Args:
            context: Discovery context
            
        Returns:
            True if artifact should be discovered
        """
        # Base discovery chance (very low for maintaining rarity)
        base_chance = 0.05  # 5% base chance
        
        # Modify based on context
        modifiers = 1.0
        
        # Increase chance for significant locations
        if context.get("location_significance", 0) > 7:
            modifiers *= 2.0
            
        # Increase chance for major story moments
        if context.get("narrative_significance", 0) > 8:
            modifiers *= 1.5
            
        # Increase chance if player has been searching specifically
        if context.get("active_search", False):
            modifiers *= 3.0
            
        # Decrease chance if artifact was recently found
        if self.artifact_history:
            last_discovery = datetime.fromisoformat(self.artifact_history[-1]["timestamp"])
            hours_since = (datetime.utcnow() - last_discovery).total_seconds() / 3600
            if hours_since < 24:
                modifiers *= 0.1  # Much less likely within 24 hours
                
        final_chance = min(base_chance * modifiers, 0.5)  # Cap at 50%
        
        return random.random() < final_chance
    
    async def _get_relevant_lore(self, location: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant lore for artifact generation."""
        try:
            # Lazy import database connection
            from db.connection import get_db_connection_context
            
            async with get_db_connection_context() as conn:
                # Get location lore
                location_data = await conn.fetchrow("""
                    SELECT description, cultural_significance, hidden_aspects
                    FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2 
                    AND location_name = $3
                """, self.user_id, self.conversation_id, location)
                
                # Get active factions
                factions = await conn.fetch("""
                    SELECT name, type, power_level
                    FROM Factions
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY power_level DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
                # Get recent historical events
                events = await conn.fetch("""
                    SELECT name, description, significance
                    FROM HistoricalEvents
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY significance DESC
                    LIMIT 3
                """, self.user_id, self.conversation_id)
                
                return {
                    "location": dict(location_data) if location_data else {},
                    "factions": [dict(f) for f in factions],
                    "historical_events": [dict(e) for e in events],
                    "current_context": context
                }
        except Exception as e:
            logger.error(f"Error getting relevant lore: {e}")
            return {}
    
    async def _store_artifact_canonically(self, artifact_data: GeneratedArtifact) -> str:
        """Store artifact using the canon system."""
        try:
            # Lazy imports
            from db.connection import get_db_connection_context
            from lore.core import canon
            from lore.core.context import CanonicalContext
            from embedding.vector_store import generate_embedding
            
            # Create canonical context
            ctx = CanonicalContext(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            async with get_db_connection_context() as conn:
                # Generate unique artifact ID
                artifact_id = f"artifact_{hashlib.md5(f'{artifact_data.name}{datetime.utcnow()}'.encode()).hexdigest()[:8]}"
                
                # Check for duplicates using semantic matching
                embedding_text = f"{artifact_data.name} {artifact_data.description} {artifact_data.properties.origin_story}"
                embedding = await generate_embedding(embedding_text)
                
                # Check for similar artifacts
                similar = await conn.fetchrow("""
                    SELECT artifact_id, name, 1 - (embedding <=> $1) AS similarity
                    FROM Artifacts
                    WHERE user_id = $2 AND conversation_id = $3
                    AND 1 - (embedding <=> $1) > 0.85
                    ORDER BY embedding <=> $1
                    LIMIT 1
                """, embedding, self.user_id, self.conversation_id)
                
                if similar:
                    logger.warning(f"Similar artifact exists: {similar['name']} (similarity: {similar['similarity']:.2f})")
                    return similar['artifact_id']
                
                # Store artifact
                await conn.execute("""
                    INSERT INTO Artifacts (
                        user_id, conversation_id, artifact_id, name, description,
                        artifact_type, properties, discovery_location, discovery_date,
                        current_owner, is_active, power_level, rarity,
                        lore_connections, faction_affiliations, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    self.user_id, self.conversation_id, artifact_id,
                    artifact_data.name, artifact_data.description,
                    artifact_data.artifact_type,
                    json.dumps(artifact_data.properties.dict()),
                    artifact_data.discovery_context.location if artifact_data.discovery_context else "Unknown",
                    datetime.utcnow(),
                    None,  # No owner initially
                    artifact_data.properties.current_state == "active",
                    artifact_data.properties.power_level,
                    artifact_data.properties.rarity,
                    json.dumps(artifact_data.lore_connections),
                    json.dumps(artifact_data.faction_affiliations),
                    embedding
                )
                
                # Log canonical event
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Legendary artifact '{artifact_data.name}' discovered",
                    tags=["artifact", "discovery", artifact_data.properties.rarity],
                    significance=9 if artifact_data.properties.rarity == "legendary" else 7
                )
                
                return artifact_id
                
        except Exception as e:
            logger.error(f"Error storing artifact canonically: {e}")
            raise
    
    async def analyze_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """
        Analyze an artifact using the analysis agent.
        
        Args:
            artifact_id: ID of the artifact to analyze
            
        Returns:
            Analysis results
        """
        try:
            # Lazy import
            from agents import Runner
            
            artifact = self.active_artifacts.get(artifact_id)
            if not artifact:
                return {"error": "Artifact not found"}
            
            # Check cache
            if artifact_id in self.analysis_cache:
                return self.analysis_cache[artifact_id]
            
            # Build analysis prompt
            analysis_prompt = f"""
            Artifact: {json.dumps(artifact, indent=2)}
            
            Perform a detailed analysis of this artifact:
            1. Identify all properties (obvious and hidden)
            2. Assess its power and potential
            3. Uncover historical context and significance
            4. Determine compatibility with current characters and factions
            5. Identify any warnings or dangers
            """
            
            # Run analysis
            result = await Runner.run(
                self.analysis_agent,
                analysis_prompt,
                context=self.agent_context
            )
            
            analysis = result.final_output_as(ArtifactAnalysisResult)
            
            # Cache analysis
            self.analysis_cache[artifact_id] = analysis.dict()
            
            # Update agent learning
            self._update_agent_learning("artifact_analysis", analysis.dict())
            
            return analysis.dict()
            
        except Exception as e:
            logger.error(f"Error analyzing artifact: {e}")
            return {"error": str(e)}
    
    async def integrate_artifact(
        self,
        artifact_id: str,
        conflict_id: int,
        integration_type: str = "power"
    ) -> Dict[str, Any]:
        """
        Integrate an artifact with a conflict (now synthesis-aware).
        WITH GOVERNANCE TRACKING.
        """
        try:
            from logic.conflict_system.conflict_synthesizer import ConflictSynthesizer
            from agents import Runner, RunContextWrapper
            from nyx.nyx_governance import AgentType
            from nyx.governance_helpers import with_governance
            
            @with_governance(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                action_type="integrate_artifact",
                action_description="Integrating an artifact with a conflict",
                id_from_context=lambda ctx: "artifact_integration"
            )
            async def _integrate_with_governance():
                artifact = self.active_artifacts.get(artifact_id)
                if not artifact:
                    return {"success": False, "error": "Artifact not found"}
                
                # Check if this conflict is part of a synthesis
                synthesizer = ConflictSynthesizer(self.user_id, self.conversation_id)
                
                async with get_db_connection_context() as conn:
                    # Check for synthesis
                    synthesis = await conn.fetchrow("""
                        SELECT synthesis_id, component_conflicts, synthesis_type
                        FROM conflict_synthesis
                        WHERE $1 = ANY(component_conflicts::int[])
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, conflict_id)
                
                if synthesis:
                    # This conflict is part of a synthesis - artifact affects the whole
                    synthesis_event = {
                        "type": "artifact_integration",
                        "artifact": artifact,
                        "target_conflict": conflict_id,
                        "integration_type": integration_type
                    }
                    
                    result = await synthesizer.manage_synthesis_progression(
                        synthesis['synthesis_id'],
                        synthesis_event
                    )
                    
                    return {
                        "success": True,
                        "artifact_id": artifact_id,
                        "synthesis_affected": True,
                        "synthesis_id": synthesis['synthesis_id'],
                        "cascade_effects": result.get('cascade_effects', []),
                        "narrative_impact": result.get('narrative_impact', '')
                    }
                else:
                    # Single conflict integration (old path)
                    ctx_wrapper = RunContextWrapper(self.user_id, self.conversation_id)
                    from logic.conflict_system.conflict_tools import get_conflict_details
                    conflict = await get_conflict_details(ctx_wrapper, conflict_id)
                    
                    if not conflict:
                        return {"success": False, "error": "Conflict not found"}
                
                # Build integration prompt
                integration_prompt = f"""
                Artifact: {json.dumps(artifact, indent=2)}
                Conflict: {json.dumps(conflict, indent=2)}
                Integration Type: {integration_type}
                
                Plan how this artifact should integrate with the conflict:
                1. Determine the type of integration (power shift, revelation, catalyst, resolution)
                2. Identify which story elements will be affected
                3. Predict narrative outcomes
                4. Assess risks
                5. Rate the expected narrative impact (1-10)
                """
                
                # Get integration plan
                result = await Runner.run(
                    self.integration_agent,
                    integration_prompt,
                    context=self.agent_context
                )
                
                plan = result.final_output_as(ArtifactIntegrationPlan)
                
                # Execute integration
                integration_result = await self._execute_integration(artifact, conflict, plan.dict())
                
                # Update conflict if successful
                if integration_result.get("success"):
                    await self._update_conflict_with_artifact(conflict_id, artifact_id, integration_result)
                
                # Update agent performance
                self._update_agent_performance("artifact_integration", integration_result.get("success", False))
                
                return integration_result
            
            # Execute the governed function
            return await _integrate_with_governance()
            
        except Exception as e:
            logger.error(f"Error integrating artifact: {e}")
            self._update_agent_performance("artifact_integration", False)
            return {"success": False, "error": str(e)}
    
    async def grant_artifact_to_player(
        self,
        artifact_id: str,
        player_name: str = "Chase"
    ) -> Dict[str, Any]:
        """
        Grant an artifact to a player, adding it to their inventory.
        
        Args:
            artifact_id: ID of the artifact
            player_name: Name of the player
            
        Returns:
            Result of the grant operation
        """
        try:
            # Lazy imports
            from logic.inventory_system_sdk import add_item
            from db.connection import get_db_connection_context
            from lore.core import canon
            from lore.core.context import CanonicalContext
            
            artifact = self.active_artifacts.get(artifact_id)
            if not artifact:
                return {"success": False, "error": "Artifact not found"}
            
            # Add to player inventory using inventory system
            inventory_result = await add_item(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                player_name=player_name,
                item_name=artifact["name"],
                description=artifact["description"],
                effect=f"Power Level: {artifact['properties']['power_level']}",
                category="artifact",
                quantity=1
            )
            
            if inventory_result.get("success"):
                # Update artifact owner in database
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE Artifacts
                        SET current_owner = $1, is_active = true
                        WHERE artifact_id = $2 AND user_id = $3 AND conversation_id = $4
                    """, player_name, artifact_id, self.user_id, self.conversation_id)
                    
                    # Update artifact bonding
                    artifact["properties"]["bonded_to"] = player_name
                    artifact["properties"]["current_state"] = "active"
                    
                    # Log the event
                    ctx = CanonicalContext(
                        user_id=self.user_id,
                        conversation_id=self.conversation_id
                    )
                    
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"Artifact '{artifact['name']}' claimed by {player_name}",
                        tags=["artifact", "ownership", "player_item"],
                        significance=8
                    )
            
            return {
                "success": inventory_result.get("success", False),
                "artifact_name": artifact["name"],
                "player": player_name,
                "bonding_complete": True
            }
            
        except Exception as e:
            logger.error(f"Error granting artifact to player: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_historical_patterns(self, action_type: str) -> List[Dict[str, Any]]:
        """Get historical patterns for an action type."""
        patterns = []
        
        # Filter history by action type
        type_history = [e for e in self.artifact_history if e.get("action_type") == action_type]
        
        # Analyze patterns
        if type_history:
            # Get success patterns
            success_patterns = [e for e in type_history if "error" not in e.get("result", {})]
            if success_patterns:
                patterns.append({
                    "type": "success",
                    "count": len(success_patterns),
                    "examples": success_patterns[-3:]
                })
                
            # Get failure patterns
            failure_patterns = [e for e in type_history if "error" in e.get("result", {})]
            if failure_patterns:
                patterns.append({
                    "type": "failure",
                    "count": len(failure_patterns),
                    "examples": failure_patterns[-3:]
                })
                
        return patterns
        
    def _update_agent_performance(self, action: str, success: bool):
        """Update agent performance metrics."""
        if action not in self.agent_performance:
            self.agent_performance[action] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0
            }
            
        metrics = self.agent_performance[action]
        metrics["total"] += 1
        
        if success:
            metrics["successful"] += 1
        else:
            metrics["failed"] += 1
            
        metrics["success_rate"] = metrics["successful"] / metrics["total"]
        
    def _update_agent_learning(self, action: str, result: Dict[str, Any]):
        """Update agent learning state."""
        if action not in self.agent_learning:
            self.agent_learning[action] = {
                "patterns": [],
                "strategies": {},
                "adaptations": []
            }
            
        learning = self.agent_learning[action]
        
        # Extract patterns
        if "patterns" in result:
            learning["patterns"].extend(result["patterns"])
            
        # Update strategies
        if "strategy" in result:
            strategy = result["strategy"]
            if strategy not in learning["strategies"]:
                learning["strategies"][strategy] = {
                    "uses": 0,
                    "successes": 0,
                    "failures": 0
                }
            learning["strategies"][strategy]["uses"] += 1
            if "error" not in result:
                learning["strategies"][strategy]["successes"] += 1
            else:
                learning["strategies"][strategy]["failures"] += 1
                
        # Record adaptations
        if "adaptation" in result:
            learning["adaptations"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "details": result["adaptation"]
            })
            
    async def _execute_integration(
        self,
        artifact: Dict[str, Any],
        conflict: Dict[str, Any],
        integration_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute artifact integration based on agent plan."""
        try:
            # Implementation based on integration type
            integration_type = integration_plan.get("integration_type", "catalyst")
            
            result = {
                "success": True,
                "integration_type": integration_type,
                "artifact_name": artifact["name"],
                "conflict_id": conflict.get("id"),
                "narrative_impact": integration_plan.get("narrative_impact", 5),
                "outcomes": []
            }
            
            # Apply integration effects based on type
            if integration_type == "power_shift":
                # Artifact shifts power balance in conflict
                result["outcomes"].append(
                    f"The {artifact['name']} has shifted the balance of power"
                )
                result["power_change"] = {
                    "magnitude": artifact["properties"]["power_level"],
                    "beneficiary": "TBD based on wielder"
                }
                
            elif integration_type == "revelation":
                # Artifact reveals hidden information
                result["outcomes"].append(
                    f"The {artifact['name']} reveals hidden truths about the conflict"
                )
                result["revelations"] = artifact["properties"].get("hidden_properties", [])
                
            elif integration_type == "catalyst":
                # Artifact accelerates or triggers events
                result["outcomes"].append(
                    f"The {artifact['name']} acts as a catalyst, accelerating events"
                )
                result["acceleration_factor"] = 1.5
                
            elif integration_type == "resolution":
                # Artifact provides means to resolve conflict
                result["outcomes"].append(
                    f"The {artifact['name']} offers a path to resolution"
                )
                result["resolution_paths"] = integration_plan.get("expected_outcomes", [])
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing integration: {e}")
            return {"success": False, "error": str(e)}
        
    async def _update_conflict_with_artifact(
        self,
        conflict_id: int,
        artifact_id: str,
        integration_result: Dict[str, Any]
    ):
        """Update conflict state after artifact integration."""
        try:
            # Lazy imports
            from agents import RunContextWrapper
            from logic.conflict_system.conflict_tools import (
                add_resolution_path,
                update_conflict_progress
            )
            
            ctx_wrapper = RunContextWrapper(self.user_id, self.conversation_id)
            
            # Add artifact as a resolution path or modifier
            if integration_result.get("resolution_paths"):
                for path in integration_result["resolution_paths"]:
                    await add_resolution_path(
                        ctx_wrapper,
                        conflict_id,
                        path,
                        f"Enabled by artifact: {integration_result['artifact_name']}"
                    )
            
            # Update conflict progress if there was impact
            if integration_result.get("narrative_impact", 0) > 5:
                await update_conflict_progress(
                    ctx_wrapper,
                    conflict_id,
                    integration_result["narrative_impact"] * 10,  # Convert to percentage
                    f"Artifact integration: {integration_result['artifact_name']}"
                )
                
        except Exception as e:
            logger.error(f"Error updating conflict with artifact: {e}")

# =====================================================
# Database Schema Creation
# =====================================================

async def ensure_artifacts_table(conn):
    """Ensure the Artifacts table exists."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS Artifacts (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            artifact_id VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            artifact_type VARCHAR(50),
            properties JSONB,
            discovery_location VARCHAR(255),
            discovery_date TIMESTAMP,
            current_owner VARCHAR(255),
            is_active BOOLEAN DEFAULT false,
            power_level INTEGER CHECK (power_level >= 1 AND power_level <= 10),
            rarity VARCHAR(50),
            lore_connections JSONB,
            faction_affiliations JSONB,
            embedding VECTOR(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_artifacts_lookup
        ON Artifacts(user_id, conversation_id, artifact_id)
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_artifacts_owner
        ON Artifacts(user_id, conversation_id, current_owner)
    """)
    
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_artifacts_embedding_hnsw
        ON Artifacts
        USING hnsw (embedding vector_cosine_ops)
    """)

# =====================================================
# Export the manager
# =====================================================

__all__ = ['ArtifactManager', 'ensure_artifacts_table']

