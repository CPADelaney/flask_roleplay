# lore/lore_orchestrator.py

"""
Lore Orchestrator - Unified Entry Point for All Lore Functionality

This module provides a single, comprehensive interface to all lore system components,
managing initialization, resource allocation, and coordination between subsystems.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set, AsyncGenerator
from datetime import datetime
import json
from enum import Enum
from dataclasses import dataclass

# Core imports
from db.connection import get_db_connection_context

# Nyx governance
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission

logger = logging.getLogger(__name__)

# Singleton instance storage
_ORCHESTRATOR_INSTANCES: Dict[Tuple[int, int], "LoreOrchestrator"] = {}


@dataclass
class OrchestratorConfig:
    """Configuration for the Lore Orchestrator"""
    enable_matriarchal_theme: bool = True
    enable_governance: bool = True
    enable_metrics: bool = True
    enable_validation: bool = True
    cache_ttl: int = 3600
    max_parallel_operations: int = 10
    auto_initialize: bool = True
    resource_limits: Dict[str, Any] = None
    

class LoreOrchestrator:
    """
    Master orchestrator that provides unified access to all lore functionality.
    Acts as the single entry point for external systems to interact with lore.
    """
    
    @classmethod
    def get_instance(cls, user_id: int, conversation_id: int, config: Optional[OrchestratorConfig] = None) -> "LoreOrchestrator":
        """Get or create a singleton instance for the given user/conversation."""
        key = (user_id, conversation_id)
        
        if key not in _ORCHESTRATOR_INSTANCES:
            _ORCHESTRATOR_INSTANCES[key] = cls(user_id, conversation_id, config)
        elif config:
            # Update config if provided
            _ORCHESTRATOR_INSTANCES[key].config = config
            
        return _ORCHESTRATOR_INSTANCES[key]
    
    def __init__(self, user_id: int, conversation_id: int, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the Lore Orchestrator.
        
        Args:
            user_id: User ID for context
            conversation_id: Conversation ID for context
            config: Optional configuration settings
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = config or OrchestratorConfig()
        self.initialized = False
        
        # Core components (initialized on demand)
        self._lore_system = None
        self._matriarchal_system = None
        self._dynamic_generator = None
        self._setting_analyzer = None
        
        # Integration components
        self._npc_integration = None
        self._conflict_integration = None
        self._context_enhancer = None
        
        # Framework components
        self._matriarchal_framework = None
        self._matriarchal_power_framework = None  # NEW
        
        # System components
        self._lore_dynamics_system = None  # NEW
        self._regional_culture_system = None  # NEW
        
        # Management components
        self._config_manager = None
        self._error_handler = None
        self._resource_manager = None
        self._validation_manager = None
        
        # Agent components
        self._agent_context = None
        self._directive_handler = None
        self._quest_agent = None
        self._narrative_agent = None
        self._environment_agent = None
        self._foundation_agent = None
        self._faction_agent = None
        
        # Extended systems (lazy loaded)
        self._national_conflict_system = None
        self._religious_distribution_system = None
        self._lore_update_system = None
        
        # Governance
        self._governor = None
        
        # Component factory
        self._component_factory = None
        
        # Initialization tracking
        self._init_lock = asyncio.Lock()
        self._component_init_status: Dict[str, bool] = {}
    
    async def initialize(self) -> bool:
        """
        Initialize the orchestrator and all configured components.
        
        Returns:
            True if initialization successful
        """
        async with self._init_lock:
            if self.initialized:
                return True
            
            try:
                logger.info(f"Initializing Lore Orchestrator for user {self.user_id}, conversation {self.conversation_id}")
                
                # Initialize governance if enabled
                if self.config.enable_governance:
                    await self._initialize_governance()
                
                # Initialize core configuration
                await self._initialize_config()
                
                # Initialize error handling
                await self._initialize_error_handling()
                
                # Initialize resource management
                await self._initialize_resource_management()
                
                # Initialize validation if enabled
                if self.config.enable_validation:
                    await self._initialize_validation()
                
                # Initialize metrics if enabled
                if self.config.enable_metrics:
                    await self._initialize_metrics()
                
                # Mark as initialized
                self.initialized = True
                logger.info("Lore Orchestrator initialization complete")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize Lore Orchestrator: {e}")
                return False
    
    async def _initialize_governance(self):
        """Initialize governance system."""
        self._governor = await get_central_governance(self.user_id, self.conversation_id)
        self._component_init_status['governance'] = True
        logger.info("Governance system initialized")
    
    async def _initialize_config(self):
        """Initialize configuration management."""
        from lore.config import ConfigManager
        self._config_manager = ConfigManager()
        await self._config_manager.load_config()
        self._component_init_status['config'] = True
        logger.info("Configuration management initialized")
    
    async def _initialize_error_handling(self):
        """Initialize error handling."""
        from lore.error_manager import ErrorHandler
        self._error_handler = ErrorHandler(self.user_id, self.conversation_id, self._config_manager.config)
        await self._error_handler.start_monitoring()
        self._component_init_status['error_handling'] = True
        logger.info("Error handling initialized")
    
    async def _initialize_resource_management(self):
        """Initialize resource management."""
        from lore.resource_manager import ResourceManager
        self._resource_manager = ResourceManager(
            self.user_id, 
            self.conversation_id,
            config=await self._config_manager.get_lore_config() if self._config_manager else None
        )
        await self._resource_manager.start()
        self._component_init_status['resource_management'] = True
        logger.info("Resource management initialized")
    
    async def _initialize_validation(self):
        """Initialize validation system."""
        from lore.validation import validation_manager
        self._validation_manager = validation_manager
        await self._validation_manager.initialize()
        self._component_init_status['validation'] = True
        logger.info("Validation system initialized")
    
    async def _initialize_metrics(self):
        """Initialize metrics collection."""
        # Metrics manager is already a singleton
        from lore.metrics import metrics_manager
        self._component_init_status['metrics'] = True
        logger.info("Metrics collection initialized")
    
    # ===== CORE LORE OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_world",
        action_description="Generating complete world lore",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def generate_complete_world(self, ctx, environment_desc: str, use_matriarchal_theme: Optional[bool] = None) -> Dict[str, Any]:
        """
        Generate a complete world with all lore components.
        
        Args:
            environment_desc: Description of the environment/setting
            use_matriarchal_theme: Override config setting for matriarchal theme
            
        Returns:
            Complete world lore package
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        use_theme = use_matriarchal_theme if use_matriarchal_theme is not None else self.config.enable_matriarchal_theme
        
        try:
            if use_theme:
                # Use matriarchal system
                system = await self._get_matriarchal_system()
                return await system.generate_complete_world(ctx, environment_desc)
            else:
                # Use standard generator
                generator = await self._get_dynamic_generator()
                return await generator.generate_complete_lore(environment_desc)
                
        except Exception as e:
            if self._error_handler:
                from lore.error_manager import LoreError, ErrorType
                error = LoreError(f"Failed to generate world: {str(e)}", ErrorType.UNKNOWN)
                await self._error_handler.handle_error(error)
            raise
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_world",
        action_description="Evolving world with narrative event",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def evolve_world_with_event(self, ctx, event_description: str, affected_location_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Evolve the world based on a narrative event.
        
        Args:
            event_description: Description of the event
            affected_location_id: Optional specific location affected
            
        Returns:
            Evolution results and updates
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        if self.config.enable_matriarchal_theme:
            system = await self._get_matriarchal_system()
            return await system.handle_narrative_event(
                ctx,
                event_description,
                affected_location_id
            )
        else:
            generator = await self._get_dynamic_generator()
            evolution = await generator.lore_evolution.evolve_lore_with_event(event_description)
            return evolution
    
    # ===== MATRIARCHAL POWER FRAMEWORK OPERATIONS (NEW) =====
    
    async def generate_matriarchal_core_principles(self) -> Dict[str, Any]:
        """
        Generate core principles for a matriarchal world.
        
        Returns:
            Core principles structure
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        framework = await self._get_matriarchal_power_framework()
        return await framework.generate_core_principles()
    
    async def generate_hierarchical_constraints(self) -> Dict[str, Any]:
        """
        Generate hierarchical constraints for matriarchal society.
        
        Returns:
            Hierarchical constraints structure
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        framework = await self._get_matriarchal_power_framework()
        return await framework.generate_hierarchical_constraints()
    
    async def apply_matriarchal_power_lens(self, foundation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply matriarchal power lens to foundation lore.
        
        Args:
            foundation_data: Foundation lore to transform
            
        Returns:
            Transformed lore with matriarchal themes
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        framework = await self._get_matriarchal_power_framework()
        return await framework.apply_power_lens(foundation_data)
    
    async def generate_power_expressions(self) -> List[Dict[str, Any]]:
        """
        Generate power expressions for matriarchal society.
        
        Returns:
            List of power expressions
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        framework = await self._get_matriarchal_power_framework()
        return await framework.generate_power_expressions()
    
    async def develop_matriarchal_narrative(self, narrative_theme: str, initial_scene: str) -> AsyncGenerator[str, None]:
        """
        Develop a narrative through iterative dialogue with matriarchal themes.
        
        Args:
            narrative_theme: Theme of the narrative
            initial_scene: Starting scene
            
        Yields:
            Narrative segments as they develop
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        framework = await self._get_matriarchal_power_framework()
        async for segment in framework.develop_narrative_through_dialogue(narrative_theme, initial_scene):
            yield segment
    
    # ===== LORE DYNAMICS OPERATIONS (NEW) =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore based on event",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def evolve_lore_with_event(self, ctx, event_description: str) -> Dict[str, Any]:
        """
        Evolve world lore based on a narrative event using the dynamics system.
        
        Args:
            event_description: Description of the event
            
        Returns:
            Evolution results with affected elements and updates
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        dynamics = await self._get_lore_dynamics_system()
        return await dynamics.evolve_lore_with_event(ctx, event_description)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_emergent_event",
        action_description="Generating emergent world event",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def generate_emergent_event(self, ctx) -> Dict[str, Any]:
        """
        Generate a random emergent event in the world.
        
        Returns:
            Generated event data with lore updates
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        dynamics = await self._get_lore_dynamics_system()
        return await dynamics.generate_emergent_event(ctx)
    
    async def mature_lore_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        """
        Natural evolution of lore over time.
        
        Args:
            days_passed: Number of days to simulate
            
        Returns:
            Maturation summary with changes
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        dynamics = await self._get_lore_dynamics_system()
        
        # Create mock context for governance
        ctx = self._create_mock_context()
        
        return await dynamics.mature_lore_over_time(days_passed)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_world_over_time",
        action_description="Evolving world over time period",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def evolve_world_over_time(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
        """
        Evolve the entire world across a specified time period.
        
        Args:
            days_passed: Number of days to simulate
            
        Returns:
            Complete evolution results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        dynamics = await self._get_lore_dynamics_system()
        return await dynamics.evolve_world_over_time(ctx, days_passed)
    
    # ===== REGIONAL CULTURE OPERATIONS (NEW - Additional methods) =====
    # These are new methods that weren't in the original orchestrator
    
    async def summarize_culture(self, nation_id: int, format_type: str = "brief") -> str:
        """
        Generate a textual summary of a nation's culture.
        
        Args:
            nation_id: ID of the nation
            format_type: Format type (brief, detailed, narrative)
            
        Returns:
            Cultural summary text
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        return await culture_system.summarize_culture(nation_id, format_type)
    
    async def detect_cultural_conflicts(self, nation_id1: int, nation_id2: int) -> Dict[str, Any]:
        """
        Analyze potential cultural conflicts between two nations.
        
        Args:
            nation_id1: First nation ID
            nation_id2: Second nation ID
            
        Returns:
            Cultural conflict analysis
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        return await culture_system.detect_cultural_conflicts(nation_id1, nation_id2)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="simulate_cultural_diffusion",
        action_description="Simulating cultural diffusion",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def simulate_cultural_diffusion(self, ctx, nation1_id: int, nation2_id: int, years: int = 50) -> Dict[str, Any]:
        """
        Simulate cultural diffusion between two nations over time.
        
        Args:
            nation1_id: First nation ID
            nation2_id: Second nation ID
            years: Years to simulate
            
        Returns:
            Cultural diffusion results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        return await culture_system.simulate_cultural_diffusion(ctx, nation1_id, nation2_id, years)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_dialect",
        action_description="Evolving regional dialect",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def evolve_dialect(self, ctx, language_id: int, region_id: int, years: int = 100) -> Dict[str, Any]:
        """
        Evolve a regional dialect of a language.
        
        Args:
            language_id: ID of the language
            region_id: ID of the region
            years: Years to simulate
            
        Returns:
            Dialect evolution results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        return await culture_system.evolve_dialect(ctx, language_id, region_id, years)
    
    async def generate_diplomatic_protocol(self, nation_id1: int, nation_id2: int) -> Dict[str, Any]:
        """
        Generate diplomatic protocol guide for interactions between two nations.
        
        Args:
            nation_id1: First nation ID
            nation_id2: Second nation ID
            
        Returns:
            Diplomatic protocol guide
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        return await culture_system.generate_diplomatic_protocol(nation_id1, nation_id2)
    
    # ===== NPC INTEGRATION =====
    
    async def integrate_lore_with_npc(self, npc_id: int, cultural_background: str, faction_affiliations: List[str]) -> Dict[str, Any]:
        """
        Integrate lore with a specific NPC.
        
        Args:
            npc_id: ID of the NPC
            cultural_background: NPC's cultural background
            faction_affiliations: List of faction names
            
        Returns:
            Integration results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_npc_integration(npc_id)
        
        # Create a mock context for governance
        ctx = self._create_mock_context(npc_id=npc_id)
        
        return await integration.initialize_npc_lore_knowledge(
            ctx,
            npc_id,
            cultural_background,
            faction_affiliations
        )
    
    async def process_npc_lore_interaction(self, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Process a lore-related interaction between player and NPC.
        
        Args:
            npc_id: ID of the NPC
            player_input: Player's input/question
            
        Returns:
            Interaction results and NPC response
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_npc_integration(npc_id)
        
        # Create a mock context for governance
        ctx = self._create_mock_context(npc_id=npc_id)
        
        return await integration.process_npc_lore_interaction(ctx, npc_id, player_input)
    
    async def apply_dialect_to_npc_text(self, text: str, dialect_id: int, intensity: str = 'medium', npc_id: Optional[int] = None) -> str:
        """
        Apply dialect features to NPC text.
        
        Args:
            text: Original text
            dialect_id: ID of the dialect to apply
            intensity: Intensity of dialect application ('light', 'medium', 'strong')
            npc_id: Optional NPC ID for personalized dialect features
            
        Returns:
            Modified text with dialect features applied
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_npc_integration(npc_id)
        return await integration.apply_dialect_to_text(text, dialect_id, intensity, npc_id)
    
    async def get_npc_data(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get NPC data for a list of NPC IDs.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to their data
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import get_npc_data
        return await get_npc_data(npc_ids)
    
    async def determine_relevant_lore_for_npc(self, npc_id: int, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Determine which lore elements are relevant to a specific NPC.
        
        Args:
            npc_id: The ID of the NPC
            context: Optional context dictionary
            
        Returns:
            Dict containing relevant lore elements and their relevance scores
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import determine_relevant_lore
        return await determine_relevant_lore(npc_id, context)
    
    async def integrate_npc_lore(
        self, 
        npc_id: int, 
        relevant_lore: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Integrate relevant lore with an NPC, updating their knowledge and beliefs.
        
        Args:
            npc_id: The ID of the NPC
            relevant_lore: Dictionary of relevant lore elements
            context: Optional context dictionary
            
        Returns:
            Dict containing integration results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import integrate_npc_lore
        return await integrate_npc_lore(npc_id, relevant_lore, context)
    
    # ===== LOCATION OPERATIONS =====
    
    async def get_location_context(self, location_name: str) -> Dict[str, Any]:
        """
        Get comprehensive context for a location including lore.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Complete location context
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        enhancer = await self._get_context_enhancer()
        location_data = {"location": location_name}
        return await enhancer.enhance_context(location_data)
    
    async def generate_scene_description(self, location_name: str) -> Dict[str, Any]:
        """
        Generate an atmospheric scene description with integrated lore.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Scene description with lore elements
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        enhancer = await self._get_context_enhancer()
        return await enhancer.generate_scene_description(location_name)
    
    async def get_location_data(self, location_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve location-specific data including environment, NPCs, and lore.
        
        Args:
            location_name: Name of the location
            context: Optional context dictionary
            
        Returns:
            Dict containing location data
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_agents import get_location_data
        return await get_location_data(location_name, context)
    
    # ===== CONFLICT MANAGEMENT =====
    
    async def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """
        Get all active conflicts in the world.
        
        Returns:
            List of active conflicts
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_conflict_integration()
        
        # Query active conflicts from database
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Conflicts 
                WHERE user_id = $1 AND conversation_id = $2 
                AND status = 'active'
            """, self.user_id, self.conversation_id)
            
            conflicts = []
            for row in rows:
                conflict_id = row['id']
                conflict_lore = await integration.get_conflict_lore(conflict_id)
                conflicts.append({
                    **dict(row),
                    'lore': conflict_lore
                })
            
            return conflicts
    
    async def generate_faction_conflict(self, faction_a_id: int, faction_b_id: int) -> Dict[str, Any]:
        """
        Generate a conflict between two factions.
        
        Args:
            faction_a_id: First faction ID
            faction_b_id: Second faction ID
            
        Returns:
            Generated conflict data
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_conflict_integration()
        return await integration.generate_faction_conflict(faction_a_id, faction_b_id)
    
    async def get_faction_conflicts(self, faction_id: int) -> List[Dict[str, Any]]:
        """
        Get conflicts involving a specific faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            List of conflicts
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        integration = await self._get_conflict_integration()
        return await integration.get_faction_conflicts(faction_id)
    
    # ===== CULTURAL SYSTEMS (Original signatures preserved for backwards compatibility) =====
    
    async def generate_languages(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate languages for the world.
        
        Args:
            count: Number of languages to generate
            
        Returns:
            List of generated languages
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context()
        
        return await culture_system.generate_languages(ctx, count)
    
    async def generate_cultural_norms(self, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate cultural norms for a specific nation.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of generated cultural norms
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context(nation_id=nation_id)
        
        return await culture_system.generate_cultural_norms(ctx, nation_id)
    
    async def generate_etiquette(self, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate etiquette systems for a specific nation.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of generated etiquette systems
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context(nation_id=nation_id)
        
        return await culture_system.generate_etiquette(ctx, nation_id)
    
    async def get_nation_culture(self, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive cultural information about a nation.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with nation's cultural information
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        culture_system = await self._get_regional_culture_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context(nation_id=nation_id)
        
        return await culture_system.get_nation_culture(ctx, nation_id)
    
    # ===== NATIONAL CONFLICTS AND DOMESTIC ISSUES =====
    
    async def generate_domestic_issues(self, nation_id: int, count: int = 2) -> List[Dict[str, Any]]:
        """
        Generate domestic issues for a specific nation.
        
        Args:
            nation_id: ID of the nation
            count: Number of issues to generate
            
        Returns:
            List of generated domestic issues
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        conflict_system = await self._get_national_conflict_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context(nation_id=nation_id, count=count)
        
        return await conflict_system.generate_domestic_issues(ctx, nation_id, count)
    
    async def generate_initial_conflicts(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate initial conflicts between nations.
        
        Args:
            count: Number of conflicts to generate
            
        Returns:
            List of generated conflicts
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        conflict_system = await self._get_national_conflict_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context(count=count)
        
        return await conflict_system.generate_initial_conflicts(ctx, count)
    
    async def get_nation_issues(self, nation_id: int) -> List[Dict[str, Any]]:
        """
        Get all domestic issues for a nation.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of domestic issues
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        conflict_system = await self._get_national_conflict_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context(nation_id=nation_id)
        
        return await conflict_system.get_nation_issues(ctx, nation_id)
    
    async def evolve_all_conflicts(self, days_passed: int = 7) -> Dict[str, Any]:
        """
        Evolve all active conflicts and domestic issues over time.
        
        Args:
            days_passed: Number of days that have passed
            
        Returns:
            Evolution results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        conflict_system = await self._get_national_conflict_system()
        
        # Create mock context with the expected structure for governance
        ctx = self._create_mock_context(
            action='evolve_conflicts',
            days_passed=days_passed
        )
        
        # Note: If evolve_all_conflicts is not implemented in NationalConflictSystem,
        # this will need to be implemented or use an alternative method
        if hasattr(conflict_system, 'evolve_all_conflicts'):
            return await conflict_system.evolve_all_conflicts(ctx, days_passed)
        else:
            # Fallback: Get active conflicts and return them as-is
            logger.warning("evolve_all_conflicts not implemented, returning current conflicts")
            return {
                'conflicts': await conflict_system.get_active_conflicts(ctx),
                'evolved': False,
                'message': 'Evolution not yet implemented'
            }
    
    async def get_active_national_conflicts(self) -> List[Dict[str, Any]]:
        """
        Get all active national/international conflicts.
        
        Returns:
            List of active national conflicts
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        conflict_system = await self._get_national_conflict_system()
        
        # Create mock context for governance
        ctx = self._create_mock_context()
        
        return await conflict_system.get_active_conflicts(ctx)
    
    # ===== RELIGIOUS SYSTEMS =====
    
    async def distribute_religions(self) -> List[Dict[str, Any]]:
        """
        Distribute religions across nations.
        
        Returns:
            List of national religion distributions
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        religious_system = await self._get_religious_distribution_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context()
        
        return await religious_system.distribute_religions(ctx)
    
    async def get_nation_religion(self, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive religious information about a nation.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with nation's religious information
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        religious_system = await self._get_religious_distribution_system()
        
        # Create a mock context for governance
        ctx = self._create_mock_context(nation_id=nation_id)
        
        return await religious_system.get_nation_religion(ctx, nation_id)
    
    # ===== LORE UPDATE SYSTEM =====
    
    async def generate_lore_updates(
        self,
        affected_elements: List[Dict[str, Any]],
        event_description: str,
        player_character: Dict[str, Any] = None,
        dominant_npcs: List[Dict[str, Any]] = None,
        world_state: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate sophisticated updates for affected lore elements.
        
        Args:
            affected_elements: List of affected lore elements
            event_description: Description of the event
            player_character: Optional player character data
            dominant_npcs: Optional list of ruling NPCs
            world_state: Optional current world state data
            
        Returns:
            List of detailed updates with cascading effects
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        update_system = await self._get_lore_update_system()
        
        # Create a mock context for governance with event details
        ctx = self._create_mock_context(
            event_description=event_description[:100],  # Truncate for context
            affected_count=len(affected_elements)
        )
        
        return await update_system.generate_lore_updates(
            ctx,
            affected_elements,
            event_description,
            player_character,
            dominant_npcs,
            world_state
        )
    
    # ===== QUEST MANAGEMENT =====
    
    async def get_quest_context(self, quest_id: int) -> Dict[str, Any]:
        """
        Get comprehensive quest context including lore.
        
        Args:
            quest_id: ID of the quest
            
        Returns:
            Quest context with lore
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_quest_agent()
        return await agent.get_quest_context(quest_id)
    
    async def update_quest_stage(self, quest_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """
        Update quest progression.
        
        Args:
            quest_id: ID of the quest
            stage: New stage
            data: Stage data
            
        Returns:
            Success status
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_quest_agent()
        return await agent.update_quest_stage(quest_id, stage, data)
    
    # ===== NARRATIVE MANAGEMENT =====
    
    async def get_narrative_context(self, narrative_id: int) -> Dict[str, Any]:
        """
        Get comprehensive narrative context.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Narrative context
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_narrative_agent()
        return await agent.get_narrative_context(narrative_id)
    
    async def update_narrative_stage(self, narrative_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """
        Update narrative progression.
        
        Args:
            narrative_id: ID of the narrative
            stage: New stage
            data: Stage data
            
        Returns:
            Success status
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_narrative_agent()
        return await agent.update_narrative_stage(narrative_id, stage, data)
    
    # ===== ENVIRONMENT MANAGEMENT =====
    
    async def get_environment_context(self, location_id: int) -> Dict[str, Any]:
        """
        Get environmental context for a location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Environmental context
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_environment_agent()
        return await agent.get_environment_context(location_id)
    
    async def update_environment_state(self, location_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update environmental state.
        
        Args:
            location_id: ID of the location
            updates: State updates
            
        Returns:
            Success status
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_environment_agent()
        return await agent.update_environment_state(location_id, updates)
    
    async def update_game_time(self, time_data: Dict[str, Any]) -> bool:
        """
        Update game time and trigger related events.
        
        Args:
            time_data: Time update data
            
        Returns:
            Success status
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_environment_agent()
        return await agent.update_game_time(time_data)
    
    # ===== COMPONENT GENERATION =====
    
    async def generate_component(self, component_type: str, context: Dict[str, Any], config: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate a specific lore component.
        
        Args:
            component_type: Type of component (character, location, event)
            context: Generation context
            config: Optional component configuration
            
        Returns:
            Generated component
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.lore_generator import ComponentGeneratorFactory, ComponentConfig
        factory = await self._get_component_factory()
        
        # Convert config if needed
        if config and not isinstance(config, ComponentConfig):
            config = ComponentConfig(**config) if isinstance(config, dict) else None
        
        generator = ComponentGeneratorFactory.create_generator(
            component_type, 
            self.user_id, 
            self.conversation_id, 
            config
        )
        await generator.initialize()
        return await generator.generate(context)
    
    # ===== SETTING ANALYSIS =====
    
    async def analyze_setting(self) -> Dict[str, Any]:
        """
        Analyze the current setting and generate organizations.
        
        Returns:
            Setting analysis with generated organizations
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        analyzer = await self._get_setting_analyzer()
        
        # Create a mock context for governance
        ctx = self._create_mock_context()
        
        demographics = await analyzer.analyze_setting_demographics(ctx)
        organizations = await analyzer.generate_organizations(ctx)
        
        return {
            "demographics": demographics,
            "organizations": organizations
        }
    
    async def analyze_setting_demographics(self) -> Dict[str, Any]:
        """
        Analyze the demographics and social structure of the setting.
        
        Returns:
            Demographics analysis
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        agent = await self._get_foundation_agent()
        npc_data = await agent.aggregate_npc_data()
        return await agent.analyze_setting_demographics(npc_data)
    
    # ===== DIRECTIVE HANDLING =====
    
    async def process_directives(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Process all active directives for lore agents.
        
        Args:
            force_check: Whether to force checking directives
            
        Returns:
            Processing results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        directive_handler = await self._get_directive_handler()
        return await directive_handler.process_directives(force_check)
    
    async def check_permission(self, action_type: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action is permitted based on directives.
        
        Args:
            action_type: Type of action to check
            details: Optional action details
            
        Returns:
            Permission status dictionary
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        directive_handler = await self._get_directive_handler()
        return await directive_handler.check_permission(action_type, details)
    
    # ===== VALIDATION =====
    
    async def validate_lore_data(self, lore_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate lore data against schemas.
        
        Args:
            lore_type: Type of lore data
            data: Data to validate
            
        Returns:
            Validation results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        if not self._validation_manager:
            return {"valid": True, "message": "Validation disabled"}
        
        result = await self._validation_manager.validate(lore_type, data)
        return {
            "valid": result.is_valid,
            "errors": [e.to_dict() for e in result.errors],
            "warnings": result.warnings
        }
    
    # ===== METRICS =====
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            System metrics
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        from lore.metrics import metrics_manager
        return await metrics_manager.get_metrics_summary()
    
    # ===== RESOURCE MANAGEMENT =====
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Returns:
            Resource statistics
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        if self._resource_manager:
            return await self._resource_manager.get_resource_stats()
        return {}
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """
        Optimize resource usage.
        
        Returns:
            Optimization results
        """
        if not self.initialized and self.config.auto_initialize:
            await self.initialize()
        
        if self._resource_manager:
            return await self._resource_manager.optimize_resources()
        return {"status": "Resource management disabled"}
    
    # ===== CLEANUP =====
    
    async def cleanup(self):
        """Clean up all resources and shutdown components."""
        logger.info(f"Cleaning up Lore Orchestrator for user {self.user_id}, conversation {self.conversation_id}")
        
        # Stop monitoring
        if self._error_handler:
            await self._error_handler.stop_monitoring()
        
        if self._resource_manager:
            await self._resource_manager.stop()
        
        # Cleanup validation
        if self._validation_manager:
            await self._validation_manager.cleanup()
        
        # Cleanup components
        for component in [
            self._lore_system,
            self._matriarchal_system,
            self._dynamic_generator,
            self._npc_integration,
            self._conflict_integration,
            self._context_enhancer,
            self._regional_culture_system,
            self._national_conflict_system,
            self._religious_distribution_system,
            self._lore_update_system,
            self._matriarchal_power_framework,
            self._lore_dynamics_system
        ]:
            if component and hasattr(component, 'cleanup'):
                await component.cleanup()
        
        # Clear instance from cache
        key = (self.user_id, self.conversation_id)
        if key in _ORCHESTRATOR_INSTANCES:
            del _ORCHESTRATOR_INSTANCES[key]
        
        self.initialized = False
        logger.info("Lore Orchestrator cleanup complete")
    
    # ===== UTILITY METHODS =====
    
    def _create_mock_context(self, **attributes) -> object:
        """
        Create a mock context object for governance decorators.
        
        This utility method creates lightweight mock objects that satisfy
        the requirements of @with_governance decorated functions without
        requiring full governance context imports.
        
        Args:
            **attributes: Key-value pairs to set as object attributes
            
        Returns:
            Mock object with specified attributes
            
        Examples:
            # Simple context
            ctx = self._create_mock_context()
            
            # Context with NPC ID
            ctx = self._create_mock_context(npc_id=123)
            
            # Context with nested structure
            ctx = self._create_mock_context(
                context={'user_id': 1, 'conversation_id': 2}
            )
        """
        # Default context structure many governance decorators expect
        if 'context' not in attributes:
            attributes['context'] = {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            }
        
        # Create and return mock object with attributes
        return type('MockContext', (object,), attributes)()
    
    # ===== COMPONENT GETTERS (Lazy Initialization) =====
    
    async def _get_lore_system(self):
        """Get or initialize the core lore system."""
        if not self._lore_system:
            from lore.lore_system import LoreSystem
            self._lore_system = LoreSystem.get_instance(self.user_id, self.conversation_id)
            await self._lore_system.initialize()
        return self._lore_system
    
    async def _get_matriarchal_system(self):
        """Get or initialize the matriarchal lore system."""
        if not self._matriarchal_system:
            from lore.main import MatriarchalLoreSystem
            self._matriarchal_system = MatriarchalLoreSystem(self.user_id, self.conversation_id)
            await self._matriarchal_system.ensure_initialized()
        return self._matriarchal_system
    
    async def _get_dynamic_generator(self):
        """Get or initialize the dynamic lore generator."""
        if not self._dynamic_generator:
            from lore.lore_generator import DynamicLoreGenerator
            self._dynamic_generator = DynamicLoreGenerator.get_instance(
                self.user_id, 
                self.conversation_id,
                self._governor
            )
            await self._dynamic_generator.initialize()
        return self._dynamic_generator
    
    async def _get_setting_analyzer(self):
        """Get or initialize the setting analyzer."""
        if not self._setting_analyzer:
            from lore.setting_analyzer import SettingAnalyzer
            self._setting_analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
            await self._setting_analyzer.initialize_governance()
        return self._setting_analyzer
    
    async def _get_npc_integration(self, npc_id: Optional[int] = None):
        """Get or initialize NPC lore integration."""
        if not self._npc_integration:
            from lore.integration import NPCLoreIntegration
            self._npc_integration = NPCLoreIntegration(self.user_id, self.conversation_id, npc_id)
            self._npc_integration.governor = self._governor
            await self._npc_integration.initialize()
        return self._npc_integration
    
    async def _get_conflict_integration(self):
        """Get or initialize conflict integration."""
        if not self._conflict_integration:
            from lore.integration import ConflictIntegration
            self._conflict_integration = ConflictIntegration(self.user_id, self.conversation_id)
            self._conflict_integration.governor = self._governor
            await self._conflict_integration.initialize()
        return self._conflict_integration
    
    async def _get_context_enhancer(self):
        """Get or initialize context enhancer."""
        if not self._context_enhancer:
            from lore.integration import ContextEnhancer
            self._context_enhancer = ContextEnhancer(self.user_id, self.conversation_id)
            self._context_enhancer.governor = self._governor
            await self._context_enhancer.initialize()
        return self._context_enhancer
    
    async def _get_regional_culture_system(self):
        """Get or initialize regional culture system (NEW)."""
        if not self._regional_culture_system:
            from lore.systems.regional_culture import RegionalCultureSystem
            self._regional_culture_system = RegionalCultureSystem(self.user_id, self.conversation_id)
            await self._regional_culture_system.initialize_tables()
            await self._regional_culture_system.initialize_governance()
        return self._regional_culture_system
    
    async def _get_national_conflict_system(self):
        """Get or initialize national conflict system."""
        if not self._national_conflict_system:
            from lore.matriarchal_lore_system import NationalConflictSystem
            self._national_conflict_system = NationalConflictSystem(self.user_id, self.conversation_id)
            await self._national_conflict_system.initialize_tables()
            await self._national_conflict_system.initialize_governance()
        return self._national_conflict_system
    
    async def _get_religious_distribution_system(self):
        """Get or initialize religious distribution system."""
        if not self._religious_distribution_system:
            from lore.matriarchal_lore_system import ReligiousDistributionSystem
            self._religious_distribution_system = ReligiousDistributionSystem(self.user_id, self.conversation_id)
            await self._religious_distribution_system.initialize_tables()
            await self._religious_distribution_system.initialize_governance()
        return self._religious_distribution_system
    
    async def _get_lore_update_system(self):
        """Get or initialize lore update system."""
        if not self._lore_update_system:
            from lore.matriarchal_lore_system import LoreUpdateSystem
            self._lore_update_system = LoreUpdateSystem(self.user_id, self.conversation_id)
            await self._lore_update_system.initialize_governance()
        return self._lore_update_system
    
    async def _get_matriarchal_power_framework(self):
        """Get or initialize matriarchal power framework (NEW)."""
        if not self._matriarchal_power_framework:
            from lore.frameworks.matriarchal import MatriarchalPowerStructureFramework
            self._matriarchal_power_framework = MatriarchalPowerStructureFramework(
                self.user_id, 
                self.conversation_id
            )
            # Framework doesn't have an explicit initialize, but ensure base is initialized
            await self._matriarchal_power_framework.ensure_initialized()
        return self._matriarchal_power_framework
    
    async def _get_lore_dynamics_system(self):
        """Get or initialize lore dynamics system (NEW)."""
        if not self._lore_dynamics_system:
            from lore.systems.dynamics import LoreDynamicsSystem
            self._lore_dynamics_system = LoreDynamicsSystem(self.user_id, self.conversation_id)
            await self._lore_dynamics_system.ensure_initialized()
            # Set governor if available
            if self._governor:
                self._lore_dynamics_system.governor = self._governor
                await self._lore_dynamics_system.register_with_governance()
        return self._lore_dynamics_system
    
    async def _get_agent_context(self):
        """Get or initialize agent context."""
        if not self._agent_context:
            from lore.lore_agents import LoreAgentContext
            self._agent_context = LoreAgentContext(self.user_id, self.conversation_id)
            await self._agent_context.start()
        return self._agent_context
    
    async def _get_directive_handler(self):
        """Get or initialize directive handler."""
        if not self._directive_handler:
            from lore.lore_agents import LoreDirectiveHandler
            self._directive_handler = LoreDirectiveHandler(
                self.user_id,
                self.conversation_id,
                AgentType.NARRATIVE_CRAFTER,
                "lore_orchestrator"
            )
            await self._directive_handler.initialize()
        return self._directive_handler
    
    async def _get_quest_agent(self):
        """Get or initialize quest agent."""
        if not self._quest_agent:
            from lore.lore_agents import QuestAgent
            lore_system = await self._get_lore_system()
            self._quest_agent = QuestAgent(lore_system)
            await self._quest_agent.initialize()
        return self._quest_agent
    
    async def _get_narrative_agent(self):
        """Get or initialize narrative agent."""
        if not self._narrative_agent:
            from lore.lore_agents import NarrativeAgent
            lore_system = await self._get_lore_system()
            self._narrative_agent = NarrativeAgent(lore_system)
            await self._narrative_agent.initialize()
        return self._narrative_agent
    
    async def _get_environment_agent(self):
        """Get or initialize environment agent."""
        if not self._environment_agent:
            from lore.lore_agents import EnvironmentAgent
            lore_system = await self._get_lore_system()
            self._environment_agent = EnvironmentAgent(lore_system)
            await self._environment_agent.initialize()
        return self._environment_agent
    
    async def _get_foundation_agent(self):
        """Get or initialize foundation agent."""
        if not self._foundation_agent:
            from lore.lore_agents import FoundationAgent
            lore_system = await self._get_lore_system()
            self._foundation_agent = FoundationAgent(lore_system)
            await self._foundation_agent.initialize()
        return self._foundation_agent
    
    async def _get_faction_agent(self):
        """Get or initialize faction agent."""
        if not self._faction_agent:
            from lore.lore_agents import FactionAgent
            lore_system = await self._get_lore_system()
            self._faction_agent = FactionAgent(lore_system)
            await self._faction_agent.initialize()
        return self._faction_agent
    
    async def _get_component_factory(self):
        """Get or initialize component factory."""
        if not self._component_factory:
            from lore.lore_generator import ComponentGeneratorFactory
            self._component_factory = ComponentGeneratorFactory
        return self._component_factory


# ===== CONVENIENCE FUNCTIONS =====

async def get_lore_orchestrator(user_id: int, conversation_id: int, config: Optional[OrchestratorConfig] = None) -> LoreOrchestrator:
    """
    Get a lore orchestrator instance, initializing if needed.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        config: Optional configuration
        
    Returns:
        Initialized LoreOrchestrator instance
    """
    orchestrator = LoreOrchestrator.get_instance(user_id, conversation_id, config)
    if not orchestrator.initialized and orchestrator.config.auto_initialize:
        await orchestrator.initialize()
    return orchestrator


async def generate_world(user_id: int, conversation_id: int, environment_desc: str, **kwargs) -> Dict[str, Any]:
    """
    Quick function to generate a complete world.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        environment_desc: Environment description
        **kwargs: Additional options
        
    Returns:
        Complete world lore
    """
    orchestrator = await get_lore_orchestrator(user_id, conversation_id)
    
    # Create a mock context for governance
    ctx = orchestrator._create_mock_context()
    
    return await orchestrator.generate_complete_world(ctx, environment_desc, **kwargs)


async def evolve_world(user_id: int, conversation_id: int, event_description: str, **kwargs) -> Dict[str, Any]:
    """
    Quick function to evolve the world with an event.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        event_description: Event description
        **kwargs: Additional options
        
    Returns:
        Evolution results
    """
    orchestrator = await get_lore_orchestrator(user_id, conversation_id)
    
    # Create a mock context for governance
    ctx = orchestrator._create_mock_context()
    
    return await orchestrator.evolve_world_with_event(ctx, event_description, **kwargs)


# ===== MODULE INITIALIZATION =====

def setup_lore_orchestrator():
    """Setup function for module initialization."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Lore Orchestrator module loaded")


# Run setup on module import
setup_lore_orchestrator()
