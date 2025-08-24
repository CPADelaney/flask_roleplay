# lore/lore_orchestrator.py

"""
Lore Orchestrator - Unified Entry Point for All Lore Functionality

This module provides a single, comprehensive interface to all lore system components,
managing initialization, resource allocation, and coordination between subsystems.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set
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

# Core lore components
from lore.lore_generator import (
    DynamicLoreGenerator,
    WorldBuilder,
    FactionGenerator,
    LoreEvolution,
    ComponentGeneratorFactory,
    ComponentConfig
)
from lore.lore_system import LoreSystem
from lore.main import MatriarchalLoreSystem
from lore.setting_analyzer import SettingAnalyzer

# Integration components
from lore.integration import (
    NPCLoreIntegration,
    ConflictIntegration,
    ContextEnhancer
)

# Framework components
from lore.frameworks.matriarchal import MatriarchalPowerStructureFramework

# Management components
from lore.config import ConfigManager, LoreConfig
from lore.error_manager import ErrorHandler, LoreError, ErrorType
from lore.resource_manager import ResourceManager
from lore.metrics import MetricsManager, metrics_manager
from lore.validation import ValidationManager, validation_manager

# Data access layer
from lore.data_access import (
    NPCDataAccess,
    LocationDataAccess,
    FactionDataAccess,
    LoreKnowledgeAccess
)

# Agent components
from lore.lore_agents import (
    LoreAgentContext,
    LoreDirectiveHandler,
    BaseLoreAgent,
    QuestAgent,
    NarrativeAgent,
    EnvironmentAgent,
    FoundationAgent,
    FactionAgent,
    create_complete_lore_with_governance,
    integrate_lore_with_npcs_with_governance,
    generate_scene_description_with_lore_and_governance
)

# Schemas
from lore.unified_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput
)

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
        self._lore_system: Optional[LoreSystem] = None
        self._matriarchal_system: Optional[MatriarchalLoreSystem] = None
        self._dynamic_generator: Optional[DynamicLoreGenerator] = None
        self._setting_analyzer: Optional[SettingAnalyzer] = None
        
        # Integration components
        self._npc_integration: Optional[NPCLoreIntegration] = None
        self._conflict_integration: Optional[ConflictIntegration] = None
        self._context_enhancer: Optional[ContextEnhancer] = None
        
        # Framework components
        self._matriarchal_framework: Optional[MatriarchalPowerStructureFramework] = None
        
        # Management components
        self._config_manager: Optional[ConfigManager] = None
        self._error_handler: Optional[ErrorHandler] = None
        self._resource_manager: Optional[ResourceManager] = None
        self._validation_manager: Optional[ValidationManager] = None
        
        # Agent components
        self._agent_context: Optional[LoreAgentContext] = None
        self._directive_handler: Optional[LoreDirectiveHandler] = None
        self._quest_agent: Optional[QuestAgent] = None
        self._narrative_agent: Optional[NarrativeAgent] = None
        self._environment_agent: Optional[EnvironmentAgent] = None
        self._foundation_agent: Optional[FoundationAgent] = None
        self._faction_agent: Optional[FactionAgent] = None
        
        # Governance
        self._governor = None
        
        # Component factory
        self._component_factory: Optional[ComponentGeneratorFactory] = None
        
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
        self._config_manager = ConfigManager()
        await self._config_manager.load_config()
        self._component_init_status['config'] = True
        logger.info("Configuration management initialized")
    
    async def _initialize_error_handling(self):
        """Initialize error handling."""
        self._error_handler = ErrorHandler(self.user_id, self.conversation_id, self._config_manager.config)
        await self._error_handler.start_monitoring()
        self._component_init_status['error_handling'] = True
        logger.info("Error handling initialized")
    
    async def _initialize_resource_management(self):
        """Initialize resource management."""
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
        self._validation_manager = validation_manager
        await self._validation_manager.initialize()
        self._component_init_status['validation'] = True
        logger.info("Validation system initialized")
    
    async def _initialize_metrics(self):
        """Initialize metrics collection."""
        # Metrics manager is already a singleton
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
        ctx = type('obj', (object,), {'npc_id': npc_id})()
        
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
        ctx = type('obj', (object,), {'npc_id': npc_id})()
        
        return await integration.process_npc_lore_interaction(ctx, npc_id, player_input)
    
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
    
    # ===== COMPONENT GENERATION =====
    
    async def generate_component(self, component_type: str, context: Dict[str, Any], config: Optional[ComponentConfig] = None) -> Dict[str, Any]:
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
        
        factory = await self._get_component_factory()
        generator = factory.create_generator(component_type, self.user_id, self.conversation_id, config)
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
        ctx = type('obj', (object,), {})()
        
        demographics = await analyzer.analyze_setting_demographics(ctx)
        organizations = await analyzer.generate_organizations(ctx)
        
        return {
            "demographics": demographics,
            "organizations": organizations
        }
    
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
            self._context_enhancer
        ]:
            if component and hasattr(component, 'cleanup'):
                await component.cleanup()
        
        # Clear instance from cache
        key = (self.user_id, self.conversation_id)
        if key in _ORCHESTRATOR_INSTANCES:
            del _ORCHESTRATOR_INSTANCES[key]
        
        self.initialized = False
        logger.info("Lore Orchestrator cleanup complete")
    
    # ===== COMPONENT GETTERS (Lazy Initialization) =====
    
    async def _get_lore_system(self) -> LoreSystem:
        """Get or initialize the core lore system."""
        if not self._lore_system:
            self._lore_system = LoreSystem.get_instance(self.user_id, self.conversation_id)
            await self._lore_system.initialize()
        return self._lore_system
    
    async def _get_matriarchal_system(self) -> MatriarchalLoreSystem:
        """Get or initialize the matriarchal lore system."""
        if not self._matriarchal_system:
            self._matriarchal_system = MatriarchalLoreSystem(self.user_id, self.conversation_id)
            await self._matriarchal_system.ensure_initialized()
        return self._matriarchal_system
    
    async def _get_dynamic_generator(self) -> DynamicLoreGenerator:
        """Get or initialize the dynamic lore generator."""
        if not self._dynamic_generator:
            self._dynamic_generator = DynamicLoreGenerator.get_instance(
                self.user_id, 
                self.conversation_id,
                self._governor
            )
            await self._dynamic_generator.initialize()
        return self._dynamic_generator
    
    async def _get_setting_analyzer(self) -> SettingAnalyzer:
        """Get or initialize the setting analyzer."""
        if not self._setting_analyzer:
            self._setting_analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
            await self._setting_analyzer.initialize_governance()
        return self._setting_analyzer
    
    async def _get_npc_integration(self, npc_id: Optional[int] = None) -> NPCLoreIntegration:
        """Get or initialize NPC lore integration."""
        if not self._npc_integration:
            self._npc_integration = NPCLoreIntegration(self.user_id, self.conversation_id, npc_id)
            self._npc_integration.governor = self._governor
            await self._npc_integration.initialize()
        return self._npc_integration
    
    async def _get_conflict_integration(self) -> ConflictIntegration:
        """Get or initialize conflict integration."""
        if not self._conflict_integration:
            self._conflict_integration = ConflictIntegration(self.user_id, self.conversation_id)
            self._conflict_integration.governor = self._governor
            await self._conflict_integration.initialize()
        return self._conflict_integration
    
    async def _get_context_enhancer(self) -> ContextEnhancer:
        """Get or initialize context enhancer."""
        if not self._context_enhancer:
            self._context_enhancer = ContextEnhancer(self.user_id, self.conversation_id)
            self._context_enhancer.governor = self._governor
            await self._context_enhancer.initialize()
        return self._context_enhancer
    
    async def _get_agent_context(self) -> LoreAgentContext:
        """Get or initialize agent context."""
        if not self._agent_context:
            self._agent_context = LoreAgentContext(self.user_id, self.conversation_id)
            await self._agent_context.start()
        return self._agent_context
    
    async def _get_quest_agent(self) -> QuestAgent:
        """Get or initialize quest agent."""
        if not self._quest_agent:
            lore_system = await self._get_lore_system()
            self._quest_agent = QuestAgent(lore_system)
            await self._quest_agent.initialize()
        return self._quest_agent
    
    async def _get_narrative_agent(self) -> NarrativeAgent:
        """Get or initialize narrative agent."""
        if not self._narrative_agent:
            lore_system = await self._get_lore_system()
            self._narrative_agent = NarrativeAgent(lore_system)
            await self._narrative_agent.initialize()
        return self._narrative_agent
    
    async def _get_environment_agent(self) -> EnvironmentAgent:
        """Get or initialize environment agent."""
        if not self._environment_agent:
            lore_system = await self._get_lore_system()
            self._environment_agent = EnvironmentAgent(lore_system)
            await self._environment_agent.initialize()
        return self._environment_agent
    
    async def _get_foundation_agent(self) -> FoundationAgent:
        """Get or initialize foundation agent."""
        if not self._foundation_agent:
            lore_system = await self._get_lore_system()
            self._foundation_agent = FoundationAgent(lore_system)
            await self._foundation_agent.initialize()
        return self._foundation_agent
    
    async def _get_faction_agent(self) -> FactionAgent:
        """Get or initialize faction agent."""
        if not self._faction_agent:
            lore_system = await self._get_lore_system()
            self._faction_agent = FactionAgent(lore_system)
            await self._faction_agent.initialize()
        return self._faction_agent
    
    async def _get_component_factory(self) -> ComponentGeneratorFactory:
        """Get or initialize component factory."""
        if not self._component_factory:
            self._component_factory = ComponentGeneratorFactory()
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
    ctx = type('obj', (object,), {})()
    
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
    ctx = type('obj', (object,), {})()
    
    return await orchestrator.evolve_world_with_event(ctx, event_description, **kwargs)


# ===== MODULE INITIALIZATION =====

def setup_lore_orchestrator():
    """Setup function for module initialization."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Lore Orchestrator module loaded")


# Run setup on module import
setup_lore_orchestrator()
